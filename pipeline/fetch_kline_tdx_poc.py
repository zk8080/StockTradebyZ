"""TDX K线抓取 PoC（最小改造验证）

目标：
- 用 TDX 数据源拉取 A 股日线 OHLCV
- 导出到 data/raw/<code>.csv
- 使下游 pipeline.cli preselect 与 dashboard/export_kline_charts.py 可直接复用

注意：
- 这是 PoC 脚本，不改动现有 pipeline/fetch_kline.py 主流程。
- TDX 数据通常为不复权口径，与 tushare qfq 可能不同。

用法示例：
  python -m pipeline.fetch_kline_tdx_poc --codes 300215,000581,301218 --start 2020-01-01 --out data/raw

"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
from pytdx.hq import TdxHq_API


# 一组常用行情服务器（可按需补充/替换）
TDX_SERVERS: List[Tuple[str, int]] = [
    ("119.147.212.81", 7709),
    ("119.147.212.81", 7721),
    ("183.60.224.178", 7709),
    ("183.60.224.178", 7721),
    ("113.105.73.88", 7709),
    ("113.105.73.88", 7721),
]


def _market_from_code(code: str) -> int:
    """TDX market: 0=SZ, 1=SH.

    注意：在 pytdx 中，很多深市代码（0/3/30/31 开头）使用 market=0，
    沪市 6 开头使用 market=1。

    最小映射：
    - 6xxxxxx → SH (1)
    - 其他   → SZ (0)
    """
    code = str(code).zfill(6)
    return 1 if code.startswith("6") else 0


def _parse_date(s: str) -> pd.Timestamp:
    return pd.to_datetime(s)


def _connect_any(api: TdxHq_API, servers: Iterable[Tuple[str, int]]) -> Tuple[str, int]:
    last_err: Optional[Exception] = None
    for ip, port in servers:
        try:
            ok = api.connect(ip, port)
            if ok:
                return ip, port
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"无法连接任意 TDX 行情服务器，最后错误: {last_err}")


def _fetch_daily_bars(api: TdxHq_API, code: str, count: int = 800) -> pd.DataFrame:
    """抓取日线 bars。

    category=9 表示日线。start=0 通常表示从最近开始取 count 根。
    PoC 阶段先取最近 count 根，再在本地按 start_date 过滤。

    兼容性处理：不同服务端/品种可能对 market 映射有差异。
    这里先按映射尝试；若无数据，再自动切换 market 重试一次。
    """
    code = str(code).zfill(6)
    market = _market_from_code(code)

    bars = api.get_security_bars(category=9, market=market, code=code, start=0, count=count)  # type: ignore[arg-type]
    if not bars:
        alt_market = 1 - market
        bars = api.get_security_bars(category=9, market=alt_market, code=code, start=0, count=count)  # type: ignore[arg-type]

    df = pd.DataFrame(bars or [])
    if df.empty:
        return df

    # pytdx 返回字段通常包含: open, close, high, low, vol, amount, datetime
    df = df.rename(columns={"datetime": "date", "vol": "volume"})
    keep = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"TDX bars 缺少字段: {missing}; got={list(df.columns)}")

    df = df[keep].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["date"]).drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def export_codes(
    codes: List[str],
    out_dir: Path,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    bars_count: int = 800,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    api = TdxHq_API(heartbeat=True)
    ip, port = _connect_any(api, TDX_SERVERS)
    print(f"[INFO] Connected to TDX server: {ip}:{port}")

    try:
        for code in codes:
            df = _fetch_daily_bars(api, code, count=bars_count)
            if df.empty:
                print(f"[WARN] {code} 无数据，跳过")
                continue

            if start_date is not None:
                df = df[df["date"] >= start_date]
            if end_date is not None:
                df = df[df["date"] <= end_date]

            # 写出契约字段（小写）
            df_out = df[["date", "open", "high", "low", "close", "volume"]].copy()
            # date 输出为 YYYY-MM-DD，便于下游解析
            df_out["date"] = df_out["date"].dt.strftime("%Y-%m-%d")

            path = out_dir / f"{str(code).zfill(6)}.csv"
            df_out.to_csv(path, index=False)
            print(f"[OK]  {code} → {path}")
    finally:
        try:
            api.disconnect()
        except Exception:
            pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pipeline.fetch_kline_tdx_poc", description="TDX 日线抓取 PoC")
    p.add_argument("--codes", required=True, help="逗号分隔股票代码，如 300215,000581")
    p.add_argument("--out", default="data/raw", help="输出目录（默认 data/raw）")
    p.add_argument("--start", default=None, help="起始日期（YYYY-MM-DD），可选")
    p.add_argument("--end", default=None, help="结束日期（YYYY-MM-DD），可选")
    p.add_argument("--count", type=int, default=1200, help="从 TDX 拉取最近多少根日线（默认 1200）")
    return p


def main() -> None:
    args = build_parser().parse_args()
    codes = [c.strip() for c in str(args.codes).split(",") if c.strip()]
    out_dir = Path(args.out)
    start_date = _parse_date(args.start) if args.start else None
    end_date = _parse_date(args.end) if args.end else None

    export_codes(codes=codes, out_dir=out_dir, start_date=start_date, end_date=end_date, bars_count=int(args.count))


if __name__ == "__main__":
    main()
