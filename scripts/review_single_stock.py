#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agent"))
sys.path.insert(0, str(ROOT / "dashboard"))

from agent.model_review import ModelReviewer, load_config  # noqa: E402
from dashboard.components.charts import make_daily_chart  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="单股 AI 点评入口（图不存在时自动导图）")
    p.add_argument("--code", required=True, help="股票代码，如 600644")
    p.add_argument("--pick-date", required=True, help="点评日期，如 2026-03-20")
    p.add_argument("--config", default=str(ROOT / "config" / "model_review.yaml"))
    p.add_argument("--raw-dir", default=str(ROOT / "data" / "raw"))
    p.add_argument("--kline-dir", default=str(ROOT / "data" / "kline"))
    p.add_argument("--output", default="", help="输出文件路径（默认 data/review_single/<date>/<code>.json）")
    p.add_argument("--bars", type=int, default=120, help="自动导图时的日线 bars 数（默认 120）")
    return p.parse_args()


def load_raw(code: str, raw_dir: Path) -> pd.DataFrame:
    csv_path = raw_dir / f"{code}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"raw 数据不存在：{csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]
    if "date" not in df.columns:
        raise ValueError(f"raw 文件缺少 date 字段：{csv_path}")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def ensure_day_chart(code: str, pick_date: str, raw_dir: Path, kline_dir: Path, bars: int) -> Path:
    date_dir = kline_dir / pick_date
    jpg = date_dir / f"{code}_day.jpg"
    png = date_dir / f"{code}_day.png"
    if jpg.exists():
        return jpg
    if png.exists():
        return png

    df = load_raw(code, raw_dir)
    fig = make_daily_chart(df, code, bars=bars, height=700)
    date_dir.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(jpg), format="jpg", width=1400, height=700, scale=2)
    return jpg


def main() -> int:
    args = parse_args()
    code = args.code.strip()
    pick_date = args.pick_date.strip()
    raw_dir = Path(args.raw_dir)
    kline_dir = Path(args.kline_dir)

    cfg = load_config(Path(args.config))
    reviewer = ModelReviewer(cfg)

    day_chart = ensure_day_chart(code, pick_date, raw_dir, kline_dir, args.bars)
    result = reviewer.review_stock(code=code, day_chart=day_chart, prompt=reviewer.prompt)
    result["pick_date"] = pick_date

    out_path = Path(args.output) if args.output else (ROOT / "data" / "review_single" / pick_date / f"{code}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[INFO] code={code} date={pick_date}")
    print(f"[INFO] chart={day_chart}")
    print(f"[INFO] score={result.get('total_score')} verdict={result.get('verdict')} signal={result.get('signal_type')}")
    print(f"[INFO] output={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
