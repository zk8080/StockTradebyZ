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
from dashboard.components.charts import make_daily_chart, prepare_daily_indicators  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="单股 AI 点评入口（图不存在时自动导图）")
    p.add_argument("--code", required=True, help="股票代码，如 600644")
    p.add_argument("--pick-date", required=True, help="点评日期，如 2026-03-20")
    p.add_argument("--config", default=str(ROOT / "config" / "model_review.yaml"))
    p.add_argument("--raw-dir", default=str(ROOT / "data" / "raw"))
    p.add_argument("--kline-dir", default=str(ROOT / "data" / "kline"))
    p.add_argument("--output", default="", help="输出文件路径（默认 data/review_single/<date>/<code>.json）")
    p.add_argument("--bars", type=int, default=120, help="自动导图时的日线 bars 数（默认 120）")
    p.add_argument("--reuse-existing-chart", action="store_true", help="若已存在点评日期图表则直接复用")
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


def slice_to_pick_date(df: pd.DataFrame, pick_date: str) -> pd.DataFrame:
    cutoff = pd.Timestamp(pick_date)
    sliced = df[df["date"] <= cutoff].copy().reset_index(drop=True)
    if sliced.empty:
        raise ValueError(f"{pick_date} 之前无可用日线数据")
    return sliced


def write_simple_daily_chart(df: pd.DataFrame, code: str, pick_date: str, out_path: Path, bars: int) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    width = 1400
    height = 700
    left = 60
    right = 20
    top = 40
    price_height = 470
    volume_height = 120
    gap = 20
    volume_top = top + price_height + gap

    df = prepare_daily_indicators(df)
    if bars > 0:
        df = df.tail(bars).reset_index(drop=True)

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    n = max(len(df), 1)
    bar_w = max(3, (width - left - right) // n)
    xs = [left + i * bar_w + bar_w // 2 for i in range(len(df))]

    zxdq = df["_zxdq"].dropna()
    zxdkx = df["_zxdkx"].dropna()
    price_min = float(df["low"].min())
    price_max = float(df["high"].max())
    if not zxdq.empty:
        price_max = max(price_max, float(zxdq.max()))
    if not zxdkx.empty:
        price_min = min(price_min, float(zxdkx.min()))
        price_max = max(price_max, float(zxdkx.max()))
    price_pad = (price_max - price_min) * 0.06 or 1.0
    price_min -= price_pad
    price_max += price_pad
    volume_max = float(df["volume"].max()) * 1.1 or 1.0

    def y_price(value: float) -> float:
        return top + (price_max - value) / (price_max - price_min) * price_height

    def y_volume(value: float) -> float:
        return volume_top + volume_height - (value / volume_max) * volume_height

    draw.text((left, 10), f"{code} Daily to {pick_date} bars={len(df)}", fill="black", font=font)
    draw.rectangle([left, top, width - right, top + price_height], outline="#bbbbbb", width=1)
    draw.rectangle([left, volume_top, width - right, volume_top + volume_height], outline="#bbbbbb", width=1)

    # Keep a minimal but readable fallback chart for retrospective sample checks.
    for column, color, line_width in (("_zxdq", "#e67e22", 2), ("_zxdkx", "#2980b9", 2)):
        points: list[tuple[int, float]] = []
        for x, (_, row) in zip(xs, df.iterrows()):
            value = row[column]
            if pd.notna(value):
                points.append((x, y_price(float(value))))
        if len(points) >= 2:
            draw.line(points, fill=color, width=line_width)

    for x, (_, row) in zip(xs, df.iterrows()):
        open_px = float(row["open"])
        high_px = float(row["high"])
        low_px = float(row["low"])
        close_px = float(row["close"])
        volume = float(row["volume"])
        color = "#d62728" if close_px >= open_px else "#2ca02c"
        draw.line((x, y_price(high_px), x, y_price(low_px)), fill=color, width=1)
        half = max(1, bar_w // 3)
        body_top = y_price(max(open_px, close_px))
        body_bottom = y_price(min(open_px, close_px))
        if abs(body_top - body_bottom) < 1:
            body_bottom = body_top + 1
        draw.rectangle((x - half, body_top, x + half, body_bottom), outline=color, fill=color)
        draw.rectangle((x - half, y_volume(volume), x + half, volume_top + volume_height), outline=color, fill=color)

    for idx in {0, len(df) // 2, len(df) - 1}:
        if 0 <= idx < len(df):
            label = df.iloc[idx]["date"].strftime("%Y-%m-%d")
            draw.text((xs[idx] - 25, volume_top + volume_height + 5), label, fill="black", font=font)

    legend_y = top + 5
    for i, (label, color) in enumerate((("zxdq", "#e67e22"), ("zxdkx", "#2980b9"))):
        legend_x = width - right - 180 + i * 85
        draw.line((legend_x, legend_y + 8, legend_x + 18, legend_y + 8), fill=color, width=2)
        draw.text((legend_x + 24, legend_y + 2), label, fill="black", font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


def ensure_day_chart(code: str, pick_date: str, raw_dir: Path, kline_dir: Path, bars: int, reuse_existing_chart: bool) -> Path:
    date_dir = kline_dir / pick_date
    jpg = date_dir / f"{code}_day.jpg"
    png = date_dir / f"{code}_day.png"
    if reuse_existing_chart and jpg.exists():
        return jpg
    if reuse_existing_chart and png.exists():
        return png

    df = slice_to_pick_date(load_raw(code, raw_dir), pick_date)
    date_dir.mkdir(parents=True, exist_ok=True)

    try:
        fig = make_daily_chart(df, code, bars=bars, height=700)
        fig.write_image(str(jpg), format="jpg", width=1400, height=700, scale=2)
        return jpg
    except Exception as exc:
        print(f"[WARN] Plotly 导图失败，回退到 Pillow 简图：{exc}", file=sys.stderr)
        return write_simple_daily_chart(df, code, pick_date, png, bars)


def main() -> int:
    args = parse_args()
    code = args.code.strip()
    pick_date = args.pick_date.strip()
    raw_dir = Path(args.raw_dir)
    kline_dir = Path(args.kline_dir)

    cfg = load_config(Path(args.config))
    reviewer = ModelReviewer(cfg)

    day_chart = ensure_day_chart(code, pick_date, raw_dir, kline_dir, args.bars, args.reuse_existing_chart)
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
