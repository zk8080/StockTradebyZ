#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REVIEW_ROOT = ROOT / "data" / "review_single"
DEFAULT_NAME_MAP = ROOT / "data" / "meta" / "code_name_map.json"
DEFAULT_OUT_DIR = ROOT / "data" / "meta"
SH_TZ = timezone(timedelta(hours=8))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def pick_date_to_ts_ms(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=SH_TZ)
    return int(dt.timestamp() * 1000)


def load_name_map(path: Path) -> dict[str, dict[str, str]]:
    data = load_json(path)
    out: dict[str, dict[str, str]] = {}
    for code, item in data.items():
        if isinstance(item, dict):
            out[str(code)] = {
                "name": str(item.get("name", "")).strip(),
                "exchange": str(item.get("exchange", "")).strip(),
            }
        else:
            out[str(code)] = {"name": str(item).strip(), "exchange": ""}
    return out


def parse_market(exchange: str) -> str:
    exchange = exchange.lower().strip()
    if exchange in {"sh", "sz", "bj"}:
        return "A股"
    if exchange in {"hk"}:
        return "港股"
    if exchange in {"us", "nasdaq", "nyse"}:
        return "美股"
    return "A股"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Bitable payload for single-stock AI review result.")
    p.add_argument("--code", required=True)
    p.add_argument("--pick-date", required=True)
    p.add_argument("--review-root", default=str(DEFAULT_REVIEW_ROOT))
    p.add_argument("--name-map", default=str(DEFAULT_NAME_MAP))
    p.add_argument("--out", default="")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    code = args.code.strip()
    pick_date = args.pick_date.strip()
    review_path = Path(args.review_root) / pick_date / f"{code}.json"
    if not review_path.exists():
        raise FileNotFoundError(f"single review json not found: {review_path}")

    review = load_json(review_path)
    name_map = load_name_map(Path(args.name_map))
    meta = name_map.get(code, {})
    name = meta.get("name", "")
    market = parse_market(meta.get("exchange", ""))
    title = f"{pick_date}-{code}-{name}" if name else f"{pick_date}-{code}"
    ts_ms = pick_date_to_ts_ms(pick_date)

    payload = {
        "pick_date": pick_date,
        "pick_date_ts_ms": ts_ms,
        "code": code,
        "record_key": f"{pick_date}:{code}",
        "record": {
            "record_title": title,
            "fields": {
                "记录标题": title,
                "点评日期": ts_ms,
                "股票代码": code,
                "股票名称": name,
                "市场": market,
                "AI总分": float(review.get("total_score", 0) or 0),
                "AI结论": str(review.get("verdict", "")).strip(),
                "信号类型": str(review.get("signal_type", "")).strip(),
                "AI点评": str(review.get("comment", "")).strip(),
                "人工备注": "",
            },
        },
    }

    out_path = Path(args.out) if args.out else (DEFAULT_OUT_DIR / f"bitable_single_review_{pick_date}_{code}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"wrote {out_path}")
    print(f"title={title}")
    print(f"score={payload['record']['fields']['AI总分']} verdict={payload['record']['fields']['AI结论']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
