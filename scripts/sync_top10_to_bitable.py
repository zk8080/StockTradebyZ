#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REVIEW_DIR = ROOT / "data" / "review"
DEFAULT_NAME_MAP = ROOT / "data" / "meta" / "code_name_map.json"
DEFAULT_OUT_DIR = ROOT / "data" / "meta"
SH_TZ = timezone(timedelta(hours=8))


@dataclass
class RankedItem:
    rank: int
    code: str
    name: str
    score: float
    verdict: str
    signal_type: str
    comment: str
    title: str


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


def collect_ranked_items(review_dir: Path, name_map: dict[str, dict[str, str]], top_n: int) -> tuple[str, list[RankedItem], dict[str, Any]]:
    suggestion_path = review_dir / "suggestion.json"
    summary: dict[str, Any] = {"date": review_dir.name, "recommendations": []}
    if suggestion_path.exists():
        try:
            summary = load_json(suggestion_path)
        except Exception:
            summary = {"date": review_dir.name, "recommendations": []}

    # always rank from all per-stock review json, not only suggestion recommendations
    rows = []
    for p in sorted(review_dir.glob("*.json")):
        if p.name == "suggestion.json":
            continue
        try:
            rows.append(load_json(p))
        except Exception:
            continue
    if not rows:
        raise FileNotFoundError(f"No review json found under {review_dir}")
    rows.sort(key=lambda x: (x.get("total_score", 0), x.get("code", "")), reverse=True)
    pick_date = str(summary.get("date") or review_dir.name)
    items = []
    for i, r in enumerate(rows[:top_n], start=1):
        code = str(r.get("code", "")).strip()
        name = name_map.get(code, {}).get("name", "")
        items.append(
            RankedItem(
                rank=i,
                code=code,
                name=name,
                score=float(r.get("total_score", 0) or 0),
                verdict=str(r.get("verdict", "")).strip(),
                signal_type=str(r.get("signal_type", "")).strip(),
                comment=str(r.get("comment", "")).strip(),
                title=f"{pick_date}-{code}-{name}" if name else f"{pick_date}-{code}",
            )
        )
    return pick_date, items, summary


def build_payload(pick_date: str, items: list[RankedItem], summary: dict[str, Any], source: str) -> dict[str, Any]:
    ts_ms = pick_date_to_ts_ms(pick_date)
    detail_records = [
        {
            "record_title": x.title,
            "fields": {
                "记录标题": x.title,
                "选股日期": ts_ms,
                "股票代码": x.code,
                "策略类型": "b1",
                "排名": x.rank,
                "AI总分": x.score,
                "AI结论": x.verdict,
                "信号类型": x.signal_type,
                "AI点评": x.comment,
                "是否推荐": x.score >= 4.0,
                "人工备注": "",
            },
        }
        for x in items
    ]
    summary_record = {
        "fields": {
            "选股日期": ts_ms,
            "数据源": source,
            "raw数量": None,
            "候选数量": summary.get("total_reviewed") or None,
            "AI复评数量": summary.get("total_reviewed") or len(items),
            "推荐数量": len(summary.get("recommendations", [])) if isinstance(summary.get("recommendations"), list) else sum(1 for x in items if x.score >= 4.0),
            "Top1代码": items[0].code if len(items) > 0 else "",
            "Top2代码": items[1].code if len(items) > 1 else "",
            "Top3代码": items[2].code if len(items) > 2 else "",
            "备注": f"{pick_date} {source} Top{len(items)} sync payload",
        }
    }
    return {
        "pick_date": pick_date,
        "pick_date_ts_ms": ts_ms,
        "source": source,
        "top_n": len(items),
        "detail_records": detail_records,
        "summary_record": summary_record,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare deterministic TopN Bitable sync payload from review results.")
    p.add_argument("--pick-date", required=True, help="Review date, e.g. 2026-03-20")
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--review-root", default=str(DEFAULT_REVIEW_DIR))
    p.add_argument("--name-map", default=str(DEFAULT_NAME_MAP))
    p.add_argument("--source", default="tdx")
    p.add_argument("--out", default="", help="Optional output json path")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    review_dir = Path(args.review_root) / args.pick_date
    name_map = load_name_map(Path(args.name_map))
    pick_date, items, summary = collect_ranked_items(review_dir, name_map, args.top_n)
    payload = build_payload(pick_date, items, summary, args.source)

    out_path = Path(args.out) if args.out else (DEFAULT_OUT_DIR / f"bitable_top{args.top_n}_{pick_date}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"wrote {out_path}")
    print(f"pick_date={pick_date} top_n={len(items)}")
    for x in items[: min(10, len(items))]:
        print(f"{x.rank}. {x.title} score={x.score} verdict={x.verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
