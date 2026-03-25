#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ExistingRow:
    record_id: str
    rank: int
    title: str
    code: str


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_existing(existing_path: Path) -> dict[int, ExistingRow]:
    if not existing_path.exists():
        return {}
    payload = load_json(existing_path)
    records = payload.get("records", payload if isinstance(payload, list) else [])
    out: dict[int, ExistingRow] = {}
    for item in records:
        fields = item.get("fields", {}) if isinstance(item, dict) else {}
        rank = int(fields.get("排名", 0) or 0)
        if rank <= 0:
            continue
        out[rank] = ExistingRow(
            record_id=str(item.get("record_id") or item.get("id") or "").strip(),
            rank=rank,
            title=str(fields.get("记录标题", "")).strip(),
            code=str(fields.get("股票代码", "")).strip(),
        )
    return out


def build_plan(desired: dict[str, Any], existing_by_rank: dict[int, ExistingRow]) -> dict[str, Any]:
    desired_rows = desired.get("detail_records", [])
    creates = []
    updates = []
    desired_ranks = set()

    for row in desired_rows:
        fields = row.get("fields", {})
        rank = int(fields.get("排名", 0) or 0)
        if rank <= 0:
            continue
        desired_ranks.add(rank)
        existing = existing_by_rank.get(rank)
        if not existing:
            creates.append(row)
            continue
        updates.append(
            {
                "record_id": existing.record_id,
                "fields": fields,
                "reason": "update_by_rank",
            }
        )

    deletes = []
    for rank, row in existing_by_rank.items():
        if rank not in desired_ranks:
            deletes.append(
                {
                    "record_id": row.record_id,
                    "rank": rank,
                    "title": row.title,
                    "reason": "rank_out_of_top_n",
                }
            )

    return {
        "pick_date": desired.get("pick_date"),
        "top_n": desired.get("top_n"),
        "creates": creates,
        "updates": updates,
        "deletes": deletes,
        "summary_record": desired.get("summary_record"),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare deterministic Bitable upsert plan from desired payload + existing snapshot.")
    p.add_argument("--desired", required=True, help="Path to bitable_topN payload json")
    p.add_argument("--existing", required=True, help="Path to existing snapshot json (exported/list result)")
    p.add_argument("--out", default="", help="Output plan path")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    desired_path = Path(args.desired)
    existing_path = Path(args.existing)
    out_path = Path(args.out) if args.out else desired_path.with_name(desired_path.stem + "_plan.json")

    desired = load_json(desired_path)
    existing_by_rank = parse_existing(existing_path)
    plan = build_plan(desired, existing_by_rank)

    out_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    print(f"creates={len(plan['creates'])} updates={len(plan['updates'])} deletes={len(plan['deletes'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
