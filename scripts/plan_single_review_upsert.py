#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

SH_TZ = timezone(timedelta(hours=8))


@dataclass
class ExistingRow:
    record_id: str
    pick_date: str
    code: str
    title: str


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def timestamp_to_pick_date(value: float) -> str:
    ts = float(value)
    if abs(ts) >= 1_000_000_000_000:
        ts /= 1000.0
    return datetime.fromtimestamp(ts, tz=SH_TZ).date().isoformat()


def normalize_pick_date(value: Any) -> str:
    if value is None or isinstance(value, bool):
        return ""
    if isinstance(value, (int, float)):
        return timestamp_to_pick_date(value)

    text = str(value).strip()
    if not text:
        return ""
    if text.isdigit():
        return timestamp_to_pick_date(float(text))

    normalized = text.replace("/", "-")
    try:
        return datetime.strptime(normalized[:10], "%Y-%m-%d").date().isoformat()
    except ValueError:
        pass

    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return normalized

    if dt.tzinfo is None:
        return dt.date().isoformat()
    return dt.astimezone(SH_TZ).date().isoformat()


def build_record_key(pick_date: str, code: str) -> str:
    return f"{pick_date}:{code}"


def parse_existing(existing_path: Path) -> dict[str, list[ExistingRow]]:
    if not existing_path.exists():
        return {}

    payload = load_json(existing_path)
    if isinstance(payload, dict):
        records = payload.get("records", [])
    elif isinstance(payload, list):
        records = payload
    else:
        records = []
    out: dict[str, list[ExistingRow]] = {}
    for item in records:
        if not isinstance(item, dict):
            continue
        fields = item.get("fields", {})
        if not isinstance(fields, dict):
            continue

        pick_date = normalize_pick_date(fields.get("点评日期"))
        code = str(fields.get("股票代码", "")).strip()
        if not pick_date or not code:
            continue

        key = build_record_key(pick_date, code)
        out.setdefault(key, []).append(
            ExistingRow(
                record_id=str(item.get("record_id") or item.get("id") or "").strip(),
                pick_date=pick_date,
                code=code,
                title=str(fields.get("记录标题", "")).strip(),
            )
        )
    return out


def parse_desired(desired: dict[str, Any]) -> tuple[str, str, str, dict[str, Any]]:
    record = desired.get("record", {})
    if not isinstance(record, dict):
        raise ValueError("desired payload missing record object")
    fields = record.get("fields", {})
    if not isinstance(fields, dict):
        raise ValueError("desired payload record missing fields object")

    pick_date = str(desired.get("pick_date", "")).strip() or normalize_pick_date(fields.get("点评日期"))
    code = str(desired.get("code", "")).strip() or str(fields.get("股票代码", "")).strip()
    if not pick_date:
        raise ValueError("desired payload missing pick_date / 点评日期")
    if not code:
        raise ValueError("desired payload missing code / 股票代码")

    record_key = str(desired.get("record_key", "")).strip() or build_record_key(pick_date, code)
    return pick_date, code, record_key, record


def build_plan(desired: dict[str, Any], existing_by_key: dict[str, list[ExistingRow]]) -> dict[str, Any]:
    pick_date, code, record_key, record = parse_desired(desired)
    matches = existing_by_key.get(record_key, [])

    creates: list[dict[str, Any]] = []
    updates: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []

    if not matches:
        action = "create"
        creates.append(record)
    elif len(matches) == 1:
        action = "update"
        updates.append(
            {
                "record_id": matches[0].record_id,
                "fields": record.get("fields", {}),
                "reason": "update_by_pick_date_code",
            }
        )
    else:
        action = "conflict"
        conflicts = [
            {
                "record_id": row.record_id,
                "pick_date": row.pick_date,
                "code": row.code,
                "title": row.title,
                "reason": "multiple_records_for_pick_date_code",
            }
            for row in matches
        ]

    return {
        "pick_date": pick_date,
        "code": code,
        "record_key": record_key,
        "action": action,
        "desired_record": record,
        "creates": creates,
        "updates": updates,
        "conflicts": conflicts,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare deterministic single-review Bitable upsert plan from desired payload + existing snapshot."
    )
    p.add_argument("--desired", required=True, help="Path to bitable_single_review payload json")
    p.add_argument("--existing", required=True, help="Path to existing snapshot json (exported/list result)")
    p.add_argument("--out", default="", help="Output plan path")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    desired_path = Path(args.desired)
    existing_path = Path(args.existing)
    out_path = Path(args.out) if args.out else desired_path.with_name(desired_path.stem + "_plan.json")

    desired = load_json(desired_path)
    existing_by_key = parse_existing(existing_path)
    plan = build_plan(desired, existing_by_key)

    out_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    print(
        f"action={plan['action']} creates={len(plan['creates'])} "
        f"updates={len(plan['updates'])} conflicts={len(plan['conflicts'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
