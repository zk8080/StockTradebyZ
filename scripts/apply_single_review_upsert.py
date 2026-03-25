#!/usr/bin/env python3
"""Build the last local hop before a real Feishu Bitable write.

OpenClaw mapping:
- operation=create -> feishu_bitable_app_table_record.create
- operation=update -> feishu_bitable_app_table_record.update
- operation=conflict -> executable=false, do not call any write tool
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_TABLE_TARGET = "bitable_single_review"
CREATE_TOOL_NAME = "feishu_bitable_app_table_record.create"
UPDATE_TOOL_NAME = "feishu_bitable_app_table_record.update"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def pick_single_item(items: Any) -> dict[str, Any]:
    if isinstance(items, list) and len(items) == 1 and isinstance(items[0], dict):
        return items[0]
    return {}


def require_fields(record: Any, context: str) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError(f"{context} missing record object")
    fields = record.get("fields", {})
    if not isinstance(fields, dict) or not fields:
        raise ValueError(f"{context} missing fields")
    return fields


def normalize_optional_text(value: str) -> str | None:
    text = str(value).strip()
    return text or None


def build_tool_arguments(
    *,
    operation: str,
    app_token: str | None,
    table_id: str | None,
    fields: dict[str, Any] | None,
    record_id: str | None,
) -> tuple[str | None, dict[str, Any] | None]:
    if operation == "create":
        return (
            CREATE_TOOL_NAME,
            {
                "app_token": app_token,
                "table_id": table_id,
                "fields": fields,
            },
        )

    if operation == "update":
        return (
            UPDATE_TOOL_NAME,
            {
                "app_token": app_token,
                "table_id": table_id,
                "record_id": record_id,
                "fields": fields,
            },
        )

    return None, None


def build_apply_output(plan: dict[str, Any], table_target: str, app_token: str | None, table_id: str | None) -> tuple[dict[str, Any], int]:
    action = str(plan.get("action", "")).strip().lower()
    pick_date = str(plan.get("pick_date", "")).strip()
    code = str(plan.get("code", "")).strip()
    record_key = str(plan.get("record_key", "")).strip()

    if action == "create":
        create = pick_single_item(plan.get("creates"))
        desired_record = plan.get("desired_record", {}) if isinstance(plan.get("desired_record"), dict) else {}
        record = create or desired_record
        fields = require_fields(record, "create plan")
        tool_name, tool_arguments = build_tool_arguments(
            operation="create",
            app_token=app_token,
            table_id=table_id,
            fields=fields,
            record_id=None,
        )
        output = {
            "action": action,
            "executable": True,
            "operation": "create",
            "table_target": table_target,
            "app_token": app_token,
            "table_id": table_id,
            "pick_date": pick_date,
            "code": code,
            "record_key": record_key,
            "record_id": None,
            "record_title": str(record.get("record_title", "")).strip() if isinstance(record, dict) else "",
            "fields": fields,
            "reason": "create_by_missing_pick_date_code",
            "tool_name": tool_name,
            "tool_arguments": tool_arguments,
        }
        return output, 0

    if action == "update":
        update = pick_single_item(plan.get("updates"))
        fields = require_fields(update, "update plan")
        record_id = str(update.get("record_id", "")).strip() if isinstance(update, dict) else ""
        if not record_id:
            raise ValueError("update plan missing record_id")
        tool_name, tool_arguments = build_tool_arguments(
            operation="update",
            app_token=app_token,
            table_id=table_id,
            fields=fields,
            record_id=record_id,
        )
        output = {
            "action": action,
            "executable": True,
            "operation": "update",
            "table_target": table_target,
            "app_token": app_token,
            "table_id": table_id,
            "pick_date": pick_date,
            "code": code,
            "record_key": record_key,
            "record_id": record_id,
            "record_title": "",
            "fields": fields,
            "reason": str(update.get("reason", "")).strip() if isinstance(update, dict) else "update_by_pick_date_code",
            "tool_name": tool_name,
            "tool_arguments": tool_arguments,
        }
        return output, 0

    if action == "conflict":
        conflicts = plan.get("conflicts", []) if isinstance(plan.get("conflicts"), list) else []
        reason = "multiple_records_for_pick_date_code"
        if conflicts and isinstance(conflicts[0], dict):
            reason = str(conflicts[0].get("reason", "")).strip() or reason
        output = {
            "action": action,
            "executable": False,
            "operation": "conflict",
            "table_target": table_target,
            "app_token": app_token,
            "table_id": table_id,
            "pick_date": pick_date,
            "code": code,
            "record_key": record_key,
            "record_id": None,
            "record_title": "",
            "fields": None,
            "reason": reason,
            "tool_name": None,
            "tool_arguments": None,
            "conflicts": conflicts,
        }
        return output, 2

    raise ValueError(f"unsupported plan action: {action or '<empty>'}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert single-review upsert plan into a standardized local apply instruction."
    )
    p.add_argument("--plan", required=True, help="Path to single-review plan json")
    p.add_argument("--out", default="", help="Output apply json path")
    p.add_argument("--table-target", default=DEFAULT_TABLE_TARGET, help="Logical target table name")
    p.add_argument("--app-token", default="", help="Optional Feishu app token placeholder for downstream tool mapping")
    p.add_argument("--table-id", default="", help="Optional Feishu table id placeholder for downstream tool mapping")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    plan_path = Path(args.plan)
    out_path = Path(args.out) if args.out else plan_path.with_name(plan_path.stem + "_apply.json")

    plan = load_json(plan_path)
    apply_output, exit_code = build_apply_output(
        plan,
        table_target=args.table_target,
        app_token=normalize_optional_text(args.app_token),
        table_id=normalize_optional_text(args.table_id),
    )

    out_path.write_text(json.dumps(apply_output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    print(
        f"operation={apply_output['operation']} executable={str(apply_output['executable']).lower()} "
        f"reason={apply_output['reason']}"
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
