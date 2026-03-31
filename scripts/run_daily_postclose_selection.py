#!/usr/bin/env python3
"""Run the minimum viable post-close project-side flow.

Default flow:
1. Fetch latest daily kline via tdx-api
2. Run preselect to refresh candidates_latest.json / candidates_YYYY-MM-DD.json
3. Run B1 AI review (B1 score >= 80 only)
4. Emit a machine-readable JSON summary and persist it under data/postclose/

This script is intentionally project-side only. It does NOT write Feishu tables.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable


def _run(step_name: str, cmd: list[str]) -> None:
    print(f"\n{'=' * 60}")
    print(f"[步骤] {step_name}")
    print(f"  命令: {' '.join(cmd)}")
    print(f"{'=' * 60}")
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _compute_raw_max_date(raw_dir: Path) -> str:
    max_raw_date = ""
    for path in raw_dir.glob("*.csv"):
        last = ""
        with path.open("r", encoding="utf-8") as f:
            rows = csv.DictReader(f)
            for row in rows:
                last = row.get("date") or last
        if last and last > max_raw_date:
            max_raw_date = last
    return max_raw_date


def _matched_strategies(row: dict[str, Any]) -> list[str]:
    extra = row.get("extra", {}) or {}
    matched = extra.get("matched_strategies") or []
    if matched:
        return [str(x) for x in matched]
    strategy = row.get("strategy")
    return [str(strategy)] if strategy else []


def _hits_b1(row: dict[str, Any]) -> bool:
    return "b1" in _matched_strategies(row)


def _hits_b2(row: dict[str, Any]) -> bool:
    return "b2" in _matched_strategies(row)


def _hits_combo(row: dict[str, Any]) -> bool:
    matched = set(_matched_strategies(row))
    return "b2_xg_combo" in matched or {"b2", "brick_reversal_xg"}.issubset(matched)


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _load_candidates_summary() -> dict[str, Any]:
    latest = ROOT / "data" / "candidates" / "candidates_latest.json"
    if not latest.exists():
        raise FileNotFoundError(f"missing candidates file: {latest}")

    data = _read_json(latest)
    pick_date = str(data.get("pick_date") or "")
    candidates = data.get("candidates") or []
    dated = ROOT / "data" / "candidates" / f"candidates_{pick_date}.json"

    b1_candidates = [c for c in candidates if _hits_b1(c)]
    b2_candidates = [c for c in candidates if _hits_b2(c)]
    combo_candidates = [c for c in candidates if _hits_combo(c)]
    return {
        "pick_date": pick_date,
        "candidate_count": len(candidates),
        "b1_count": len(b1_candidates),
        "b2_count": len(b2_candidates),
        "combo_count": len(combo_candidates),
        "latest_candidates_path": _rel(latest),
        "dated_candidates_path": _rel(dated),
        "dated_candidates_exists": dated.exists(),
    }


def _load_b1_review_summary(pick_date: str) -> dict[str, Any]:
    review_dir = ROOT / "data" / "review_b1" / pick_date
    suggestion = review_dir / "suggestion.json"

    reviewed_count = 0
    recommendation_count = 0
    if suggestion.exists():
        data = _read_json(suggestion)
        reviewed_count = int(data.get("total_reviewed") or 0)
        recommendation_count = len(data.get("recommendations") or [])
    else:
        reviewed_count = len([p for p in review_dir.glob("*.json") if p.name != "suggestion.json"])

    return {
        "b1_ai_review_dir": _rel(review_dir),
        "b1_ai_review_suggestion_path": _rel(suggestion),
        "b1_ai_review_done": suggestion.exists() or reviewed_count > 0,
        "b1_ai_review_count": reviewed_count,
        "b1_ai_recommendation_count": recommendation_count,
    }


def _write_summary(summary: dict[str, Any]) -> None:
    pick_date = str(summary.get("pick_date") or "unknown")
    out_dir = ROOT / "data" / "postclose" / pick_date
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "summary.json"
    summary["summary_path"] = _rel(summary_path)
    latest_path = ROOT / "data" / "postclose" / "latest_summary.json"
    summary["latest_summary_path"] = _rel(latest_path)

    payload = json.dumps(summary, ensure_ascii=False, indent=2)
    summary_path.write_text(payload, encoding="utf-8")
    latest_path.write_text(payload, encoding="utf-8")



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run project-side daily post-close flow")
    p.add_argument("--skip-fetch", action="store_true", help="Skip kline fetch step")
    p.add_argument("--skip-b1-review", action="store_true", help="Skip B1 AI review step")
    p.add_argument("--skip-export", action="store_true", help="Pass --skip-export to B1 AI review runner")
    p.add_argument("--fetch-limit", type=int, default=200, help="Fetch limit for tdx-api runner")
    p.add_argument("--fetch-concurrency", type=int, default=20, help="Fetch concurrency for tdx-api runner")
    p.add_argument("--reviewer", choices=["openai", "gemini"], default="openai", help="B1 AI reviewer")
    return p.parse_args()



def main() -> None:
    args = parse_args()

    if not args.skip_fetch:
        _run(
            "1/3 拉取最新 K 线（tdx-api fetch_kline_tdx_api）",
            [
                PYTHON,
                str(ROOT / "pipeline" / "fetch_kline_tdx_api.py"),
                "--all-a",
                "--limit",
                str(args.fetch_limit),
                "--concurrency",
                str(args.fetch_concurrency),
            ],
        )

    _run(
        "2/3 执行 preselect",
        [PYTHON, "-m", "pipeline.cli", "preselect"],
    )

    if not args.skip_b1_review:
        cmd = [
            PYTHON,
            str(ROOT / "scripts" / "run_b1_review_flow.py"),
            "--reviewer",
            args.reviewer,
        ]
        if args.skip_export:
            cmd.append("--skip-export")
        _run("3/3 执行 B1 AI 复评", cmd)

    candidate_summary = _load_candidates_summary()
    pick_date = candidate_summary["pick_date"]
    raw_max_date = _compute_raw_max_date(ROOT / "data" / "raw")

    review_summary = {
        "b1_ai_review_dir": _rel(ROOT / "data" / "review_b1" / pick_date),
        "b1_ai_review_suggestion_path": _rel(ROOT / "data" / "review_b1" / pick_date / "suggestion.json"),
        "b1_ai_review_done": False,
        "b1_ai_review_count": 0,
        "b1_ai_recommendation_count": 0,
    }
    if not args.skip_b1_review:
        review_summary = _load_b1_review_summary(pick_date)

    summary: dict[str, Any] = {
        "pick_date": pick_date,
        "raw_max_date": raw_max_date,
        "preselect_done": True,
        "candidate_count": candidate_summary["candidate_count"],
        "b1_count": candidate_summary["b1_count"],
        "b2_count": candidate_summary["b2_count"],
        "combo_count": candidate_summary["combo_count"],
        "b1_ai_review_done": review_summary["b1_ai_review_done"],
        "b1_ai_review_count": review_summary["b1_ai_review_count"],
        "b1_ai_recommendation_count": review_summary["b1_ai_recommendation_count"],
        "final_ready_for_bitable_sync": bool(candidate_summary["dated_candidates_exists"]),
        "latest_candidates_path": candidate_summary["latest_candidates_path"],
        "dated_candidates_path": candidate_summary["dated_candidates_path"],
        "b1_ai_review_dir": review_summary["b1_ai_review_dir"],
        "b1_ai_review_suggestion_path": review_summary["b1_ai_review_suggestion_path"],
        "skip_fetch": args.skip_fetch,
        "skip_b1_review": args.skip_b1_review,
        "reviewer": args.reviewer,
    }

    _write_summary(summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
