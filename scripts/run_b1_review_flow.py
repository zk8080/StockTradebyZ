#!/usr/bin/env python3
"""Run B1 AI review flow.

Default flow:
1. Export latest candidate charts into data/kline/{pick_date}
2. Run B1-specific AI review with config/model_review_b1.yaml

Examples:
  python scripts/run_b1_review_flow.py
  python scripts/run_b1_review_flow.py --reviewer gemini
  python scripts/run_b1_review_flow.py --skip-export
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run B1 AI review flow")
    parser.add_argument(
        "--reviewer",
        choices=["openai", "gemini"],
        default="openai",
        help="Which AI reviewer to use (default: openai)",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip chart export and directly run review",
    )
    args = parser.parse_args()

    if not args.skip_export:
        _run(
            "1/2 导出最新候选图表",
            [PYTHON, str(ROOT / "dashboard" / "export_kline_charts.py")],
        )

    if args.reviewer == "gemini":
        config_path = ROOT / "config" / "gemini_review_b1.yaml"
        review_cmd = [PYTHON, str(ROOT / "agent" / "gemini_review.py"), "--config", str(config_path)]
    else:
        config_path = ROOT / "config" / "model_review_b1.yaml"
        review_cmd = [PYTHON, str(ROOT / "agent" / "model_review.py"), "--config", str(config_path)]

    _run(
        "2/2 执行 B1 AI 复评（仅 B1>=80）",
        review_cmd,
    )

    print("\n✅ B1 AI复评流程完成。")
    print(f"   reviewer={args.reviewer}")
    print("   scope=matched_strategies contains b1 AND b1_score >= 80")


if __name__ == "__main__":
    main()
