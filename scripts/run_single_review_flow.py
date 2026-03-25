#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="单股 AI 点评闭环流程入口（点评结果 + Bitable payload）")
    p.add_argument("--code", required=True, help="股票代码，如 600644")
    p.add_argument("--pick-date", required=True, help="点评日期，如 2026-03-20")
    p.add_argument("--config", default=str(ROOT / "config" / "model_review.yaml"))
    p.add_argument("--raw-dir", default=str(ROOT / "data" / "raw"))
    p.add_argument("--kline-dir", default=str(ROOT / "data" / "kline"))
    p.add_argument("--review-root", default=str(ROOT / "data" / "review_single"))
    p.add_argument("--name-map", default=str(ROOT / "data" / "meta" / "code_name_map.json"))
    p.add_argument("--output-review", default="", help="可选：单股点评 JSON 输出路径")
    p.add_argument("--output-payload", default="", help="可选：Bitable payload 输出路径")
    p.add_argument("--bars", type=int, default=120, help="自动导图时的日线 bars 数")
    return p.parse_args()


def run(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    args = parse_args()

    review_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "review_single_stock.py"),
        "--code", args.code,
        "--pick-date", args.pick_date,
        "--config", args.config,
        "--raw-dir", args.raw_dir,
        "--kline-dir", args.kline_dir,
        "--bars", str(args.bars),
    ]
    if args.output_review:
        review_cmd.extend(["--output", args.output_review])

    payload_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "sync_single_review_to_bitable.py"),
        "--code", args.code,
        "--pick-date", args.pick_date,
        "--review-root", args.review_root,
        "--name-map", args.name_map,
    ]
    if args.output_payload:
        payload_cmd.extend(["--out", args.output_payload])

    run(review_cmd)
    run(payload_cmd)

    print("[DONE] 单股 AI 点评结果与 Bitable payload 已生成。")
    print("[NOTE] 真正写入飞书多维表格仍由主流程执行。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
