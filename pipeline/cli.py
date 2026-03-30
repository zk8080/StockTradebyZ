"""
pipeline/cli.py
统一命令行入口。

用法：
  python -m pipeline.cli preselect
  python -m pipeline.cli preselect --date 2025-12-31
  python -m pipeline.cli preselect --config config/rules_preselect.yaml --data data/raw
  python -m pipeline.cli b1-eval --code 300683 --date 2026-03-23
  python -m pipeline.cli b1-legacy-eval --code 300683 --date 2026-03-23

子命令：
  preselect        运行量化初选，写入 data/candidates/
  b1-eval          评估单只股票在指定日期的当前正式 B1（原 B1-E）
  b1-legacy-eval   评估单只股票在指定日期的 legacy/确认版旧 B1
"""
from __future__ import annotations

import argparse
import datetime
import logging
import sys
from pathlib import Path

# 将 pipeline 目录加入 path（直接用 python cli.py 时需要）
sys.path.insert(0, str(Path(__file__).parent))

from select_stock import (
    evaluate_b1_for_code,
    evaluate_b1_legacy_for_code,
    run_preselect,
    resolve_preselect_output_dir,
)
from schemas import CandidateRun
from pipeline_io import save_candidates

# ── 日志配置 ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("cli")


def _add_log_file(log_dir: str, pick_date: str) -> None:
    """可选：追加文件日志到 data/logs/pipeline_YYYY-MM-DD.log。"""
    p = Path(log_dir)
    p.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(p / f"pipeline_{pick_date}.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(fh)


# =============================================================================
# preselect 子命令
# =============================================================================

def cmd_preselect(args: argparse.Namespace) -> None:
    logger.info("===== 量化初选开始 =====")

    pick_ts, candidates = run_preselect(
        config_path=args.config or None,
        data_dir=args.data or None,
        end_date=args.end_date or None,
        pick_date=args.date or None,
    )

    pick_date_str = pick_ts.strftime("%Y-%m-%d")
    run_date_str = datetime.date.today().isoformat()

    # 可选日志文件
    if args.log_dir:
        _add_log_file(args.log_dir, pick_date_str)

    run = CandidateRun(
        run_date=run_date_str,
        pick_date=pick_date_str,
        candidates=candidates,
        meta={
            "config": args.config,
            "data_dir": args.data,
            "total": len(candidates),
        },
    )

    resolved_output_dir = resolve_preselect_output_dir(
        config_path=args.config or None,
        output_dir=args.output or None,
    )

    paths = save_candidates(
        run,
        candidates_dir=resolved_output_dir,
    )

    logger.info("===== 初选完成 =====")
    logger.info("选股日期  : %s", pick_date_str)
    logger.info("候选数量  : %d 只", len(candidates))
    for key, path in paths.items():
        logger.info("%-8s → %s", key, path)

    # 终端摘要
    if candidates:
        print(f"\n{'代码':>8}  {'策略':>6}  {'收盘价':>8}  {'砖型增长':>10}")
        print("-" * 44)
        for c in candidates:
            bg = f"{c.brick_growth:.2f}x" if c.brick_growth is not None else "  —"
            print(f"{c.code:>8}  {c.strategy:>6}  {c.close:>8.2f}  {bg:>10}")
    else:
        print("\n(今日无候选股票)")


def _print_eval_result(title: str, result: dict, strategy_note: str) -> None:
    print(f"\n[{title}]")
    print(f"strategy        : {result.get('strategy', '—')}")
    print(f"note            : {strategy_note}")
    print(f"code            : {result['code']}")
    print(f"requested_date  : {result['requested_date']}")
    print(f"resolved_date   : {result['resolved_date']}")
    print(f"hit             : {'YES' if result['hit'] else 'NO'}")
    print(f"close           : {result['close']:.3f}")
    print(f"turnover_n      : {result['turnover_n']:.3f}")
    print(f"score_total     : {result.get('score_total')}")
    print("scores          :")
    if result.get("scores"):
        for key, val in result["scores"].items():
            print(f"  - {key}: {val}")
    else:
        print("  - n/a")

    print("hard_filters    :")
    if result.get("hard_filter_reasons"):
        for reason in result["hard_filter_reasons"]:
            print(f"  - {reason}")
    else:
        print("  - PASS")

    print("metrics         :")
    for key, val in result.get("metrics", {}).items():
        print(f"  - {key}: {val}")


def cmd_b1_eval(args: argparse.Namespace) -> None:
    """评估单只股票在指定日期的当前正式 B1 命中与分数。"""
    try:
        result = evaluate_b1_for_code(
            code=args.code,
            pick_date=args.date,
            config_path=args.config or None,
            data_dir=args.data or None,
            end_date=args.end_date or None,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("B1 评估失败: %s", exc)
        sys.exit(1)

    _print_eval_result("B1 Eval", result, "当前正式 B1（原 B1-E / 启动版）")


def cmd_b1_legacy_eval(args: argparse.Namespace) -> None:
    """评估单只股票在指定日期的 legacy/确认版旧 B1。"""
    try:
        result = evaluate_b1_legacy_for_code(
            code=args.code,
            pick_date=args.date,
            config_path=args.config or None,
            data_dir=args.data or None,
            end_date=args.end_date or None,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("B1 legacy 评估失败: %s", exc)
        sys.exit(1)

    _print_eval_result("B1 Legacy Eval", result, "legacy / 确认版旧 B1，仅供对比研究")


# =============================================================================
# CLI 解析
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline.cli",
        description="AgentTrader 量化初选 CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("preselect", help="运行量化初选")
    p.add_argument("--config", default=None, help="rules_preselect.yaml 路径")
    p.add_argument("--data",   default=None, help="CSV 数据目录（覆盖配置文件）")
    p.add_argument("--date",   default=None, help="选股基准日期 YYYY-MM-DD（默认最新）")
    p.add_argument("--end-date", dest="end_date", default=None,
                   help="数据截断日期（回测用）")
    p.add_argument("--output", default=None, help="候选输出目录（默认 data/candidates/）")
    p.add_argument("--log-dir", dest="log_dir", default=None,
                   help="流水日志目录（默认 data/logs/）")

    p = sub.add_parser(
        "b1-eval",
        aliases=["b1e-eval"],
        help="评估单只股票在指定日期的当前正式 B1（原 B1-E）",
    )
    p.add_argument("--code", required=True, help="股票代码，如 300683")
    p.add_argument("--date", required=True, help="评估日期 YYYY-MM-DD")
    p.add_argument("--config", default=None, help="rules_preselect.yaml 路径")
    p.add_argument("--data", default=None, help="CSV 数据目录（覆盖配置文件）")
    p.add_argument("--end-date", dest="end_date", default=None,
                   help="数据截断日期（回看用）")

    p = sub.add_parser("b1-legacy-eval", help="评估单只股票在指定日期的 legacy/确认版旧 B1")
    p.add_argument("--code", required=True, help="股票代码，如 300683")
    p.add_argument("--date", required=True, help="评估日期 YYYY-MM-DD")
    p.add_argument("--config", default=None, help="rules_preselect.yaml 路径")
    p.add_argument("--data", default=None, help="CSV 数据目录（覆盖配置文件）")
    p.add_argument("--end-date", dest="end_date", default=None,
                   help="数据截断日期（回看用）")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "preselect":
        cmd_preselect(args)
    elif args.command in {"b1-eval", "b1e-eval"}:
        cmd_b1_eval(args)
    elif args.command == "b1-legacy-eval":
        cmd_b1_legacy_eval(args)
    else:
        parser.print_help()
        sys.exit(1)
        
def test():
    """简单测试函数，验证 CLI 逻辑（不依赖外部数据）。"""
    class Args:
        command = "preselect"
        config = None
        data = None
        date = None
        end_date = None
        output = "./data/candidates"
        log_dir = "./data/logs"

    args = Args()
    cmd_preselect(args)


if __name__ == "__main__":
    main()  
    # test()
