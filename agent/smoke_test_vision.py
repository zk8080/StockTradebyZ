"""
smoke_test_vision.py
~~~~~~~~~~~~~~~~~~~~
最小化视觉接口冒烟测试：
- 读取本地图像
- 调用 OpenAI 兼容模型
- 解析 JSON 输出
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from model_review import ModelReviewer, load_config

REQUIRED_KEYS = {
    "trend_reasoning",
    "position_reasoning",
    "volume_reasoning",
    "abnormal_move_reasoning",
    "signal_reasoning",
    "scores",
    "total_score",
    "signal_type",
    "verdict",
    "comment",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenAI 兼容视觉接口冒烟测试")
    parser.add_argument("--image", required=True, help="本地图像路径")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent.parent / "config" / "model_review.yaml"),
        help="配置文件路径（默认 config/model_review.yaml）",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[ERROR] 图片不存在：{image_path}")
        sys.exit(1)

    config = load_config(Path(args.config))
    reviewer = ModelReviewer(config)

    result = reviewer.review_stock(code="SMOKE", day_chart=image_path, prompt=reviewer.prompt)

    missing = [k for k in REQUIRED_KEYS if k not in result]
    if missing:
        print(f"[ERROR] JSON 缺少字段：{missing}")
        sys.exit(1)

    print("[OK] 视觉调用成功，JSON 结构完整。")
    print(f"verdict={result.get('verdict')} total_score={result.get('total_score')}")


if __name__ == "__main__":
    main()
