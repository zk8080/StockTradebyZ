"""
model_review.py
~~~~~~~~~~~~~~~
使用 OpenAI 兼容接口对候选股票进行图表分析评分。
继承自 BaseReviewer 基础架构。

用法：
    python agent/model_review.py
    python agent/model_review.py --config config/model_review.yaml

配置：
    默认读取 config/model_review.yaml。

环境变量：
    REVIEW_API_KEY 或 LC_OPENAI_API_KEY —— API Key（必填）
    REVIEW_BASE_URL —— OpenAI 兼容 API Base URL（可选）
    REVIEW_MODEL —— 覆盖配置里的 model（可选）
    REVIEW_API_STYLE —— responses | chat_completions（可选）

输出：
    ./data/review/{pick_date}/{code}.json   每支股票的评分 JSON
    ./data/review/{pick_date}/suggestion.json  汇总推荐建议
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI

from base_reviewer import BaseReviewer

# ────────────────────────────────────────────────
# 配置加载
# ────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = _ROOT / "config" / "model_review.yaml"

DEFAULT_CONFIG: dict[str, Any] = {
    # 路径参数（相对路径默认基于项目根目录）
    "candidates": "data/candidates/candidates_latest.json",
    "kline_dir": "data/kline",
    "output_dir": "data/review",
    "prompt_path": "agent/prompt.md",
    # OpenAI 兼容模型参数
    "model": "gpt-4o-mini",
    "base_url": "",
    "api_style": "responses",  # responses | chat_completions
    "request_delay": 5,
    "skip_existing": False,
    "suggest_min_score": 4.0,
}


def _resolve_cfg_path(path_like: str | Path, base_dir: Path = _ROOT) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else (base_dir / p)


def _env_first(*keys: str) -> str:
    for key in keys:
        value = os.environ.get(key, "")
        if value:
            return value
    return ""


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    cfg_path = config_path or _DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"找不到配置文件：{cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = {**DEFAULT_CONFIG, **raw}

    # 环境变量覆盖（便于 Docker 部署）
    cfg["base_url"] = _env_first("REVIEW_BASE_URL", "OPENAI_BASE_URL", "LC_OPENAI_BASE_URL") or cfg.get("base_url", "")
    cfg["model"] = _env_first("REVIEW_MODEL") or cfg.get("model", "gpt-4o-mini")
    cfg["api_style"] = _env_first("REVIEW_API_STYLE") or cfg.get("api_style", "responses")

    # BaseReviewer 依赖这些路径字段为 Path 对象
    cfg["candidates"] = _resolve_cfg_path(cfg["candidates"])
    cfg["kline_dir"] = _resolve_cfg_path(cfg["kline_dir"])
    cfg["output_dir"] = _resolve_cfg_path(cfg["output_dir"])
    cfg["prompt_path"] = _resolve_cfg_path(cfg["prompt_path"])

    return cfg


class ModelReviewer(BaseReviewer):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)

        api_key = _env_first("REVIEW_API_KEY", "LC_OPENAI_API_KEY", "OPENAI_API_KEY")
        if not api_key:
            print(
                "[ERROR] 未找到环境变量 REVIEW_API_KEY 或 LC_OPENAI_API_KEY，请先设置后重试。",
                file=sys.stderr,
            )
            sys.exit(1)

        base_url = self.config.get("base_url") or None
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    @staticmethod
    def image_to_data_url(path: Path) -> str:
        suffix = path.suffix.lower()
        mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
        mime_type = mime_map.get(suffix, "image/jpeg")
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{data}"

    @staticmethod
    def _extract_text_from_response(response: Any) -> str:
        if hasattr(response, "output_text") and response.output_text:
            return response.output_text

        # 兼容某些 OpenAI 兼容实现（responses 格式）
        try:
            output = response.output or []
            texts: list[str] = []
            for item in output:
                for content in getattr(item, "content", []) or []:
                    if getattr(content, "type", "") in {"output_text", "text"}:
                        texts.append(getattr(content, "text", ""))
            if texts:
                return "\n".join(t for t in texts if t)
        except Exception:
            pass

        # chat completions
        try:
            return response.choices[0].message.content or ""
        except Exception:
            return ""

    def _call_responses(self, prompt: str, user_text: str, image_url: str) -> str:
        response = self.client.responses.create(
            model=self.config.get("model", "gpt-4o-mini"),
            input=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "【日线图】"},
                        {"type": "input_image", "image_url": image_url},
                        {"type": "input_text", "text": user_text},
                    ],
                },
            ],
            temperature=0.2,
        )
        return self._extract_text_from_response(response)

    def _call_chat_completions(self, prompt: str, user_text: str, image_url: str) -> str:
        response = self.client.chat.completions.create(
            model=self.config.get("model", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "【日线图】"},
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": user_text},
                    ],
                },
            ],
            temperature=0.2,
        )
        return self._extract_text_from_response(response)

    def review_stock(self, code: str, day_chart: Path, prompt: str) -> dict:
        user_text = (
            f"股票代码：{code}\n\n"
            "以下是该股票的 **日线图**，请按照系统提示中的框架进行分析，"
            "并严格按照要求输出 JSON。"
        )

        image_url = self.image_to_data_url(day_chart)
        api_style = str(self.config.get("api_style", "responses")).strip().lower()

        if api_style == "chat_completions":
            response_text = self._call_chat_completions(prompt, user_text, image_url)
        else:
            response_text = self._call_responses(prompt, user_text, image_url)

        if not response_text:
            raise RuntimeError(f"模型返回空响应，无法解析 JSON（code={code}）")

        result = self.extract_json(response_text)
        result["code"] = code
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenAI 兼容图表复评")
    parser.add_argument(
        "--config",
        default=str(_DEFAULT_CONFIG_PATH),
        help="配置文件路径（默认 config/model_review.yaml）",
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))
    reviewer = ModelReviewer(config)
    reviewer.run()


if __name__ == "__main__":
    main()
