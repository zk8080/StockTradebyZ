"""
base_reviewer.py
~~~~~~~~~~~~~~~~
提供 LLM 图表分析的基础架构：
- 加载配置和 prompt
- 读取候选股票列表
- 按配置筛选候选股票
- 查找本地 K 线图
- 遍历调用子类实现的单股评分模型
- 结果汇总和输出
"""

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class BaseReviewer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.prompt = self.load_prompt(Path(config["prompt_path"]))
        self.kline_dir = Path(config["kline_dir"])
        self.output_dir = Path(config["output_dir"])

    @staticmethod
    def load_prompt(prompt_path: Path) -> str:
        return prompt_path.read_text(encoding="utf-8")

    @staticmethod
    def load_candidates(path: Path) -> dict:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        try:
            if value is None or value == "":
                return None
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _contains_any(text: str, keywords: list[str]) -> bool:
        if not text:
            return False
        return any(keyword in text for keyword in keywords)

    @staticmethod
    def _contains_unnegated(text: str, keywords: list[str]) -> bool:
        if not text:
            return False
        negations = ("不", "没", "无", "未", "非", "并非", "不是", "不属", "不算", "没有", "未见", "并不", "并无", "不能")
        for keyword in keywords:
            start = 0
            while True:
                idx = text.find(keyword, start)
                if idx == -1:
                    break
                prefix = text[max(0, idx - 16):idx]
                if not any(neg in prefix for neg in negations):
                    return True
                start = idx + len(keyword)
        return False

    @staticmethod
    def _matched_strategies(candidate: dict) -> list[str]:
        extra = candidate.get("extra", {}) or {}
        matched = extra.get("matched_strategies") or []
        if matched:
            return [str(x) for x in matched]
        strategy = candidate.get("strategy")
        return [str(strategy)] if strategy else []

    @staticmethod
    def _strategy_match(candidate: dict, required: list[str], mode: str = "all") -> bool:
        if not required:
            return True
        matched = set(BaseReviewer._matched_strategies(candidate))
        required_set = {str(x) for x in required}
        if mode == "any":
            return bool(matched & required_set)
        return required_set.issubset(matched)

    @staticmethod
    def _score_sort_key(candidate: dict, score_field: str) -> tuple[float, str]:
        extra = candidate.get("extra", {}) or {}
        score = BaseReviewer._safe_float(extra.get(score_field))
        if score is None:
            score = -999.0
        return (-score, str(candidate.get("code", "")))

    def filter_candidates(self, candidates: List[dict]) -> List[dict]:
        filtered = list(candidates)
        original_count = len(filtered)

        required = [str(x) for x in (self.config.get("required_matched_strategies") or [])]
        strategy_mode = str(self.config.get("matched_strategy_mode", "all")).strip().lower() or "all"
        if required:
            filtered = [c for c in filtered if self._strategy_match(c, required, strategy_mode)]

        min_b1_score = self._safe_float(self.config.get("min_b1_score"))
        if min_b1_score is not None:
            filtered = [
                c for c in filtered
                if (self._safe_float((c.get("extra", {}) or {}).get("b1_score")) or -999.0) >= min_b1_score
            ]

        sort_by = str(self.config.get("sort_by", "")).strip().lower()
        if sort_by == "b1_score_desc":
            filtered.sort(key=lambda c: self._score_sort_key(c, "b1_score"))
        elif sort_by == "total_score_desc":
            filtered.sort(key=lambda c: self._score_sort_key(c, "total_score"))

        max_candidates = self.config.get("max_candidates")
        if max_candidates is not None:
            try:
                max_n = int(max_candidates)
                if max_n > 0:
                    filtered = filtered[:max_n]
            except Exception:
                pass

        if required or min_b1_score is not None or max_candidates is not None or sort_by:
            print(
                "[INFO] 候选过滤完成："
                f"原始 {original_count} 支 -> 保留 {len(filtered)} 支 | "
                f"required_matched_strategies={required or '[]'} | "
                f"min_b1_score={min_b1_score if min_b1_score is not None else 'None'} | "
                f"sort_by={sort_by or 'None'} | "
                f"max_candidates={max_candidates if max_candidates is not None else 'None'}"
            )

        return filtered

    def find_chart_images(self, pick_date: str, code: str) -> Optional[Path]:
        date_dir = self.kline_dir / pick_date
        day_chart = date_dir / f"{code}_day.jpg"
        if not day_chart.exists():
            day_chart_png = date_dir / f"{code}_day.png"
            day_chart = day_chart_png if day_chart_png.exists() else None
        return day_chart

    @staticmethod
    def extract_json(text: str) -> dict:
        """Extract the first JSON object from model output.

        Tolerates common LLM formatting issues:
        - ```json code blocks
        - leading/trailing text
        - trailing commas
        - single quotes (best-effort)
        """

        code_block = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if code_block:
            text = code_block.group(1)

        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError(f"未能在模型输出中找到 JSON 对象:\n{text}")
        candidate = text[start:end].strip()

        try:
            return json.loads(candidate)
        except Exception:
            pass

        repaired = candidate
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        if "'" in repaired and '"' not in repaired:
            repaired = repaired.replace("'", '"')

        try:
            return json.loads(repaired)
        except Exception as e:
            raise ValueError(f"模型输出 JSON 解析失败: {e}\n原始片段:\n{candidate}\n\n修复后片段:\n{repaired}")

    def normalize_result(self, result: dict) -> dict:
        """Normalize model output with lightweight deterministic post-processing.

        Current uses:
        - Recompute total_score from configured score_weights when all dimension scores exist
        - Keep priority aligned with verdict
        - Keep signal_type aligned with pattern_label for known B1 labels
        """
        if not isinstance(result, dict):
            return result

        weights = self.config.get("score_weights") or {}
        scores = result.get("scores") or {}
        if isinstance(weights, dict) and isinstance(scores, dict) and weights:
            weighted_total = 0.0
            valid = True
            for key, weight in weights.items():
                score_value = self._safe_float(scores.get(key))
                weight_value = self._safe_float(weight)
                if score_value is None or weight_value is None:
                    valid = False
                    break
                weighted_total += score_value * weight_value
            if valid:
                result["total_score"] = round(weighted_total, 1)

        verdict = str(result.get("verdict", "")).strip().upper()
        if verdict in {"A", "B", "C"}:
            result["verdict"] = verdict
            result["priority"] = verdict

        review_schema = str(self.config.get("review_schema", "")).strip().lower()
        pattern_label = str(result.get("pattern_label", "")).strip()
        signal_map = {
            "standard_b1": "b1_standard",
            "borderline_b1": "b1_borderline",
            "false_b1": "b1_false_start",
        }
        if review_schema == "b1":
            result = self._normalize_b1_result(result)
            pattern_label = str(result.get("pattern_label", "")).strip()
        if review_schema == "b1" and pattern_label in signal_map:
            result["signal_type"] = signal_map[pattern_label]

        return result

    def _normalize_b1_result(self, result: dict) -> dict:
        scores = result.get("scores") or {}
        if not isinstance(scores, dict):
            return result

        cfg = self.config.get("b1_label_consistency") or {}
        min_total_for_standard = self._safe_float(cfg.get("promote_standard_min_total_score")) or 3.8
        min_similarity_for_standard = self._safe_float(cfg.get("promote_standard_min_similarity")) or 4.0
        min_pattern_fit = self._safe_float(cfg.get("promote_standard_min_pattern_fit")) or 4.0
        min_volume_quality = self._safe_float(cfg.get("promote_standard_min_volume_pattern_quality")) or 4.0
        min_restart_momentum = self._safe_float(cfg.get("promote_standard_min_restart_momentum")) or 4.0
        min_zxdkx = self._safe_float(cfg.get("promote_standard_min_zxdkx_integrity")) or 3.0

        pattern_fit = self._safe_float(scores.get("pattern_fit"))
        volume_quality = self._safe_float(scores.get("volume_pattern_quality"))
        zxdkx_integrity = self._safe_float(scores.get("zxdkx_integrity"))
        restart_momentum = self._safe_float(scores.get("restart_momentum"))
        total_score = self._safe_float(result.get("total_score"))
        similar_to_ideal_b1 = self._safe_float(result.get("similar_to_ideal_b1"))
        pattern_label = str(result.get("pattern_label", "")).strip()
        verdict = str(result.get("verdict", "")).strip().upper()

        text_blob_parts = [
            str(result.get("pattern_reasoning", "")),
            str(result.get("volume_reasoning", "")),
            str(result.get("zxdkx_reasoning", "")),
            str(result.get("momentum_reasoning", "")),
            str(result.get("risk_reward_reasoning", "")),
            str(result.get("priority_reasoning", "")),
            str(result.get("comment", "")),
        ]
        risk_tags = [str(tag).strip() for tag in (result.get("risk_tags") or []) if str(tag).strip()]
        text_blob_parts.extend(risk_tags)
        text_blob = " ".join(part for part in text_blob_parts if part).lower()

        structural_damage_keywords = [
            "更像反弹",
            "放量阴线",
            "放量下跌",
            "出货",
            "派发",
            "高位续命",
            "伪启动",
            "结构破坏",
            "动能不足",
            "缩量不足",
        ]
        qualifies_for_standard = (
            total_score is not None
            and total_score >= min_total_for_standard
            and pattern_fit is not None
            and pattern_fit >= min_pattern_fit
            and volume_quality is not None
            and volume_quality >= min_volume_quality
            and restart_momentum is not None
            and restart_momentum >= min_restart_momentum
            and (zxdkx_integrity is None or zxdkx_integrity >= min_zxdkx)
            and (similar_to_ideal_b1 is None or similar_to_ideal_b1 >= min_similarity_for_standard)
        )
        has_structural_damage = self._contains_unnegated(text_blob, structural_damage_keywords)
        structure_too_weak_for_standard = (
            (pattern_fit is not None and pattern_fit <= 3)
            or (volume_quality is not None and volume_quality <= 3)
            or (restart_momentum is not None and restart_momentum <= 3)
            or (zxdkx_integrity is not None and zxdkx_integrity <= 2)
        )

        if pattern_label == "borderline_b1" and qualifies_for_standard and not has_structural_damage:
            result["pattern_label"] = "standard_b1"
            pattern_label = "standard_b1"
        elif pattern_label == "standard_b1":
            if has_structural_damage and (total_score is None or total_score < min_total_for_standard):
                result["pattern_label"] = "borderline_b1"
                pattern_label = "borderline_b1"
            elif structure_too_weak_for_standard:
                result["pattern_label"] = "borderline_b1"
                pattern_label = "borderline_b1"
        elif pattern_label == "false_b1" and qualifies_for_standard and not has_structural_damage:
            result["pattern_label"] = "borderline_b1"
            pattern_label = "borderline_b1"

        if pattern_label == "false_b1":
            result["verdict"] = "C"
            result["priority"] = "C"
        elif pattern_label == "borderline_b1" and verdict == "A":
            result["verdict"] = "B"
            result["priority"] = "B"
        elif pattern_label == "standard_b1" and verdict == "C":
            result["verdict"] = "B"
            result["priority"] = "B"

        return result

    def review_stock(self, code: str, day_chart: Path, prompt: str, candidate: Optional[dict] = None) -> dict:
        """子类需实现此方法，调用具体的 LLM 进行打分，并返回 JSON 解析字典。"""
        raise NotImplementedError("子类必须实现 review_stock 方法")

    def generate_suggestion(self, pick_date: str, all_results: List[dict], min_score: float) -> dict:
        passed = [r for r in all_results if r.get("total_score", 0) >= min_score]
        excluded = [r["code"] for r in all_results if r.get("total_score", 0) < min_score]

        passed.sort(key=lambda r: r.get("total_score", 0), reverse=True)

        recommendations = [
            {
                "rank": i + 1,
                "code": r["code"],
                "verdict": r.get("verdict", ""),
                "priority": r.get("priority", r.get("verdict", "")),
                "pattern_label": r.get("pattern_label", ""),
                "total_score": r.get("total_score", 0),
                "signal_type": r.get("signal_type", ""),
                "comment": r.get("comment", ""),
            }
            for i, r in enumerate(passed)
        ]

        return {
            "date": pick_date,
            "min_score_threshold": min_score,
            "total_reviewed": len(all_results),
            "recommendations": recommendations,
            "excluded": excluded,
        }

    def run(self):
        candidates_data = self.load_candidates(Path(self.config["candidates"]))
        pick_date: str = candidates_data["pick_date"]
        raw_candidates: List[dict] = candidates_data["candidates"]
        print(f"[INFO] pick_date={pick_date}，候选股票数={len(raw_candidates)}")

        candidates = self.filter_candidates(raw_candidates)
        if not candidates:
            print("[WARN] 经过筛选后没有待复评股票，流程结束。")
            return

        out_dir = self.output_dir / pick_date
        out_dir.mkdir(parents=True, exist_ok=True)

        all_results: List[dict] = []
        failed_codes: List[str] = []

        for i, candidate in enumerate(candidates, 1):
            code: str = candidate["code"]
            out_file = out_dir / f"{code}.json"

            if self.config.get("skip_existing", False) and out_file.exists():
                print(f"[{i}/{len(candidates)}] {code} — 已存在，跳过。")
                with open(out_file, encoding="utf-8") as f:
                    result = json.load(f)
                all_results.append(result)
                continue

            day_chart = self.find_chart_images(pick_date, code)
            if day_chart is None:
                print(f"[{i}/{len(candidates)}] {code} — 缺少日线图，跳过。")
                failed_codes.append(code)
                continue

            print(f"[{i}/{len(candidates)}] {code} — 正在分析 ...", end=" ", flush=True)

            try:
                result = self.review_stock(
                    code=code,
                    day_chart=day_chart,
                    prompt=self.prompt,
                    candidate=candidate,
                )
                result = self.normalize_result(result)
                result.setdefault("code", code)
                result.setdefault("pick_date", pick_date)
                result.setdefault("source_strategy", candidate.get("strategy", ""))
                result.setdefault("matched_strategies", self._matched_strategies(candidate))
                extra = candidate.get("extra", {}) or {}
                if "b1_score" in extra and "b1_score" not in result:
                    result["b1_score"] = extra.get("b1_score")

                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                all_results.append(result)
                verdict = result.get("verdict", "?")
                score = result.get("total_score", "?")
                print(f"完成 — verdict={verdict}, score={score}")
            except Exception as e:
                print(f"失败 — {e}")
                failed_codes.append(code)

            if i < len(candidates):
                time.sleep(self.config.get("request_delay", 5))

        print(f"\n[INFO] 评分完成：成功 {len(all_results)} 支，失败/跳过 {len(failed_codes)} 支")
        if failed_codes:
            print(f"[WARN] 未处理股票：{failed_codes}")

        if not all_results:
            print("[ERROR] 没有可用的评分结果，跳过汇总。")
            return

        print("\n[INFO] 正在生成汇总推荐建议 ...")
        min_score = self.config.get("suggest_min_score", 4.0)
        suggestion = self.generate_suggestion(
            pick_date=pick_date,
            all_results=all_results,
            min_score=min_score,
        )
        suggestion_file = out_dir / "suggestion.json"
        with open(suggestion_file, "w", encoding="utf-8") as f:
            json.dump(suggestion, f, ensure_ascii=False, indent=2)
        print(f"[INFO] 汇总推荐已写入: {suggestion_file}")
        print(f"       推荐股票数（score≥{min_score}）: {len(suggestion['recommendations'])}")

        print("\n✅ 全部完成。")
        print(f"   输出目录: {out_dir}")
