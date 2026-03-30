"""
pipeline/select_stock.py
量化初选核心逻辑。

职责：
  - 读取 rules_preselect.yaml 参数
  - 加载 data/raw/*.csv 日线数据
  - 运行当前正式 B1、legacy/确认版旧 B1、B2 和砖型图策略
  - 返回 List[Candidate]（纯 Python 对象，不写文件）
  - 写文件由 cli.py 调用 io.py 完成
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml 

from schemas import Candidate
from Selector import (
    B1Selector,
    BrickChartSelector,
    compute_brick_chart,
    compute_kdj,
    compute_weekly_ma_bull,
    compute_zx_lines,
)
from pipeline_core import MarketDataPreparer, TopTurnoverPoolBuilder

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG = _PROJECT_ROOT / "config" / "rules_preselect.yaml"
_B1_COMPONENT_SCORE_MAX = 20.0
_B1_SCORE_DEFAULT_WEIGHTS = {
    "volume_alignment": 18.0,
    "down_shrink": 22.0,
    "up_expand": 22.0,
    "zx_retest": 26.0,
    "volume_impulse": 12.0,
    "center_uplift": 0.0,
}
# Former B1-E compatibility aliases. The startup-style logic is now the official B1.
_B1E_COMPONENT_SCORE_MAX = _B1_COMPONENT_SCORE_MAX
_B1E_SCORE_DEFAULT_WEIGHTS = _B1_SCORE_DEFAULT_WEIGHTS


def _resolve_cfg_path(path_like: str | Path, base_dir: Path = _PROJECT_ROOT) -> Path:
    """将配置中的相对路径解析为项目根目录下的绝对路径。"""
    p = Path(path_like)
    return p if p.is_absolute() else (base_dir / p)


# =============================================================================
# 配置 & 数据加载
# =============================================================================

def load_config(config_path: Optional[str] = None) -> dict:
    """加载 rules_preselect.yaml，返回原始 dict."""
    path = _resolve_cfg_path(config_path) if config_path else _DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def resolve_preselect_output_dir(
    *,
    config_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Path:
    """返回候选输出目录，优先级：CLI参数 > 配置文件 global.output_dir > 默认值。"""
    if output_dir:
        return _resolve_cfg_path(output_dir)
    cfg = load_config(config_path)
    g = cfg.get("global", {})
    return _resolve_cfg_path(g.get("output_dir", "./data/candidates"))


def load_raw_data(
    data_dir: str,
    end_date: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """读取 data_dir 下所有 *.csv，统一处理列名/日期/排序."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"data_dir 不存在: {data_dir}")

    end_ts = pd.to_datetime(end_date) if end_date else None
    data: Dict[str, pd.DataFrame] = {}

    for fname in os.listdir(data_dir):
        if not fname.lower().endswith(".csv"):
            continue
        code = fname.rsplit(".", 1)[0]
        fpath = os.path.join(data_dir, fname)

        df = pd.read_csv(fpath).copy()
        df.columns = [c.lower() for c in df.columns]
        if "date" not in df.columns:
            logger.warning("跳过 %s：没有 date 列", fname)
            continue

        df = df.assign(date=pd.to_datetime(df["date"]))
        df = df.sort_values("date").reset_index(drop=True).copy()

        if end_ts is not None:
            df = df[df["date"] <= end_ts].reset_index(drop=True)

        if not df.empty:
            data[code] = df

    if not data:
        raise ValueError(f"未找到任何 CSV 数据: {data_dir}")

    logger.info("读取股票数量: %d", len(data))
    return data


def load_single_code_data(
    code: str,
    data_dir: str,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """读取单只股票 CSV 并统一处理列名/日期/排序。"""
    fpath = Path(data_dir) / f"{code}.csv"
    if not fpath.is_file():
        raise FileNotFoundError(f"股票数据不存在: {fpath}")

    end_ts = pd.to_datetime(end_date) if end_date else None
    df = pd.read_csv(fpath).copy()
    df.columns = [c.lower() for c in df.columns]
    if "date" not in df.columns:
        raise ValueError(f"{fpath.name} 没有 date 列")

    df = df.assign(date=pd.to_datetime(df["date"]))
    df = df.sort_values("date").reset_index(drop=True).copy()
    if end_ts is not None:
        df = df[df["date"] <= end_ts].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"{code} 在 end_date={end_date} 下无可用数据")
    return df


def prepare_single_code_data(
    df: pd.DataFrame,
    n_turnover_days: int,
) -> pd.DataFrame:
    """单只股票通用预处理：turnover_n + DatetimeIndex。"""
    out = df.copy()
    out.columns = [c.lower() for c in out.columns]
    for col in ("date", "open", "close", "volume"):
        if col not in out.columns:
            raise ValueError(f"缺少必要列: {col}")
    out = out.assign(date=pd.to_datetime(out["date"]))
    out = out.sort_values("date").reset_index(drop=True).copy()
    signed_turnover = (out["open"] + out["close"]) / 2.0 * out["volume"]
    out = out.assign(
        signed_turnover=signed_turnover,
        turnover_n=signed_turnover.rolling(int(n_turnover_days), min_periods=1).sum(),
    )
    return out.set_index("date", drop=False)


# =============================================================================
# 工具函数
# =============================================================================

def _sorted_zx(m1: int, m2: int, m3: int, m4: int) -> Tuple[int, int, int, int]:
    """保证均线参数从小到大排列."""
    a = sorted([int(m1), int(m2), int(m3), int(m4)])
    return a[0], a[1], a[2], a[3]


def _resolve_pick_date(
    prepared: Dict[str, pd.DataFrame],
    pick_date: Optional[str] = None,
) -> pd.Timestamp:
    """确定选股基准日期：None → 最晚可用交易日，否则向前搜索最近日期."""
    all_dates = sorted(
        {d for df in prepared.values() if isinstance(df.index, pd.DatetimeIndex) for d in df.index}
    )
    if not all_dates:
        raise ValueError("prepared 数据中没有可用日期。")
    if pick_date is None:
        return all_dates[-1]

    target = pd.to_datetime(pick_date)
    arr = np.array(all_dates, dtype="datetime64[ns]")
    idx = int(np.searchsorted(arr, target.to_datetime64(), side="right")) - 1
    if idx < 0:
        raise ValueError(f"pick_date={pick_date} 早于最早可用日期={all_dates[0].date()}")
    return all_dates[idx]


def _calc_warmup(cfg: dict, buffer: int) -> int:
    """根据启用策略的参数计算最长所需 warmup bars."""
    warmup = 120

    cfg_b1 = cfg.get("b1", {})
    if cfg_b1.get("enabled", True):
        warmup = max(warmup, _calc_official_b1_warmup(cfg_b1, buffer))

    cfg_b1_legacy = cfg.get("b1_legacy", {})
    if cfg_b1_legacy.get("enabled", False):
        warmup = max(warmup, int(cfg_b1_legacy.get("zx_m4", 371)) + buffer)

    cfg_b2 = cfg.get("b2", {})
    if cfg_b2.get("enabled", False):
        warmup = max(warmup, _calc_official_b1_warmup(cfg_b1, buffer))

    cfg_brick = cfg.get("brick", {})
    if cfg_brick.get("enabled", True):
        warmup = max(
            warmup,
            int(cfg_brick.get("wma_long", 120)) * 5 + buffer,
            int(cfg_brick.get("zxdkx_m4", 114)) + buffer,
        )

    return warmup


def _calc_upper_shadow_ratio(row: pd.Series) -> Optional[float]:
    """计算上影线比例：(high - max(open, close)) / close。"""
    close = float(row.get("close", np.nan))
    high = float(row.get("high", np.nan))
    open_ = float(row.get("open", np.nan))
    if not np.isfinite(close) or close <= 0 or not np.isfinite(high) or not np.isfinite(open_):
        return None
    upper_shadow = max(0.0, high - max(open_, close))
    return upper_shadow / close


def _clip01(value: float) -> float:
    return float(min(max(value, 0.0), 1.0))


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return default
    return val if np.isfinite(val) else default


def _normalize_linear(value: Optional[float], floor: float, ceiling: float) -> float:
    if value is None or not np.isfinite(value):
        return 0.0
    if ceiling <= floor:
        return 0.0
    return _clip01((float(value) - floor) / (ceiling - floor))


def _normalize_inverse(value: Optional[float], floor: float, ceiling: float) -> float:
    if value is None or not np.isfinite(value):
        return 0.0
    if ceiling <= floor:
        return 0.0
    return _clip01((ceiling - float(value)) / (ceiling - floor))


def _score_b2_metrics(
    *,
    ret: float,
    j: float,
    j_delta: float,
    upper_shadow_ratio: Optional[float],
    volume_ratio: Optional[float],
    score_cfg: Optional[dict] = None,
) -> dict:
    score_cfg = score_cfg or {}
    weights = score_cfg.get("weights", {})

    ret_score = _normalize_linear(ret, _safe_float(score_cfg.get("return_floor", 0.04)), _safe_float(score_cfg.get("return_ceiling", 0.10), 0.10)) * _safe_float(weights.get("return", 30), 30)
    j_pos_score = _normalize_inverse(j, 0.0, _safe_float(score_cfg.get("j_ceiling", 55), 55)) * _safe_float(weights.get("j_position", 20), 20)
    j_delta_score = _normalize_linear(j_delta, 0.0, _safe_float(score_cfg.get("j_delta_ceiling", 40), 40)) * _safe_float(weights.get("j_delta", 20), 20)
    volume_score = _normalize_linear(volume_ratio, _safe_float(score_cfg.get("volume_ratio_floor", 1.0), 1.0), _safe_float(score_cfg.get("volume_ratio_ceiling", 3.0), 3.0)) * _safe_float(weights.get("volume_ratio", 20), 20)
    upper_shadow_score = _normalize_inverse(upper_shadow_ratio, 0.0, _safe_float(score_cfg.get("upper_shadow_ceiling", 0.03), 0.03)) * _safe_float(weights.get("upper_shadow", 10), 10)

    total = round(ret_score + j_pos_score + j_delta_score + volume_score + upper_shadow_score, 1)
    return {
        "b2_score": total,
        "b2_score_version": str(score_cfg.get("version", "v1")),
        "b2_score_breakdown": {
            "return": round(ret_score, 1),
            "j_position": round(j_pos_score, 1),
            "j_delta": round(j_delta_score, 1),
            "volume_ratio": round(volume_score, 1),
            "upper_shadow": round(upper_shadow_score, 1),
        },
    }


def _apply_brick_scores(candidates: List[Candidate], score_cfg: Optional[dict] = None) -> None:
    if not candidates:
        return
    score_cfg = score_cfg or {}
    base_score = _safe_float(score_cfg.get("base_score", 60), 60)
    max_score = _safe_float(score_cfg.get("max_score", 100), 100)
    growths = [float(c.brick_growth) for c in candidates if c.brick_growth is not None and np.isfinite(c.brick_growth)]
    if not growths:
        return
    g_min = min(growths)
    g_max = max(growths)

    for c in candidates:
        c.extra = dict(c.extra or {})
        bg = c.brick_growth if c.brick_growth is not None and np.isfinite(c.brick_growth) else None
        if bg is None:
            score = round(base_score, 1)
        elif g_max <= g_min:
            score = round((base_score + max_score) / 2.0, 1)
        else:
            ratio = (float(bg) - g_min) / (g_max - g_min)
            score = round(base_score + _clip01(ratio) * (max_score - base_score), 1)
        c.extra.update({
            "brick_score": score,
            "brick_score_version": str(score_cfg.get("version", "v1")),
            "brick_score_base": round(base_score, 1),
            "brick_score_min_growth": round(g_min, 6),
            "brick_score_max_growth": round(g_max, 6),
        })


def _score_combo_candidate(candidate: Candidate, score_cfg: Optional[dict] = None) -> float:
    score_cfg = score_cfg or {}
    weights = score_cfg.get("weights", {})
    b2_weight = _safe_float(weights.get("b2", 0.6), 0.6)
    brick_weight = _safe_float(weights.get("brick", 0.4), 0.4)
    bonus = _safe_float(score_cfg.get("resonance_bonus", 5), 5)
    max_score = _safe_float(score_cfg.get("max_score", 100), 100)
    b2_score = _safe_float((candidate.extra or {}).get("b2_score"), 0.0)
    xg_score = _safe_float((candidate.extra or {}).get("xg_score"), 0.0)
    brick_score = _safe_float((candidate.extra or {}).get("brick_score"), xg_score)
    return round(min(max_score, b2_weight * b2_score + brick_weight * brick_score + bonus), 1)


def _score_xg_metrics(
    *,
    red_height: float,
    green_height: float,
    close: float,
    yellow_line: float,
    brick_level: float,
    score_cfg: Optional[dict] = None,
) -> dict:
    score_cfg = score_cfg or {}
    weights = score_cfg.get("weights", {})

    recovery_ratio = red_height / green_height if np.isfinite(green_height) and green_height > 0 else 0.0
    yellow_margin = close / yellow_line - 1.0 if np.isfinite(yellow_line) and yellow_line > 0 else 0.0

    red_height_score = _normalize_linear(red_height, 0.0, _safe_float(score_cfg.get("red_height_ceiling", 20), 20)) * _safe_float(weights.get("red_height", 35), 35)
    recovery_ratio_score = _normalize_linear(recovery_ratio, _safe_float(score_cfg.get("recovery_ratio_floor", 2 / 3), 2 / 3), _safe_float(score_cfg.get("recovery_ratio_ceiling", 3.0), 3.0)) * _safe_float(weights.get("recovery_ratio", 30), 30)
    yellow_margin_score = _normalize_linear(yellow_margin, _safe_float(score_cfg.get("yellow_margin_floor", 0.0), 0.0), _safe_float(score_cfg.get("yellow_margin_ceiling", 0.15), 0.15)) * _safe_float(weights.get("yellow_margin", 20), 20)
    brick_level_score = _normalize_linear(brick_level, 0.0, _safe_float(score_cfg.get("brick_level_ceiling", 20), 20)) * _safe_float(weights.get("brick_level", 15), 15)

    total = round(red_height_score + recovery_ratio_score + yellow_margin_score + brick_level_score, 1)
    return {
        "xg_score": total,
        "xg_score_version": str(score_cfg.get("version", "v1")),
        "xg_score_breakdown": {
            "red_height": round(red_height_score, 1),
            "recovery_ratio": round(recovery_ratio_score, 1),
            "yellow_margin": round(yellow_margin_score, 1),
            "brick_level": round(brick_level_score, 1),
        },
        "xg_recovery_ratio": round(recovery_ratio, 6),
        "xg_yellow_margin": round(yellow_margin, 6),
    }


def _is_j_turn_up(j_t: float, j_t1: float, j_t2: float, *, mode: str = "combo", j_jump_min: float = 8.0) -> bool:
    """判断 J 值是否明显拐头向上。"""
    vals = [j_t, j_t1, j_t2]
    if any(not np.isfinite(v) for v in vals):
        return False
    up2 = j_t > j_t1 > j_t2
    jump = (j_t > j_t1) and ((j_t - j_t1) >= j_jump_min)
    mode = str(mode or "combo").lower()
    if mode == "up2":
        return up2
    if mode == "jump":
        return jump
    return up2 or jump


def _has_recent_b1_hit(pf: pd.DataFrame, pick_date: pd.Timestamp, lookback_days: int) -> bool:
    """检查最近 N 个交易日内是否命中过当前正式 B1。"""
    if pick_date not in pf.index:
        return False
    end_idx = pf.index.get_loc(pick_date)
    if isinstance(end_idx, slice):
        return False
    start_idx = max(0, int(end_idx) - max(1, lookback_days) + 1)
    window = pf.iloc[start_idx:int(end_idx) + 1]
    if window.empty:
        return False
    cfg_b1 = getattr(pf, "attrs", {}).get("b1_cfg", None)
    for trade_date in window.index:
        evaluation = evaluate_b1_from_prepared(pf, pd.Timestamp(trade_date), cfg_b1)
        if evaluation.get("hit", False):
            return True
    return False


def _calc_official_b1_warmup(cfg_b1: Optional[dict], buffer: int) -> int:
    """当前正式 B1 所需的 warmup。"""
    cfg = _normalize_b1_cfg(cfg_b1)
    top_distribution_cfg = dict(cfg.get("top_distribution", {}))
    volume_impulse_cfg = dict(cfg.get("volume_impulse", {}))
    lookback_for_warmup = max(
        int(cfg.get("lookback_days", 20)),
        int(top_distribution_cfg.get("top_lookback_days", 20)),
        int(volume_impulse_cfg.get("lookback_days", 20)),
    )
    warmup = int(cfg.get("zx_m4", 114)) + lookback_for_warmup + buffer
    weekly_tag_cfg = cfg.get("weekly_tag", {})
    if weekly_tag_cfg.get("enabled", True):
        warmup = max(warmup, int(weekly_tag_cfg.get("wma_long", 120)) * 5 + buffer)
    return warmup


def _normalize_b1_cfg(cfg_b1: Optional[dict] = None) -> dict:
    raw = dict(cfg_b1 or {})
    weekly_tag = dict(raw.get("weekly_tag", {}))
    top_distribution = dict(raw.get("top_distribution", {}))
    volume_impulse = dict(raw.get("volume_impulse", {}))
    score_cfg = dict(raw.get("score", {}))
    raw_weights = dict(score_cfg.get("weights", {}))
    base_lookback_days = int(raw.get("lookback_days", 20))
    soft_impulse_ratio = float(volume_impulse.get("soft_volume_ratio", 1.6))
    strict_impulse_ratio = max(
        soft_impulse_ratio,
        float(volume_impulse.get("strict_volume_ratio", 2.0)),
    )
    fresh_impulse_days = max(0, int(volume_impulse.get("fresh_days", 8)))
    stale_impulse_days = max(fresh_impulse_days, int(volume_impulse.get("stale_days", 12)))
    weights: Dict[str, float] = {}
    for key, default in _B1_SCORE_DEFAULT_WEIGHTS.items():
        weights[key] = max(0.0, _safe_float(raw_weights.get(key), default))
    if sum(weights.values()) <= 0:
        weights = dict(_B1_SCORE_DEFAULT_WEIGHTS)
    total_weight = float(sum(weights.values()))
    if total_weight > 0 and not np.isclose(total_weight, 100.0):
        weights = {key: round(val * 100.0 / total_weight, 6) for key, val in weights.items()}
    return {
        "enabled": bool(raw.get("enabled", True)),
        "zx_m1": int(raw.get("zx_m1", 14)),
        "zx_m2": int(raw.get("zx_m2", 28)),
        "zx_m3": int(raw.get("zx_m3", 57)),
        "zx_m4": int(raw.get("zx_m4", 114)),
        "zxdq_span": int(raw.get("zxdq_span", 10)),
        "kdj_n": int(raw.get("kdj_n", 9)),
        "j_threshold": float(raw.get("j_threshold", 15.0)),
        "j_q_threshold": float(raw.get("j_q_threshold", 0.10)),
        "lookback_days": base_lookback_days,
        "retest_lookback_days": int(raw.get("retest_lookback_days", 5)),
        "vol_ma_days": int(raw.get("vol_ma_days", 5)),
        "up_expand_volume_factor": float(raw.get("up_expand_volume_factor", 1.3)),
        "down_shrink_volume_factor": float(raw.get("down_shrink_volume_factor", 0.95)),
        "down_shrink_return_floor": float(raw.get("down_shrink_return_floor", -0.03)),
        "distribution_return_threshold": float(raw.get("distribution_return_threshold", -0.03)),
        "distribution_volume_factor": float(raw.get("distribution_volume_factor", 1.2)),
        "reject_distribution_days": int(raw.get("reject_distribution_days", 2)),
        "heavy_breakdown_volume_factor": float(raw.get("heavy_breakdown_volume_factor", 1.2)),
        "max_break_days": int(raw.get("max_break_days", 1)),
        "top_distribution": {
            "enabled": bool(top_distribution.get("enabled", True)),
            "recent_window_days": int(top_distribution.get("recent_window_days", 10)),
            "top_lookback_days": int(top_distribution.get("top_lookback_days", 20)),
            "confirm_days": int(top_distribution.get("confirm_days", 3)),
            "high_volume_factor": float(top_distribution.get("high_volume_factor", 1.75)),
            "top_close_to_high_ratio": float(top_distribution.get("top_close_to_high_ratio", 0.91)),
            "top_high_to_high_ratio": float(top_distribution.get("top_high_to_high_ratio", 0.98)),
            "upper_shadow_ratio": float(top_distribution.get("upper_shadow_ratio", 0.50)),
            "small_return_threshold": float(top_distribution.get("small_return_threshold", 0.03)),
            "continuation_high_ratio": float(top_distribution.get("continuation_high_ratio", 1.005)),
            "pullback_close_ratio": float(top_distribution.get("pullback_close_ratio", 0.97)),
        },
        "volume_impulse": {
            "enabled": bool(volume_impulse.get("enabled", True)),
            "lookback_days": int(volume_impulse.get("lookback_days", base_lookback_days)),
            "soft_volume_ratio": soft_impulse_ratio,
            "strict_volume_ratio": strict_impulse_ratio,
            "min_return": float(volume_impulse.get("min_return", 0.015)),
            "min_close_position": float(volume_impulse.get("min_close_position", 0.60)),
            "max_upper_shadow_ratio": float(volume_impulse.get("max_upper_shadow_ratio", 0.35)),
            "high_area_threshold": float(volume_impulse.get("high_area_threshold", 0.75)),
            "fresh_days": fresh_impulse_days,
            "stale_days": stale_impulse_days,
            "strong_ratio": float(volume_impulse.get("strong_ratio", 2.2)),
            "explosive_ratio": float(volume_impulse.get("explosive_ratio", 3.0)),
            "neutral_score": int(volume_impulse.get("neutral_score", 4)),
        },
        "weekly_tag": {
            "enabled": bool(weekly_tag.get("enabled", True)),
            "wma_short": int(weekly_tag.get("wma_short", 20)),
            "wma_mid": int(weekly_tag.get("wma_mid", 60)),
            "wma_long": int(weekly_tag.get("wma_long", 120)),
        },
        "score": {
            "version": str(score_cfg.get("version", "v4")),
            "weights": weights,
        },
    }


def _normalize_b1e_cfg(cfg_b1e: Optional[dict] = None) -> dict:
    """兼容旧名称：原 B1-E 现已升级为正式 B1。"""
    return _normalize_b1_cfg(cfg_b1e)


def _resolve_pick_date_in_df(df: pd.DataFrame, pick_date: Optional[str] = None) -> pd.Timestamp:
    """单只股票版本的 pick_date 解析：若无该日，则向前取最近交易日。"""
    if not isinstance(df.index, pd.DatetimeIndex) or df.empty:
        raise ValueError("df 必须使用 DatetimeIndex 且不能为空。")
    if pick_date is None:
        return pd.Timestamp(df.index[-1])

    target = pd.to_datetime(pick_date)
    arr = df.index.to_numpy(dtype="datetime64[ns]")
    idx = int(np.searchsorted(arr, target.to_datetime64(), side="right")) - 1
    if idx < 0:
        raise ValueError(f"pick_date={pick_date} 早于最早可用日期={df.index[0].date()}")
    return pd.Timestamp(df.index[idx])


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or numerator <= 0:
        return 0.0 if np.isfinite(denominator) and denominator > 0 else 1.0
    if not np.isfinite(denominator) or denominator <= 0:
        return float("inf")
    return float(numerator / denominator)


def _calc_intraday_upper_shadow_ratio(
    open_price: float,
    high_price: float,
    low_price: float,
    close_price: float,
) -> float:
    if not all(np.isfinite(val) for val in (open_price, high_price, low_price, close_price)):
        return float("nan")
    bar_range = high_price - low_price
    if bar_range <= 0:
        return 0.0
    return max(0.0, float(high_price - max(open_price, close_price)) / float(bar_range))


def _detect_b1_top_distribution(
    pf: pd.DataFrame,
    loc: int,
    cfg: dict,
) -> Optional[dict]:
    """识别近期高位巨量弱收且后续无强延续的派发型顶部柱。"""
    td_cfg = dict(cfg.get("top_distribution", {}))
    if not td_cfg.get("enabled", True) or loc <= 0:
        return None

    recent_window_days = max(3, int(td_cfg.get("recent_window_days", 10)))
    top_lookback_days = max(5, int(td_cfg.get("top_lookback_days", 20)))
    confirm_days = max(1, int(td_cfg.get("confirm_days", 3)))
    start_loc = max(top_lookback_days - 1, loc - recent_window_days)

    for cand_loc in range(loc - 1, start_loc - 1, -1):
        bar = pf.iloc[cand_loc]
        bar_open = float(bar.get("open", np.nan))
        bar_high = float(bar.get("high", np.nan))
        bar_low = float(bar.get("low", np.nan))
        bar_close = float(bar.get("close", np.nan))
        bar_volume = float(bar.get("volume", np.nan))
        vol_ma5 = float(bar.get("vol_ma5", np.nan))
        prev_close = float(bar.get("prev_close", np.nan))

        if not (
            all(np.isfinite(val) for val in (bar_open, bar_high, bar_low, bar_close, bar_volume, vol_ma5, prev_close))
            and vol_ma5 > 0
            and prev_close > 0
        ):
            continue

        high_window = pf["high"].iloc[cand_loc - top_lookback_days + 1:cand_loc + 1].astype(float)
        recent_high = float(high_window.max()) if not high_window.empty else float("nan")
        if not np.isfinite(recent_high) or recent_high <= 0:
            continue

        vol_ratio = bar_volume / vol_ma5
        close_to_recent_high = bar_close / recent_high
        high_to_recent_high = bar_high / recent_high
        if vol_ratio < float(td_cfg["high_volume_factor"]):
            continue
        if (
            close_to_recent_high < float(td_cfg["top_close_to_high_ratio"])
            and high_to_recent_high < float(td_cfg["top_high_to_high_ratio"])
        ):
            continue

        upper_shadow_ratio = _calc_intraday_upper_shadow_ratio(bar_open, bar_high, bar_low, bar_close)
        daily_return = bar_close / prev_close - 1.0
        weak_close_tags: List[str] = []

        # 高位巨量柱若只是“看起来翻红”，但实体很弱/长上影/明显滞涨，
        # 再配合后续无延续，通常更像派发而不是启动延续。
        if bar_close < bar_open and bar_close > prev_close:
            weak_close_tags.append("pseudo_bearish_true_bull")
        if np.isfinite(upper_shadow_ratio) and upper_shadow_ratio >= float(td_cfg["upper_shadow_ratio"]):
            weak_close_tags.append("long_upper_shadow")
        if np.isfinite(daily_return) and daily_return <= float(td_cfg["small_return_threshold"]):
            weak_close_tags.append("small_return_stall")
        if not weak_close_tags:
            continue

        future_end = min(loc, cand_loc + confirm_days)
        future = pf.iloc[cand_loc + 1:future_end + 1]
        if future.empty:
            continue

        future_high_max = float(future["high"].astype(float).max())
        future_close_min = float(future["close"].astype(float).min())
        has_strong_continuation = bool(
            np.isfinite(future_high_max)
            and future_high_max >= bar_high * float(td_cfg["continuation_high_ratio"])
        )
        has_clear_pullback = bool(
            np.isfinite(future_close_min)
            and future_close_min <= bar_close * float(td_cfg["pullback_close_ratio"])
        )

        if (not has_strong_continuation) and has_clear_pullback:
            trade_date = pd.Timestamp(pf.index[cand_loc]).strftime("%Y-%m-%d")
            return {
                "bar_date": trade_date,
                "vol_ratio": round(vol_ratio, 4),
                "close_to_recent_high": round(close_to_recent_high, 4),
                "high_to_recent_high": round(high_to_recent_high, 4),
                "upper_shadow_ratio": round(upper_shadow_ratio, 4) if np.isfinite(upper_shadow_ratio) else None,
                "daily_return": round(daily_return, 6) if np.isfinite(daily_return) else None,
                "future_high_max": round(future_high_max, 4) if np.isfinite(future_high_max) else None,
                "future_close_min": round(future_close_min, 4) if np.isfinite(future_close_min) else None,
                "weak_close_tags": weak_close_tags,
            }
    return None


def _score_b1_volume_alignment(
    ratio_avg: float,
    ratio_sum: float,
    up_expand_to_down_shrink_avg_vol_ratio: float,
) -> int:
    """基于整体涨跌量能关系打底，再用放量上涨/缩量下跌的直接对比做一档微调。"""
    base_score: int
    if ratio_avg > 1.5 and ratio_sum > 1.2:
        base_score = 20
    elif ratio_avg > 1.3 and ratio_sum > 1.0:
        base_score = 16
    elif ratio_avg > 1.1 and ratio_sum > 0.9:
        base_score = 12
    elif ratio_avg > 0.9:
        base_score = 8
    else:
        base_score = 3

    if np.isfinite(up_expand_to_down_shrink_avg_vol_ratio):
        if up_expand_to_down_shrink_avg_vol_ratio >= 1.3:
            return min(20, base_score + 4)
        if up_expand_to_down_shrink_avg_vol_ratio < 1.0:
            return max(3, base_score - 4)
    return base_score


def _score_b1_down_shrink(count: int) -> int:
    if count >= 8:
        return 20
    if count >= 6:
        return 16
    if count >= 4:
        return 12
    if count >= 2:
        return 8
    return 3


def _score_b1_up_expand(count: int) -> int:
    if 3 <= count <= 6:
        return 20
    if count in (2, 7):
        return 16
    if count in (1, 8):
        return 10
    return 5


def _score_b1_volume_impulse(
    soft_count: int,
    strict_count: int,
    max_ratio: float,
    latest_days_ago: Optional[int],
    latest_impulse_is_strict: bool,
    high_area_count: int,
    cfg: Optional[dict] = None,
) -> int:
    impulse_cfg = dict(cfg or {})
    neutral_score = max(0, int(impulse_cfg.get("neutral_score", 4)))
    if soft_count <= 0:
        return min(int(_B1_COMPONENT_SCORE_MAX), neutral_score)

    raw = 8
    if strict_count >= 2:
        raw += 5
    elif strict_count == 1:
        raw += 4
    elif soft_count >= 2:
        raw += 4
    else:
        raw += 2

    if soft_count >= 3:
        raw += 2
    elif soft_count >= 2:
        raw += 1

    soft_ratio = float(impulse_cfg.get("soft_volume_ratio", 1.6))
    strong_ratio = max(soft_ratio, float(impulse_cfg.get("strong_ratio", 2.2)))
    explosive_ratio = max(strong_ratio, float(impulse_cfg.get("explosive_ratio", 3.0)))
    if np.isfinite(max_ratio):
        if max_ratio >= explosive_ratio:
            raw += 3
        elif max_ratio >= strong_ratio:
            raw += 2
        elif max_ratio >= soft_ratio:
            raw += 1

    fresh_days = max(0, int(impulse_cfg.get("fresh_days", 8)))
    stale_days = max(fresh_days, int(impulse_cfg.get("stale_days", 12)))
    if latest_days_ago is not None:
        if latest_days_ago <= max(0, fresh_days // 2):
            raw += 3
        elif latest_days_ago <= fresh_days:
            raw += 2
        elif latest_days_ago <= stale_days:
            raw += 1

    if latest_impulse_is_strict:
        raw += 2

    if high_area_count > 0:
        raw -= min(2, high_area_count)

    return int(min(_B1_COMPONENT_SCORE_MAX, max(neutral_score, raw)))


def _evaluate_b1_volume_impulse(
    window: pd.DataFrame,
    cfg: Optional[dict] = None,
) -> dict:
    impulse_cfg = dict(cfg or {})
    neutral_score = max(0, int(impulse_cfg.get("neutral_score", 4)))
    default = {
        "impulse_count_soft": 0,
        "impulse_count_strict": 0,
        "max_impulse_ratio": float("nan"),
        "latest_impulse_days_ago": None,
        "latest_impulse_is_strict": False,
        "impulse_high_area_count": 0,
        "raw_score": neutral_score,
    }
    if not impulse_cfg.get("enabled", True):
        default["raw_score"] = 0
        return default
    if window.empty:
        return default

    impulse_window = window.tail(max(1, int(impulse_cfg.get("lookback_days", len(window))))).copy()
    if impulse_window.empty:
        return default

    core_mask = (
        (impulse_window["close"].astype(float) > impulse_window["prev_close"].astype(float))
        & (impulse_window["close"].astype(float) > impulse_window["open"].astype(float))
        & (impulse_window["daily_return"].astype(float) >= float(impulse_cfg.get("min_return", 0.015)))
        & (impulse_window["close_position_ratio"].astype(float) >= float(impulse_cfg.get("min_close_position", 0.60)))
        & (
            impulse_window["upper_shadow_bar_ratio"].astype(float)
            <= float(impulse_cfg.get("max_upper_shadow_ratio", 0.35))
        )
        & (impulse_window["prev_volume"].astype(float) > 0)
    )
    soft_mask = core_mask & (
        impulse_window["volume_ratio_prev"].astype(float)
        >= float(impulse_cfg.get("soft_volume_ratio", 1.6))
    )
    strict_mask = core_mask & (
        impulse_window["volume_ratio_prev"].astype(float)
        >= float(impulse_cfg.get("strict_volume_ratio", 2.0))
    )

    high_price = float(impulse_window["high"].astype(float).max())
    low_price = float(impulse_window["low"].astype(float).min())
    if np.isfinite(high_price) and np.isfinite(low_price) and high_price > low_price:
        location_ratio = ((impulse_window["close"].astype(float) - low_price) / (high_price - low_price)).clip(0.0, 1.0)
    else:
        location_ratio = pd.Series(np.nan, index=impulse_window.index, dtype=float)
    high_area_mask = soft_mask & (
        location_ratio >= float(impulse_cfg.get("high_area_threshold", 0.75))
    )

    impulse_count_soft = int(soft_mask.sum())
    impulse_count_strict = int(strict_mask.sum())
    max_impulse_ratio = (
        float(impulse_window.loc[soft_mask, "volume_ratio_prev"].astype(float).max())
        if impulse_count_soft > 0
        else float("nan")
    )

    latest_impulse_days_ago: Optional[int] = None
    latest_impulse_is_strict = False
    soft_positions = np.flatnonzero(soft_mask.to_numpy())
    if soft_positions.size > 0:
        latest_pos = int(soft_positions[-1])
        latest_impulse_days_ago = len(impulse_window) - 1 - latest_pos
        latest_impulse_is_strict = bool(strict_mask.iloc[latest_pos])

    impulse_high_area_count = int(high_area_mask.sum())
    raw_score = _score_b1_volume_impulse(
        impulse_count_soft,
        impulse_count_strict,
        max_impulse_ratio,
        latest_impulse_days_ago,
        latest_impulse_is_strict,
        impulse_high_area_count,
        cfg=impulse_cfg,
    )
    return {
        "impulse_count_soft": impulse_count_soft,
        "impulse_count_strict": impulse_count_strict,
        "max_impulse_ratio": max_impulse_ratio,
        "latest_impulse_days_ago": latest_impulse_days_ago,
        "latest_impulse_is_strict": latest_impulse_is_strict,
        "impulse_high_area_count": impulse_high_area_count,
        "raw_score": raw_score,
    }


def _score_b1_zx_retest(break_days: int, shrink_break_days: int, heavy_breakdowns: int) -> int:
    if break_days == 0:
        return 20
    if break_days == 1 and shrink_break_days == 1:
        return 18
    if break_days == 1:
        return 12
    if break_days == 2 and heavy_breakdowns == 0:
        return 8
    return 2


def _score_b1_center_uplift(center_uplift: float) -> int:
    # 价格中枢抬升仅保留为弱辅助信号；不应因为中枢略低就否掉未有效跌破知行线的标的。
    if center_uplift > 0.05:
        return 20
    if center_uplift >= 0.0:
        return 18
    if center_uplift >= -0.04:
        return 16
    if center_uplift >= -0.08:
        return 12
    return 8


def _scale_b1_weighted_scores(raw_scores: Dict[str, int], score_cfg: Optional[dict] = None) -> Tuple[Dict[str, float], float]:
    """将各原始信号分值按配置权重缩放为最终贡献分。"""
    score_cfg = score_cfg or {}
    weights = dict(score_cfg.get("weights", {}))
    weighted_scores: Dict[str, float] = {}

    for key, raw_score in raw_scores.items():
        weight = max(0.0, _safe_float(weights.get(key), 0.0))
        contribution = _clip01(_safe_float(raw_score, 0.0) / _B1_COMPONENT_SCORE_MAX) * weight
        weighted_scores[key] = round(contribution, 1)

    return weighted_scores, round(sum(weighted_scores.values()), 1)


def prepare_b1_df(
    df: pd.DataFrame,
    cfg_b1: Optional[dict] = None,
) -> pd.DataFrame:
    """为当前正式 B1（原 B1-E）预计算 ZX/KDJ/量价辅助列。"""
    cfg = _normalize_b1_cfg(cfg_b1)
    out = df.copy().sort_index()

    zs, zk = compute_zx_lines(
        out,
        cfg["zx_m1"], cfg["zx_m2"], cfg["zx_m3"], cfg["zx_m4"],
        zxdq_span=cfg["zxdq_span"],
    )

    kdj = compute_kdj(out, n=cfg["kdj_n"])

    vol_ma_days = max(1, int(cfg["vol_ma_days"]))
    prev_close = out["close"].shift(1)
    prev_volume = out["volume"].shift(1)
    bar_range = out["high"] - out["low"]
    valid_bar_range = bar_range.where(bar_range > 0)
    close_position_ratio = ((out["close"] - out["low"]) / valid_bar_range).clip(lower=0.0, upper=1.0)
    upper_shadow_bar_ratio = (
        (out["high"] - out[["open", "close"]].max(axis=1)) / valid_bar_range
    ).clip(lower=0.0, upper=1.0)
    out = out.assign(
        zxdq=zs,
        zxdkx=zk,
        K=kdj["K"],
        D=kdj["D"],
        J=kdj["J"],
        vol_ma5=out["volume"].rolling(vol_ma_days, min_periods=vol_ma_days).mean(),
        prev_close=prev_close,
        prev_volume=prev_volume,
        volume_ratio_prev=out["volume"] / prev_volume,
        bar_range=bar_range,
        close_position_ratio=close_position_ratio,
        upper_shadow_bar_ratio=upper_shadow_bar_ratio,
        daily_return=out["close"] / prev_close - 1.0,
    )

    weekly_tag_cfg = cfg["weekly_tag"]
    if weekly_tag_cfg.get("enabled", True):
        out = out.assign(
            wma_bull=compute_weekly_ma_bull(
                out,
                ma_periods=(
                    int(weekly_tag_cfg["wma_short"]),
                    int(weekly_tag_cfg["wma_mid"]),
                    int(weekly_tag_cfg["wma_long"]),
                ),
            ).to_numpy()
        )
    out.attrs["b1_cfg"] = cfg
    return out


def prepare_b1e_df(
    df: pd.DataFrame,
    cfg_b1e: Optional[dict] = None,
) -> pd.DataFrame:
    """兼容旧名称：原 B1-E 现已升级为正式 B1。"""
    return prepare_b1_df(df, cfg_b1e)


def evaluate_b1_from_prepared(
    pf: pd.DataFrame,
    pick_date: pd.Timestamp,
    cfg_b1: Optional[dict] = None,
) -> dict:
    """评估单只股票在指定日期是否满足当前正式 B1，并返回明细。"""
    cfg = _normalize_b1_cfg(cfg_b1 or getattr(pf, "attrs", {}).get("b1_cfg"))
    if pick_date not in pf.index:
        return {
            "hit": False,
            "requested_date": pick_date.strftime("%Y-%m-%d"),
            "resolved_date": None,
            "hard_filter_reasons": [f"pick_date_not_found:{pick_date.strftime('%Y-%m-%d')}"],
        }

    loc = pf.index.get_loc(pick_date)
    if isinstance(loc, slice):
        return {
            "hit": False,
            "requested_date": pick_date.strftime("%Y-%m-%d"),
            "resolved_date": pick_date.strftime("%Y-%m-%d"),
            "hard_filter_reasons": ["pick_date_not_unique"],
        }
    loc = int(loc)

    lookback_days = max(5, int(cfg["lookback_days"]))
    retest_days = max(1, int(cfg["retest_lookback_days"]))
    top_distribution_cfg = dict(cfg.get("top_distribution", {}))
    volume_impulse_cfg = dict(cfg.get("volume_impulse", {}))
    impulse_lookback_days = max(1, int(volume_impulse_cfg.get("lookback_days", lookback_days)))
    required_loc = max(
        lookback_days,
        retest_days,
        impulse_lookback_days,
        int(top_distribution_cfg.get("top_lookback_days", 20)),
    )
    if loc < required_loc:
        return {
            "hit": False,
            "requested_date": pick_date.strftime("%Y-%m-%d"),
            "resolved_date": pick_date.strftime("%Y-%m-%d"),
            "hard_filter_reasons": [f"insufficient_history:need>{required_loc}_bars"],
        }

    window20 = pf.iloc[loc - lookback_days + 1:loc + 1].copy()
    window5 = pf.iloc[loc - retest_days + 1:loc + 1].copy()
    impulse_window = pf.iloc[loc - impulse_lookback_days + 1:loc + 1].copy()
    row = pf.iloc[loc]

    j_hist = pf["J"].iloc[:loc + 1].dropna().astype(float)
    j_today = float(row.get("J", np.nan))
    j_quantile = float(j_hist.quantile(cfg["j_q_threshold"])) if not j_hist.empty else float("nan")
    j_low_ok = bool(
        np.isfinite(j_today) and (
            j_today < cfg["j_threshold"] or (np.isfinite(j_quantile) and j_today <= j_quantile)
        )
    )

    zxdq_today = float(row.get("zxdq", np.nan))
    zxdkx_today = float(row.get("zxdkx", np.nan))
    zx_lines_ready = bool(np.isfinite(zxdq_today) and np.isfinite(zxdkx_today))
    zxdq_gt_ok = bool(zx_lines_ready and zxdq_today > zxdkx_today)
    zxdq_above_zxdkx_pct = (
        zxdq_today / zxdkx_today - 1.0
        if zx_lines_ready and zxdkx_today > 0
        else float("nan")
    )

    break_mask = (
        window5["zxdkx"].notna()
        & (window5["close"].astype(float) < window5["zxdkx"].astype(float))
    )
    shrink_break_mask = break_mask & (window5["volume"].astype(float) < window5["vol_ma5"].astype(float))
    heavy_breakdown_mask = break_mask & (
        window5["volume"].astype(float) > window5["vol_ma5"].astype(float) * cfg["heavy_breakdown_volume_factor"]
    )

    break_days = int(break_mask.sum())
    shrink_break_days = int(shrink_break_mask.sum())
    heavy_breakdown_days = int(heavy_breakdown_mask.sum())

    distribution_mask = (
        (window20["close"].astype(float) < window20["prev_close"].astype(float))
        & (window20["daily_return"].astype(float) <= cfg["distribution_return_threshold"])
        & (
            window20["volume"].astype(float)
            > window20["vol_ma5"].astype(float) * cfg["distribution_volume_factor"]
        )
    )
    distribution_days = int(distribution_mask.sum())
    top_distribution_event = _detect_b1_top_distribution(pf, loc, cfg)

    hard_filter_reasons: List[str] = []
    if not j_low_ok:
        hard_filter_reasons.append(
            f"j_not_low:J={j_today:.2f},threshold={cfg['j_threshold']:.2f},q={j_quantile:.2f}"
        )
    if not zx_lines_ready:
        hard_filter_reasons.append(
            f"zx_lines_not_ready:zxdq={zxdq_today:.3f},zxdkx={zxdkx_today:.3f}"
        )
    elif not zxdq_gt_ok:
        hard_filter_reasons.append(
            f"zxdq_not_above_zxdkx:zxdq={zxdq_today:.3f},zxdkx={zxdkx_today:.3f}"
        )
    if break_days > int(cfg["max_break_days"]):
        hard_filter_reasons.append(
            f"too_many_zx_break_days:last{retest_days}d={break_days},max={cfg['max_break_days']}"
        )
    elif break_days == 1 and shrink_break_days != 1:
        hard_filter_reasons.append("single_zx_break_not_shrink_volume")
    if distribution_days >= int(cfg["reject_distribution_days"]):
        hard_filter_reasons.append(
            f"distribution_days_too_many:last{lookback_days}d={distribution_days},reject_if>={cfg['reject_distribution_days']}"
        )
    if heavy_breakdown_days > 0:
        hard_filter_reasons.append(f"heavy_breakdown_days:last{retest_days}d={heavy_breakdown_days}")
    if top_distribution_event:
        hard_filter_reasons.append(
            "recent_top_distribution:"
            f"bar={top_distribution_event['bar_date']},"
            f"vol_ratio={top_distribution_event['vol_ratio']:.2f},"
            f"weak={'+'.join(top_distribution_event['weak_close_tags'])},"
            f"future_close_min={top_distribution_event['future_close_min']}"
        )

    up_mask = window20["close"].astype(float) > window20["prev_close"].astype(float)
    down_mask = window20["close"].astype(float) < window20["prev_close"].astype(float)
    up_volumes = window20.loc[up_mask, "volume"].astype(float)
    down_volumes = window20.loc[down_mask, "volume"].astype(float)

    up_avg = float(up_volumes.mean()) if not up_volumes.empty else 0.0
    down_avg = float(down_volumes.mean()) if not down_volumes.empty else 0.0
    up_sum = float(up_volumes.sum()) if not up_volumes.empty else 0.0
    down_sum = float(down_volumes.sum()) if not down_volumes.empty else 0.0
    ratio_avg = _safe_ratio(up_avg, down_avg)
    ratio_sum = _safe_ratio(up_sum, down_sum)

    down_shrink_mask = (
        down_mask
        & (window20["daily_return"].astype(float) > float(cfg["down_shrink_return_floor"]))
        & (
            window20["volume"].astype(float)
            < window20["vol_ma5"].astype(float) * float(cfg["down_shrink_volume_factor"])
        )
    )
    up_expand_mask = (
        (window20["close"].astype(float) > window20["prev_close"].astype(float))
        & (window20["close"].astype(float) > window20["open"].astype(float))
        & (
            window20["volume"].astype(float)
            > window20["vol_ma5"].astype(float) * float(cfg["up_expand_volume_factor"])
        )
    )
    down_shrink_volumes = window20.loc[down_shrink_mask, "volume"].astype(float)
    up_expand_volumes = window20.loc[up_expand_mask, "volume"].astype(float)
    down_shrink_days = int(down_shrink_mask.sum())
    up_expand_days = int(up_expand_mask.sum())
    down_shrink_avg_vol = float(down_shrink_volumes.mean()) if not down_shrink_volumes.empty else float("nan")
    up_expand_avg_vol = float(up_expand_volumes.mean()) if not up_expand_volumes.empty else float("nan")
    up_expand_to_down_shrink_avg_vol_ratio = (
        float(up_expand_avg_vol / down_shrink_avg_vol)
        if (
            np.isfinite(up_expand_avg_vol)
            and np.isfinite(down_shrink_avg_vol)
            and up_expand_avg_vol > 0
            and down_shrink_avg_vol > 0
        )
        else float("nan")
    )

    start5_mean = float(window20["close"].head(5).mean())
    end5_mean = float(window20["close"].tail(5).mean())
    center_uplift = end5_mean / start5_mean - 1.0 if np.isfinite(start5_mean) and start5_mean > 0 else float("nan")
    volume_impulse_metrics = _evaluate_b1_volume_impulse(impulse_window, volume_impulse_cfg)

    raw_scores = {
        "volume_alignment": _score_b1_volume_alignment(
            ratio_avg,
            ratio_sum,
            up_expand_to_down_shrink_avg_vol_ratio,
        ),
        "down_shrink": _score_b1_down_shrink(down_shrink_days),
        "up_expand": _score_b1_up_expand(up_expand_days),
        "zx_retest": _score_b1_zx_retest(break_days, shrink_break_days, heavy_breakdown_days),
        "volume_impulse": int(volume_impulse_metrics["raw_score"]),
        "center_uplift": _score_b1_center_uplift(center_uplift),
    }
    scores, total_score = _scale_b1_weighted_scores(raw_scores, cfg["score"])

    weekly_tag = None
    if "wma_bull" in pf.columns:
        weekly_tag = "bull" if bool(row.get("wma_bull", False)) else "non_bull"

    metrics = {
        "j_today": round(j_today, 4) if np.isfinite(j_today) else None,
        "j_quantile_10pct": round(j_quantile, 4) if np.isfinite(j_quantile) else None,
        "zxdq": round(zxdq_today, 6) if np.isfinite(zxdq_today) else None,
        "zxdkx": round(zxdkx_today, 6) if np.isfinite(zxdkx_today) else None,
        "zxdq_above_zxdkx_pct": round(zxdq_above_zxdkx_pct, 6) if np.isfinite(zxdq_above_zxdkx_pct) else None,
        "ratio_avg": round(ratio_avg, 4) if np.isfinite(ratio_avg) else None,
        "ratio_sum": round(ratio_sum, 4) if np.isfinite(ratio_sum) else None,
        "down_shrink_avg_vol": round(down_shrink_avg_vol, 4) if np.isfinite(down_shrink_avg_vol) else None,
        "up_expand_avg_vol": round(up_expand_avg_vol, 4) if np.isfinite(up_expand_avg_vol) else None,
        "up_expand_to_down_shrink_avg_vol_ratio": (
            round(up_expand_to_down_shrink_avg_vol_ratio, 4)
            if np.isfinite(up_expand_to_down_shrink_avg_vol_ratio) else None
        ),
        "down_shrink_days": down_shrink_days,
        "up_expand_days": up_expand_days,
        "zx_break_days_5d": break_days,
        "shrink_break_days_5d": shrink_break_days,
        "heavy_breakdown_days_5d": heavy_breakdown_days,
        "distribution_days_20d": distribution_days,
        "center_uplift": round(center_uplift, 6) if np.isfinite(center_uplift) else None,
        "impulse_count_strict": int(volume_impulse_metrics["impulse_count_strict"]),
        "impulse_count_soft": int(volume_impulse_metrics["impulse_count_soft"]),
        "max_impulse_ratio": (
            round(float(volume_impulse_metrics["max_impulse_ratio"]), 4)
            if np.isfinite(volume_impulse_metrics["max_impulse_ratio"]) else None
        ),
        "latest_impulse_days_ago": volume_impulse_metrics["latest_impulse_days_ago"],
        "latest_impulse_is_strict": bool(volume_impulse_metrics["latest_impulse_is_strict"]),
        "impulse_high_area_count": int(volume_impulse_metrics["impulse_high_area_count"]),
        "volume_impulse_raw_score": int(volume_impulse_metrics["raw_score"]),
        "top_distribution_hit": bool(top_distribution_event),
        "top_distribution_bar_date": (top_distribution_event or {}).get("bar_date"),
        "top_distribution_vol_ratio": (top_distribution_event or {}).get("vol_ratio"),
        "top_distribution_close_to_recent_high": (top_distribution_event or {}).get("close_to_recent_high"),
        "top_distribution_upper_shadow_ratio": (top_distribution_event or {}).get("upper_shadow_ratio"),
        "top_distribution_weak_close_tags": (top_distribution_event or {}).get("weak_close_tags"),
        "weekly_tag": weekly_tag,
    }

    return {
        "hit": not hard_filter_reasons,
        "requested_date": pick_date.strftime("%Y-%m-%d"),
        "resolved_date": pick_date.strftime("%Y-%m-%d"),
        "hard_filter_reasons": hard_filter_reasons,
        "scores": scores,
        "score_total": total_score,
        "metrics": metrics,
        "extra": {
            "b1_score": total_score,
            "b1_score_version": cfg["score"]["version"],
            "b1_score_breakdown": scores,
            "b1_score_raw_breakdown": raw_scores,
            "b1_score_weights": dict(cfg["score"]["weights"]),
            "b1_metrics": metrics,
            "b1_hard_filter_reasons": hard_filter_reasons,
            "weekly_tag": weekly_tag,
        },
    }


def evaluate_b1e_from_prepared(
    pf: pd.DataFrame,
    pick_date: pd.Timestamp,
    cfg_b1e: Optional[dict] = None,
) -> dict:
    """兼容旧名称：原 B1-E 现已升级为正式 B1。"""
    return evaluate_b1_from_prepared(pf, pick_date, cfg_b1e)


# =============================================================================
# B1 策略
# =============================================================================

def _build_b1_legacy_selector(cfg_b1_legacy: dict) -> B1Selector:
    zx_m1, zx_m2, zx_m3, zx_m4 = _sorted_zx(
        cfg_b1_legacy["zx_m1"],
        cfg_b1_legacy["zx_m2"],
        cfg_b1_legacy["zx_m3"],
        cfg_b1_legacy["zx_m4"],
    )
    return B1Selector(
        j_threshold=float(cfg_b1_legacy["j_threshold"]),
        j_q_threshold=float(cfg_b1_legacy["j_q_threshold"]),
        kdj_n=int(cfg_b1_legacy.get("kdj_n", 9)),
        zx_m1=zx_m1, zx_m2=zx_m2, zx_m3=zx_m3, zx_m4=zx_m4,
        zxdq_span=int(cfg_b1_legacy.get("zxdq_span", 10)),
    )


def run_b1_legacy(
    prepared: Dict[str, pd.DataFrame],
    pick_date: pd.Timestamp,
    pool_codes: List[str],
    cfg_b1_legacy: dict,
) -> List[Candidate]:
    """在流动性池内运行 legacy/确认版旧 B1，返回 Candidate 列表.

    优化：对每只股票先调用 prepare_df() 预计算所有指标列，
    再用 vec_picks_from_prepared() 直接查表，避免重复计算。
    """
    selector = _build_b1_legacy_selector(cfg_b1_legacy)

    date_str = pick_date.strftime("%Y-%m-%d")
    candidates: List[Candidate] = []

    for code in pool_codes:
        df = prepared.get(code)
        if df is None or pick_date not in df.index:
            continue
        try:
            pf = selector.prepare_df(df)
            if selector.vec_picks_from_prepared(pf, start=pick_date, end=pick_date):
                row = pf.loc[pick_date]
                candidates.append(Candidate(
                    code=code,
                    date=date_str,
                    strategy="b1_legacy",
                    close=float(row["close"]),
                    turnover_n=float(row["turnover_n"]),
                ))
        except Exception as exc:
            logger.debug("B1 legacy skip %s: %s", code, exc)

    logger.info("B1 legacy 选出: %d 只", len(candidates))
    return candidates


def run_b1(
    prepared: Dict[str, pd.DataFrame],
    pick_date: pd.Timestamp,
    pool_codes: List[str],
    cfg_b1: Optional[dict] = None,
) -> List[Candidate]:
    """在流动性池内运行当前正式 B1（原 B1-E），返回按总分降序的 Candidate 列表。"""
    cfg = _normalize_b1_cfg(cfg_b1)
    date_str = pick_date.strftime("%Y-%m-%d")
    candidates: List[Candidate] = []

    for code in pool_codes:
        df = prepared.get(code)
        if df is None or pick_date not in df.index:
            continue
        try:
            pf = prepare_b1_df(df, cfg)
            evaluation = evaluate_b1_from_prepared(pf, pick_date, cfg)
            if not evaluation.get("hit", False):
                continue

            row = pf.loc[pick_date]
            candidates.append(Candidate(
                code=code,
                date=date_str,
                strategy="b1",
                close=float(row["close"]),
                turnover_n=float(row["turnover_n"]),
                extra=dict(evaluation.get("extra") or {}),
            ))
        except Exception as exc:
            logger.debug("B1 skip %s: %s", code, exc)

    candidates.sort(
        key=lambda c: (
            -_safe_float((c.extra or {}).get("b1_score"), -999),
            -_safe_float(((c.extra or {}).get("b1_score_breakdown") or {}).get("zx_retest"), -999),
            -_safe_float(((c.extra or {}).get("b1_score_breakdown") or {}).get("down_shrink"), -999),
            -_safe_float(((c.extra or {}).get("b1_score_breakdown") or {}).get("up_expand"), -999),
            -_safe_float(((c.extra or {}).get("b1_score_breakdown") or {}).get("volume_impulse"), -999),
            -_safe_float(((c.extra or {}).get("b1_score_breakdown") or {}).get("volume_alignment"), -999),
            -_safe_float(
                ((c.extra or {}).get("b1_metrics") or {}).get("up_expand_to_down_shrink_avg_vol_ratio"),
                -999,
            ),
            -_safe_float(((c.extra or {}).get("b1_metrics") or {}).get("zxdq_above_zxdkx_pct"), -999),
            -_safe_float(((c.extra or {}).get("b1_metrics") or {}).get("ratio_sum"), -999),
        )
    )
    logger.info("正式 B1 选出: %d 只", len(candidates))
    return candidates


def run_b1e(
    prepared: Dict[str, pd.DataFrame],
    pick_date: pd.Timestamp,
    pool_codes: List[str],
    cfg_b1e: Optional[dict] = None,
) -> List[Candidate]:
    """兼容旧名称：原 B1-E 现已升级为正式 B1。"""
    return run_b1(prepared, pick_date, pool_codes, cfg_b1e)


def evaluate_b1_for_code(
    *,
    code: str,
    pick_date: str,
    config_path: Optional[str] = None,
    data_dir: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    """评估单只股票在指定日期的当前正式 B1 结果，供调试/回看使用。"""
    cfg = load_config(config_path)
    g = cfg.get("global", {})
    _data_dir = str(_resolve_cfg_path(data_dir or g.get("data_dir", "./data/raw")))
    n_turnover_days = int(g.get("n_turnover_days", 43))

    raw_df = load_single_code_data(code, _data_dir, end_date=end_date)
    prepared_df = prepare_single_code_data(raw_df, n_turnover_days=n_turnover_days)
    resolved_date = _resolve_pick_date_in_df(prepared_df, pick_date)
    pf = prepare_b1_df(prepared_df, cfg.get("b1", {}))
    evaluation = evaluate_b1_from_prepared(pf, resolved_date, cfg.get("b1", {}))

    row = pf.loc[resolved_date]
    result = {
        "code": code,
        "strategy": "b1",
        "requested_date": pick_date,
        "resolved_date": resolved_date.strftime("%Y-%m-%d"),
        "close": round(float(row["close"]), 6),
        "turnover_n": round(float(row.get("turnover_n", 0.0)), 6),
        "hit": bool(evaluation.get("hit", False)),
        "hard_filter_reasons": list(evaluation.get("hard_filter_reasons", [])),
        "scores": dict(evaluation.get("scores") or {}),
        "score_total": evaluation.get("score_total"),
        "metrics": dict(evaluation.get("metrics") or {}),
    }
    return result


def evaluate_b1e_for_code(
    *,
    code: str,
    pick_date: str,
    config_path: Optional[str] = None,
    data_dir: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    """兼容旧名称：原 B1-E 现已升级为正式 B1。"""
    return evaluate_b1_for_code(
        code=code,
        pick_date=pick_date,
        config_path=config_path,
        data_dir=data_dir,
        end_date=end_date,
    )


def evaluate_b1_legacy_for_code(
    *,
    code: str,
    pick_date: str,
    config_path: Optional[str] = None,
    data_dir: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    """评估单只股票在指定日期的 legacy/确认版旧 B1 结果。"""
    cfg = load_config(config_path)
    g = cfg.get("global", {})
    _data_dir = str(_resolve_cfg_path(data_dir or g.get("data_dir", "./data/raw")))
    n_turnover_days = int(g.get("n_turnover_days", 43))
    cfg_b1_legacy = dict(cfg.get("b1_legacy", {}))
    if not cfg_b1_legacy:
        raise ValueError("配置中缺少 b1_legacy 段。")

    selector = _build_b1_legacy_selector(cfg_b1_legacy)
    raw_df = load_single_code_data(code, _data_dir, end_date=end_date)
    prepared_df = prepare_single_code_data(raw_df, n_turnover_days=n_turnover_days)
    resolved_date = _resolve_pick_date_in_df(prepared_df, pick_date)
    pf = selector.prepare_df(prepared_df)

    if resolved_date not in pf.index:
        raise ValueError(f"{code} 在 {resolved_date.date()} 无可用数据")
    loc = pf.index.get_loc(resolved_date)
    if isinstance(loc, slice):
        raise ValueError(f"{code} 在 {resolved_date.date()} 存在重复日期")
    loc = int(loc)
    row = pf.iloc[loc]
    hit = bool(selector.vec_picks_from_prepared(pf, start=resolved_date, end=resolved_date))

    j_today = float(row.get("J", np.nan))
    j_hist = pf["J"].iloc[:loc + 1].dropna().astype(float)
    j_quantile = float(j_hist.quantile(float(cfg_b1_legacy.get("j_q_threshold", 0.10)))) if not j_hist.empty else float("nan")
    close_t = float(row.get("close", np.nan))
    zxdq_t = float(row.get("zxdq", np.nan))
    zxdkx_t = float(row.get("zxdkx", np.nan))
    max_vol_mask = selector._max_vol_filter.vec_mask(pf) if selector._max_vol_filter is not None else np.ones(len(pf), dtype=bool)

    hard_filter_reasons: List[str] = []
    if not (
        np.isfinite(j_today)
        and (
            j_today < float(cfg_b1_legacy.get("j_threshold", 15.0))
            or (np.isfinite(j_quantile) and j_today <= j_quantile)
        )
    ):
        hard_filter_reasons.append("j_not_low_enough")
    if not (np.isfinite(close_t) and np.isfinite(zxdkx_t) and close_t > zxdkx_t):
        hard_filter_reasons.append("close_not_above_zxdkx")
    if not (np.isfinite(zxdq_t) and np.isfinite(zxdkx_t) and zxdq_t > zxdkx_t):
        hard_filter_reasons.append("zxdq_not_above_zxdkx")
    if not bool(row.get("wma_bull", False)):
        hard_filter_reasons.append("weekly_ma_not_bull")
    if loc >= len(max_vol_mask) or not bool(max_vol_mask[loc]):
        hard_filter_reasons.append("max_vol_day_bearish")

    return {
        "code": code,
        "strategy": "b1_legacy",
        "requested_date": pick_date,
        "resolved_date": resolved_date.strftime("%Y-%m-%d"),
        "close": round(float(row["close"]), 6),
        "turnover_n": round(float(row.get("turnover_n", 0.0)), 6),
        "hit": hit,
        "hard_filter_reasons": [] if hit else hard_filter_reasons,
        "scores": {},
        "score_total": None,
        "metrics": {
            "j_today": round(j_today, 4) if np.isfinite(j_today) else None,
            "j_quantile_threshold": round(j_quantile, 4) if np.isfinite(j_quantile) else None,
            "zxdq": round(zxdq_t, 6) if np.isfinite(zxdq_t) else None,
            "zxdkx": round(zxdkx_t, 6) if np.isfinite(zxdkx_t) else None,
            "close_gt_zxdkx": bool(np.isfinite(close_t) and np.isfinite(zxdkx_t) and close_t > zxdkx_t),
            "zxdq_gt_zxdkx": bool(np.isfinite(zxdq_t) and np.isfinite(zxdkx_t) and zxdq_t > zxdkx_t),
            "weekly_ma_bull": bool(row.get("wma_bull", False)),
            "max_vol_day_not_bearish": bool(loc < len(max_vol_mask) and max_vol_mask[loc]),
        },
    }


# =============================================================================
# B2 策略
# =============================================================================

def run_b2(
    prepared: Dict[str, pd.DataFrame],
    pick_date: pd.Timestamp,
    pool_codes: List[str],
    cfg_b2: dict,
    cfg_b1: dict,
) -> List[Candidate]:
    """在流动性池内运行 B2 策略（B1超跌位启动确认）。"""
    cfg_b1 = _normalize_b1_cfg(cfg_b1)

    lookback_days = int(cfg_b2.get("b1_lookback_days", 3))
    min_return = float(cfg_b2.get("min_return", 0.04))
    require_volume_up_vs_prev = bool(cfg_b2.get("require_volume_up_vs_prev", True))
    j_max = float(cfg_b2.get("j_max", 55))
    j_turn_mode = str(cfg_b2.get("j_turn_mode", "combo"))
    j_jump_min = float(cfg_b2.get("j_jump_min", 8))
    max_upper_shadow_ratio = float(cfg_b2.get("max_upper_shadow_ratio", 0.03))
    score_cfg = dict(cfg_b2.get("score", {}))

    date_str = pick_date.strftime("%Y-%m-%d")
    candidates: List[Candidate] = []

    for code in pool_codes:
        df = prepared.get(code)
        if df is None or pick_date not in df.index:
            continue
        try:
            pf = prepare_b1_df(df, cfg_b1)
            if pick_date not in pf.index:
                continue
            loc = pf.index.get_loc(pick_date)
            if isinstance(loc, slice) or int(loc) < 2:
                continue
            if not _has_recent_b1_hit(pf, pick_date, lookback_days):
                continue

            row_t = pf.iloc[int(loc)]
            row_t1 = pf.iloc[int(loc) - 1]
            row_t2 = pf.iloc[int(loc) - 2]

            close_t = float(row_t["close"])
            close_t1 = float(row_t1["close"])
            vol_t = float(row_t["volume"])
            vol_t1 = float(row_t1["volume"])
            j_t = float(row_t["J"])
            j_t1 = float(row_t1["J"])
            j_t2 = float(row_t2["J"])

            if not np.isfinite(close_t) or not np.isfinite(close_t1) or close_t1 <= 0:
                continue
            ret = close_t / close_t1 - 1.0
            if ret <= min_return:
                continue
            if require_volume_up_vs_prev and (not np.isfinite(vol_t) or not np.isfinite(vol_t1) or vol_t <= vol_t1):
                continue
            if not np.isfinite(j_t) or j_t >= j_max:
                continue
            if not _is_j_turn_up(j_t, j_t1, j_t2, mode=j_turn_mode, j_jump_min=j_jump_min):
                continue

            upper_shadow_ratio = _calc_upper_shadow_ratio(row_t)
            if upper_shadow_ratio is None or upper_shadow_ratio > max_upper_shadow_ratio:
                continue

            volume_ratio = None
            if np.isfinite(vol_t) and np.isfinite(vol_t1) and vol_t1 > 0:
                volume_ratio = vol_t / vol_t1

            extra = {
                "b2_return": round(ret, 6),
                "b2_j": round(j_t, 4),
                "b2_j_delta": round(j_t - j_t1, 4),
                "upper_shadow_ratio": round(float(upper_shadow_ratio), 6),
                "volume_ratio": round(float(volume_ratio), 6) if volume_ratio is not None else None,
                "b1_lookback_days": lookback_days,
            }
            extra.update(
                _score_b2_metrics(
                    ret=ret,
                    j=j_t,
                    j_delta=j_t - j_t1,
                    upper_shadow_ratio=upper_shadow_ratio,
                    volume_ratio=volume_ratio,
                    score_cfg=score_cfg,
                )
            )

            candidates.append(Candidate(
                code=code,
                date=date_str,
                strategy="b2",
                close=close_t,
                turnover_n=float(row_t["turnover_n"]),
                extra=extra,
            ))
        except Exception as exc:
            logger.debug("B2 skip %s: %s", code, exc)

    candidates.sort(
        key=lambda c: (
            -_safe_float((c.extra or {}).get("b2_score"), -999),
            -_safe_float((c.extra or {}).get("b2_return"), -999),
            (c.extra or {}).get("upper_shadow_ratio") if (c.extra or {}).get("upper_shadow_ratio") is not None else 999,
            -_safe_float((c.extra or {}).get("volume_ratio"), -999),
        )
    )
    logger.info("B2 选出: %d 只", len(candidates))
    return candidates


# =============================================================================
# 砖型图策略
# =============================================================================

def run_brick(
    prepared: Dict[str, pd.DataFrame],
    pick_date: pd.Timestamp,
    pool_codes: List[str],
    cfg_brick: dict,
) -> List[Candidate]:
    """在流动性池内运行砖型图策略，返回按 brick_growth 降序的 Candidate 列表.

    优化：对每只股票先调用 prepare_df() 预计算 brick/zxdq/wma_bull 等列，
    再用 vec_picks_from_prepared() 直接查表，brick_growth 也直接读预计算列，
    避免重复计算。
    """
    selector = BrickChartSelector(
        daily_return_threshold=float(cfg_brick.get("daily_return_threshold", 0.05)),
        brick_growth_ratio=float(cfg_brick.get("brick_growth_ratio", 1.0)),
        min_prior_green_bars=int(cfg_brick.get("min_prior_green_bars", 2)),
        zxdq_ratio=cfg_brick.get("zxdq_ratio"),
        zxdq_span=int(cfg_brick.get("zxdq_span", 10)),
        require_zxdq_gt_zxdkx=bool(cfg_brick.get("require_zxdq_gt_zxdkx", True)),
        zxdkx_m1=int(cfg_brick.get("zxdkx_m1", 14)),
        zxdkx_m2=int(cfg_brick.get("zxdkx_m2", 28)),
        zxdkx_m3=int(cfg_brick.get("zxdkx_m3", 57)),
        zxdkx_m4=int(cfg_brick.get("zxdkx_m4", 114)),
        require_weekly_ma_bull=bool(cfg_brick.get("require_weekly_ma_bull", True)),
        wma_short=int(cfg_brick.get("wma_short", 20)),
        wma_mid=int(cfg_brick.get("wma_mid", 60)),
        wma_long=int(cfg_brick.get("wma_long", 120)),
        n=int(cfg_brick.get("n", 4)),
        m1=int(cfg_brick.get("m1", 4)),
        m2=int(cfg_brick.get("m2", 6)),
        m3=int(cfg_brick.get("m3", 6)),
        t=float(cfg_brick.get("t", 4.0)),
        shift1=float(cfg_brick.get("shift1", 90.0)),
        shift2=float(cfg_brick.get("shift2", 100.0)),
        sma_w1=int(cfg_brick.get("sma_w1", 1)),
        sma_w2=int(cfg_brick.get("sma_w2", 1)),
        sma_w3=int(cfg_brick.get("sma_w3", 1)),
    )

    date_str = pick_date.strftime("%Y-%m-%d")
    candidates: List[Candidate] = []
    score_cfg = dict(cfg_brick.get("score", {}))

    for code in pool_codes:        
        df = prepared.get(code)        
        if df is None or pick_date not in df.index:
            continue
        try:
            pf = selector.prepare_df(df)
            if selector.vec_picks_from_prepared(pf, start=pick_date, end=pick_date):
                row = pf.loc[pick_date]
                bg = float(row["brick_growth"]) if "brick_growth" in pf.columns else selector.brick_growth_on_date(pf, pick_date)
                candidates.append(Candidate(
                    code=code,
                    date=date_str,
                    strategy="brick",
                    close=float(row["close"]),
                    turnover_n=float(row["turnover_n"]),
                    brick_growth=bg if np.isfinite(bg) else None,
                    extra={
                        "brick_growth": round(bg, 6) if np.isfinite(bg) else None,
                    },
                ))
        except Exception as exc:
            logger.debug("Brick skip %s: %s", code, exc)

    _apply_brick_scores(candidates, score_cfg=score_cfg)
    candidates.sort(
        key=lambda c: (
            -_safe_float((c.extra or {}).get("brick_score"), -999),
            -(c.brick_growth or -999),
        ),
        reverse=False,
    )
    logger.info("Brick 选出: %d 只", len(candidates))
    return candidates


# =============================================================================
# B2 + Brick 组合策略
# =============================================================================

def run_brick_reversal_xg(
    prepared: Dict[str, pd.DataFrame],
    pick_date: pd.Timestamp,
    pool_codes: List[str],
    cfg_xg: dict,
) -> List[Candidate]:
    """按用户新公式实现：昨天绿柱 AND 今天红柱 AND 高度达标 AND 黄线达标。"""
    zx_m1 = int(cfg_xg.get("zx_m1", 14))
    zx_m2 = int(cfg_xg.get("zx_m2", 28))
    zx_m3 = int(cfg_xg.get("zx_m3", 57))
    zx_m4 = int(cfg_xg.get("zx_m4", 114))
    height_ratio = float(cfg_xg.get("height_ratio", 2 / 3))
    date_str = pick_date.strftime("%Y-%m-%d")
    score_cfg = dict(cfg_xg.get("score", {}))

    candidates: List[Candidate] = []
    for code in pool_codes:
        df = prepared.get(code)
        if df is None or pick_date not in df.index:
            continue
        try:
            pf = df.copy().sort_index()
            loc = pf.index.get_loc(pick_date)
            if isinstance(loc, slice) or int(loc) < 2:
                continue

            zxdk = (
                pf["close"].rolling(zx_m1, min_periods=zx_m1).mean()
                + pf["close"].rolling(zx_m2, min_periods=zx_m2).mean()
                + pf["close"].rolling(zx_m3, min_periods=zx_m3).mean()
                + pf["close"].rolling(zx_m4, min_periods=zx_m4).mean()
            ) / 4.0
            pf["xg_yellow_line"] = zxdk

            c7 = compute_brick_chart(
                pf,
                n=int(cfg_xg.get("n", 4)),
                m1=int(cfg_xg.get("m1", 4)),
                m2=int(cfg_xg.get("m2", 6)),
                m3=int(cfg_xg.get("m3", 6)),
                t=float(cfg_xg.get("t", 4.0)),
                shift1=float(cfg_xg.get("shift1", 90.0)),
                shift2=float(cfg_xg.get("shift2", 100.0)),
                sma_w1=int(cfg_xg.get("sma_w1", 1)),
                sma_w2=int(cfg_xg.get("sma_w2", 1)),
                sma_w3=int(cfg_xg.get("sma_w3", 1)),
            )
            pf["xg_brick"] = c7

            row = pf.iloc[int(loc)]
            prev = pf.iloc[int(loc) - 1]
            prev2 = pf.iloc[int(loc) - 2]

            yellow = float(row.get("xg_yellow_line", np.nan))
            if not np.isfinite(yellow):
                continue
            yellow_ok = float(row["close"]) > yellow
            if not yellow_ok:
                continue

            c0 = float(row.get("xg_brick", 0.0))
            c1 = float(prev.get("xg_brick", 0.0))
            c2 = float(prev2.get("xg_brick", 0.0))
            today_red = c0 > c1
            yesterday_green = c1 < c2
            red_height = c0 - c1
            green_height = c2 - c1
            height_ok = red_height >= green_height * height_ratio
            if not (yesterday_green and today_red and height_ok):
                continue

            extra = {
                "xg_yellow_line": round(yellow, 6),
                "xg_brick": round(c0, 6),
                "xg_brick_prev": round(c1, 6),
                "xg_brick_prev2": round(c2, 6),
                "xg_red_height": round(red_height, 6),
                "xg_green_height": round(green_height, 6),
                "xg_height_ratio_need": round(green_height * height_ratio, 6),
                "xg_yesterday_green": yesterday_green,
                "xg_today_red": today_red,
                "xg_height_ok": height_ok,
                "xg_yellow_ok": yellow_ok,
            }
            extra.update(
                _score_xg_metrics(
                    red_height=red_height,
                    green_height=green_height,
                    close=float(row["close"]),
                    yellow_line=yellow,
                    brick_level=c0,
                    score_cfg=score_cfg,
                )
            )

            candidates.append(Candidate(
                code=code,
                date=date_str,
                strategy="brick_reversal_xg",
                close=float(row["close"]),
                turnover_n=float(row.get("turnover_n", 0.0)),
                extra=extra,
            ))
        except Exception as exc:
            logger.debug("BrickReversalXG skip %s: %s", code, exc)

    candidates.sort(
        key=lambda c: (
            -_safe_float((c.extra or {}).get("xg_score"), -999),
            -_safe_float((c.extra or {}).get("xg_red_height"), -999),
            -_safe_float((c.extra or {}).get("xg_brick"), -999),
        )
    )
    logger.info("BrickReversalXG 选出: %d 只", len(candidates))
    return candidates


def run_b2_brick(
    b2_candidates: List[Candidate],
    brick_candidates: List[Candidate],
    cfg_b2_brick: Optional[dict] = None,
) -> List[Candidate]:
    """组合策略：同一天同时命中 B2 和 brick_reversal_xg。策略名输出为 b2_xg_combo。"""
    b2_map = {c.code: c for c in b2_candidates}
    brick_map = {c.code: c for c in brick_candidates}
    overlap_codes = [code for code in b2_map.keys() if code in brick_map]
    score_cfg = dict((cfg_b2_brick or {}).get("score", {}))

    combined: List[Candidate] = []
    for code in overlap_codes:
        b2 = b2_map[code]
        brick = brick_map[code]
        extra = dict(b2.extra or {})
        extra["brick_growth"] = (brick.extra or {}).get("xg_brick", brick.brick_growth)
        extra["xg_score"] = (brick.extra or {}).get("xg_score")
        extra["xg_score_version"] = (brick.extra or {}).get("xg_score_version")
        extra["brick_score"] = (brick.extra or {}).get("xg_score", (brick.extra or {}).get("brick_score"))
        extra["matched_strategies"] = ["b2", "brick_reversal_xg", "b2_xg_combo"]

        candidate = Candidate(
            code=code,
            date=b2.date,
            strategy="b2_xg_combo",
            close=b2.close,
            turnover_n=b2.turnover_n,
            brick_growth=brick.brick_growth,
            extra=extra,
        )
        candidate.extra["combo_score"] = _score_combo_candidate(candidate, score_cfg=score_cfg)
        candidate.extra["combo_score_version"] = str(score_cfg.get("version", "v1"))
        candidate.extra["combo_strategy"] = "b2_xg_combo"
        combined.append(candidate)

    combined.sort(
        key=lambda c: (
            -_safe_float((c.extra or {}).get("combo_score"), -999),
            -_safe_float((c.extra or {}).get("b2_score"), -999),
            -(c.brick_growth or -999),
            (c.extra or {}).get("upper_shadow_ratio") if (c.extra or {}).get("upper_shadow_ratio") is not None else 999,
        )
    )
    logger.info("B2+Brick 选出: %d 只", len(combined))
    return combined


# =============================================================================
# 主入口
# =============================================================================

def run_preselect(
    *,
    config_path: Optional[str] = None,
    data_dir: Optional[str] = None,
    end_date: Optional[str] = None,
    pick_date: Optional[str] = None,
) -> Tuple[pd.Timestamp, List[Candidate]]:
    """
    量化初选主函数，返回 (pick_date_ts, List[Candidate])。
    不写任何文件，由 cli.py 负责落盘。

    参数
    ----
    config_path : rules_preselect.yaml 路径（None = 默认）
    data_dir    : CSV 目录（None = 读配置）
    end_date    : 数据截断日期（回测用）
    pick_date   : 选股基准日期（None = 自动最新）
    """
    cfg = load_config(config_path)
    g = cfg.get("global", {})

    _data_dir = str(_resolve_cfg_path(data_dir or g.get("data_dir", "./data/raw")))
    top_m = int(g.get("top_m", 20))
    n_turnover_days = int(g.get("n_turnover_days", 43))
    min_bars_buffer = int(g.get("min_bars_buffer", 10))

    # 1) 加载原始数据
    raw_data = load_raw_data(_data_dir, end_date=end_date)

    # 2) 计算 warmup_bars
    warmup = _calc_warmup(cfg, min_bars_buffer)

    # 3) 通用数据预处理
    preparer = MarketDataPreparer(
        end_date=pd.to_datetime(end_date) if end_date else None,
        warmup_bars=warmup,
        n_turnover_days=n_turnover_days,
        selector=None,
    )
    prepared = preparer.prepare(raw_data)

    # 4) 确定选股日期
    pick_ts = _resolve_pick_date(prepared, pick_date)
    logger.info("选股日期: %s", pick_ts.date())

    # 5) 构建流动性池
    pool_codes = TopTurnoverPoolBuilder(top_m=top_m).build(prepared).get(pick_ts, [])
    if not pool_codes:
        logger.warning("流动性池为空，pick_date=%s", pick_ts.date())
        return pick_ts, []

    logger.info("流动性池: %d 只", len(pool_codes))

    # 6) 运行各策略
    all_candidates: List[Candidate] = []
    b1_candidates: List[Candidate] = []
    b1_legacy_candidates: List[Candidate] = []
    b2_candidates: List[Candidate] = []
    brick_candidates: List[Candidate] = []
    brick_reversal_xg_candidates: List[Candidate] = []
    b2_xg_combo_candidates: List[Candidate] = []

    if cfg.get("b1", {}).get("enabled", True):
        b1_candidates = run_b1(prepared, pick_ts, pool_codes, cfg["b1"])
        all_candidates.extend(b1_candidates)

    if cfg.get("b1_legacy", {}).get("enabled", False):
        b1_legacy_candidates = run_b1_legacy(prepared, pick_ts, pool_codes, cfg["b1_legacy"])
        all_candidates.extend(b1_legacy_candidates)

    if cfg.get("b2", {}).get("enabled", False):
        b2_candidates = run_b2(prepared, pick_ts, pool_codes, cfg["b2"], cfg["b1"])
        all_candidates.extend(b2_candidates)

    if cfg.get("brick", {}).get("enabled", True):
        brick_candidates = run_brick(prepared, pick_ts, pool_codes, cfg["brick"])
        all_candidates.extend(brick_candidates)

    if cfg.get("brick_reversal_xg", {}).get("enabled", False):
        brick_reversal_xg_candidates = run_brick_reversal_xg(prepared, pick_ts, pool_codes, cfg["brick_reversal_xg"])
        all_candidates.extend(brick_reversal_xg_candidates)

    if cfg.get("b2_brick", {}).get("enabled", False):
        b2_xg_combo_candidates = run_b2_brick(b2_candidates, brick_reversal_xg_candidates, cfg.get("b2_brick", {}))
        all_candidates.extend(b2_xg_combo_candidates)

    # 7) 去重并保留多策略命中信息（优先级：b2_xg_combo > brick_reversal_xg > b2 > brick > b1 > b1_legacy）
    priority = {"b2_xg_combo": 6, "brick_reversal_xg": 5, "b2": 4, "brick": 3, "b1": 2, "b1_legacy": 1}
    merged: Dict[str, Candidate] = {}

    for c in all_candidates:
        existing = merged.get(c.code)
        if existing is None:
            c.extra = dict(c.extra or {})
            c.extra["matched_strategies"] = [c.strategy]
            merged[c.code] = c
            continue

        existing_extra = dict(existing.extra or {})
        current_extra = dict(c.extra or {})
        matched = list(existing_extra.get("matched_strategies", [existing.strategy]))
        if c.strategy not in matched:
            matched.append(c.strategy)

        # 关键规则：即便最终主策略被更高优先级策略替换，也保留已命中策略自己的评分与指标。
        merged_extra = dict(existing_extra)
        merged_extra.update(current_extra)
        merged_extra["matched_strategies"] = matched
        existing.extra = merged_extra

        if priority.get(c.strategy, 0) > priority.get(existing.strategy, 0):
            replacement = Candidate(
                code=c.code,
                date=c.date,
                strategy=c.strategy,
                close=c.close,
                turnover_n=c.turnover_n,
                brick_growth=c.brick_growth,
                extra=merged_extra,
            )
            merged[c.code] = replacement

    deduped = list(merged.values())

    logger.info("初选完成，候选股票: %d 只", len(deduped))
    return pick_ts, deduped
