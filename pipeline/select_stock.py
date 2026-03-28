"""
pipeline/select_stock.py
量化初选核心逻辑。

职责：
  - 读取 rules_preselect.yaml 参数
  - 加载 data/raw/*.csv 日线数据
  - 运行 B1 策略（KDJ + 知行均线）和砖型图策略
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
from Selector import B1Selector, BrickChartSelector, compute_brick_chart
from pipeline_core import MarketDataPreparer, TopTurnoverPoolBuilder

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG = _PROJECT_ROOT / "config" / "rules_preselect.yaml"


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

        df = pd.read_csv(fpath)
        df.columns = [c.lower() for c in df.columns]
        if "date" not in df.columns:
            logger.warning("跳过 %s：没有 date 列", fname)
            continue

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        if end_ts is not None:
            df = df[df["date"] <= end_ts].reset_index(drop=True)

        if not df.empty:
            data[code] = df

    if not data:
        raise ValueError(f"未找到任何 CSV 数据: {data_dir}")

    logger.info("读取股票数量: %d", len(data))
    return data


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
        warmup = max(warmup, int(cfg_b1.get("zx_m4", 371)) + buffer)

    cfg_b2 = cfg.get("b2", {})
    if cfg_b2.get("enabled", False):
        warmup = max(warmup, int(cfg_b1.get("zx_m4", 371)) + buffer)

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
    """检查最近 N 个交易日内是否命中过 B1。"""
    if "_vec_pick" not in pf.columns or pick_date not in pf.index:
        return False
    end_idx = pf.index.get_loc(pick_date)
    if isinstance(end_idx, slice):
        return False
    start_idx = max(0, int(end_idx) - max(1, lookback_days) + 1)
    window = pf.iloc[start_idx:int(end_idx) + 1]
    if window.empty:
        return False
    return bool(window["_vec_pick"].astype(bool).any())


# =============================================================================
# B1 策略
# =============================================================================

def run_b1(
    prepared: Dict[str, pd.DataFrame],
    pick_date: pd.Timestamp,
    pool_codes: List[str],
    cfg_b1: dict,
) -> List[Candidate]:
    """在流动性池内运行 B1 策略，返回 Candidate 列表.

    优化：对每只股票先调用 prepare_df() 预计算所有指标列，
    再用 vec_picks_from_prepared() 直接查表，避免重复计算。
    """
    zx_m1, zx_m2, zx_m3, zx_m4 = _sorted_zx(
        cfg_b1["zx_m1"], cfg_b1["zx_m2"], cfg_b1["zx_m3"], cfg_b1["zx_m4"]
    )
    selector = B1Selector(
        j_threshold=float(cfg_b1["j_threshold"]),
        j_q_threshold=float(cfg_b1["j_q_threshold"]),
        zx_m1=zx_m1, zx_m2=zx_m2, zx_m3=zx_m3, zx_m4=zx_m4,
    )

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
                    strategy="b1",
                    close=float(row["close"]),
                    turnover_n=float(row["turnover_n"]),
                ))
        except Exception as exc:
            logger.debug("B1 skip %s: %s", code, exc)

    logger.info("B1 选出: %d 只", len(candidates))
    return candidates


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
    zx_m1, zx_m2, zx_m3, zx_m4 = _sorted_zx(
        cfg_b1["zx_m1"], cfg_b1["zx_m2"], cfg_b1["zx_m3"], cfg_b1["zx_m4"]
    )
    selector = B1Selector(
        j_threshold=float(cfg_b1["j_threshold"]),
        j_q_threshold=float(cfg_b1["j_q_threshold"]),
        zx_m1=zx_m1, zx_m2=zx_m2, zx_m3=zx_m3, zx_m4=zx_m4,
    )

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
            pf = selector.prepare_df(df)
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
    b2_candidates: List[Candidate] = []
    brick_candidates: List[Candidate] = []
    brick_reversal_xg_candidates: List[Candidate] = []
    b2_xg_combo_candidates: List[Candidate] = []

    if cfg.get("b1", {}).get("enabled", True):
        b1_candidates = run_b1(prepared, pick_ts, pool_codes, cfg["b1"])
        all_candidates.extend(b1_candidates)

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

    # 7) 去重并保留多策略命中信息（优先级：b2_xg_combo > brick_reversal_xg > b2 > brick > b1）
    priority = {"b2_xg_combo": 6, "brick_reversal_xg": 5, "b2": 4, "brick": 2, "b1": 1}
    merged: Dict[str, Candidate] = {}

    for c in all_candidates:
        existing = merged.get(c.code)
        if existing is None:
            c.extra = dict(c.extra or {})
            c.extra["matched_strategies"] = [c.strategy]
            merged[c.code] = c
            continue

        matched = list(existing.extra.get("matched_strategies", [existing.strategy]))
        if c.strategy not in matched:
            matched.append(c.strategy)
        existing.extra["matched_strategies"] = matched

        if priority.get(c.strategy, 0) > priority.get(existing.strategy, 0):
            replacement = Candidate(
                code=c.code,
                date=c.date,
                strategy=c.strategy,
                close=c.close,
                turnover_n=c.turnover_n,
                brick_growth=c.brick_growth,
                extra=dict(c.extra or {}),
            )
            replacement.extra["matched_strategies"] = matched
            merged[c.code] = replacement

    deduped = list(merged.values())

    logger.info("初选完成，候选股票: %d 只", len(deduped))
    return pick_ts, deduped
