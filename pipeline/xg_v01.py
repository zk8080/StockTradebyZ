from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "candidates" / "candidates_latest.json"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "bitable_sync"

PATTERN_BONUS_MAP = {
    "n_breakout": 3,
    "box_breakout": 3,
    "trend_continuation": 1,
    "unknown": 0,
}


def box_breakout_subtype_and_quality(features: Dict[str, object], risk_note: str) -> Tuple[str, float]:
    clean_break = bool(
        safe_float(features.get("breakout_high_margin_prev15"), -9.0) >= 0.0
        or safe_float(features.get("breakout_close_margin_prev15"), -9.0) >= -0.01
    )
    dynamic_break = bool(
        safe_float(features.get("breakout_high_margin_last5"), -9.0) >= -0.005
        or safe_float(features.get("breakout_close_margin_last5"), -9.0) >= -0.015
    )
    subtype = "clean" if clean_break else "dynamic"

    contract = safe_float(features.get("range_contract_ratio"), 0.0)
    center = safe_float(features.get("center_shift_pct"), 0.0)
    upper = safe_float(features.get("upper_shadow_ratio"), 0.0)
    vol = safe_float(features.get("volume_ratio_10"), 0.0)
    day = safe_float(features.get("day_change"), 0.0)

    score = 0.0
    score += 2.0 if contract <= 0.50 else (1.0 if contract <= 0.65 else 0.0)
    score += 1.2 if 0.0 <= center <= 0.12 else (0.6 if center > 0 else 0.0)
    score += 1.0 if upper <= 0.015 else (0.5 if upper <= 0.025 else 0.0)
    score += 1.0 if 1.0 <= vol <= 1.8 else (0.5 if 0.95 <= vol <= 2.2 else 0.0)
    score += 0.8 if 0.045 <= day <= 0.075 else (0.4 if 0.045 <= day <= 0.085 else 0.0)
    score += 1.5 if clean_break else (0.8 if dynamic_break else 0.0)
    score += 0.5 if risk_note == "none" else 0.0
    return subtype, round(score, 3)

RISK_PENALTY_MAP = {
    "none": 0,
    "weak_close": 1,
    "volume_unconfirmed": 1,
    "upper_shadow": 2,
    "overextended": 2,
    "pattern_ambiguous": 1,
    "insufficient_kline": 2,
}

ACTION_BUCKET_THRESHOLDS = {
    "main_candidate_min": 68.0,
    "watchlist_min": 52.0,
    "delete_first_max": 45.0,
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def candidate_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if not isinstance(payload, dict):
        return []
    rows = payload.get("candidates") or payload.get("items") or []
    return [x for x in rows if isinstance(x, dict)]


def pick_date_of(payload: Any, items: list[dict[str, Any]]) -> str:
    if isinstance(payload, dict):
        value = str(payload.get("pick_date") or "").strip()
        if value:
            return value
    for item in items:
        value = str(item.get("date") or "").strip()
        if value:
            return value
    return ""


def matched_strategies(item: dict[str, Any]) -> list[str]:
    extra = item.get("extra") or {}
    matched = extra.get("matched_strategies") or []
    if isinstance(matched, list) and matched:
        return [str(x) for x in matched if str(x).strip()]
    strategy = str(item.get("strategy") or "").strip()
    return [strategy] if strategy else []


def is_xg_hit(item: dict[str, Any]) -> bool:
    strategy = str(item.get("strategy") or "")
    matched = set(matched_strategies(item))
    return strategy == "brick_reversal_xg" or (
        strategy == "b2_xg_combo" and "brick_reversal_xg" in matched
    )


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def pct_text(value: float | None) -> str:
    if value is None:
        return "?"
    return f"{value * 100:.1f}%"


class RawKlineStore:
    def __init__(self, raw_dir: Path) -> None:
        self.raw_dir = raw_dir
        self._cache: dict[str, pd.DataFrame] = {}

    def load(self, code: str) -> pd.DataFrame:
        if code in self._cache:
            return self._cache[code]

        path = self.raw_dir / f"{code}.csv"
        if not path.exists():
            df = pd.DataFrame()
            self._cache[code] = df
            return df

        try:
            df = pd.read_csv(path)
            if df.empty:
                self._cache[code] = df
                return df
            df.columns = [str(c).lower() for c in df.columns]
            required = ["date", "open", "high", "low", "close", "volume"]
            if not set(required).issubset(set(df.columns)):
                self._cache[code] = pd.DataFrame()
                return self._cache[code]
            df = df.loc[:, required].copy(deep=True)
            df.loc[:, "date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True).copy(deep=True)
            for col in ["open", "high", "low", "close", "volume"]:
                df.loc[:, col] = pd.to_numeric(df[col], errors="coerce")
            self._cache[code] = df
            return df
        except Exception:
            self._cache[code] = pd.DataFrame()
            return self._cache[code]


def kline_features(raw_df: pd.DataFrame, pick_date: str) -> dict[str, Any]:
    if raw_df.empty:
        return {"has_kline": False, "has_pick_date": False, "bars": 0}

    ts = pd.Timestamp(pick_date)
    df = raw_df.loc[raw_df["date"] <= ts].copy()
    if df.empty:
        return {"has_kline": True, "has_pick_date": False, "bars": 0}

    df = df.sort_values("date").reset_index(drop=True)
    last_trade_date = df.iloc[-1]["date"]
    has_pick_date = bool(last_trade_date.normalize() == ts.normalize())
    if not has_pick_date:
        return {
            "has_kline": True,
            "has_pick_date": False,
            "bars": int(len(df)),
            "last_trade_date": last_trade_date.strftime("%Y-%m-%d"),
        }

    bars = int(len(df))
    if bars < 25:
        return {"has_kline": True, "has_pick_date": True, "bars": bars}

    current = df.iloc[-1]
    prev15 = df.iloc[-16:-1]
    prev20 = df.iloc[-21:-1]
    older_swing = df.iloc[-16:-6]
    recent_pullback = df.iloc[-6:-1]
    first5 = prev15.head(5)
    last5 = prev15.tail(5)
    prev10_ex_last5 = prev15.iloc[:-5]
    last7 = prev15.tail(7)

    close = safe_float(current["close"])
    high = safe_float(current["high"])
    low = safe_float(current["low"])
    open_ = safe_float(current["open"])
    volume = safe_float(current["volume"])

    ma5 = safe_float(df["close"].tail(5).mean())
    ma10 = safe_float(df["close"].tail(10).mean())
    ma20 = safe_float(df["close"].tail(20).mean())
    vol_ma10 = safe_float(df["volume"].iloc[-11:-1].mean())

    prev15_high = safe_float(prev15["high"].max())
    prev15_low = safe_float(prev15["low"].min())
    prev20_high = safe_float(prev20["high"].max())
    older_high = safe_float(older_swing["high"].max()) if not older_swing.empty else 0.0
    pullback_low = safe_float(recent_pullback["low"].min()) if not recent_pullback.empty else 0.0
    first5_close_mean = safe_float(first5["close"].mean()) if not first5.empty else 0.0
    last5_close_mean = safe_float(last5["close"].mean()) if not last5.empty else 0.0
    first5_high = safe_float(first5["high"].max()) if not first5.empty else 0.0
    first5_low = safe_float(first5["low"].min()) if not first5.empty else 0.0
    last5_high = safe_float(last5["high"].max()) if not last5.empty else 0.0
    last5_low = safe_float(last5["low"].min()) if not last5.empty else 0.0
    prev10_ex_last5_vol_ma = safe_float(prev10_ex_last5["volume"].mean()) if not prev10_ex_last5.empty else 0.0
    last7_high = safe_float(last7["high"].max()) if not last7.empty else 0.0
    last7_low = safe_float(last7["low"].min()) if not last7.empty else 0.0

    bar_range = max(high - low, 1e-9)
    volume_ratio_10 = volume / vol_ma10 if vol_ma10 > 0 else 0.0
    box_range_ratio = (prev15_high - prev15_low) / prev15_low if prev15_low > 0 else 0.0
    pullback_depth = (older_high - pullback_low) / older_high if older_high > 0 else 0.0
    upper_shadow_ratio = max(0.0, high - max(open_, close)) / close if close > 0 else 0.0
    close_strength = (close - low) / bar_range
    extension_vs_ma20 = close / ma20 - 1.0 if ma20 > 0 else 0.0
    day_change = close / safe_float(df.iloc[-2]["close"], close) - 1.0 if bars >= 2 and safe_float(df.iloc[-2]["close"], 0.0) > 0 else 0.0
    first5_range_ratio = (first5_high - first5_low) / first5_low if first5_low > 0 else 0.0
    last5_range_ratio = (last5_high - last5_low) / last5_low if last5_low > 0 else 0.0
    range_contract_ratio = last5_range_ratio / first5_range_ratio if first5_range_ratio > 0 else 0.0
    center_shift_pct = last5_close_mean / first5_close_mean - 1.0 if first5_close_mean > 0 else 0.0
    vol_last5_vs_prev10 = safe_float(last5["volume"].mean()) / prev10_ex_last5_vol_ma if prev10_ex_last5_vol_ma > 0 else 0.0
    breakout_close_margin_prev15 = close / prev15_high - 1.0 if prev15_high > 0 else 0.0
    breakout_high_margin_prev15 = high / prev15_high - 1.0 if prev15_high > 0 else 0.0
    breakout_close_margin_last5 = close / last5_high - 1.0 if last5_high > 0 else 0.0
    breakout_high_margin_last5 = high / last5_high - 1.0 if last5_high > 0 else 0.0
    last7_range_ratio = (last7_high - last7_low) / last7_low if last7_low > 0 else 0.0
    flatness_score = (last7_range_ratio + last5_range_ratio) / 2.0 if last7_range_ratio > 0 and last5_range_ratio > 0 else max(last7_range_ratio, last5_range_ratio)
    platform_days_ge3 = bool(last7_range_ratio > 0 and last5_range_ratio > 0)

    return {
        "has_kline": True,
        "has_pick_date": True,
        "bars": bars,
        "trade_date": last_trade_date.strftime("%Y-%m-%d"),
        "close": round(close, 6),
        "open": round(open_, 6),
        "high": round(high, 6),
        "low": round(low, 6),
        "ma5": round(ma5, 6),
        "ma10": round(ma10, 6),
        "ma20": round(ma20, 6),
        "volume_ratio_10": round(volume_ratio_10, 6),
        "box_range_ratio": round(box_range_ratio, 6),
        "pullback_depth": round(pullback_depth, 6),
        "upper_shadow_ratio": round(upper_shadow_ratio, 6),
        "close_strength": round(close_strength, 6),
        "extension_vs_ma20": round(extension_vs_ma20, 6),
        "day_change": round(day_change, 6),
        "first5_range_ratio": round(first5_range_ratio, 6),
        "last5_range_ratio": round(last5_range_ratio, 6),
        "range_contract_ratio": round(range_contract_ratio, 6),
        "center_shift_pct": round(center_shift_pct, 6),
        "vol_last5_vs_prev10": round(vol_last5_vs_prev10, 6),
        "last7_range_ratio": round(last7_range_ratio, 6),
        "flatness_score": round(flatness_score, 6),
        "platform_days_ge3": bool(platform_days_ge3),
        "breakout_close_margin_prev15": round(breakout_close_margin_prev15, 6),
        "breakout_high_margin_prev15": round(breakout_high_margin_prev15, 6),
        "breakout_close_margin_last5": round(breakout_close_margin_last5, 6),
        "breakout_high_margin_last5": round(breakout_high_margin_last5, 6),
        "breakout_prev15": bool(close >= prev15_high * 0.995) if prev15_high > 0 else False,
        "breakout_prev20": bool(close >= prev20_high * 0.995) if prev20_high > 0 else False,
        "close_above_ma5": bool(close >= ma5) if ma5 > 0 else False,
        "close_above_ma10": bool(close >= ma10) if ma10 > 0 else False,
        "close_above_ma20": bool(close >= ma20) if ma20 > 0 else False,
        "bullish_ma_stack": bool(ma5 >= ma10 >= ma20) if ma20 > 0 else False,
    }


def classify_pattern(
    candidate: dict[str, Any],
    features: dict[str, Any],
) -> tuple[str, str]:
    extra = candidate.get("extra") or {}
    xg_yellow_margin = safe_float(extra.get("xg_yellow_margin"), 0.0)

    if not features.get("has_kline") or not features.get("has_pick_date"):
        return "unknown", "缺少可复用日线数据，先保留为人工确认。"
    if int(features.get("bars") or 0) < 25:
        return "unknown", "历史K线不足 25 根，先保留为人工确认。"

    box_center_ok = safe_float(features.get("center_shift_pct")) >= 0.0
    box_day_ok = 0.045 <= safe_float(features.get("day_change")) <= 0.085
    box_close_ok = safe_float(features.get("close_strength")) >= 0.65
    box_shadow_ok = safe_float(features.get("upper_shadow_ratio")) <= 0.028
    box_volume_ok = 0.95 <= safe_float(features.get("volume_ratio_10")) <= 2.3
    box_platform_thick_ok = bool(features.get("platform_days_ge3"))
    box_break_ok = (
        safe_float(features.get("breakout_high_margin_prev15")) >= 0.0
        or safe_float(features.get("breakout_close_margin_prev15")) >= -0.02
        or safe_float(features.get("breakout_high_margin_last5")) >= -0.015
        or safe_float(features.get("breakout_close_margin_last5")) >= -0.03
    )

    if box_center_ok and box_day_ok and box_close_ok and box_shadow_ok and box_volume_ok and box_platform_thick_ok and box_break_ok:
        return (
            "box_breakout",
            "近15日存在平台整理，价格重心未下移，且当日以中强阳线脱离平台。",
        )

    return "unknown", "当前 XG 先只聚焦 box_breakout，其他形态暂不参与正式分类。"


def assess_risk(
    pattern_type: str,
    features: dict[str, Any],
    candidate: dict[str, Any],
) -> tuple[str, int]:
    extra = candidate.get("extra") or {}
    xg_yellow_margin = safe_float(extra.get("xg_yellow_margin"), 0.0)

    if not features.get("has_kline") or not features.get("has_pick_date"):
        return "insufficient_kline", RISK_PENALTY_MAP["insufficient_kline"]
    if int(features.get("bars") or 0) < 25:
        return "insufficient_kline", RISK_PENALTY_MAP["insufficient_kline"]
    if safe_float(features.get("upper_shadow_ratio")) >= 0.04:
        return "upper_shadow", RISK_PENALTY_MAP["upper_shadow"]
    if safe_float(features.get("extension_vs_ma20")) >= 0.15 or xg_yellow_margin >= 0.18:
        return "overextended", RISK_PENALTY_MAP["overextended"]
    if pattern_type in {"n_breakout", "box_breakout"} and safe_float(features.get("close_strength")) < 0.60:
        return "weak_close", RISK_PENALTY_MAP["weak_close"]
    if safe_float(features.get("volume_ratio_10")) < 0.90:
        return "volume_unconfirmed", RISK_PENALTY_MAP["volume_unconfirmed"]
    if pattern_type == "unknown":
        return "pattern_ambiguous", RISK_PENALTY_MAP["pattern_ambiguous"]
    return "none", 0


def decide_action_bucket(pattern_type: str, final_score: float, risk_note: str, risk_penalty: int) -> str:
    if risk_note == "insufficient_kline":
        return "manual_review"
    if pattern_type == "unknown":
        return "manual_review" if final_score >= ACTION_BUCKET_THRESHOLDS["delete_first_max"] else "delete_first"
    if final_score >= ACTION_BUCKET_THRESHOLDS["main_candidate_min"] and risk_penalty <= 1:
        return "main_candidate"
    if final_score >= ACTION_BUCKET_THRESHOLDS["watchlist_min"]:
        return "watchlist"
    if final_score < ACTION_BUCKET_THRESHOLDS["delete_first_max"] or (risk_penalty >= 2 and final_score < 55.0):
        return "delete_first"
    return "manual_review"


def enrich_candidate(
    candidate: dict[str, Any],
    *,
    pick_date: str,
    raw_store: RawKlineStore,
) -> dict[str, Any]:
    code = str(candidate.get("code") or "").strip()
    extra = candidate.get("extra") or {}
    strategy = str(candidate.get("strategy") or "").strip()
    matched = matched_strategies(candidate)
    effective_pick_date = str(candidate.get("date") or pick_date or "").strip()

    raw_df = raw_store.load(code)
    features = kline_features(raw_df, effective_pick_date)
    pattern_type, confirmation_note = classify_pattern(candidate, features)
    pattern_bonus = int(PATTERN_BONUS_MAP[pattern_type])
    risk_note, risk_penalty = assess_risk(pattern_type, features, candidate)
    xg_score_raw = round(safe_float(extra.get("xg_score"), 0.0), 1)
    final_score = round(xg_score_raw + pattern_bonus - risk_penalty, 1)
    box_subtype = ""
    quality_score = 0.0
    order_score = final_score
    if pattern_type == "box_breakout":
        box_subtype, quality_score = box_breakout_subtype_and_quality(features, risk_note)
        order_score = round(final_score + quality_score, 3)
    action_bucket = decide_action_bucket(pattern_type, final_score, risk_note, risk_penalty)

    item = {
        "code": code,
        "pick_date": effective_pick_date,
        "xg_score_raw": xg_score_raw,
        "pattern_type": pattern_type,
        "pattern_bonus": pattern_bonus,
        "confirmation_note": confirmation_note,
        "risk_note": risk_note,
        "risk_penalty": risk_penalty,
        "final_score": final_score,
        "box_subtype": box_subtype,
        "quality_score": quality_score,
        "order_score": order_score,
        "action_bucket": action_bucket,
        "strategy": strategy,
        "matched_strategies": matched,
    }

    if features.get("has_kline"):
        item["kline_context"] = {
            "bars": int(features.get("bars") or 0),
            "trade_date": features.get("trade_date") or features.get("last_trade_date") or effective_pick_date,
            "breakout_prev15": bool(features.get("breakout_prev15", False)),
            "breakout_prev20": bool(features.get("breakout_prev20", False)),
            "box_range_ratio": safe_float(features.get("box_range_ratio"), 0.0),
            "pullback_depth": safe_float(features.get("pullback_depth"), 0.0),
            "volume_ratio_10": safe_float(features.get("volume_ratio_10"), 0.0),
            "upper_shadow_ratio": safe_float(features.get("upper_shadow_ratio"), 0.0),
            "extension_vs_ma20": safe_float(features.get("extension_vs_ma20"), 0.0),
            "day_change": safe_float(features.get("day_change"), 0.0),
            "range_contract_ratio": safe_float(features.get("range_contract_ratio"), 0.0),
            "center_shift_pct": safe_float(features.get("center_shift_pct"), 0.0),
            "vol_last5_vs_prev10": safe_float(features.get("vol_last5_vs_prev10"), 0.0),
            "breakout_close_margin_prev15": safe_float(features.get("breakout_close_margin_prev15"), 0.0),
            "breakout_high_margin_prev15": safe_float(features.get("breakout_high_margin_prev15"), 0.0),
            "breakout_close_margin_last5": safe_float(features.get("breakout_close_margin_last5"), 0.0),
            "breakout_high_margin_last5": safe_float(features.get("breakout_high_margin_last5"), 0.0),
            "box_subtype": box_subtype,
            "quality_score": quality_score,
            "order_score": order_score,
            "last5_range_ratio": safe_float(features.get("last5_range_ratio"), 0.0),
            "last7_range_ratio": safe_float(features.get("last7_range_ratio"), 0.0),
            "flatness_score": safe_float(features.get("flatness_score"), 0.0),
        }
    return item


def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pattern_counts": dict(Counter(str(x.get("pattern_type") or "unknown") for x in items)),
        "risk_counts": dict(Counter(str(x.get("risk_note") or "none") for x in items)),
        "action_bucket_counts": dict(Counter(str(x.get("action_bucket") or "manual_review") for x in items)),
    }


def enrich_xg_candidates(
    input_path: Path,
    *,
    raw_dir: Path = DEFAULT_RAW_DIR,
    top_n: int = 0,
) -> dict[str, Any]:
    payload = load_json(input_path)
    items = candidate_items(payload)
    pick_date = pick_date_of(payload, items)
    xg_items = [x for x in items if is_xg_hit(x)]
    xg_items.sort(
        key=lambda row: (
            -safe_float((row.get("extra") or {}).get("xg_score"), 0.0),
            str(row.get("code") or ""),
        )
    )

    raw_store = RawKlineStore(raw_dir)
    enriched = [enrich_candidate(row, pick_date=pick_date, raw_store=raw_store) for row in xg_items]
    enriched = [row for row in enriched if str(row.get("pattern_type") or "") == "box_breakout"]
    enriched.sort(
        key=lambda row: (
            -safe_float(row.get("order_score"), safe_float(row.get("final_score"), 0.0)),
            -safe_float(row.get("final_score"), 0.0),
            -safe_float(row.get("quality_score"), 0.0),
            -safe_float(row.get("xg_score_raw"), 0.0),
            str(row.get("code") or ""),
        )
    )

    if top_n > 0:
        enriched = enriched[:top_n]

    for idx, row in enumerate(enriched, start=1):
        row["rank"] = idx

    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return {
        "version": "xg_v0.1",
        "generated_at": generated_at,
        "pick_date": pick_date,
        "source_candidates_path": str(input_path),
        "raw_dir": str(raw_dir),
        "count": len(enriched),
        "risk_note_allow_list": list(RISK_PENALTY_MAP.keys()),
        "pattern_bonus_map": dict(PATTERN_BONUS_MAP),
        "risk_penalty_map": dict(RISK_PENALTY_MAP),
        "action_bucket_thresholds": dict(ACTION_BUCKET_THRESHOLDS),
        "summary": summarize(enriched),
        "items": enriched,
    }
