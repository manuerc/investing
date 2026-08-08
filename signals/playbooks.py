"""The three setup archetypes and their scoring.

Each playbook returns a score in 0..100 on bars where its hard preconditions
hold, and NaN elsewhere. Scores are computed on closed bars only.
"""

import numpy as np
import pandas as pd

from .regime import MULTIPLIER, RISK_ON


def _b(s: pd.Series) -> pd.Series:
    """Boolean series with NaN treated as False. `.shift()` on a bool column
    yields object dtype with NaN, hence the explicit eq(True)."""
    return s if s.dtype == bool else s.eq(True)


def score_A(x: pd.DataFrame, rs_top30: pd.Series) -> tuple[pd.Series, pd.DataFrame]:
    """Pullback in an established uptrend."""
    pre = (
        _b(x["close"] > x["ema200"])
        & _b(x["ema50"] > x["ema200"])
        & _b(x["ema50_slope"] > 0)
    )

    low5 = x["low"].rolling(5, min_periods=5).min()
    pull_deep = _b(low5 <= x["ema50"] * 1.01)
    pull_shallow = _b(low5 <= x["ema20"]) & ~pull_deep

    comp = pd.DataFrame(index=x.index)
    comp["pullback_deep"] = np.where(pull_deep, 20, 0)
    comp["pullback_shallow"] = np.where(pull_shallow, 12, 0)
    comp["rsi_reset"] = np.where(
        _b((x["rsi14"] >= 40) & (x["rsi14"] <= 55)),
        20,
        np.where(_b((x["rsi14"] >= 35) & (x["rsi14"] < 40)), 15, 0),
    )
    comp["reversal_bar"] = np.where(_b(x["bull_reversal"]), 15, 0)
    comp["rel_strength"] = np.where(_b(rs_top30), 15, 0)
    comp["volume"] = np.where(_b(x["vol_ratio"] > 1.2), 10, 0)
    comp["atr_contracting"] = np.where(_b(x["atr_falling"]), 10, 0)
    comp["macd_turn"] = np.where(_b(x["macd_turning_up"]), 10, 0)

    score = comp.sum(axis=1).where(pre, np.nan)
    return score, comp


def score_B(x: pd.DataFrame, rs_pos: pd.Series) -> tuple[pd.Series, pd.DataFrame]:
    """Volatility-squeeze breakout."""
    # The squeeze must have existed on the bars BEFORE the breakout bar.
    pre = _b(x["close"] > x["ema200"]) & _b(x["squeeze_5d"].shift(1))

    range10 = (x["high"].rolling(10, min_periods=10).max()
               - x["low"].rolling(10, min_periods=10).min()).shift(1)

    comp = pd.DataFrame(index=x.index)
    comp["breaks_range"] = np.where(_b(x["close"] > x["range_high_20"]), 25, 0)
    comp["volume_thrust"] = np.where(_b(x["vol_ratio"] >= 1.8), 20, 0)
    comp["near_52w_high"] = np.where(_b(x["near_52w_high"]), 15, 0)
    comp["close_high_in_bar"] = np.where(_b(x["bar_pos"] > 0.66), 10, 0)
    comp["adx_cross"] = np.where(_b(x["adx_cross20"]), 10, 0)
    comp["tight_range"] = np.where(_b(range10 < 3.0 * x["atr14"]), 10, 0)
    comp["rel_strength"] = np.where(_b(rs_pos), 10, 0)

    score = comp.sum(axis=1).where(pre, np.nan)
    return score, comp


def score_C(x: pd.DataFrame, regime: pd.Series) -> tuple[pd.Series, pd.DataFrame]:
    """Oversold mean reversion inside a structurally healthy asset."""
    pre = (
        _b(x["close"] > x["ema200_w"])
        & (_b(x["rsi2"] < 5) | _b(x["rsi14"] < 30))
        & (regime.reindex(x.index) == RISK_ON)
    )

    near_swing = _b((x["close"] - x["swing_low_20"]).abs() / x["atr14"] < 1.0)
    near_ema200 = _b((x["close"] - x["ema200"]).abs() / x["atr14"] < 1.0)

    comp = pd.DataFrame(index=x.index)
    comp["rsi_divergence"] = np.where(_b(x["rsi_bull_div"]), 30, 0)
    comp["support_nearby"] = np.where(near_swing | near_ema200, 15, 0)
    comp["capitulation"] = np.where(_b(x["drop_3d_atr"] >= 2.0), 15, 0)
    comp["volume_climax"] = np.where(_b(x["vol_ratio"] > 2.0), 15, 0)
    comp["outside_lower_bb"] = np.where(_b(x["bb_pctb"] < 0), 15, 0)
    comp["reversal_bar"] = np.where(_b(x["bull_reversal"]), 10, 0)

    score = comp.sum(axis=1).where(pre, np.nan)
    return score, comp


def score_all(
    x: pd.DataFrame, regime: pd.Series, rs_top30: pd.Series, rs_pos: pd.Series, enabled: list[str]
) -> pd.DataFrame:
    """Score every enabled playbook and apply the regime multiplier."""
    mult = regime.reindex(x.index).map(MULTIPLIER).astype(float)

    out = pd.DataFrame(index=x.index)
    comps = {}
    if "A" in enabled:
        out["A"], comps["A"] = score_A(x, rs_top30)
    if "B" in enabled:
        out["B"], comps["B"] = score_B(x, rs_pos)
    if "C" in enabled:
        out["C"], comps["C"] = score_C(x, regime)

    out = out.mul(mult, axis=0)

    cols = [c for c in ("A", "B", "C") if c in out.columns]
    best = out[cols].max(axis=1)
    out["best_score"] = best
    out["best_playbook"] = out[cols].fillna(-1.0).idxmax(axis=1).where(best.notna())
    out["regime"] = regime.reindex(x.index)
    return out, comps
