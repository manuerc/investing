"""Exit rules, researched on the WIDE universe.

The first pass at exits was fitted on the 15 personal watchlist names — 338
trades from a small set of correlated stocks. That is the same mistake the
entry research was designed to avoid. Here the rule is chosen on the 82-name
research universe (~3,200 signals) and the watchlist is held out to validate.

Every rule is judged against the base rate at ITS OWN average holding period.
Without that, "hold longer" always wins in a rising market — which is exactly
how the RSI>70 exit fooled the first analysis.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from research.events import build_events
from signals.data import load_ohlcv
from signals.indicators import atr, ema, rsi

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
MAXBARS = 63
WATCHLIST = {"MELI", "MSFT", "NVDA", "JNJ", "META", "NOW", "KO", "EWZ",
             "IBM", "GOOGL", "AAPL"}


def build_paths(min_conditions: int = 5, maxbars: int = MAXBARS) -> pd.DataFrame:
    """One row per signal, carrying the forward price path and entry-time state."""
    ev = build_events(min_conditions=min_conditions)
    rows = []
    for asset, g in ev.groupby("asset"):
        try:
            px = load_ohlcv(asset, "2009-01-01", "2026-08-08")
        except Exception:
            continue
        o, h, l, c = (px[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        e20 = ema(px["close"], 20).to_numpy(float)
        r14 = rsi(px["close"], 14).to_numpy(float)
        a14 = atr(px["high"], px["low"], px["close"], 14).to_numpy(float)
        idx = {d: i for i, d in enumerate(px.index)}
        for _, r in g.iterrows():
            i = idx.get(r["date"])
            if i is None or i + maxbars + 1 >= len(px):
                continue
            j = i + 1                                   # entry at next open
            rows.append({
                "asset": asset, "date": r["date"], "entry": o[j],
                "atr_pct": a14[i] / c[i], "n_conditions": r["n_conditions"],
                "rsi14_0": r14[i], "atr_rank": r.get("atr_pct_rank252", np.nan),
                "high": h[j:j + maxbars], "low": l[j:j + maxbars],
                "close": c[j:j + maxbars], "ema20": e20[j:j + maxbars],
                "rsi14": r14[j:j + maxbars], "atr": a14[i],
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------- exit rules

def ex_fixed(p, n):
    k = min(n, len(p["close"])) - 1
    return p["close"][k] / p["entry"] - 1, k + 1


def ex_target_atr(p, mult, cap=MAXBARS):
    """Take profit at `mult` ATR above entry; time-stop at the cap."""
    tgt = p["entry"] + mult * p["atr"]
    for k in range(min(cap, len(p["high"]))):
        if p["high"][k] >= tgt:
            return tgt / p["entry"] - 1, k + 1
    k = min(cap, len(p["close"])) - 1
    return p["close"][k] / p["entry"] - 1, k + 1


def ex_mean_revert(p, cap=MAXBARS):
    """Exit when price closes back above its own 20-day mean: the reversion is done."""
    for k in range(min(cap, len(p["close"]))):
        if np.isfinite(p["ema20"][k]) and p["close"][k] > p["ema20"][k]:
            return p["close"][k] / p["entry"] - 1, k + 1
    k = min(cap, len(p["close"])) - 1
    return p["close"][k] / p["entry"] - 1, k + 1


def ex_rsi(p, level, cap=MAXBARS):
    for k in range(min(cap, len(p["rsi14"]))):
        if np.isfinite(p["rsi14"][k]) and p["rsi14"][k] > level:
            return p["close"][k] / p["entry"] - 1, k + 1
    k = min(cap, len(p["close"])) - 1
    return p["close"][k] / p["entry"] - 1, k + 1


def ex_stop_target(p, tgt_mult, stop_mult, cap=MAXBARS):
    tgt = p["entry"] + tgt_mult * p["atr"]
    stp = p["entry"] - stop_mult * p["atr"]
    for k in range(min(cap, len(p["high"]))):
        if p["low"][k] <= stp:
            return stp / p["entry"] - 1, k + 1          # stop first if both hit
        if p["high"][k] >= tgt:
            return tgt / p["entry"] - 1, k + 1
    k = min(cap, len(p["close"])) - 1
    return p["close"][k] / p["entry"] - 1, k + 1


def ex_scale_out(p, n1, n2):
    """Half out at n1 bars, half at n2."""
    a, _ = ex_fixed(p, n1)
    b, _ = ex_fixed(p, n2)
    return 0.5 * a + 0.5 * b, (n1 + n2) / 2


def ex_vol_scaled(p, base_bars):
    """Hold longer when the asset is slow, shorter when it is fast."""
    n = int(round(base_bars * np.clip(0.025 / max(p["atr_pct"], 1e-4), 0.5, 2.0)))
    return ex_fixed(p, max(3, min(n, MAXBARS)))


RULES = {
    "fijo 5": lambda p: ex_fixed(p, 5),
    "fijo 10": lambda p: ex_fixed(p, 10),
    "fijo 15": lambda p: ex_fixed(p, 15),
    "fijo 21": lambda p: ex_fixed(p, 21),
    "fijo 30": lambda p: ex_fixed(p, 30),
    "fijo 42": lambda p: ex_fixed(p, 42),
    "vuelta a la media (EMA20)": ex_mean_revert,
    "RSI>60": lambda p: ex_rsi(p, 60),
    "RSI>70": lambda p: ex_rsi(p, 70),
    "objetivo 1 ATR": lambda p: ex_target_atr(p, 1.0),
    "objetivo 2 ATR": lambda p: ex_target_atr(p, 2.0),
    "objetivo 3 ATR": lambda p: ex_target_atr(p, 3.0),
    "obj 2 / stop 2 ATR": lambda p: ex_stop_target(p, 2.0, 2.0),
    "obj 3 / stop 3 ATR": lambda p: ex_stop_target(p, 3.0, 3.0),
    "obj 2 / stop 4 ATR": lambda p: ex_stop_target(p, 2.0, 4.0),
    "escalonada 10 y 30": lambda p: ex_scale_out(p, 10, 30),
    "escalonada 5 y 21": lambda p: ex_scale_out(p, 5, 21),
    "plazo escalado por vol": lambda p: ex_vol_scaled(p, 21),
}
