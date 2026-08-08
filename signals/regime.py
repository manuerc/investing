"""Global risk-on / risk-off gate, computed once per proxy."""

import pandas as pd

from .indicators import ema, slope

RISK_ON, NEUTRAL, RISK_OFF = "RISK_ON", "NEUTRAL", "RISK_OFF"

# Score multiplier applied to the winning playbook score.
MULTIPLIER = {RISK_ON: 1.00, NEUTRAL: 0.80, RISK_OFF: 0.0}


def compute_regime(proxy: pd.DataFrame, kind: str) -> pd.Series:
    """kind: 'stock' (SPY rules) or 'crypto' (BTC rules)."""
    c = proxy["close"]
    e50, e200 = ema(c, 50), ema(c, 200)

    above200 = c > e200
    if kind == "stock":
        confirm = slope(e50, 10) > 0
    else:
        confirm = e50 > e200

    out = pd.Series(NEUTRAL, index=c.index, dtype=object)
    out[above200 & confirm] = RISK_ON
    out[~above200] = RISK_OFF
    out[e200.isna()] = NEUTRAL
    return out
