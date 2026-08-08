"""Walk-forward evaluation of signals.

Success definition (user-provided): the position gains +10% in USD within
3 months. We evaluate it as a race between target and stop:

    WIN     -> touched +10% before touching the ATR stop, inside the horizon
    LOSS    -> touched the stop first
    TIMEOUT -> neither, position closed at the horizon

Entry is the OPEN of the bar AFTER the signal bar, so no lookahead.
"""

import numpy as np
import pandas as pd

WIN, LOSS, TIMEOUT = "WIN", "LOSS", "TIMEOUT"


def simulate(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    open_: np.ndarray,
    entry_i: int,
    stop_dist: float,
    horizon: int,
    target_pct: float,
) -> dict | None:
    """Simulate one long trade entered at open[entry_i]. None if not evaluable."""
    n = len(close)
    if entry_i >= n or not np.isfinite(stop_dist) or stop_dist <= 0:
        return None
    last = min(entry_i + horizon - 1, n - 1)
    if last - entry_i + 1 < horizon:
        return None  # not enough forward data -> would bias the KPI

    entry = open_[entry_i]
    target = entry * (1.0 + target_pct)
    stop = entry - stop_dist

    outcome, bars_held = TIMEOUT, last - entry_i + 1
    for i in range(entry_i, last + 1):
        hit_stop = low[i] <= stop
        hit_target = high[i] >= target
        if hit_stop and hit_target:
            outcome, bars_held = LOSS, i - entry_i + 1  # conservative: stop first
            break
        if hit_stop:
            outcome, bars_held = LOSS, i - entry_i + 1
            break
        if hit_target:
            outcome, bars_held = WIN, i - entry_i + 1
            break

    window_h = high[entry_i : last + 1]
    window_l = low[entry_i : last + 1]
    mfe = window_h.max() / entry - 1.0
    mae = window_l.min() / entry - 1.0
    ret_horizon = close[last] / entry - 1.0

    r_unit = stop_dist / entry
    if outcome == WIN:
        r_mult = target_pct / r_unit
    elif outcome == LOSS:
        r_mult = -1.0
    else:
        r_mult = ret_horizon / r_unit

    return {
        "outcome": outcome,
        "entry": entry,
        "stop": stop,
        "target": target,
        "bars_held": bars_held,
        "mfe": mfe,
        "mae": mae,
        "ret_horizon": ret_horizon,
        "r_mult": r_mult,
        "hit10_no_stop": bool(mfe >= target_pct),
        "risk_pct": r_unit,
    }


def generate_signals(
    scores: pd.DataFrame, eligible: pd.Series, setup_th: float, exit_th: float, cooldown: int
) -> list[dict]:
    """State machine: alert only on the transition into SETUP, with cooldown."""
    signals = []
    idx = scores.index
    for pb in [c for c in ("A", "B", "C") if c in scores.columns]:
        s = scores[pb].to_numpy(dtype=float)
        elig = eligible.reindex(idx).fillna(False).to_numpy(dtype=bool)
        in_setup = False
        last_fire = -10**9
        for i in range(len(idx)):
            v = s[i]
            if not np.isfinite(v):
                continue
            if in_setup and v < exit_th:
                in_setup = False
            if not in_setup and v >= setup_th and elig[i] and (i - last_fire) >= cooldown:
                signals.append({"i": i, "date": idx[i], "playbook": pb, "score": float(v)})
                in_setup = True
                last_fire = i
    return sorted(signals, key=lambda d: d["i"])


def evaluate_asset(
    alias: str,
    x: pd.DataFrame,
    scores: pd.DataFrame,
    eligible: pd.Series,
    cfg: dict,
    horizon: int,
    setup_th: float,
) -> tuple[list[dict], list[dict]]:
    """Returns (signal trades, base-rate trades) for one asset."""
    bt = cfg["backtest"]
    o = x["open"].to_numpy(float)
    h = x["high"].to_numpy(float)
    l = x["low"].to_numpy(float)
    c = x["close"].to_numpy(float)
    atr_ = x["atr14"].to_numpy(float)

    sig_rows = []
    for sg in generate_signals(
        scores, eligible, setup_th, cfg["thresholds"]["exit_hysteresis"], bt["cooldown_bars"]
    ):
        i = sg["i"]
        res = simulate(h, l, c, o, i + 1, bt["stop_atr_mult"] * atr_[i], horizon, bt["target_pct"])
        if res is None:
            sig_rows.append({**sg, "asset": alias, "outcome": "INCOMPLETE"})
            continue
        sig_rows.append({**sg, "asset": alias, "regime": scores["regime"].iloc[i], **res})

    # Base rate: identical trade rules entered on every eligible bar.
    base_rows = []
    elig = eligible.to_numpy(bool)
    for i in range(len(x)):
        if not elig[i] or not np.isfinite(atr_[i]):
            continue
        res = simulate(h, l, c, o, i + 1, bt["stop_atr_mult"] * atr_[i], horizon, bt["target_pct"])
        if res is not None:
            base_rows.append({"asset": alias, "date": x.index[i], **res})

    return sig_rows, base_rows


def summarize(trades: pd.DataFrame, label: str) -> dict:
    """KPI block for a set of trades."""
    t = trades[trades["outcome"].isin([WIN, LOSS, TIMEOUT])]
    n = len(t)
    if n == 0:
        return {"set": label, "n": 0}
    wins = (t["outcome"] == WIN).sum()
    losses = (t["outcome"] == LOSS).sum()
    return {
        "set": label,
        "n": n,
        "success_rate": wins / n,
        "loss_rate": losses / n,
        "timeout_rate": (t["outcome"] == TIMEOUT).sum() / n,
        "hit10_no_stop": t["hit10_no_stop"].mean(),
        "avg_R": t["r_mult"].mean(),
        "expectancy_pct": (t["r_mult"] * t["risk_pct"]).mean(),
        "avg_ret_3m": t["ret_horizon"].mean(),
        "median_ret_3m": t["ret_horizon"].median(),
        "avg_mfe": t["mfe"].mean(),
        "avg_mae": t["mae"].mean(),
        "avg_bars_to_exit": t["bars_held"].mean(),
    }
