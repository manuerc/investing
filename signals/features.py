"""Point-in-time feature library + forward targets.

Every feature is scale-free so it is comparable across assets and across time.
Every feature at bar t uses only data up to and including t.
Targets use shift(-h) and are the only forward-looking columns.
"""

import numpy as np
import pandas as pd

from .indicators import adx, atr, bollinger, ema, macd_hist, rsi, sma


def _rank(s: pd.Series, window: int) -> pd.Series:
    """Percentile rank of the current value within its trailing window."""
    return s.rolling(window, min_periods=window // 2).rank(pct=True)


def _slope(s: pd.Series, n: int) -> pd.Series:
    return (s - s.shift(n)) / s.shift(n).abs().replace(0.0, np.nan)


def _streak_down(close: pd.Series, cap: int = 10) -> pd.Series:
    down = (close.diff() < 0).astype(int)
    grp = (down != down.shift()).cumsum()
    return (down.groupby(grp).cumsum() * down).clip(upper=cap)


def _streak_up(close: pd.Series, cap: int = 10) -> pd.Series:
    up = (close.diff() > 0).astype(int)
    grp = (up != up.shift()).cumsum()
    return (up.groupby(grp).cumsum() * up).clip(upper=cap)


def build_features(df: pd.DataFrame, bench_close: pd.Series | None) -> pd.DataFrame:
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
    f = pd.DataFrame(index=df.index)

    e20, e50, e200 = ema(c, 20), ema(c, 50), ema(c, 200)
    atr14 = atr(h, l, c, 14)
    ret1 = c.pct_change()

    # --- trend / position -------------------------------------------------
    f["px_ema20"] = c / e20 - 1
    f["px_ema50"] = c / e50 - 1
    f["px_ema200"] = c / e200 - 1
    f["ema50_ema200"] = e50 / e200 - 1
    f["ema50_slope21"] = _slope(e50, 21)
    f["ema200_slope63"] = _slope(e200, 63)
    f["dist_ema200_atr"] = (c - e200) / atr14
    f["days_above_ema200_63"] = (c > e200).rolling(63, min_periods=30).mean()

    # --- momentum ---------------------------------------------------------
    f["ret_1"] = ret1                                        # needed by the strategy backtest
    for n in (5, 21, 63, 126, 252):
        f[f"ret_{n}"] = c.pct_change(n)
    f["mom_12_1"] = c.shift(21) / c.shift(252) - 1          # classic 12-1 momentum
    f["mom_accel"] = f["ret_21"] - f["ret_63"] / 3.0

    # --- mean reversion ---------------------------------------------------
    f["rsi2"] = rsi(c, 2)
    f["rsi14"] = rsi(c, 14)
    sd20 = c.rolling(20, min_periods=20).std(ddof=0)
    f["zscore_20"] = (c - sma(c, 20)) / sd20.replace(0.0, np.nan)
    _, _, bb_bw, bb_pctb = bollinger(c, 20, 2.0)
    f["bb_pctb"] = bb_pctb
    f["down_streak"] = _streak_down(c)
    f["drawdown_252"] = c / c.rolling(252, min_periods=100).max() - 1

    # --- volatility -------------------------------------------------------
    f["atr_pct"] = atr14 / c
    f["atr_pct_rank252"] = _rank(f["atr_pct"], 252)
    f["bb_bandwidth"] = bb_bw
    f["bb_bw_rank120"] = _rank(bb_bw, 120)                  # low = squeeze
    rv21 = ret1.rolling(21, min_periods=21).std(ddof=0)
    rv63 = ret1.rolling(63, min_periods=63).std(ddof=0)
    f["vol_ratio_21_63"] = rv21 / rv63.replace(0.0, np.nan)
    f["vol_of_vol"] = rv21.rolling(63, min_periods=63).std(ddof=0) / rv21.replace(0.0, np.nan)

    # --- volume -----------------------------------------------------------
    vol20 = sma(v, 20)
    f["vol_ratio_20"] = v / vol20.replace(0.0, np.nan)
    obv = (np.sign(ret1).fillna(0.0) * v).cumsum()
    f["obv_slope21"] = (obv - obv.shift(21)) / vol20.replace(0.0, np.nan) / 21.0
    f["dollar_vol_trend"] = sma(c * v, 20) / sma(c * v, 60).replace(0.0, np.nan)

    # --- trend strength ---------------------------------------------------
    f["adx14"] = adx(h, l, c, 14)
    f["macd_hist_norm"] = macd_hist(c) / c
    f["macd_hist_slope"] = f["macd_hist_norm"] - f["macd_hist_norm"].shift(5)

    # --- structure --------------------------------------------------------
    f["dist_52w_high"] = c / h.rolling(252, min_periods=100).max() - 1
    f["dist_52w_low"] = c / l.rolling(252, min_periods=100).min() - 1
    f["bar_pos"] = (c - l) / (h - l).replace(0.0, np.nan)      # IBS: where close sits in the bar
    f["gap_atr"] = (o - c.shift(1)) / atr14
    f["up_streak"] = _streak_up(c)

    # --- relative strength ------------------------------------------------
    if bench_close is not None:
        b = bench_close.reindex(df.index).ffill()
        f["rs_21"] = c.pct_change(21) - b.pct_change(21)
        f["rs_63"] = c.pct_change(63) - b.pct_change(63)
        f["rs_252"] = c.pct_change(252) - b.pct_change(252)
        f["corr_bench_63"] = ret1.rolling(63, min_periods=40).corr(b.pct_change())
        f["bench_px_ema200"] = b / ema(b, 200) - 1

    return f


def build_targets(df: pd.DataFrame, bench_close: pd.Series | None,
                  horizons=(3, 5, 10, 21, 63, 126)) -> pd.DataFrame:
    """Forward returns. The ONLY forward-looking block in the codebase."""
    c, h, l = df["close"], df["high"], df["low"]
    t = pd.DataFrame(index=df.index)
    b = bench_close.reindex(df.index).ffill() if bench_close is not None else None

    for n in horizons:
        fwd = c.shift(-n) / c - 1
        t[f"fwd_ret_{n}"] = fwd
        if b is not None:
            t[f"fwd_excess_{n}"] = fwd - (b.shift(-n) / b - 1)
        # path stats over bars [t+1, t+n]
        t[f"fwd_maxup_{n}"] = h.rolling(n, min_periods=n).max().shift(-n) / c - 1
        t[f"fwd_maxdd_{n}"] = l.rolling(n, min_periods=n).min().shift(-n) / c - 1
    return t
