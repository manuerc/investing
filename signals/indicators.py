"""Indicator computations. Pure pandas/numpy, no TA-Lib dependency.

Every function returns a Series aligned to the input index and uses only
information available up to and including each bar (no lookahead).
"""

import numpy as np
import pandas as pd


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def sma(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=window).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder smoothing
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(100.0).where(avg_loss.notna(), np.nan)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    atr_ = atr(high, low, close, period)
    alpha = 1 / period
    plus_di = 100 * plus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean() / atr_
    minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean() / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()


def macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line - sig


def bollinger(close: pd.Series, window: int = 20, n_std: float = 2.0):
    mid = sma(close, window)
    sd = close.rolling(window, min_periods=window).std(ddof=0)
    upper = mid + n_std * sd
    lower = mid - n_std * sd
    bandwidth = (upper - lower) / mid
    pct_b = (close - lower) / (upper - lower).replace(0.0, np.nan)
    return upper, lower, bandwidth, pct_b


def rolling_percentile(s: pd.Series, window: int) -> pd.Series:
    """Percentile rank (0-1) of the current value within its trailing window."""
    return s.rolling(window, min_periods=window // 2).apply(
        lambda w: (w[:-1] < w[-1]).mean(), raw=True
    )


def slope(s: pd.Series, lookback: int = 10) -> pd.Series:
    """Normalized slope: pct change of the series over `lookback` bars."""
    return (s - s.shift(lookback)) / s.shift(lookback).abs().replace(0.0, np.nan)


def weekly_ema(close: pd.Series, span: int = 200) -> pd.Series:
    """EMA over weekly resampled closes, forward-filled back onto the daily index."""
    wk = close.resample("W-FRI").last()
    wk_ema = wk.ewm(span=span, adjust=False, min_periods=span).mean()
    return wk_ema.reindex(close.index, method="ffill")


def bullish_reversal_bar(o, h, l, c) -> pd.Series:
    """Close above previous bar's high, or a bullish engulfing."""
    breakout = c > h.shift(1)
    engulfing = (c > o) & (o.shift(1) > c.shift(1)) & (c >= o.shift(1)) & (o <= c.shift(1))
    return breakout | engulfing


def rsi_bullish_divergence(close: pd.Series, rsi_s: pd.Series, lookback: int = 20) -> pd.Series:
    """Price makes a lower low vs `lookback` bars ago while RSI makes a higher low."""
    price_ll = close == close.rolling(lookback, min_periods=lookback).min()
    prev_price_min = close.shift(1).rolling(lookback, min_periods=lookback).min()
    prev_rsi_at_min = rsi_s.shift(1).rolling(lookback, min_periods=lookback).min()
    return price_ll & (close < prev_price_min) & (rsi_s > prev_rsi_at_min)


def rsi_bearish_divergence(close: pd.Series, rsi_s: pd.Series, lookback: int = 20) -> pd.Series:
    price_hh = close == close.rolling(lookback, min_periods=lookback).max()
    prev_price_max = close.shift(1).rolling(lookback, min_periods=lookback).max()
    prev_rsi_at_max = rsi_s.shift(1).rolling(lookback, min_periods=lookback).max()
    return price_hh & (close > prev_price_max) & (rsi_s < prev_rsi_at_max)


def vwap_anchored(high: pd.Series, low: pd.Series, close: pd.Series,
                  volume: pd.Series, anchor: str = "D") -> pd.Series:
    """VWAP that resets at each `anchor` boundary (default: daily, UTC).

    Crypto trades 24/7 so there is no session open to anchor to like on an
    equity exchange; the daily UTC boundary is the closest stand-in.
    """
    typical = (high + low + close) / 3.0
    tp_vol = typical * volume
    key = typical.index.tz_convert("UTC").floor(anchor) if typical.index.tz is not None \
        else typical.index.floor(anchor)
    cum_tp_vol = tp_vol.groupby(key).cumsum()
    cum_vol = volume.groupby(key).cumsum()
    return cum_tp_vol / cum_vol.replace(0.0, np.nan)


def poc_price(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
             bins: int = 48) -> float | None:
    """Point of control: the price bucket with the most traded volume.

    Approximated from bar volume assigned to each bar's typical price, since
    we only have OHLCV bars, not a real tick-level volume profile.
    """
    typical = (high + low + close) / 3.0
    lo, hi = typical.min(), typical.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    edges = np.linspace(lo, hi, bins + 1)
    idx = np.clip(np.digitize(typical.to_numpy(), edges) - 1, 0, bins - 1)
    vol_per_bin = np.bincount(idx, weights=volume.to_numpy(), minlength=bins)
    top = int(vol_per_bin.argmax())
    return float((edges[top] + edges[top + 1]) / 2.0)


def fibonacci_levels(high: pd.Series, low: pd.Series) -> dict:
    """Retracement levels for the last swing between the window's high and low.

    Direction follows whichever extreme happened more recently: if the swing
    high came after the swing low (an up-leg), levels retrace DOWN from the
    high; otherwise they retrace UP from the low.
    """
    hi_idx, lo_idx = high.idxmax(), low.idxmin()
    hi, lo = float(high.loc[hi_idx]), float(low.loc[lo_idx])
    span = hi - lo
    up_leg = hi_idx > lo_idx
    levels = {}
    for ratio in (0.382, 0.5, 0.618, 0.786):
        levels[ratio] = hi - span * ratio if up_leg else lo + span * ratio
    return {"direction": "up" if up_leg else "down", "high": hi, "low": lo, "levels": levels}


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Attach every indicator the playbooks need to an OHLCV frame."""
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
    x = df.copy()

    x["ema20"] = ema(c, 20)
    x["ema50"] = ema(c, 50)
    x["ema200"] = ema(c, 200)
    x["ema50_slope"] = slope(x["ema50"], 10)
    x["ema200_w"] = weekly_ema(c, 200)

    x["rsi14"] = rsi(c, 14)
    x["rsi2"] = rsi(c, 2)
    x["atr14"] = atr(h, l, c, 14)
    x["atr_pct"] = x["atr14"] / c
    x["atr_falling"] = x["atr14"] < x["atr14"].shift(5)
    x["adx14"] = adx(h, l, c, 14)
    x["adx_cross20"] = (x["adx14"] > 20) & (x["adx14"].shift(1) <= 20)

    x["macd_hist"] = macd_hist(c)
    x["macd_turning_up"] = (x["macd_hist"] > x["macd_hist"].shift(1)) & (
        x["macd_hist"].shift(1) <= x["macd_hist"].shift(2)
    )

    bb_u, bb_l, bb_bw, bb_pctb = bollinger(c, 20, 2.0)
    x["bb_upper"], x["bb_lower"] = bb_u, bb_l
    x["bb_bandwidth"] = bb_bw
    x["bb_pctb"] = bb_pctb
    x["bb_bw_pctile"] = rolling_percentile(bb_bw, 120)
    x["squeeze"] = x["bb_bw_pctile"] < 0.20
    x["squeeze_5d"] = x["squeeze"].rolling(5, min_periods=5).sum() >= 5

    x["vol_sma20"] = sma(v, 20)
    x["vol_ratio"] = v / x["vol_sma20"].replace(0.0, np.nan)
    x["dollar_vol20"] = sma(c * v, 20)

    x["high_52w"] = h.rolling(252, min_periods=100).max()
    x["near_52w_high"] = c >= 0.95 * x["high_52w"]
    # Highest high of the 20 bars BEFORE the current one -> breakout reference
    x["range_high_20"] = h.shift(1).rolling(20, min_periods=20).max()
    x["swing_low_20"] = l.shift(1).rolling(20, min_periods=20).min()

    x["bull_reversal"] = bullish_reversal_bar(o, h, l, c)
    x["rsi_bull_div"] = rsi_bullish_divergence(c, x["rsi14"], 20)
    x["rsi_bear_div"] = rsi_bearish_divergence(c, x["rsi14"], 20)

    x["bar_pos"] = (c - l) / (h - l).replace(0.0, np.nan)   # close within bar range
    x["drop_3d_atr"] = (c.shift(3) - c) / x["atr14"]
    x["ret_63"] = c.pct_change(63)

    return x
