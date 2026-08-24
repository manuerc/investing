"""Scan for the experimental 'discrecional' crypto alert channel.

Codifies, as checkable conditions, the tools described in a trading YouTube
video: RSI overbought/oversold + divergence, EMA20 as dynamic support/
resistance, a VWAP breakout, proximity to the volume POC, and proximity to a
Fibonacci retracement zone. The video itself is fully discretionary ("I look
at the chart and decide") — this only surfaces which of those conditions are
true right now. It is not a validated signal and carries no backtest; the
call is left to whoever reads the alert.

Alerts only fire on the bar where a condition first turns true (edge
detection), same philosophy as the rest of the bot: alert on change, not on
persisting state, so RSI staying overbought for six hours in a row doesn't
send six identical alerts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from signals.data import load_ohlcv_intraday
from signals.indicators import (
    ema, rsi, rsi_bullish_divergence, rsi_bearish_divergence,
    vwap_anchored, poc_price, fibonacci_levels,
)


@dataclass
class DiscretionaryAlert:
    asset: str
    symbol: str
    bar_time: str
    close: float
    detail: dict[str, bool]
    triggered: list[str]
    kind: str = "discretionary_crypto"


def drop_open_hour_bar(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the current, still-forming hourly candle."""
    if df.empty:
        return df
    now = pd.Timestamp.now(tz="UTC")
    if df.index[-1].floor("h") >= now.floor("h"):
        return df.iloc[:-1]
    return df


def _cross_up(s: pd.Series, level) -> pd.Series:
    lvl = level if isinstance(level, pd.Series) else pd.Series(level, index=s.index)
    return (s > lvl) & (s.shift(1) <= lvl.shift(1))


def _cross_down(s: pd.Series, level) -> pd.Series:
    lvl = level if isinstance(level, pd.Series) else pd.Series(level, index=s.index)
    return (s < lvl) & (s.shift(1) >= lvl.shift(1))


def _near(s: pd.Series, level: float, pct: float) -> pd.Series:
    return (s - level).abs() / s <= pct


def compute_conditions(df1h: pd.DataFrame, cfg: dict) -> dict[str, pd.Series]:
    c, h, l, v = df1h["close"], df1h["high"], df1h["low"], df1h["volume"]
    r = rsi(c, cfg["rsi"]["period"])
    e = ema(c, cfg["ema"]["period"])
    vwap = vwap_anchored(h, l, c, v, cfg["vwap"]["anchor"])

    poc_cfg = cfg["poc"]
    win = df1h.tail(poc_cfg["lookback_bars"])
    poc = poc_price(win["high"], win["low"], win["close"], win["volume"], poc_cfg["bins"])

    fib_cfg = cfg["fibonacci"]
    h4 = df1h.resample(fib_cfg["timeframe"]).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    fib_win = h4.tail(fib_cfg["lookback_bars"])
    fib = fibonacci_levels(fib_win["high"], fib_win["low"]) if len(fib_win) >= 5 else None

    conditions = {
        "RSI(14) H1 sobrecompra (>{})".format(cfg["rsi"]["overbought"]):
            _cross_up(r, cfg["rsi"]["overbought"]),
        "RSI(14) H1 sobreventa (<{})".format(cfg["rsi"]["oversold"]):
            _cross_down(r, cfg["rsi"]["oversold"]),
        "Divergencia alcista RSI/precio":
            rsi_bullish_divergence(c, r, cfg["rsi"]["divergence_lookback"]),
        "Divergencia bajista RSI/precio":
            rsi_bearish_divergence(c, r, cfg["rsi"]["divergence_lookback"]),
        "Toque de EMA20 (soporte/resistencia dinámica)":
            _near(c, e, cfg["ema"]["touch_pct"]) if not e.isna().all() else pd.Series(False, index=c.index),
        "Ruptura de VWAP al alza":
            _cross_up(c, vwap),
        "Ruptura de VWAP a la baja":
            _cross_down(c, vwap),
    }
    if poc is not None:
        near_poc = _near(c, poc, poc_cfg["proximity_pct"])
        conditions["Cerca del POC (zona de mucho volumen negociado)"] = near_poc & ~near_poc.shift(1).fillna(False)
    if fib is not None:
        for ratio in fib_cfg["levels"]:
            level = fib["levels"][ratio]
            near = _near(c, level, fib_cfg["proximity_pct"])
            key = f"Cerca de retroceso Fibo {ratio * 100:.1f}% ({fib['direction']})"
            conditions[key] = (near & ~near.shift(1).fillna(False)).reindex(c.index, fill_value=False)
    return conditions


def scan_discretionary(cfg: dict) -> list[dict]:
    """Returns, per asset, the last closed bar's condition state.

    Callers decide dedup/alerting policy from `bar_time` + `triggered`.
    """
    out = []
    for item in cfg["crypto"]["watchlist"]:
        try:
            df = load_ohlcv_intraday(item["symbol"], cfg["intraday_period"], cfg["intraday_interval"])
        except Exception as e:
            print(f"[DISCRECIONAL] {item['alias']}: sin datos ({e})")
            continue
        df = drop_open_hour_bar(df)
        if len(df) < 300:
            print(f"[DISCRECIONAL] {item['alias']}: historia insuficiente ({len(df)} velas H1)")
            continue

        conditions = compute_conditions(df, cfg)
        detail = {name: bool(s.iloc[-1]) if pd.notna(s.iloc[-1]) else False
                  for name, s in conditions.items()}
        triggered = [name for name, v in detail.items() if v]

        out.append({
            "asset": item["alias"], "symbol": item["symbol"],
            "bar_time": df.index[-1].isoformat(),
            "close": float(df["close"].iloc[-1]),
            "detail": detail, "triggered": triggered,
        })
    return out
