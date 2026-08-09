"""Signal computation for the daily job.

Implements exactly the two rules validated in the research scripts:

  EQUITIES  oversold ladder, requires close > EMA200, alert at >= min_conditions
  CRYPTO    momentum tilt, weight = clip(0.5 + pctile252(close/EMA200)), banded

Everything reads CLOSED bars only. The equity entry is the NEXT session's open,
so a signal computed after today's close is actionable tomorrow morning.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date

import numpy as np
import pandas as pd

from signals.data import load_ohlcv
from signals.indicators import atr, ema, rsi


@dataclass
class EquitySignal:
    asset: str
    symbol: str
    bar_date: str
    close: float
    conditions: int
    detail: dict[str, bool]
    p_up: float | None
    p_up_oos: float | None
    base_rate: float
    hold_bars: int
    atr_pct: float
    kind: str = "equity_buy"


@dataclass
class CryptoSignal:
    asset: str
    symbol: str
    bar_date: str
    close: float
    score: float
    target_weight: float
    prev_weight: float | None
    direction: str
    kind: str = "crypto_weight"


@dataclass
class RegimeSignal:
    proxy: str
    bar_date: str
    state: str
    prev_state: str | None
    close: float
    ema200: float
    kind: str = "regime"


def _pctile(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=window // 2).rank(pct=True)


def _streak_down(close: pd.Series) -> pd.Series:
    down = (close.diff() < 0).astype(int)
    grp = (down != down.shift()).cumsum()
    return down.groupby(grp).cumsum() * down


def drop_open_bar(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Remove the last bar if that session has not closed yet.

    yfinance happily returns a partial candle for the current day. Acting on it
    would break the "closed bars only" guarantee and make the signal flip
    during the day. Crypto daily candles close at 00:00 UTC, so today's crypto
    bar is ALWAYS open. US equities close at 21:00 UTC at the latest (20:00
    during DST), so today's bar is usable once we are past 21:15 UTC.
    """
    if df.empty:
        return df
    now = pd.Timestamp.now(tz="UTC")
    last = df.index[-1].date()
    if last < now.date():
        return df                                  # last bar is from a past day
    if kind == "crypto":
        return df.iloc[:-1]
    closed = now.hour > 21 or (now.hour == 21 and now.minute >= 15)
    return df if closed else df.iloc[:-1]


def load_history(symbol: str, days: int, as_of: str | None = None,
                 kind: str = "stock") -> pd.DataFrame:
    """`as_of` truncates history to that date — used to replay a past session."""
    end = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=days * 2)   # calendar span for `days` sessions
    df = load_ohlcv(symbol, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"),
                    force=as_of is None)
    if as_of:
        return df.loc[:pd.Timestamp(as_of)]
    return drop_open_bar(df, kind)


def market_score(bench: pd.DataFrame, window: int = 252) -> pd.Series:
    """How beaten down SPY is vs its own EMA200 history. 1.0 = most beaten down."""
    stretch = bench["close"] / ema(bench["close"], 200) - 1
    return 1.0 - _pctile(stretch, window)


def equity_conditions(df: pd.DataFrame, mkt: pd.Series) -> tuple[dict[str, bool], bool]:
    """The six validated conditions, evaluated on the last closed bar."""
    c, h, l = df["close"], df["high"], df["low"]
    i = -1

    e200 = ema(c, 200)
    in_uptrend = bool(c.iloc[i] > e200.iloc[i]) if np.isfinite(e200.iloc[i]) else False

    r2, r14 = rsi(c, 2), rsi(c, 14)
    sd20 = c.rolling(20, min_periods=20).std(ddof=0)
    z20 = (c - c.rolling(20, min_periods=20).mean()) / sd20.replace(0.0, np.nan)
    ibs = (c - l) / (h - l).replace(0.0, np.nan)
    streak = _streak_down(c)
    m = mkt.reindex(df.index).ffill()

    def ok(v) -> bool:
        return bool(v) if pd.notna(v) else False

    detail = {
        "RSI(2) < 10": ok(r2.iloc[i] < 10),
        "IBS < 0,20": ok(ibs.iloc[i] < 0.20),
        "RSI(14) < 40": ok(r14.iloc[i] < 40),
        "z-score 20d < -1,5": ok(z20.iloc[i] < -1.5),
        "3+ ruedas bajando": ok(streak.iloc[i] >= 3),
        "SPY castigado": ok(m.iloc[i] > 0.70),
    }
    return detail, in_uptrend


def scan_equities(cfg: dict, as_of: str | None = None) -> tuple[list[EquitySignal], list[dict]]:
    """Returns (signals above threshold, status of every name for /status)."""
    days = cfg["history_days"]
    bench = load_history(cfg["equities"]["benchmark"], days, as_of)
    mkt = market_score(bench)
    cal = cfg["calibration"]
    min_c = cfg["equities"]["min_conditions"]

    signals, status = [], []
    for item in cfg["equities"]["watchlist"]:
        try:
            df = load_history(item["symbol"], days, as_of)
        except Exception as e:
            print(f"[ENGINE] {item['alias']}: sin datos ({e})")
            continue
        if len(df) < 250:
            print(f"[ENGINE] {item['alias']}: historia insuficiente ({len(df)} ruedas)")
            continue

        detail, uptrend = equity_conditions(df, mkt)
        count = sum(detail.values()) if uptrend else -1
        a14 = atr(df["high"], df["low"], df["close"], 14)
        e20 = ema(df["close"], 20)
        above20 = (df["close"] > e20).fillna(False)

        row = {
            "asset": item["alias"], "conditions": count, "uptrend": uptrend,
            "close": float(df["close"].iloc[-1]),
            "bar_date": df.index[-1].date().isoformat(),
            "detail": detail,
            # recent sessions so the exit counter measures REAL bars elapsed,
            # not how many times the job happened to run
            # `above20` drives the exit: the trade closes once price has
            # reverted back above its own 20-day mean.
            "bars": [{"date": d.date().isoformat(), "open": float(o), "close": float(cl),
                      "above20": bool(a)}
                     for d, o, cl, a in zip(df.index[-120:], df["open"].iloc[-120:],
                                            df["close"].iloc[-120:], above20.iloc[-120:])],
        }
        status.append(row)

        if count >= min_c:
            tier = cal["by_conditions"].get(min(count, 6))
            signals.append(EquitySignal(
                asset=item["alias"], symbol=item["symbol"],
                bar_date=row["bar_date"], close=row["close"],
                conditions=count, detail=detail,
                p_up=tier["p_up"] if tier else None,
                p_up_oos=tier["oos"] if tier else None,
                base_rate=cal["base_rate_10d"],
                hold_bars=cfg["equities"]["hold_bars"],
                atr_pct=float(a14.iloc[-1] / df["close"].iloc[-1]),
            ))
    return signals, status


def scan_crypto(cfg: dict, prev_weights: dict[str, float],
                as_of: str | None = None) -> tuple[list[CryptoSignal], list[dict]]:
    cc = cfg["crypto"]
    signals, status = [], []
    for item in cc["watchlist"]:
        try:
            df = load_history(item["symbol"], cfg["history_days"], as_of, "crypto")
        except Exception as e:
            print(f"[ENGINE] {item['alias']}: sin datos ({e})")
            continue
        if len(df) < 250:
            continue

        stretch = df["close"] / ema(df["close"], 200) - 1
        score = _pctile(stretch, cc["rank_window"]).iloc[-1]
        if pd.isna(score):
            continue
        target = float(np.clip(0.5 + score, cc["weight_min"], cc["weight_max"]))
        prev = prev_weights.get(item["alias"])

        status.append({"asset": item["alias"], "score": float(score),
                       "target_weight": target, "prev_weight": prev,
                       "close": float(df["close"].iloc[-1]),
                       "bar_date": df.index[-1].date().isoformat()})

        if prev is None or abs(target - prev) > cc["rebalance_band"]:
            signals.append(CryptoSignal(
                asset=item["alias"], symbol=item["symbol"],
                bar_date=df.index[-1].date().isoformat(),
                close=float(df["close"].iloc[-1]),
                score=float(score), target_weight=target, prev_weight=prev,
                direction=("inicial" if prev is None
                           else "aumentar" if target > prev else "reducir"),
            ))
    return signals, status


def scan_regime(cfg: dict, prev_state: str | None,
                as_of: str | None = None) -> RegimeSignal | None:
    df = load_history(cfg["regime"]["proxy"], cfg["history_days"], as_of)
    e200 = ema(df["close"], 200)
    state = "RISK_ON" if df["close"].iloc[-1] > e200.iloc[-1] else "RISK_OFF"
    if state == prev_state:
        return None
    return RegimeSignal(
        proxy=cfg["regime"]["proxy"], bar_date=df.index[-1].date().isoformat(),
        state=state, prev_state=prev_state,
        close=float(df["close"].iloc[-1]), ema200=float(e200.iloc[-1]),
    )
