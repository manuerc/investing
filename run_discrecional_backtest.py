"""Event-study backtest for the discrecional (H1 crypto) alert conditions.

Same house style as run_screen.py / run_crypto.py, adapted to intraday:
  - one row per historical trigger of a condition, forward return at several
    horizons, judged against the asset's own UNCONDITIONAL forward return at
    the same horizon — not against zero
  - temporal IS/OOS split per asset (time series, never shuffled)
  - direction-aware: bullish conditions score the raw forward return,
    bearish conditions score its negative, so a positive number always means
    "the move went the way the condition implied"
  - leave-one-out across assets on the conditions that look promising OOS

Caveat bigger than anywhere else in this repo: yfinance only serves ~730
days of hourly history, so this is ~2 years of data vs. the 16 years used
for the daily model. Treat every number here as much less certain,
especially for rare triggers (divergence, Fibonacci zones) that may have
single-digit counts per asset.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from signals.data import load_ohlcv_intraday
from signals.indicators import (
    ema, rsi, rsi_bullish_divergence, rsi_bearish_divergence,
    vwap_anchored, poc_price, fibonacci_levels,
)
from bot.discretionary import _cross_up, _cross_down, _near, drop_open_hour_bar

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)

ASSETS = {"BTC": "BTC-USD", "ETH": "ETH-USD", "BNB": "BNB-USD",
          "SOL": "SOL-USD", "XRP": "XRP-USD", "ADA": "ADA-USD"}
HORIZONS = [4, 8, 24, 72]        # hours ahead
OOS_FRAC = 0.25
COST_ROUNDTRIP = 0.001            # ~conservative taker fees; excludes perp funding

RSI_PERIOD, RSI_OB, RSI_OS = 14, 70, 30
DIV_LOOKBACK = 24
EMA_PERIOD, EMA_TOUCH_PCT = 20, 0.005
POC_WINDOW, POC_BINS, POC_STEP, POC_PROX = 720, 48, 24, 0.005
FIB_TIMEFRAME, FIB_LOOKBACK, FIB_PROX = "4h", 42, 0.003


def rolling_poc(df1h: pd.DataFrame) -> pd.Series:
    """POC recomputed every POC_STEP bars from the trailing POC_WINDOW, then ffilled."""
    poc = pd.Series(np.nan, index=df1h.index)
    for i in range(POC_WINDOW, len(df1h), POC_STEP):
        win = df1h.iloc[i - POC_WINDOW:i]
        poc.iloc[i] = poc_price(win["high"], win["low"], win["close"], win["volume"], POC_BINS)
    return poc.ffill()


def rolling_fib(df1h: pd.DataFrame) -> pd.DataFrame:
    """Fib levels recomputed once per H4 bar from the trailing FIB_LOOKBACK, ffilled onto H1."""
    h4 = df1h.resample(FIB_TIMEFRAME).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    lvl = {r: pd.Series(np.nan, index=h4.index) for r in (0.618, 0.786)}
    direction = pd.Series(np.nan, index=h4.index)
    for i in range(FIB_LOOKBACK, len(h4)):
        win = h4.iloc[i - FIB_LOOKBACK:i]
        fib = fibonacci_levels(win["high"], win["low"])
        for r in (0.618, 0.786):
            lvl[r].iloc[i] = fib["levels"][r]
        direction.iloc[i] = 1.0 if fib["direction"] == "up" else -1.0
    out = pd.DataFrame({"fib618": lvl[0.618], "fib786": lvl[0.786], "fib_dir": direction})
    return out.reindex(df1h.index, method="ffill")


def build_conditions(df: pd.DataFrame) -> dict:
    """Returns {name: {"fire": bool Series, "dir": +1/-1 const or Series}}.

    `dir` encodes which direction the condition implies: aligned_return =
    dir * forward_return, so a positive aligned_return always means "the
    move went the way this condition implied", whichever direction that was.
    """
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    r = rsi(c, RSI_PERIOD)
    e = ema(c, EMA_PERIOD)
    vwap = vwap_anchored(h, l, c, v, "D")
    poc = rolling_poc(df)
    fib = rolling_fib(df)

    near_ema = _near(c, e, EMA_TOUCH_PCT) & e.notna()
    ema_first_touch = near_ema & ~near_ema.shift(1).fillna(False)
    ema_dir = pd.Series(np.where(c > e, 1.0, -1.0), index=c.index)

    near_poc = _near(c, poc, POC_PROX) & poc.notna()
    poc_first = near_poc & ~near_poc.shift(1).fillna(False)

    specs = {
        "rsi_sobrecompra": {"fire": _cross_up(r, RSI_OB), "dir": -1.0},
        "rsi_sobreventa": {"fire": _cross_down(r, RSI_OS), "dir": 1.0},
        "div_alcista": {"fire": rsi_bullish_divergence(c, r, DIV_LOOKBACK), "dir": 1.0},
        "div_bajista": {"fire": rsi_bearish_divergence(c, r, DIV_LOOKBACK), "dir": -1.0},
        "ema20_toque": {"fire": ema_first_touch, "dir": ema_dir},
        "vwap_ruptura_alza": {"fire": _cross_up(c, vwap), "dir": 1.0},
        "vwap_ruptura_baja": {"fire": _cross_down(c, vwap), "dir": -1.0},
        "poc_zona": {"fire": poc_first, "dir": 1.0},   # no directional prior, reported raw
    }
    for ratio, col in ((0.618, "fib618"), (0.786, "fib786")):
        lvl = fib[col]
        near = _near(c, lvl, FIB_PROX) & lvl.notna()
        specs[f"fib_{ratio}"] = {"fire": near & ~near.shift(1).fillna(False), "dir": fib["fib_dir"]}
    return specs


def forward_returns(close: pd.Series) -> dict[int, pd.Series]:
    return {hz: close.shift(-hz) / close - 1.0 for hz in HORIZONS}


def event_rows(asset: str, df: pd.DataFrame, specs: dict, fwd: dict) -> list[dict]:
    n = len(df)
    split = int(n * (1 - OOS_FRAC))
    period = pd.Series(np.where(np.arange(n) < split, "IS", "OOS"), index=df.index)

    rows = []
    for name, spec in specs.items():
        fire, direction = spec["fire"], spec["dir"]
        dir_s = direction if isinstance(direction, pd.Series) else pd.Series(direction, index=df.index)
        for hz in HORIZONS:
            aligned = dir_s * fwd[hz]
            ev_mask = fire & fwd[hz].notna()
            for per in ("IS", "OOS"):
                m = ev_mask & (period == per)
                base_m = fwd[hz].notna() & (period == per)
                if m.sum() == 0:
                    continue
                ev = aligned[m]
                base = (dir_s[base_m] * fwd[hz][base_m])
                rows.append({
                    "condicion": name, "asset": asset, "horizonte_h": hz, "periodo": per,
                    "n": int(m.sum()), "ret_medio": float(ev.mean()),
                    "win_rate": float((ev > 0).mean()),
                    "base_ret_medio": float(base.mean()) if len(base) else np.nan,
                    "base_win_rate": float((base > 0).mean()) if len(base) else np.nan,
                })
    return rows


def main():
    all_rows = []
    for alias, symbol in ASSETS.items():
        df = load_ohlcv_intraday(symbol, period="730d", interval="1h")
        df = drop_open_hour_bar(df)
        print(f"[DISCRECIONAL-BT] {alias}: {len(df)} velas H1  {df.index[0]} -> {df.index[-1]}")
        specs = build_conditions(df)
        fwd = forward_returns(df["close"])
        all_rows += event_rows(alias, df, specs, fwd)

    ev = pd.DataFrame(all_rows)
    ev.to_csv(OUT / "discrecional_backtest_raw.csv", index=False)

    # ---- pooled across assets ----
    pooled = (ev.groupby(["condicion", "horizonte_h", "periodo"])
              .apply(lambda g: pd.Series({
                  "n": g["n"].sum(),
                  "ret_medio": np.average(g["ret_medio"], weights=g["n"]),
                  "win_rate": np.average(g["win_rate"], weights=g["n"]),
                  "base_ret_medio": np.average(g["base_ret_medio"], weights=g["n"]),
                  "base_win_rate": np.average(g["base_win_rate"], weights=g["n"]),
              }), include_groups=False)
              .reset_index())
    pooled["edge"] = pooled["ret_medio"] - pooled["base_ret_medio"]
    pooled["edge_neto_costo"] = pooled["edge"] - COST_ROUNDTRIP
    pooled.to_csv(OUT / "discrecional_backtest_pooled.csv", index=False)

    print("\n" + "=" * 130)
    print("  POOLED — retorno alineado con la dirección implícita de cada condición, vs. base no condicional")
    print("  edge > 0 con n razonable (>=30) y OOS en la misma dirección que IS = lo único que vale mirar dos veces")
    print("=" * 130)
    for per in ("IS", "OOS"):
        print(f"\n--- {per} ---")
        p = pooled[pooled["periodo"] == per].sort_values(["horizonte_h", "edge"], ascending=[True, False])
        print(p[["condicion", "horizonte_h", "n", "ret_medio", "base_ret_medio", "edge",
                 "edge_neto_costo", "win_rate", "base_win_rate"]]
              .to_string(index=False, float_format=lambda z: f"{z:,.4f}"))

    # ---- leave-one-out on conditions that look promising in OOS at 24h ----
    promising = pooled[(pooled["periodo"] == "OOS") & (pooled["horizonte_h"] == 24)
                       & (pooled["n"] >= 30) & (pooled["edge"] > 0)]["condicion"].tolist()
    if promising:
        print("\n" + "=" * 130)
        print(f"  LEAVE-ONE-OUT (OOS, 24h) sobre condiciones prometedoras: {promising}")
        print("=" * 130)
        loo_rows = []
        for cond in promising:
            sub = ev[(ev["condicion"] == cond) & (ev["horizonte_h"] == 24) & (ev["periodo"] == "OOS")]
            for drop in ASSETS:
                s = sub[sub["asset"] != drop]
                if s["n"].sum() == 0:
                    continue
                edge = np.average(s["ret_medio"], weights=s["n"]) - np.average(s["base_ret_medio"], weights=s["n"])
                loo_rows.append({"condicion": cond, "sin": drop, "n": int(s["n"].sum()), "edge": edge})
        loo = pd.DataFrame(loo_rows)
        print(loo.to_string(index=False, float_format=lambda z: f"{z:,.4f}"))
    else:
        print("\n  Ninguna condición con n>=30 y edge>0 en OOS a 24h — nada que valga la pena mirar con leave-one-out.")


if __name__ == "__main__":
    main()
