"""Event dataset for meta-labeling.

The current signal is an unweighted count of six conditions. Meta-labeling
splits that into two jobs: the rule decides WHEN to look (candidate events),
a model decides WHICH candidates are worth taking.

Candidates are deliberately loose (>= 3 conditions) so the model has something
to discriminate. Labels use the 21-session forward return, which is what the
live exit rule actually earns — not the 10-day number used to headline the
ladder.

Two feature families are new here and not in the original screen:
  breadth_*  what the whole universe is doing that day
  xs_*       how this name ranks against its peers on the same day
Both answer "is this name falling alone, or is everything falling?", which a
single-asset indicator cannot see.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PANEL = ROOT / "data" / "panel.parquet"
MIN_CONDITIONS = 3
HORIZON = 21


CONDITION_COLS = {
    "c_rsi2": lambda d: d["rsi2"] < 10,
    "c_ibs": lambda d: d["bar_pos"] < 0.20,
    "c_rsi14": lambda d: d["rsi14"] < 40,
    "c_z20": lambda d: d["zscore_20"] < -1.5,
    "c_streak": lambda d: d["down_streak"] >= 3,
    "c_mkt": lambda d: d["_mkt_score"] > 0.70,
}


def load_panel() -> pd.DataFrame:
    p = pd.read_parquet(PANEL)
    p["date"] = pd.to_datetime(p["date"])
    return p.sort_values(["date", "asset"]).reset_index(drop=True)


def add_context(p: pd.DataFrame) -> pd.DataFrame:
    """Market-wide and cross-sectional context, computed per date."""
    p = p.copy()

    # how beaten down the benchmark is vs its own EMA200 history
    p["_mkt_score"] = 1.0 - (p.groupby("asset")["bench_px_ema200"]
                             .transform(lambda s: s.rolling(252, min_periods=126).rank(pct=True)))

    stocks = p["class"] == "stock"
    g = p[stocks].groupby("date")

    # breadth: what share of the universe is in trouble today
    breadth = pd.DataFrame({
        "breadth_above_ema200": g["px_ema200"].apply(lambda s: (s > 0).mean()),
        "breadth_rsi14_median": g["rsi14"].median(),
        "breadth_down_share": g["ret_1"].apply(lambda s: (s < 0).mean()),
        "breadth_atr_median": g["atr_pct"].median(),
    })
    p = p.merge(breadth, left_on="date", right_index=True, how="left")

    # cross-sectional rank within the day: is this name the worst, or is it everyone?
    for col, out in [("ret_1", "xs_ret1_rank"), ("ret_5", "xs_ret5_rank"),
                     ("rsi14", "xs_rsi14_rank"), ("drawdown_252", "xs_dd_rank"),
                     ("zscore_20", "xs_z20_rank")]:
        p[out] = p.groupby("date")[col].rank(pct=True)

    # is the fall idiosyncratic? own 5d return minus the universe median
    med5 = p[stocks].groupby("date")["ret_5"].median().rename("_med5")
    p = p.merge(med5, left_on="date", right_index=True, how="left")
    p["idio_ret5"] = p["ret_5"] - p["_med5"]
    return p.drop(columns=["_med5"])


def build_events(p: pd.DataFrame | None = None, min_conditions: int = MIN_CONDITIONS,
                 horizon: int = HORIZON) -> pd.DataFrame:
    p = add_context(load_panel() if p is None else p)

    for name, fn in CONDITION_COLS.items():
        p[name] = fn(p).fillna(False).astype(int)
    p["n_conditions"] = p[list(CONDITION_COLS)].sum(axis=1)

    uptrend = (p["px_ema200"] > 0).fillna(False)
    target = f"fwd_ret_{horizon}"
    ev = p[uptrend & (p["n_conditions"] >= min_conditions)
           & p[target].notna() & (p["class"] == "stock")].copy()

    ev["y"] = (ev[target] > 0).astype(int)
    ev["ret"] = ev[target]
    return ev.reset_index(drop=True)


def feature_columns(ev: pd.DataFrame) -> list[str]:
    drop = {"asset", "class", "date", "close", "y", "ret", "n_conditions", "_mkt_score"}
    return [c for c in ev.columns
            if c not in drop and not c.startswith("fwd_")
            and pd.api.types.is_numeric_dtype(ev[c])]


if __name__ == "__main__":
    ev = build_events()
    feats = feature_columns(ev)
    print(f"eventos: {len(ev):,} · activos {ev['asset'].nunique()} · features {len(feats)}")
    print(f"rango: {ev['date'].min().date()} → {ev['date'].max().date()}")
    print(f"tasa base de acierto a {HORIZON} ruedas: {ev['y'].mean():.3f} · "
          f"retorno medio {ev['ret'].mean():+.4f}")
    print("\npor cantidad de condiciones:")
    print(ev.groupby("n_conditions").agg(n=("y", "size"), acierto=("y", "mean"),
                                         ret=("ret", "mean")).to_string(
        float_format=lambda z: f"{z:,.4f}"))
