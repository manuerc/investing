"""Confidence ladder: does P(up) rise monotonically with the number of
confirming conditions? That is what lets a bot say "strong" vs "weak" instead
of firing one binary alert.

BUY  conditions (all require px > EMA200, i.e. only dip-buy inside an uptrend):
    rsi2 < 10 · IBS < 0.20 · rsi14 < 40 · z20 < -1.5 · 3+ down days · SPY beaten down
SELL conditions (mirror):
    rsi2 > 95 · IBS > 0.85 · rsi14 > 70 · z20 > 1.5 · 4+ up days · px < EMA200

Validated on the research universe, then applied unchanged to the personal
watchlist to check that it transfers.
"""

import warnings

import numpy as np
import pandas as pd

from run_screen import OUT, build_panel
from run_meanrev import OOS_START, add_mr_score
from run_allocation import build_watchlist_panel
from run_rules import decluster

warnings.filterwarnings("ignore")
pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 60)

H = 10          # medium term: two trading weeks


def buy_count(p: pd.DataFrame) -> pd.Series:
    up200 = (p["px_ema200"] > 0).fillna(False)
    conds = [
        (p["rsi2"] < 10).fillna(False),
        (p["bar_pos"] < 0.20).fillna(False),
        (p["rsi14"] < 40).fillna(False),
        (p["zscore_20"] < -1.5).fillna(False),
        (p["down_streak"] >= 3).fillna(False),
        (p["mkt_score"] > 0.70).fillna(False),
    ]
    return sum(c.astype(int) for c in conds).where(up200, -1)


def sell_count(p: pd.DataFrame) -> pd.Series:
    conds = [
        (p["rsi2"] > 95).fillna(False),
        (p["bar_pos"] > 0.85).fillna(False),
        (p["rsi14"] > 70).fillna(False),
        (p["zscore_20"] > 1.5).fillna(False),
        (p["up_streak"] >= 4).fillna(False),
        (p["px_ema200"] < 0).fillna(False),
    ]
    return sum(c.astype(int) for c in conds)


def eval_bucket(p: pd.DataFrame, mask: pd.Series, h: int) -> dict:
    tgt = f"fwd_ret_{h}"
    valid = p[tgt].notna()
    ev = p.index[mask & valid]
    if len(ev) < 25:
        return {}
    keep = []
    for _, g in p.loc[ev].groupby("asset"):
        pos = np.sort(p.index.get_indexer(g.index))
        keep.extend(decluster(None, pos, h))
    if len(keep) < 25:
        return {}
    y = p.loc[p.index[sorted(keep)], tgt]
    d = p.loc[p.index[sorted(keep)], "date"]
    base = p.loc[valid, tgt]
    n = len(y)
    p_up = (y > 0).mean()
    se = np.sqrt(p_up * (1 - p_up) / n)
    up, dn = y[y > 0], y[y <= 0]
    return {
        "n": n, "p_up": p_up, "ci_lo": p_up - 1.96 * se, "ci_hi": p_up + 1.96 * se,
        "base_up": (base > 0).mean(),
        "mean_ret": y.mean(), "base_ret": base.mean(),
        "avg_gain": up.mean() if len(up) else np.nan,
        "avg_loss": dn.mean() if len(dn) else np.nan,
        "p_up_IS": (y[d < OOS_START] > 0).mean() if (d < OOS_START).sum() >= 20 else np.nan,
        "p_up_OOS": (y[d >= OOS_START] > 0).mean() if (d >= OOS_START).sum() >= 20 else np.nan,
    }


def ladder(p: pd.DataFrame, counter, label: str, h: int = H) -> pd.DataFrame:
    cnt = counter(p)
    rows = []
    for k in range(0, 7):
        r = eval_bucket(p, cnt == k, h)
        if r:
            rows.append({"condiciones": k, **r})
    df = pd.DataFrame(rows)
    if len(df):
        df["lift"] = df["p_up"] - df["base_up"]
        df["edge_ret"] = df["mean_ret"] - df["base_ret"]
    print(f"\n--- {label} (horizonte {h} ruedas) ---")
    cols = ["condiciones", "n", "p_up", "ci_lo", "ci_hi", "base_up", "lift",
            "mean_ret", "edge_ret", "avg_gain", "avg_loss", "p_up_IS", "p_up_OOS"]
    print(df[cols].to_string(index=False, float_format=lambda z: f"{z:,.3f}"))
    return df


def main():
    research = add_mr_score(build_panel())
    research["date"] = pd.to_datetime(research["date"])
    r = research[research["class"] == "stock"].reset_index(drop=True)

    wl = add_mr_score(build_watchlist_panel())
    w = wl[wl["class"] == "stock"].reset_index(drop=True)

    print("=" * 150)
    print("  ESCALERA DE CONFIANZA — COMPRA")
    print("=" * 150)
    ladder(r, buy_count, "Universo de research (82 activos)")
    ladder(w, buy_count, "TU WATCHLIST (15 acciones) — aplicación, sin reajustar nada")

    print("\n" + "=" * 150)
    print("  ESCALERA DE CONFIANZA — VENTA")
    print("=" * 150)
    ladder(r, sell_count, "Universo de research (82 activos)")
    ladder(w, sell_count, "TU WATCHLIST (15 acciones)")

    # horizon sensitivity for the strong buy bucket
    print("\n" + "=" * 150)
    print("  SENSIBILIDAD AL HORIZONTE — compra con 4+ condiciones, universo de research")
    print("=" * 150)
    cnt = buy_count(r)
    rows = []
    for h in (3, 5, 10, 21, 63):
        res = eval_bucket(r, cnt >= 4, h)
        if res:
            rows.append({"H": h, **res, "lift": res["p_up"] - res["base_up"],
                         "edge_ret": res["mean_ret"] - res["base_ret"]})
    print(pd.DataFrame(rows)[["H", "n", "p_up", "ci_lo", "base_up", "lift", "mean_ret",
                              "edge_ret", "p_up_IS", "p_up_OOS"]]
          .to_string(index=False, float_format=lambda z: f"{z:,.3f}"))

    # alert frequency on the watchlist
    print("\n" + "=" * 150)
    print("  FRECUENCIA DE ALERTAS EN TU WATCHLIST")
    print("=" * 150)
    wc, sc = buy_count(w), sell_count(w)
    yrs = w["date"].dt.year.nunique()
    for k in (3, 4, 5):
        print(f"  compra >= {k} condiciones: {int((wc >= k).sum()):5d} eventos "
              f"= {(wc >= k).sum()/yrs:5.1f} por año en 15 acciones")
    for k in (3, 4, 5):
        print(f"  venta  >= {k} condiciones: {int((sc >= k).sum()):5d} eventos "
              f"= {(sc >= k).sum()/yrs:5.1f} por año en 15 acciones")


if __name__ == "__main__":
    main()
