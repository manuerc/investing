"""Validate the crypto momentum result on the wider research set.

The allocation test found that INVERTING the mean-reversion score on crypto
(i.e. overweight when strong, underweight when beaten down) beat buy & hold on
CAGR, Sharpe AND drawdown, in both IS and OOS. That was only 3 coins.

Here: 10 coins, per-feature, plus a walk-forward check that the result is not
an artifact of one asset or one bull run.
"""

import warnings

import numpy as np
import pandas as pd

from run_screen import OUT, build_panel
from run_meanrev import OOS_START, add_mr_score
from run_allocation import COST, stats

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)

# Candidate momentum features; weight rises with the trailing percentile rank.
CANDIDATES = ["px_ema200", "ret_63", "ret_126", "mom_12_1", "ema50_ema200",
              "dist_ema200_atr", "days_above_ema200_63", "drawdown_252"]


def rank_score(p: pd.DataFrame, col: str, window: int = 252) -> pd.Series:
    return p.groupby("asset")[col].transform(
        lambda s: s.rolling(window, min_periods=window // 2).rank(pct=True))


def tilt_by(p: pd.DataFrame, score: pd.Series, lo: float = 0.5, hi: float = 2.0):
    """Equal-weight basket, per-asset weight = clip(0.5 + score)."""
    p = p.assign(_s=score)
    sr_parts, bh_parts, dates_parts = [], [], []
    for _, g in p.groupby("asset"):
        g = g.sort_values("date")
        w = np.clip(0.5 + np.nan_to_num(g["_s"].to_numpy(float), nan=0.5), 0.0, hi)
        wl = np.roll(w, 1); wl[0] = 0.0
        turn = np.abs(np.diff(np.concatenate([[0.0], wl])))
        r = np.nan_to_num(g["ret_1"].to_numpy(float))
        sr_parts.append(pd.Series(r * wl - turn * COST, index=g["date"].to_numpy()))
        bh_parts.append(pd.Series(r, index=g["date"].to_numpy()))
    sr = pd.concat(sr_parts, axis=1).mean(axis=1).sort_index()
    bh = pd.concat(bh_parts, axis=1).mean(axis=1).sort_index()
    return sr, bh


def main():
    panel = build_panel()
    panel["date"] = pd.to_datetime(panel["date"])
    p = add_mr_score(panel)
    c = p[p["class"] == "crypto"].copy()
    print(f"[CRYPTO] {c['asset'].nunique()} monedas · {c['date'].min().date()} -> {c['date'].max().date()}")

    print("\n" + "=" * 120)
    print("  TILT POR FEATURE — peso sube con el percentil del feature (10 monedas)")
    print("=" * 120)
    rows = []
    for col in CANDIDATES:
        if col not in c.columns:
            continue
        sc = rank_score(c, col)
        sr, bh = tilt_by(c, sc)
        for period, m in (("IS", sr.index < OOS_START), ("OOS", sr.index >= OOS_START)):
            if m.sum() < 250:
                continue
            s = stats(sr[m].to_numpy(), col)
            b = stats(bh[m].to_numpy(), "buy&hold")
            rows.append({"feature": col, "periodo": period, "cagr": s["cagr"], "sharpe": s["sharpe"],
                         "maxdd": s["maxdd"], "sharpe_bh": b["sharpe"], "cagr_bh": b["cagr"],
                         "maxdd_bh": b["maxdd"], "d_sharpe": s["sharpe"] - b["sharpe"]})
    # invert the mean-reversion composite as the reference case
    sr, bh = tilt_by(c, 1.0 - c["mr_score"])
    for period, m in (("IS", sr.index < OOS_START), ("OOS", sr.index >= OOS_START)):
        if m.sum() < 250:
            continue
        s = stats(sr[m].to_numpy(), "mr_invertido"); b = stats(bh[m].to_numpy(), "bh")
        rows.append({"feature": "mr_score_invertido", "periodo": period, "cagr": s["cagr"],
                     "sharpe": s["sharpe"], "maxdd": s["maxdd"], "sharpe_bh": b["sharpe"],
                     "cagr_bh": b["cagr"], "maxdd_bh": b["maxdd"], "d_sharpe": s["sharpe"] - b["sharpe"]})

    res = pd.DataFrame(rows).sort_values(["periodo", "d_sharpe"], ascending=[True, False])
    print(res.to_string(index=False, float_format=lambda z: f"{z:,.3f}"))
    res.to_csv(OUT / "crypto_tilt.csv", index=False)

    # --- per-year consistency of the best simple feature ---
    print("\n" + "=" * 120)
    print("  CONSISTENCIA AÑO POR AÑO — tilt con px_ema200 vs buy & hold (10 monedas)")
    print("=" * 120)
    sc = rank_score(c, "px_ema200")
    sr, bh = tilt_by(c, sc)
    yr = pd.DataFrame({"strat": sr, "bh": bh})
    yr["year"] = yr.index.year
    agg = yr.groupby("year").apply(lambda g: pd.Series({
        "ret_strat": (1 + g["strat"]).prod() - 1,
        "ret_bh": (1 + g["bh"]).prod() - 1,
        "dif": (1 + g["strat"]).prod() - (1 + g["bh"]).prod(),
    }))
    print(agg.to_string(float_format=lambda z: f"{z:,.3f}"))
    print(f"\n  años en que el tilt supera a buy&hold: {(agg['dif'] > 0).sum()}/{len(agg)}")

    # --- leave-one-out: is it driven by a single coin? ---
    print("\n" + "=" * 120)
    print("  LEAVE-ONE-OUT (OOS) — se cae el resultado si saco alguna moneda?")
    print("=" * 120)
    loo = []
    for drop in sorted(c["asset"].unique()):
        sub = c[c["asset"] != drop]
        sc2 = rank_score(sub, "px_ema200")
        s2, b2 = tilt_by(sub, sc2)
        m = s2.index >= OOS_START
        loo.append({"sin": drop, "sharpe_strat": stats(s2[m].to_numpy(), "")["sharpe"],
                    "sharpe_bh": stats(b2[m].to_numpy(), "")["sharpe"]})
    l = pd.DataFrame(loo)
    l["d_sharpe"] = l["sharpe_strat"] - l["sharpe_bh"]
    print(l.to_string(index=False, float_format=lambda z: f"{z:,.3f}"))
    print(f"\n  casos donde el tilt sigue ganando: {(l['d_sharpe'] > 0).sum()}/{len(l)}")


if __name__ == "__main__":
    main()
