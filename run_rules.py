"""Rule mining with measured confidence.

Reframed question. Not "does this beat buy & hold" (a signal bot is not a
replacement for owning the assets) but:

    when this condition fires, what is P(price up in H days),
    and how much higher is that than the unconditional base rate?

Overlapping events inflate significance, so every rule is DECLUSTERED: within
an asset an event is only counted if the previous counted one was at least H
bars earlier. Confidence intervals use that independent count, not the raw one.
"""

import argparse
import warnings

import numpy as np
import pandas as pd

from run_screen import OUT, build_panel
from run_meanrev import OOS_START, add_mr_score
from run_control import ETFS

warnings.filterwarnings("ignore")
pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 300)

HORIZONS = (3, 5, 10, 21)


def build_rules(p: pd.DataFrame) -> dict[str, pd.Series]:
    up200 = p["px_ema200"] > 0
    dn200 = p["px_ema200"] < 0
    mkt_up = p["bench_px_ema200"] > 0
    hi_vol = p["atr_pct_rank252"] > 0.70
    lo_vol = p["atr_pct_rank252"] < 0.30

    r: dict[str, pd.Series] = {
        # ---------------- compra ----------------
        "C01 rsi2<5": p["rsi2"] < 5,
        "C02 rsi2<5 & px>ema200": (p["rsi2"] < 5) & up200,
        "C03 rsi2<10 & px>ema200": (p["rsi2"] < 10) & up200,
        "C04 rsi14<30": p["rsi14"] < 30,
        "C05 rsi14<30 & px>ema200": (p["rsi14"] < 30) & up200,
        "C06 z20<-2": p["zscore_20"] < -2,
        "C07 z20<-2 & px>ema200": (p["zscore_20"] < -2) & up200,
        "C08 pctB<0 & px>ema200": (p["bb_pctb"] < 0) & up200,
        "C09 3 bajas seguidas & px>ema200": (p["down_streak"] >= 3) & up200,
        "C10 4+ bajas seguidas": p["down_streak"] >= 4,
        "C11 IBS<0.2 & px>ema200": (p["bar_pos"] < 0.20) & up200,
        "C12 IBS<0.15 & rsi2<20 & px>ema200": (p["bar_pos"] < 0.15) & (p["rsi2"] < 20) & up200,
        "C13 mr_score>0.9": p["mr_score"] > 0.90,
        "C14 mr_score>0.9 & px>ema200": (p["mr_score"] > 0.90) & up200,
        "C15 mr_score>0.9 & SPY>ema200": (p["mr_score"] > 0.90) & mkt_up,
        "C16 dd252<-15% & ema200 subiendo": (p["drawdown_252"] < -0.15) & (p["ema200_slope63"] > 0),
        "C17 gap<-1 ATR & px>ema200": (p["gap_atr"] < -1.0) & up200,
        "C18 rsi2<5 & vol alta": (p["rsi2"] < 5) & hi_vol,
        "C19 rsi2<5 & vol baja": (p["rsi2"] < 5) & lo_vol,
        "C20 rsi2<10 & px>ema200 & SPY>ema200": (p["rsi2"] < 10) & up200 & mkt_up,
        "C21 rsi2<10 & px>ema200 & SPY castigado": (p["rsi2"] < 10) & up200 & (p["mkt_score"] > 0.7),
        "C22 IBS<0.2 & rsi14<40 & px>ema200": (p["bar_pos"] < 0.20) & (p["rsi14"] < 40) & up200,
        # ---------------- venta ----------------
        "V01 rsi2>95": p["rsi2"] > 95,
        "V02 rsi2>95 & px<ema200": (p["rsi2"] > 95) & dn200,
        "V03 rsi14>70": p["rsi14"] > 70,
        "V04 z20>2": p["zscore_20"] > 2,
        "V05 pctB>1": p["bb_pctb"] > 1,
        "V06 4+ subas seguidas": p["up_streak"] >= 4,
        "V07 IBS>0.85 & rsi14>70": (p["bar_pos"] > 0.85) & (p["rsi14"] > 70),
        "V08 dist_ema200>3 ATR": p["dist_ema200_atr"] > 3.0,
        "V09 mr_score<0.1": p["mr_score"] < 0.10,
        "V10 rsi2>95 & SPY<ema200": (p["rsi2"] > 95) & (~mkt_up),
        "V11 rsi14>75 & cerca max 52w": (p["rsi14"] > 75) & (p["dist_52w_high"] > -0.02),
        "V12 z20>2 & vol alta": (p["zscore_20"] > 2) & hi_vol,
    }
    return {k: v.fillna(False) for k, v in r.items()}


def decluster(dates: np.ndarray, positions: np.ndarray, min_gap: int) -> np.ndarray:
    """Greedily keep events at least `min_gap` bars apart."""
    keep, last = [], -10**9
    for pos in positions:
        if pos - last >= min_gap:
            keep.append(pos)
            last = pos
    return np.array(keep, dtype=int)


def eval_rule(p: pd.DataFrame, mask: pd.Series, h: int) -> dict:
    tgt = f"fwd_ret_{h}"
    valid = p[tgt].notna()
    ev = p.index[mask & valid]
    if len(ev) < 30:
        return {}

    # independent events only
    keep = []
    for _, g in p.loc[ev].groupby("asset"):
        pos = p.index.get_indexer(g.index)
        keep.extend(decluster(g["date"].to_numpy(), np.sort(pos), h))
    ev_i = p.index[np.array(sorted(keep))] if keep else ev[:0]
    if len(ev_i) < 30:
        return {}

    y = p.loc[ev_i, tgt]
    base = p.loc[valid, tgt]
    p_up, b_up = (y > 0).mean(), (base > 0).mean()
    n = len(y)
    se = np.sqrt(p_up * (1 - p_up) / n)

    def lift_in(m_ev, m_base):
        if m_ev.sum() < 20:
            return np.nan
        return (y[m_ev] > 0).mean() - (base[m_base] > 0).mean()

    d_ev, d_base = p.loc[ev_i, "date"], p.loc[valid, "date"]
    lift_is = lift_in(d_ev < OOS_START, d_base < OOS_START)
    lift_oos = lift_in(d_ev >= OOS_START, d_base >= OOS_START)

    yrs = []
    for yr in sorted(d_ev.dt.year.unique()):
        m_e, m_b = d_ev.dt.year == yr, d_base.dt.year == yr
        if m_e.sum() >= 10:
            yrs.append((y[m_e] > 0).mean() - (base[m_b] > 0).mean())
    yrs = np.array(yrs)

    return {
        "H": h, "n_raw": int(mask.sum()), "n_indep": n,
        "p_up": p_up, "base_up": b_up, "lift": p_up - b_up,
        "ci_lo": p_up - 1.96 * se, "ci_hi": p_up + 1.96 * se,
        "mean_ret": y.mean(), "base_ret": base.mean(),
        "edge_ret": y.mean() - base.mean(),
        "lift_IS": lift_is, "lift_OOS": lift_oos,
        "yrs_ok": (np.sign(yrs) == np.sign(p_up - b_up)).mean() if len(yrs) else np.nan,
        "n_yrs": len(yrs),
    }


def run(p: pd.DataFrame, label: str) -> pd.DataFrame:
    rules = build_rules(p)
    rows = []
    for name, mask in rules.items():
        for h in HORIZONS:
            res = eval_rule(p, mask, h)
            if res:
                rows.append({"regla": name, **res})
    df = pd.DataFrame(rows)
    df["signif"] = df["lift"].abs() / (df["p_up"] * (1 - df["p_up"]) / df["n_indep"]) ** 0.5
    df["universo"] = label
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    panel = add_mr_score(build_panel(args.force))
    panel["date"] = pd.to_datetime(panel["date"])
    p = panel[panel["class"] == "stock"].reset_index(drop=True)
    print(f"[RULES] {p['asset'].nunique()} activos · {len(p):,} filas · "
          f"{p['date'].min().date()} -> {p['date'].max().date()}")

    res = run(p, "acciones+ETFs")
    res.to_csv(OUT / "rules.csv", index=False)

    cols = ["regla", "H", "n_indep", "p_up", "base_up", "lift", "ci_lo",
            "edge_ret", "lift_IS", "lift_OOS", "yrs_ok", "signif"]

    print("\n" + "=" * 165)
    print("  REGLAS DE COMPRA — ordenadas por lift (P(sube) por encima del base rate)")
    print("  lift_IS y lift_OOS deben tener el MISMO signo; yrs_ok = fracción de años que confirman")
    print("=" * 165)
    buy = res[res["regla"].str.startswith("C")].sort_values("lift", ascending=False)
    print(buy.head(25)[cols].to_string(index=False, float_format=lambda z: f"{z:,.3f}"))

    print("\n" + "=" * 165)
    print("  REGLAS DE VENTA — lift negativo = P(sube) por DEBAJO del base rate")
    print("=" * 165)
    sell = res[res["regla"].str.startswith("V")].sort_values("lift")
    print(sell.head(20)[cols].to_string(index=False, float_format=lambda z: f"{z:,.3f}"))


if __name__ == "__main__":
    main()
