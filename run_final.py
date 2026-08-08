"""Last two checks.

A) Crypto tilt: the weight moves every day, so turnover is high. Does the edge
   survive realistic fees? Measured turnover + cost sensitivity.
B) Stocks: tilt and rotation both failed. The one strong screening result left
   is bench_px_ema200 (ETF IC -0.143, sign-consistent in 16/17 years) -> market
   timing. Tested as exposure rules on the watchlist.
"""

import warnings

import numpy as np
import pandas as pd

from run_screen import OUT, build_panel
from run_meanrev import OOS_START, add_mr_score
from run_allocation import build_watchlist_panel, stats
from run_crypto import rank_score

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)


def tilt_with_cost(p: pd.DataFrame, score: pd.Series, cost: float, band: float = 0.0):
    """band > 0 -> only rebalance when the target weight moves more than `band`."""
    p = p.assign(_s=score)
    sr_parts, bh_parts, turn_parts = [], [], []
    for _, g in p.groupby("asset"):
        g = g.sort_values("date")
        tgt = np.clip(0.5 + np.nan_to_num(g["_s"].to_numpy(float), nan=0.5), 0.0, 2.0)
        if band > 0:
            held, w = 0.0, np.empty_like(tgt)
            for i, t in enumerate(tgt):
                if abs(t - held) > band:
                    held = t
                w[i] = held
        else:
            w = tgt
        wl = np.roll(w, 1); wl[0] = 0.0
        turn = np.abs(np.diff(np.concatenate([[0.0], wl])))
        r = np.nan_to_num(g["ret_1"].to_numpy(float))
        idx = g["date"].to_numpy()
        sr_parts.append(pd.Series(r * wl - turn * cost, index=idx))
        bh_parts.append(pd.Series(r, index=idx))
        turn_parts.append(pd.Series(turn, index=idx))
    sr = pd.concat(sr_parts, axis=1).mean(axis=1).sort_index()
    bh = pd.concat(bh_parts, axis=1).mean(axis=1).sort_index()
    tn = pd.concat(turn_parts, axis=1).mean(axis=1).sort_index()
    return sr, bh, tn


def main():
    panel = add_mr_score(build_panel())
    panel["date"] = pd.to_datetime(panel["date"])
    c = panel[panel["class"] == "crypto"].copy()

    print("=" * 120)
    print("  A) CRIPTO — sensibilidad a costos y a la banda de rebalanceo (10 monedas, OOS 2020-2026)")
    print("=" * 120)
    sc = rank_score(c, "px_ema200")
    rows = []
    for band in (0.0, 0.05, 0.10, 0.20):
        for cost in (0.0010, 0.0025, 0.0050):
            sr, bh, tn = tilt_with_cost(c, sc, cost, band)
            m = sr.index >= OOS_START
            s = stats(sr[m].to_numpy(), ""); b = stats(bh[m].to_numpy(), "")
            rows.append({"banda": band, "costo_bps": cost * 1e4,
                         "turnover_anual": tn[m].mean() * 252,
                         "cagr": s["cagr"], "sharpe": s["sharpe"], "maxdd": s["maxdd"],
                         "sharpe_bh": b["sharpe"], "d_sharpe": s["sharpe"] - b["sharpe"]})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda z: f"{z:,.3f}"))

    print("\n" + "=" * 120)
    print("  B) ACCIONES — reglas de timing de mercado sobre tu watchlist")
    print("=" * 120)
    wl = add_mr_score(build_watchlist_panel())
    s = wl[wl["class"] == "stock"].copy()

    # SPY trend / stretch, shared across assets
    spy_above = s["bench_px_ema200"] > 0
    mkt_stretch = s.groupby("asset")["bench_px_ema200"].transform(
        lambda z: z.rolling(252, min_periods=126).rank(pct=True))

    rules = {
        "buy & hold (equal weight)": pd.Series(1.0, index=s.index),
        "long solo si SPY > EMA200": spy_above.astype(float),
        "tilt por sobreventa de mercado": 1.5 - mkt_stretch.fillna(0.5),
        "tilt por sobreventa del activo": 0.5 + s["mr_score"].fillna(0.5),
        "SPY>EMA200 + tilt de mercado": spy_above.astype(float) * (1.5 - mkt_stretch.fillna(0.5)),
    }

    out = []
    for name, w in rules.items():
        sr, bh, tn = tilt_with_cost(s, w - 0.5, 0.0010, band=0.05)
        for period, m in (("IS", sr.index < OOS_START), ("OOS", sr.index >= OOS_START)):
            if m.sum() < 250:
                continue
            st = stats(sr[m].to_numpy(), name)
            out.append({"regla": name, "periodo": period, "cagr": st["cagr"],
                        "sharpe": st["sharpe"], "maxdd": st["maxdd"],
                        "turnover_anual": tn[m].mean() * 252})
    res = pd.DataFrame(out)
    print(res.set_index(["periodo", "regla"]).sort_index().to_string(float_format=lambda z: f"{z:,.3f}"))
    res.to_csv(OUT / "final_rules.csv", index=False)


if __name__ == "__main__":
    main()
