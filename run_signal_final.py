"""Deployable spec for the equity buy signal.

Entry: buy_count >= K on a closed bar, enter at next open.
Checks: per-year and per-asset stability, and which EXIT rule to use given the
overbought ladder failed to predict declines.
"""

import warnings

import numpy as np
import pandas as pd

from run_screen import OUT
from run_meanrev import OOS_START, add_mr_score
from run_allocation import build_watchlist_panel
from run_ladder import buy_count
from signals.indicators import atr, ema, rsi

warnings.filterwarnings("ignore")
pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 60)

K = 5
COOLDOWN = 10


def exits_for_asset(g: pd.DataFrame, entries: list[int]) -> list[dict]:
    o, h, l, c = (g[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    e20 = ema(g["close"], 20).to_numpy(float)
    r14 = rsi(g["close"], 14).to_numpy(float)
    a14 = atr(g["high"], g["low"], g["close"], 14).to_numpy(float)
    n = len(g)
    rows = []

    for i in entries:
        if i + 1 >= n:
            continue
        entry = o[i + 1]
        rec = {"date": g["date"].iloc[i], "asset": g["asset"].iloc[0], "entry": entry}

        # fixed holds
        for hold in (5, 10, 21):
            j = min(i + hold, n - 1)
            rec[f"fijo_{hold}d"] = c[j] / entry - 1

        # exit on RSI14 > 70
        j = i + 1
        while j < n and not (np.isfinite(r14[j]) and r14[j] > 70) and j - i <= 63:
            j += 1
        rec["rsi14>70"] = c[min(j, n - 1)] / entry - 1

        # exit on close below EMA20
        j = i + 1
        while j < n and not (np.isfinite(e20[j]) and c[j] < e20[j]) and j - i <= 63:
            j += 1
        rec["cierre<ema20"] = c[min(j, n - 1)] / entry - 1

        # chandelier trailing stop at 2.5 ATR
        peak, out = entry, None
        for j in range(i + 1, min(i + 64, n)):
            peak = max(peak, h[j])
            stop = peak - 2.5 * a14[i]
            if l[j] <= stop:
                out = stop / entry - 1
                break
        rec["trailing_2.5atr"] = out if out is not None else c[min(i + 63, n - 1)] / entry - 1

        # +5% target / -3% stop, 21-bar cap
        out = None
        for j in range(i + 1, min(i + 22, n)):
            if l[j] <= entry * 0.97:
                out = -0.03
                break
            if h[j] >= entry * 1.05:
                out = 0.05
                break
        rec["tp5_sl3"] = out if out is not None else c[min(i + 21, n - 1)] / entry - 1

        rows.append(rec)
    return rows


def main():
    wl = add_mr_score(build_watchlist_panel())
    w = wl[wl["class"] == "stock"].copy()
    w["bc"] = buy_count(w)

    all_rows = []
    for asset, g in w.groupby("asset"):
        g = g.sort_values("date").reset_index(drop=True)
        fired = np.where(g["bc"].to_numpy() >= K)[0]
        entries, last = [], -10**9
        for i in fired:
            if i - last >= COOLDOWN:
                entries.append(int(i))
                last = i
        all_rows += exits_for_asset(g, entries)

    t = pd.DataFrame(all_rows)
    t["year"] = pd.to_datetime(t["date"]).dt.year
    t["period"] = np.where(pd.to_datetime(t["date"]) < OOS_START, "IS", "OOS")
    t.to_csv(OUT / "signal_trades.csv", index=False)

    exit_cols = ["fijo_5d", "fijo_10d", "fijo_21d", "rsi14>70", "cierre<ema20",
                 "trailing_2.5atr", "tp5_sl3"]

    print("=" * 130)
    print(f"  SEÑAL DE COMPRA ({K}+ condiciones) — comparación de reglas de SALIDA")
    print(f"  {len(t)} operaciones en 15 acciones, 2010-2026, cooldown {COOLDOWN} ruedas")
    print("=" * 130)
    rows = []
    for col in exit_cols:
        for per in ("IS", "OOS", "TODO"):
            s = t[col] if per == "TODO" else t[t["period"] == per][col]
            rows.append({"salida": col, "periodo": per, "n": len(s),
                         "ret_medio": s.mean(), "mediana": s.median(),
                         "%_ganadoras": (s > 0).mean(),
                         "peor": s.min(), "mejor": s.max()})
    r = pd.DataFrame(rows)
    print(r.pivot_table(index="salida", columns="periodo",
                        values=["ret_medio", "%_ganadoras"])
          .to_string(float_format=lambda z: f"{z:,.3f}"))

    best = "fijo_10d"
    print("\n" + "=" * 130)
    print(f"  ESTABILIDAD POR AÑO — salida {best}")
    print("=" * 130)
    yr = t.groupby("year")[best].agg(n="size", ret_medio="mean",
                                     pct_ganadoras=lambda s: (s > 0).mean())
    print(yr.to_string(float_format=lambda z: f"{z:,.3f}"))
    print(f"\n  años con retorno medio positivo: {(yr['ret_medio'] > 0).sum()}/{len(yr)}")

    print("\n" + "=" * 130)
    print(f"  ESTABILIDAD POR ACTIVO — salida {best}")
    print("=" * 130)
    pa = t.groupby("asset")[best].agg(n="size", ret_medio="mean",
                                      pct_ganadoras=lambda s: (s > 0).mean()).sort_values("ret_medio", ascending=False)
    print(pa.to_string(float_format=lambda z: f"{z:,.3f}"))
    print(f"\n  activos con retorno medio positivo: {(pa['ret_medio'] > 0).sum()}/{len(pa)}")


if __name__ == "__main__":
    main()
