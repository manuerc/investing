"""Correct uses of the mean-reversion signal.

The long/flat test showed the signal is real but that switching fully OFF costs
more than the edge is worth. Two better formulations, both matching what a
Discord bot on a fixed watchlist would actually do:

  1. TILT      -> always invested, weight varies with oversoldness ("when to add")
  2. ROTATION  -> each month buy the top-K most oversold names of the watchlist
                  ("which of my names do I buy now")

The watchlist panel is built separately from the research universe so the
application set never contaminates the feature selection.
"""

import warnings

import numpy as np
import pandas as pd
import yaml

from run_screen import OUT, ROOT, build_panel
from run_meanrev import COMPONENTS, OOS_START, add_mr_score, group_of
from run_control import ETFS
from signals.data import load_ohlcv
from signals.features import build_features, build_targets
from signals.universe import BENCH

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)

WARMUP, END = "2010-01-01", "2026-08-08"
COST = 0.001


def build_watchlist_panel() -> pd.DataFrame:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    bench = {k: load_ohlcv(v, WARMUP, END)["close"] for k, v in BENCH.items()}
    frames = []
    for item in cfg["universe"]:
        df = load_ohlcv(item["symbol"], WARMUP, END)
        bc = None if item["symbol"] == BENCH[item["class"]] else bench[item["class"]]
        block = pd.concat([build_features(df, bc), build_targets(df, bc)], axis=1)
        for c in ("open", "high", "low", "close"):
            block[c] = df[c]
        block["asset"] = item["alias"]
        block["class"] = item["class"]
        block["date"] = block.index
        frames.append(block.reset_index(drop=True))
    p = pd.concat(frames, ignore_index=True)
    p["date"] = pd.to_datetime(p["date"])
    return p


def stats(r: np.ndarray, label: str) -> dict:
    r = np.nan_to_num(r)
    yrs = len(r) / 252.0
    eq = (1 + r).cumprod()
    return {
        "estrategia": label,
        "cagr": eq[-1] ** (1 / yrs) - 1,
        "vol": r.std() * np.sqrt(252),
        "sharpe": r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan,
        "maxdd": (eq / np.maximum.accumulate(eq) - 1).min(),
    }


def tilt_test(p: pd.DataFrame, label: str, invert: bool = False):
    """Always invested; weight = 0.5 + score (mean exposure ~1.0) vs constant 1.0."""
    rows = []
    for period, mask_fn in (("IS", lambda d: d < OOS_START), ("OOS", lambda d: d >= OOS_START)):
        sr_all, bh_all = [], []
        for _, g in p.groupby("asset"):
            g = g.sort_values("date")
            m = mask_fn(g["date"]).to_numpy()
            if m.sum() < 250:
                continue
            s = g["mr_score"].to_numpy(float)
            s = (1.0 - s) if invert else s
            w = np.clip(0.5 + np.nan_to_num(s, nan=0.5), 0.0, 2.0)
            wl = np.roll(w, 1); wl[0] = 0.0
            turn = np.abs(np.diff(np.concatenate([[0.0], wl])))
            r = np.nan_to_num(g["ret_1"].to_numpy(float))
            sr_all.append((r * wl - turn * COST)[m])
            bh_all.append(r[m])
        if not sr_all:
            continue
        n = min(map(len, sr_all))
        sr = np.mean([x[-n:] for x in sr_all], axis=0)
        bh = np.mean([x[-n:] for x in bh_all], axis=0)
        rows.append({**stats(sr, f"tilt {label}"), "periodo": period})
        rows.append({**stats(bh, f"buy&hold {label}"), "periodo": period})
    return pd.DataFrame(rows)


def rotation_test(p: pd.DataFrame, label: str, k: int = 5, rebal: int = 21, invert: bool = False):
    """Every `rebal` bars, hold the top-k assets by score, equally weighted."""
    wide_s = p.pivot_table(index="date", columns="asset", values="mr_score")
    wide_r = p.pivot_table(index="date", columns="asset", values="ret_1")
    if invert:
        wide_s = 1.0 - wide_s
    wide_s, wide_r = wide_s.align(wide_r, join="inner")
    dates = wide_s.index

    def run(pick_top: bool):
        w = pd.DataFrame(0.0, index=dates, columns=wide_s.columns)
        cur = None
        for i, d in enumerate(dates):
            if i % rebal == 0:
                row = wide_s.loc[d].dropna()
                if len(row) >= k:
                    sel = row.nlargest(k).index if pick_top else row.nsmallest(k).index
                    cur = sel
            if cur is not None:
                w.loc[d, cur] = 1.0 / k
        wl = w.shift(1).fillna(0.0)
        turn = (wl - wl.shift(1).fillna(0.0)).abs().sum(axis=1)
        return (wl * wide_r.fillna(0.0)).sum(axis=1) - turn * COST

    top, bottom = run(True), run(False)
    eq_w = wide_r.mean(axis=1).fillna(0.0)

    rows = []
    for period, m in (("IS", dates < OOS_START), ("OOS", dates >= OOS_START)):
        if m.sum() < 250:
            continue
        rows.append({**stats(top[m].to_numpy(), f"top{k} castigados {label}"), "periodo": period})
        rows.append({**stats(bottom[m].to_numpy(), f"bottom{k} extendidos {label}"), "periodo": period})
        rows.append({**stats(eq_w[m].to_numpy(), f"equal-weight todos {label}"), "periodo": period})
    return pd.DataFrame(rows)


def show(df: pd.DataFrame, title: str):
    print(f"\n--- {title} ---")
    print(df.set_index(["periodo", "estrategia"]).to_string(float_format=lambda z: f"{z:,.3f}"))


def main():
    research = add_mr_score(build_panel())
    research["date"] = pd.to_datetime(research["date"])
    research["grp"] = group_of(research)

    wl = add_mr_score(build_watchlist_panel())
    wl_stocks = wl[wl["class"] == "stock"]
    wl_crypto = wl[wl["class"] == "crypto"]

    print("=" * 120)
    print("  1) TILT — siempre invertido, peso = 0.5 + score  (exposición media ~1.0)")
    print("=" * 120)
    show(tilt_test(research[research["grp"] == "ETF"], "ETFs"), "ETFs (universo de research)")
    show(tilt_test(wl_stocks, "watchlist"), "Tu watchlist — acciones")
    show(tilt_test(wl_crypto, "cripto (score invertido)", invert=True), "Tu watchlist — cripto, señal invertida")

    print("\n" + "=" * 120)
    print("  2) ROTACIÓN — cada mes comprar los K más castigados del watchlist")
    print("=" * 120)
    show(rotation_test(research[research["grp"] == "ETF"], "ETFs", k=4), "ETFs (11 activos, top 4)")
    show(rotation_test(wl_stocks, "watchlist", k=5), "Tu watchlist — acciones (15 activos, top 5)")

    res = pd.concat([
        tilt_test(research[research["grp"] == "ETF"], "ETFs").assign(test="tilt_etf"),
        tilt_test(wl_stocks, "watchlist").assign(test="tilt_wl"),
        rotation_test(wl_stocks, "watchlist", k=5).assign(test="rot_wl"),
    ])
    res.to_csv(OUT / "allocation_results.csv", index=False)


if __name__ == "__main__":
    main()
