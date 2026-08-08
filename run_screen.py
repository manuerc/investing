"""Univariate screening: which indicators actually predict forward returns?

Primary target: 63-day forward return IN EXCESS OF THE BENCHMARK. Absolute
forward return is dominated by market beta, which is exactly what made the
first backtest uninformative.

Reported per feature:
  ic_pooled  Spearman over all (asset, date) rows
  ic_ts      mean of per-asset time-series Spearman   -> "when to buy THIS asset"
  ic_xs      mean of per-date cross-sectional Spearman -> "which asset to buy"
  ic_is/oos  in-sample (<2020) vs out-of-sample (>=2020)
  yr_consist fraction of years whose IC has the same sign as ic_pooled
  d10_d1     top-decile minus bottom-decile mean forward excess return
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from signals.data import load_ohlcv
from signals.features import build_features, build_targets
from signals.universe import BENCH, research_universe

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)
PANEL = ROOT / "data" / "panel.parquet"

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 80)
pd.set_option("display.max_rows", 200)

WARMUP = "2010-01-01"
END = "2026-08-08"
OOS_START = "2020-01-01"


def build_panel(force: bool = False) -> pd.DataFrame:
    if PANEL.exists() and not force:
        return pd.read_parquet(PANEL)

    bench = {k: load_ohlcv(v, WARMUP, END)["close"] for k, v in BENCH.items()}
    frames = []
    for item in research_universe():
        try:
            df = load_ohlcv(item["symbol"], WARMUP, END)
        except Exception as e:
            print(f"[SCREEN] skip {item['symbol']}: {e}")
            continue
        if len(df) < 400:
            print(f"[SCREEN] skip {item['symbol']}: only {len(df)} bars")
            continue
        bc = bench[item["class"]]
        # An asset cannot have excess return against itself.
        bc_use = None if item["symbol"] == BENCH[item["class"]] else bc
        f = build_features(df, bc_use)
        t = build_targets(df, bc_use)
        block = pd.concat([f, t], axis=1)
        block["close"] = df["close"]
        block["asset"] = item["alias"]
        block["class"] = item["class"]
        block["date"] = block.index
        frames.append(block.reset_index(drop=True))

    panel = pd.concat(frames, ignore_index=True)
    panel.to_parquet(PANEL)
    print(f"[SCREEN] panel: {len(panel):,} rows x {panel.shape[1]} cols, "
          f"{panel['asset'].nunique()} assets")
    return panel


def feature_columns(panel: pd.DataFrame) -> list[str]:
    skip = {"asset", "class", "date", "close"}   # close is not scale-free
    return [c for c in panel.columns
            if c not in skip and not c.startswith("fwd_")]


def spearman(a: pd.Series, b: pd.Series) -> float:
    m = a.notna() & b.notna()
    if m.sum() < 50:
        return np.nan
    return a[m].rank().corr(b[m].rank())


def screen(panel: pd.DataFrame, target: str, min_rows: int = 5000) -> pd.DataFrame:
    feats = feature_columns(panel)
    p = panel[panel[target].notna()].copy()
    p["year"] = p["date"].dt.year

    weekly = p[p["date"].dt.dayofweek == 2]          # Wednesdays, for the XS pass
    rows = []
    for f in feats:
        s, y = p[f], p[target]
        if s.notna().sum() < min_rows:
            continue

        ic_pooled = spearman(s, y)
        ic_is = spearman(s[p["date"] < OOS_START], y[p["date"] < OOS_START])
        ic_oos = spearman(s[p["date"] >= OOS_START], y[p["date"] >= OOS_START])

        ts = p.groupby("asset").apply(lambda g: spearman(g[f], g[target]), include_groups=False)
        ic_ts = ts.mean()
        ts_pos = (np.sign(ts.dropna()) == np.sign(ic_pooled)).mean() if len(ts.dropna()) else np.nan

        xs = weekly.groupby(["date", "class"]).apply(
            lambda g: spearman(g[f], g[target]), include_groups=False)
        ic_xs = xs.mean()

        yr = p.groupby("year").apply(lambda g: spearman(g[f], g[target]), include_groups=False).dropna()
        yr_consist = (np.sign(yr) == np.sign(ic_pooled)).mean() if len(yr) else np.nan

        # Decile spread, deciles formed cross-sectionally within (date, class).
        q = p.groupby(["date", "class"])[f].transform(
            lambda g: pd.qcut(g, 10, labels=False, duplicates="drop") if g.notna().sum() >= 10 else np.nan)
        d10 = p.loc[q == 9, target].mean()
        d1 = p.loc[q == 0, target].mean()

        rows.append({
            "feature": f, "n": int(s.notna().sum()),
            "ic_pooled": ic_pooled, "ic_ts": ic_ts, "ic_xs": ic_xs,
            "ic_is": ic_is, "ic_oos": ic_oos,
            "ts_consist": ts_pos, "yr_consist": yr_consist,
            "d10": d10, "d1": d1, "d10_d1": d10 - d1,
        })

    out = pd.DataFrame(rows)
    out["abs_ic"] = out["ic_pooled"].abs()
    return out.sort_values("abs_ic", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rebuild the panel")
    ap.add_argument("--target", default="fwd_excess_63")
    args = ap.parse_args()

    panel = build_panel(args.force)
    panel["date"] = pd.to_datetime(panel["date"])
    print(f"[SCREEN] panel {len(panel):,} filas · {panel['asset'].nunique()} activos · "
          f"{panel['date'].min().date()} -> {panel['date'].max().date()}")

    for target in ("fwd_excess_63", "fwd_ret_63", "fwd_excess_126", "fwd_excess_21"):
        res = screen(panel, target)
        res.to_csv(OUT / f"screen_{target}.csv", index=False)
        cols = ["feature", "n", "ic_pooled", "ic_ts", "ic_xs", "ic_is", "ic_oos",
                "ts_consist", "yr_consist", "d10_d1"]
        print("\n" + "=" * 140)
        print(f"  TARGET: {target}   (IC positivo = valor alto del feature -> retorno futuro alto)")
        print("=" * 140)
        print(res.head(22)[cols].to_string(index=False, float_format=lambda z: f"{z:,.4f}"))


if __name__ == "__main__":
    main()
