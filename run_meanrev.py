"""Turn the screening result into a signal and test it as a real strategy.

Composite "oversoldness" score, all components point-in-time:

    mr_score = mean over components of (1 - trailing_252d_percentile_rank)

Components are the features whose IC on 63-day forward return was negative and
sign-consistent across years: rsi14, zscore_20, dist_ema200_atr, drawdown_252.
High mr_score = beaten down relative to the asset's own recent history.

Two evaluations:
  1. Decile table   -> forward 63d return by score decile, IS vs OOS
  2. Long/flat backtest -> hold only while the score is elevated, vs buy & hold
"""

import argparse
import warnings

import numpy as np
import pandas as pd

from run_screen import OUT, build_panel
from run_control import ETFS

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)

OOS_START = pd.Timestamp("2020-01-01")
COMPONENTS = ["rsi14", "zscore_20", "dist_ema200_atr", "drawdown_252"]
COST = 0.001          # 10 bps per side


def add_mr_score(panel: pd.DataFrame, components=COMPONENTS, window: int = 252) -> pd.DataFrame:
    p = panel.sort_values(["asset", "date"]).copy()
    parts = []
    for comp in components:
        r = p.groupby("asset")[comp].transform(
            lambda s: s.rolling(window, min_periods=window // 2).rank(pct=True))
        parts.append(1.0 - r)                      # invert: negative IC -> oversold is good
    p["mr_score"] = pd.concat(parts, axis=1).mean(axis=1)

    # Market-level oversoldness, shared by every asset in the class.
    p["mkt_score"] = 1.0 - p.groupby("asset")["bench_px_ema200"].transform(
        lambda s: s.rolling(window, min_periods=window // 2).rank(pct=True))
    return p


def group_of(p: pd.DataFrame) -> pd.Series:
    g = np.where(p["asset"].isin(ETFS), "ETF",
                 np.where(p["class"] == "crypto", "Cripto", "Accion"))
    return pd.Series(g, index=p.index)


def decile_table(p: pd.DataFrame, target: str = "fwd_ret_63") -> pd.DataFrame:
    d = p[p[target].notna() & p["mr_score"].notna()].copy()
    d["grp"] = group_of(d)
    d["dec"] = (d["mr_score"] * 10).clip(0, 9.999).astype(int)
    d["period"] = np.where(d["date"] < OOS_START, "IS (2010-19)", "OOS (2020-26)")

    tb = d.pivot_table(index="dec", columns=["grp", "period"], values=target, aggfunc="mean")
    base = d.pivot_table(index=[], columns=["grp", "period"], values=target, aggfunc="mean")
    tb.loc["TODOS"] = d.groupby(["grp", "period"])[target].mean()
    return tb


def backtest_long_flat(p: pd.DataFrame, entry: float, exit_: float, use_mkt: bool = False):
    """Per-asset long/flat timing on mr_score, compared against buy & hold."""
    rows = []
    for asset, g in p.groupby("asset"):
        g = g.sort_values("date")
        s = g["mr_score"].to_numpy(float)
        m = g["mkt_score"].to_numpy(float)
        r = g["ret_1"].to_numpy(float)
        dates = g["date"].to_numpy()
        n = len(g)

        pos = np.zeros(n)
        holding = False
        for i in range(n):
            if not np.isfinite(s[i]):
                pos[i] = 0.0
                continue
            gate = (not use_mkt) or (np.isfinite(m[i]) and m[i] >= 0.3)
            if not holding and s[i] >= entry and gate:
                holding = True
            elif holding and s[i] <= exit_:
                holding = False
            pos[i] = 1.0 if holding else 0.0

        pos_l = np.roll(pos, 1)          # trade on the next bar's return
        pos_l[0] = 0.0
        trades = np.abs(np.diff(np.concatenate([[0.0], pos_l])))
        strat_r = np.nan_to_num(r) * pos_l - trades * COST
        bh_r = np.nan_to_num(r)

        for period, mask in (("IS", dates < np.datetime64(OOS_START)),
                             ("OOS", dates >= np.datetime64(OOS_START))):
            if mask.sum() < 250:
                continue
            sr, br = strat_r[mask], bh_r[mask]
            yrs = mask.sum() / 252.0
            rows.append({
                "asset": asset, "grp": group_of(g).iloc[0], "period": period,
                "cagr_strat": (1 + sr).prod() ** (1 / yrs) - 1,
                "cagr_bh": (1 + br).prod() ** (1 / yrs) - 1,
                "sharpe_strat": sr.mean() / sr.std() * np.sqrt(252) if sr.std() > 0 else np.nan,
                "sharpe_bh": br.mean() / br.std() * np.sqrt(252) if br.std() > 0 else np.nan,
                "maxdd_strat": ((1 + sr).cumprod() / np.maximum.accumulate((1 + sr).cumprod()) - 1).min(),
                "maxdd_bh": ((1 + br).cumprod() / np.maximum.accumulate((1 + br).cumprod()) - 1).min(),
                "exposure": pos_l[mask].mean(),
                "n_trades": int(trades[mask].sum() / 2),
            })
    return pd.DataFrame(rows)


def summarize_bt(bt: pd.DataFrame, label: str) -> pd.DataFrame:
    out = bt.groupby(["grp", "period"]).agg(
        n_assets=("asset", "nunique"),
        cagr_strat=("cagr_strat", "mean"), cagr_bh=("cagr_bh", "mean"),
        sharpe_strat=("sharpe_strat", "mean"), sharpe_bh=("sharpe_bh", "mean"),
        maxdd_strat=("maxdd_strat", "mean"), maxdd_bh=("maxdd_bh", "mean"),
        exposure=("exposure", "mean"), trades=("n_trades", "mean"),
    )
    out["d_cagr"] = out["cagr_strat"] - out["cagr_bh"]
    out["d_sharpe"] = out["sharpe_strat"] - out["sharpe_bh"]
    out["win_rate_vs_bh"] = bt.groupby(["grp", "period"]).apply(
        lambda g: (g["sharpe_strat"] > g["sharpe_bh"]).mean())
    print(f"\n--- {label} ---")
    print(out.to_string(float_format=lambda z: f"{z:,.3f}"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    panel = build_panel(args.force)
    panel["date"] = pd.to_datetime(panel["date"])
    p = add_mr_score(panel)

    print("\n" + "=" * 130)
    print("  RETORNO 63d FORWARD POR DECIL DE mr_score (0 = caro/extendido, 9 = castigado)")
    print("=" * 130)
    print(decile_table(p).to_string(float_format=lambda z: f"{z:,.4f}"))

    print("\n" + "=" * 130)
    print("  ESTRATEGIA LONG/FLAT  ·  entra con score>=0.7, sale con score<=0.4  ·  costo 10bps/lado")
    print("=" * 130)
    bt = backtest_long_flat(p, 0.70, 0.40)
    summarize_bt(bt, "sin gate de mercado")
    bt.to_csv(OUT / "meanrev_backtest.csv", index=False)

    bt2 = backtest_long_flat(p, 0.70, 0.40, use_mkt=True)
    summarize_bt(bt2, "con gate de mercado (mkt_score >= 0.3)")


if __name__ == "__main__":
    main()
