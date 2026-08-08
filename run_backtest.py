"""Backtest the three playbooks over the configured watchlist.

Usage:
    python3 run_backtest.py                 # default threshold from config
    python3 run_backtest.py --sweep         # threshold sensitivity table
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from signals.backtest import evaluate_asset, summarize
from signals.data import load_ohlcv
from signals.indicators import enrich
from signals.playbooks import score_all
from signals.regime import compute_regime

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)


def build_panel(cfg: dict) -> dict:
    bt = cfg["backtest"]
    start, end = bt["warmup_start"], bt["signal_end"]

    proxies = {}
    for kind, sym in (("stock", cfg["regime"]["stock_proxy"]), ("crypto", cfg["regime"]["crypto_proxy"])):
        proxies[kind] = load_ohlcv(sym, start, end)

    regimes = {k: compute_regime(df, k) for k, df in proxies.items()}

    bench_ret = {}
    for kind, sym in (("stock", cfg["regime"]["stock_proxy"]), ("crypto", cfg["regime"]["crypto_proxy"])):
        bench_ret[kind] = proxies[kind]["close"].pct_change(63)

    assets = {}
    for item in cfg["universe"]:
        df = load_ohlcv(item["symbol"], start, end)
        assets[item["alias"]] = {"meta": item, "x": enrich(df)}

    # Cross-sectional relative strength, ranked within each asset class.
    for kind in ("stock", "crypto"):
        members = [a for a, v in assets.items() if v["meta"]["class"] == kind]
        rs = pd.DataFrame(
            {a: assets[a]["x"]["ret_63"] - bench_ret[kind].reindex(assets[a]["x"].index) for a in members}
        )
        rank = rs.rank(axis=1, pct=True)
        for a in members:
            # rs/rank live on the union index of the class; realign to the asset's own bars
            own = assets[a]["x"].index
            assets[a]["rs"] = rs[a].reindex(own)
            assets[a]["rs_top30"] = (rank[a] >= 0.70).reindex(own).fillna(False)
            assets[a]["rs_pos"] = (rs[a] > 0).reindex(own).fillna(False)

    return {"assets": assets, "regimes": regimes, "proxies": proxies}


def eligibility(x: pd.DataFrame, cls: str, cfg: dict) -> pd.Series:
    bt, liq = cfg["backtest"], cfg["liquidity"]
    min_dv = liq["min_dollar_vol_stock"] if cls == "stock" else liq["min_dollar_vol_crypto"]
    in_window = (x.index >= pd.Timestamp(bt["signal_start"])) & (x.index <= pd.Timestamp(bt["signal_end"]))
    return (
        pd.Series(in_window, index=x.index)
        & x["ema200"].notna()
        & x["atr14"].notna()
        & (x["dollar_vol20"] > min_dv)
        & (x["atr_pct"] > 0.01)
        & (x["atr_pct"] < 0.15)
    ).fillna(False)


def run(cfg: dict, panel: dict, setup_th: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    bt = cfg["backtest"]
    all_sig, all_base = [], []

    for alias, v in panel["assets"].items():
        cls = v["meta"]["class"]
        x = v["x"]
        regime = panel["regimes"][cls]
        horizon = bt["horizon_bars_stock"] if cls == "stock" else bt["horizon_bars_crypto"]

        scores, _ = score_all(
            x, regime, v["rs_top30"], v["rs_pos"], cfg["playbooks_enabled"]
        )
        elig = eligibility(x, cls, cfg)
        sig, base = evaluate_asset(alias, x, scores, elig, cfg, horizon, setup_th)
        for r in sig:
            r["class"] = cls
        for r in base:
            r["class"] = cls
        all_sig += sig
        all_base += base

    return pd.DataFrame(all_sig), pd.DataFrame(all_base)


def report(sig: pd.DataFrame, base: pd.DataFrame, setup_th: float) -> dict:
    done = sig[sig["outcome"] != "INCOMPLETE"]
    overall_sig = summarize(done, f"SIGNALS (score>={setup_th:g})")
    overall_base = summarize(base, "BASE RATE (any eligible day)")

    print("\n" + "=" * 92)
    print(f"  KPI PRINCIPAL — % de operaciones que alcanzan +10% USD antes del stop, en 3 meses")
    print("=" * 92)
    rows = [overall_sig, overall_base]
    df = pd.DataFrame(rows).set_index("set")
    show = ["n", "success_rate", "loss_rate", "timeout_rate", "hit10_no_stop", "avg_R",
            "expectancy_pct", "avg_ret_3m", "avg_mfe", "avg_mae", "avg_bars_to_exit"]
    print(df[show].to_string(float_format=lambda z: f"{z:,.3f}"))

    edge = overall_sig.get("success_rate", np.nan) - overall_base.get("success_rate", np.nan)
    print(f"\n  EDGE = success_rate(signals) - success_rate(base) = {edge:+.1%}")

    print("\n--- por playbook ---")
    pb_rows = [summarize(done[done["playbook"] == p], p) for p in sorted(done["playbook"].unique())]
    print(pd.DataFrame(pb_rows).set_index("set")[show].to_string(float_format=lambda z: f"{z:,.3f}"))

    print("\n--- por clase de activo ---")
    cls_rows = []
    for c in sorted(done["class"].unique()):
        cls_rows.append(summarize(done[done["class"] == c], f"signals/{c}"))
        cls_rows.append(summarize(base[base["class"] == c], f"base/{c}"))
    print(pd.DataFrame(cls_rows).set_index("set")[show].to_string(float_format=lambda z: f"{z:,.3f}"))

    print("\n--- por activo (señales vs base rate del mismo activo) ---")
    per_asset = []
    for a in sorted(done["asset"].unique()):
        s = summarize(done[done["asset"] == a], a)
        b = summarize(base[base["asset"] == a], a)
        per_asset.append({
            "asset": a,
            "n_sig": s["n"],
            "success": s.get("success_rate"),
            "base": b.get("success_rate"),
            "edge": (s.get("success_rate", np.nan) - b.get("success_rate", np.nan)),
            "avg_R": s.get("avg_R"),
            "avg_ret_3m": s.get("avg_ret_3m"),
        })
    pa = pd.DataFrame(per_asset).sort_values("edge", ascending=False)
    print(pa.to_string(index=False, float_format=lambda z: f"{z:,.3f}"))

    print("\n--- por régimen en el momento de la señal ---")
    if "regime" in done.columns:
        rg = [summarize(done[done["regime"] == r], r) for r in sorted(done["regime"].dropna().unique())]
        print(pd.DataFrame(rg).set_index("set")[show].to_string(float_format=lambda z: f"{z:,.3f}"))

    n_incomplete = (sig["outcome"] == "INCOMPLETE").sum()
    print(f"\n[BT] señales sin 3 meses de datos por delante (excluidas del KPI): {n_incomplete}")
    return {"signals": overall_sig, "base": overall_base, "edge": float(edge)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--start", help="override backtest.signal_start")
    ap.add_argument("--warmup", help="override backtest.warmup_start")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    if args.warmup:
        cfg["backtest"]["warmup_start"] = args.warmup
    if args.start:
        cfg["backtest"]["signal_start"] = args.start
    panel = build_panel(cfg)

    if args.sweep:
        print("\n=== sensibilidad al umbral de SETUP ===")
        rows = []
        for th in [50, 55, 60, 65, 70, 75, 80]:
            sig, base = run(cfg, panel, th)
            done = sig[sig["outcome"] != "INCOMPLETE"] if len(sig) else sig
            s = summarize(done, str(th)) if len(done) else {"n": 0}
            b = summarize(base, "base")
            rows.append({
                "threshold": th,
                "n_signals": s.get("n", 0),
                "success_rate": s.get("success_rate"),
                "base_rate": b.get("success_rate"),
                "edge": (s.get("success_rate", np.nan) - b.get("success_rate", np.nan)),
                "avg_R": s.get("avg_R"),
                "expectancy_pct": s.get("expectancy_pct"),
            })
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda z: f"{z:,.3f}"))
        return

    setup_th = cfg["thresholds"]["setup"]
    sig, base = run(cfg, panel, setup_th)
    sig.to_csv(OUT / "signals.csv", index=False)
    base.to_csv(OUT / "base_rate_trades.csv", index=False)
    summary = report(sig, base, setup_th)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[BT] detalle -> {OUT/'signals.csv'}")


if __name__ == "__main__":
    main()
