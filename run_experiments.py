"""Variant sweep: does ANY configuration of the framework beat buying at random?

Baselines matter more than the signals here:
  base_all      -> buy any eligible day
  base_uptrend  -> buy any day with close > EMA200            (one-line filter)
  base_up_rs    -> buy any day with close > EMA200 and top-30% relative strength
If a playbook cannot beat base_uptrend, the whole scoring layer adds nothing.
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path

from signals.backtest import evaluate_asset, generate_signals, simulate, summarize
from signals.playbooks import score_all
from run_backtest import ROOT, OUT, build_panel, eligibility

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 60)


def trades_from_mask(alias, cls, x, mask, cfg, horizon, stop_mult, target_pct):
    o, h, l, c = (x[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    atr_ = x["atr14"].to_numpy(float)
    m = mask.reindex(x.index).fillna(False).to_numpy(bool)
    rows = []
    for i in range(len(x)):
        if not m[i] or not np.isfinite(atr_[i]):
            continue
        r = simulate(h, l, c, o, i + 1, stop_mult * atr_[i], horizon, target_pct)
        if r is not None:
            rows.append({"asset": alias, "class": cls, "date": x.index[i], **r})
    return rows


def z_vs_base(p_sig, n_sig, p_base):
    """One-sample z-test of the signal success rate against the base rate."""
    if not n_sig or not np.isfinite(p_sig) or not np.isfinite(p_base):
        return np.nan
    se = np.sqrt(p_base * (1 - p_base) / n_sig)
    return (p_sig - p_base) / se if se > 0 else np.nan


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start")
    ap.add_argument("--warmup")
    a = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    if a.warmup:
        cfg["backtest"]["warmup_start"] = a.warmup
    if a.start:
        cfg["backtest"]["signal_start"] = a.start
    panel = build_panel(cfg)
    bt = cfg["backtest"]
    target = bt["target_pct"]

    # Precompute per-asset scores / masks once.
    prep = {}
    for alias, v in panel["assets"].items():
        cls = v["meta"]["class"]
        x = v["x"]
        scores, _ = score_all(x, panel["regimes"][cls], v["rs_top30"], v["rs_pos"], ["A", "B", "C"])
        prep[alias] = {
            "cls": cls, "x": x, "scores": scores,
            "elig": eligibility(x, cls, cfg),
            "rs_top30": v["rs_top30"].reindex(x.index).fillna(False),
            "horizon": bt["horizon_bars_stock"] if cls == "stock" else bt["horizon_bars_crypto"],
        }

    results = []

    def add(label, rows, kind):
        df = pd.DataFrame(rows)
        s = summarize(df, label) if len(df) else {"set": label, "n": 0}
        results.append({**s, "kind": kind})

    # ---------- baselines ----------
    for stop_mult in (2.5,):
        b_all, b_up, b_up_rs = [], [], []
        for a, p in prep.items():
            up = p["elig"] & (p["x"]["close"] > p["x"]["ema200"])
            b_all += trades_from_mask(a, p["cls"], p["x"], p["elig"], cfg, p["horizon"], stop_mult, target)
            b_up += trades_from_mask(a, p["cls"], p["x"], up, cfg, p["horizon"], stop_mult, target)
            b_up_rs += trades_from_mask(a, p["cls"], p["x"], up & p["rs_top30"], cfg, p["horizon"], stop_mult, target)
        add("base_all", b_all, "baseline")
        add("base_uptrend", b_up, "baseline")
        add("base_uptrend_rs_top30", b_up_rs, "baseline")

    base_rate = next(r["success_rate"] for r in results if r["set"] == "base_all")
    base_up_rate = next(r["success_rate"] for r in results if r["set"] == "base_uptrend")

    # ---------- variants ----------
    variants = []
    for th in (60, 65, 70, 75, 80, 85):
        variants.append({"label": f"score>={th} · all PB", "th": th, "pb": ["A", "B", "C"], "stop": 2.5})
    for pb in (["A"], ["B"], ["C"]):
        variants.append({"label": f"score>=70 · PB {pb[0]}", "th": 70, "pb": pb, "stop": 2.5})
    for stop in (2.0, 3.0, 4.0, 6.0, 99.0):
        tag = "sin stop" if stop == 99.0 else f"stop {stop}xATR"
        variants.append({"label": f"score>=70 · {tag}", "th": 70, "pb": ["A", "B", "C"], "stop": stop})
    variants.append({"label": "score>=70 · gate RS top30", "th": 70, "pb": ["A", "B", "C"], "stop": 2.5, "rs_gate": True})
    variants.append({"label": "score>=70 · PB B · stop 4xATR", "th": 70, "pb": ["B"], "stop": 4.0})
    variants.append({"label": "score>=70 · PB B · gate RS", "th": 70, "pb": ["B"], "stop": 2.5, "rs_gate": True})

    for vspec in variants:
        rows = []
        for a, p in prep.items():
            cols = [c for c in vspec["pb"] if c in p["scores"].columns]
            sub = p["scores"][cols]
            elig = p["elig"] & (p["rs_top30"] if vspec.get("rs_gate") else True)
            sigs = generate_signals(sub, elig, vspec["th"], cfg["thresholds"]["exit_hysteresis"],
                                    bt["cooldown_bars"])
            o, h, l, c = (p["x"][k].to_numpy(float) for k in ("open", "high", "low", "close"))
            atr_ = p["x"]["atr14"].to_numpy(float)
            for sg in sigs:
                i = sg["i"]
                r = simulate(h, l, c, o, i + 1, vspec["stop"] * atr_[i], p["horizon"], target)
                if r is not None:
                    rows.append({"asset": a, "class": p["cls"], **sg, **r})
        add(vspec["label"], rows, "variant")

    # ---------- report ----------
    df = pd.DataFrame(results)
    df["edge_vs_all"] = df["success_rate"] - base_rate
    df["edge_vs_uptrend"] = df["success_rate"] - base_up_rate
    df["z"] = df.apply(lambda r: z_vs_base(r.get("success_rate"), r.get("n"), base_rate), axis=1)

    cols = ["set", "kind", "n", "success_rate", "edge_vs_all", "edge_vs_uptrend", "z",
            "avg_R", "expectancy_pct", "avg_ret_3m", "hit10_no_stop", "avg_mae"]
    df = df[cols].sort_values(["kind", "expectancy_pct"], ascending=[True, False])

    print("\n" + "=" * 150)
    print("  VARIANTES — success_rate = % que toca +10% USD antes del stop, dentro de 3 meses")
    print("  |z| > 1.96 => diferencia estadisticamente significativa vs base_all")
    print("=" * 150)
    print(df.to_string(index=False, float_format=lambda z: f"{z:,.3f}"))
    df.to_csv(OUT / "experiments.csv", index=False)

    # ---------- period characterization ----------
    print("\n--- contexto del periodo 2024-01-01 .. 2026-08-07 ---")
    ctx = []
    for a, p in prep.items():
        w = p["x"].loc["2024-01-01":]
        ctx.append({
            "asset": a,
            "buy_hold_ret": w["close"].iloc[-1] / w["close"].iloc[0] - 1,
            "pct_days_above_ema200": (w["close"] > w["ema200"]).mean(),
            "max_dd": (w["close"] / w["close"].cummax() - 1).min(),
        })
    print(pd.DataFrame(ctx).sort_values("buy_hold_ret", ascending=False)
          .to_string(index=False, float_format=lambda z: f"{z:,.3f}"))

    for kind, reg in panel["regimes"].items():
        r = reg.loc["2024-01-01":]
        print(f"\n[{kind}] régimen: " + ", ".join(f"{k}={v/len(r):.0%}" for k, v in r.value_counts().items()))


if __name__ == "__main__":
    main()
