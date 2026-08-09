"""Meta-labeling with walk-forward validation.

Every number reported here is OUT OF SAMPLE. The protocol:

  for each test year Y:
      train on everything before Y, minus a purge window
      predict on Y, never refit

The purge matters. Labels are 21-session forward returns, so an event dated
21 days before the test set already "knows" part of it. Without dropping that
window the model leaks and every metric flatters itself.

The bar to clear is not 50%: it is the existing unweighted count. A model that
beats a coin flip but not the count is worthless here.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb

from research.events import HORIZON, build_events, feature_columns

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

ROOT = Path(__file__).resolve().parent.parent
FIRST_TEST_YEAR = 2015          # leaves ~5 years of training for the first fold
SELECT_FRAC = 0.20              # keep the top 20% of candidates


def walk_forward(ev: pd.DataFrame, feats: list[str], model_fn, name: str) -> pd.DataFrame:
    """Expanding-window walk-forward with a purge gap. Returns OOS predictions."""
    out = []
    years = sorted(y for y in ev["date"].dt.year.unique() if y >= FIRST_TEST_YEAR)
    for y in years:
        test_start = pd.Timestamp(f"{y}-01-01")
        purge_until = test_start - pd.Timedelta(days=int(HORIZON * 1.6))
        tr = ev[ev["date"] < purge_until]
        te = ev[(ev["date"] >= test_start) & (ev["date"] < pd.Timestamp(f"{y + 1}-01-01"))]
        if len(tr) < 800 or len(te) < 30:
            continue
        m = model_fn()
        m.fit(tr[feats].fillna(0.0), tr["y"])
        p = m.predict_proba(te[feats].fillna(0.0))[:, 1]
        out.append(te.assign(**{f"p_{name}": p}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def logistic():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=0.05, max_iter=2000, penalty="l2"))


def gbm():
    return lgb.LGBMClassifier(
        n_estimators=220, learning_rate=0.03, num_leaves=12,
        min_child_samples=120, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.6, reg_lambda=5.0, verbose=-1, random_state=7)


def evaluate(df: pd.DataFrame, col: str, label: str, frac: float = SELECT_FRAC) -> dict:
    thr = df[col].quantile(1 - frac)
    sel = df[df[col] >= thr]
    n = len(sel)
    se = np.sqrt(sel["y"].mean() * (1 - sel["y"].mean()) / n) if n else np.nan
    return {
        "modelo": label, "n_sel": n,
        "acierto": sel["y"].mean() if n else np.nan,
        "ic95": 1.96 * se if n else np.nan,
        "ret_medio": sel["ret"].mean() if n else np.nan,
        "auc": roc_auc_score(df["y"], df[col]) if df["y"].nunique() > 1 else np.nan,
        "base_acierto": df["y"].mean(), "base_ret": df["ret"].mean(),
    }


def main() -> None:
    ev = build_events()
    feats = feature_columns(ev)
    print(f"[META] {len(ev):,} eventos · {len(feats)} features · "
          f"test desde {FIRST_TEST_YEAR}\n")

    res = walk_forward(ev, feats, logistic, "log")
    res_g = walk_forward(ev, feats, gbm, "gbm")
    res = res.merge(res_g[["asset", "date", "p_gbm"]], on=["asset", "date"], how="inner")
    res["p_ens"] = (res["p_log"].rank(pct=True) + res["p_gbm"].rank(pct=True)) / 2
    res["count_score"] = res["n_conditions"]
    print(f"[META] predicciones OOS: {len(res):,} eventos "
          f"({res['date'].dt.year.min()}–{res['date'].dt.year.max()})\n")

    rows = [evaluate(res, c, l) for c, l in
            [("count_score", "conteo actual (baseline)"), ("p_log", "logística"),
             ("p_gbm", "LightGBM"), ("p_ens", "ensamble")]]
    tb = pd.DataFrame(rows)
    print("=" * 110)
    print(f"  SELECCIONANDO EL TOP {SELECT_FRAC:.0%} DE CANDIDATOS · todo fuera de muestra")
    print("=" * 110)
    print(tb.to_string(index=False, float_format=lambda z: f"{z:,.4f}"))

    best = max(("p_log", "p_gbm", "p_ens"),
               key=lambda c: evaluate(res, c, c)["acierto"])
    print(f"\n--- estabilidad año a año del mejor ({best}) vs el conteo ---")
    yr = []
    for y, g in res.groupby(res["date"].dt.year):
        if len(g) < 40:
            continue
        a = evaluate(g, best, "m")
        b = evaluate(g, "count_score", "c")
        yr.append({"año": y, "n": len(g), "modelo": a["acierto"], "conteo": b["acierto"],
                   "base": g["y"].mean(), "dif": a["acierto"] - b["acierto"]})
    ydf = pd.DataFrame(yr)
    print(ydf.to_string(index=False, float_format=lambda z: f"{z:,.3f}"))
    print(f"\n  años en que el modelo supera al conteo: "
          f"{(ydf['dif'] > 0).sum()}/{len(ydf)}")

    print("\n--- sensibilidad al corte de selección ---")
    sens = [{"top": f"{f:.0%}", **{k: v for k, v in evaluate(res, best, "m", f).items()
                                   if k in ("n_sel", "acierto", "ret_medio")}}
            for f in (0.05, 0.10, 0.20, 0.30, 0.50)]
    print(pd.DataFrame(sens).to_string(index=False, float_format=lambda z: f"{z:,.4f}"))

    res.to_parquet(ROOT / "data" / "meta_oos.parquet")
    print(f"\n[META] predicciones guardadas en data/meta_oos.parquet")


if __name__ == "__main__":
    main()
