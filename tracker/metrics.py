"""Live performance of each Discord channel, from the decision journal.

Design note — read this before adding a "is the edge real?" metric.

A power analysis on the backtest distribution (mean +3.01%, sd 10.35% per trade,
~20 trades/year) says detecting the +1.1pp edge against the base rate at 80%
power needs 685 trades: 34 years. On hit rate it is 195 years. Live data can
NEVER confirm the edge within a human timeframe, and a dashboard that implies
otherwise is lying.

What live data CAN do is detect that the system stopped behaving like the
backtest. So every metric here is tested against the BACKTEST DISTRIBUTION as
the null, not against zero — which is far more powerful — and the headline
status is a drift verdict, not a verdict on the strategy.

Everything is recomputed from prices plus the journal of what the bot said and
when. Nothing about outcomes is stored, so a fixed bug re-scores history.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from signals.data import load_ohlcv

ROOT = Path(__file__).resolve().parent.parent
BACKTEST_TRADES = ROOT / "out" / "signal_trades.csv"
ALLOC = 0.10                     # share of capital per equity trade
BOOT = 20000
SPY = "SPY"


# ------------------------------------------------------------------ helpers

def _prices(symbol: str, start: str) -> pd.DataFrame:
    end = (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    start = (pd.Timestamp(start) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    return load_ohlcv(symbol, start, end, force=True)


def _backtest_returns() -> np.ndarray:
    if not BACKTEST_TRADES.exists():
        return np.array([])
    return pd.read_csv(BACKTEST_TRADES)["fijo_21d"].dropna().to_numpy()


def _bootstrap_bands(sample: np.ndarray, n: int, alloc: float = ALLOC) -> dict:
    """Where a portfolio of `n` trades drawn from the backtest should land."""
    if len(sample) == 0 or n == 0:
        return {}
    rng = np.random.default_rng(12345)
    sims = np.array([np.prod(1 + rng.choice(sample, n, replace=True) * alloc) - 1
                     for _ in range(BOOT)])
    p = np.percentile(sims, [5, 25, 50, 75, 95])
    return {"p5": p[0], "p25": p[1], "p50": p[2], "p75": p[3], "p95": p[4],
            "_sims": sims}


def _drift(observed: float, sims: np.ndarray) -> dict:
    """Where the live result sits inside the backtest's own distribution."""
    pct = float((sims < observed).mean())
    if pct < 0.05:
        status, label = "alerta", "por debajo de lo esperado"
    elif pct < 0.25:
        status, label = "vigilar", "en la banda baja"
    elif pct > 0.95:
        status, label = "ok", "muy por encima de lo esperado"
    else:
        status, label = "ok", "dentro de lo esperado"
    return {"percentil": pct, "estado": status, "detalle": label}


# ------------------------------------------------------------------ equities

def equities(con: sqlite3.Connection) -> dict:
    rows = [dict(r) for r in con.execute(
        "SELECT * FROM positions ORDER BY signal_date")]
    if not rows:
        return {"n_cerradas": 0, "n_abiertas": 0, "operaciones": []}

    first = min(r["signal_date"] for r in rows)
    px_cache: dict[str, pd.DataFrame] = {}
    trades = []
    for r in rows:
        sym = r["asset"]
        if sym not in px_cache:
            try:
                px_cache[sym] = _prices(_symbol_for(con, sym), first)
            except Exception:
                continue
        df = px_cache.get(sym)
        entry = r["entry_price"]
        if entry is None or df is None or df.empty:
            continue
        last = float(df["close"].iloc[-1])
        closed = r["status"] == "closed"
        ret = r["pnl_pct"] if closed else last / entry - 1
        trades.append({
            "asset": sym, "signal_date": r["signal_date"], "entry_date": r["entry_date"],
            "entry": entry, "exit": r["exit_price"] if closed else last,
            "ret": ret, "bars_held": r["bars_held"], "estado": "cerrada" if closed else "abierta",
        })

    closed = [t for t in trades if t["estado"] == "cerrada"]
    rets = np.array([t["ret"] for t in closed], dtype=float)
    out = {
        "n_cerradas": len(closed), "n_abiertas": len(trades) - len(closed),
        "operaciones": trades,
    }
    if len(closed) == 0:
        return out

    equity = float(np.prod(1 + rets * ALLOC) - 1)
    se = rets.std(ddof=1) / np.sqrt(len(rets)) if len(rets) > 1 else np.nan
    out.update({
        "acierto": float((rets > 0).mean()),
        "ret_medio": float(rets.mean()),
        "ret_medio_ic95": [float(rets.mean() - 1.96 * se), float(rets.mean() + 1.96 * se)]
                          if np.isfinite(se) else None,
        "cartera_pct": equity,
        "mejor": float(rets.max()), "peor": float(rets.min()),
    })

    bt = _backtest_returns()
    if len(bt):
        bands = _bootstrap_bands(bt, len(closed))
        sims = bands.pop("_sims")
        out["backtest"] = {"acierto": float((bt > 0).mean()), "ret_medio": float(bt.mean())}
        out["bandas"] = bands
        out["drift"] = _drift(equity, sims)
    return out


def _symbol_for(con, alias: str) -> str:
    import yaml
    cfg = yaml.safe_load((ROOT / "bot" / "config.yaml").read_text())
    for it in cfg["equities"]["watchlist"]:
        if it["alias"] == alias:
            return it["symbol"]
    return alias


# ------------------------------------------------------------------ crypto

def crypto(con: sqlite3.Connection) -> dict:
    hist = pd.DataFrame([dict(r) for r in con.execute(
        "SELECT asset, weight, bar_date FROM crypto_weight_history ORDER BY bar_date")])
    if hist.empty:
        return {"n_ajustes": 0}

    import yaml
    cfg = yaml.safe_load((ROOT / "bot" / "config.yaml").read_text())
    sym = {i["alias"]: i["symbol"] for i in cfg["crypto"]["watchlist"]}
    start = hist["bar_date"].min()

    tilt_parts, hold_parts = [], []
    for alias, g in hist.groupby("asset"):
        if alias not in sym:
            continue
        px = _prices(sym[alias], start)
        px = px[px.index >= pd.Timestamp(start)]
        if px.empty:
            continue
        w = (pd.Series(g.set_index("bar_date")["weight"].to_dict())
             .rename(lambda d: pd.Timestamp(d)).sort_index()
             .reindex(px.index, method="ffill").ffill())
        ret = px["close"].pct_change().fillna(0.0)
        tilt_parts.append(ret * w.shift(1).fillna(0.0))   # act on the next bar
        hold_parts.append(ret)

    if not tilt_parts:
        return {"n_ajustes": len(hist)}

    tilt = pd.concat(tilt_parts, axis=1).mean(axis=1)
    hold = pd.concat(hold_parts, axis=1).mean(axis=1)
    return {
        "n_ajustes": int(len(hist)),
        "desde": start,
        "tilt_pct": float((1 + tilt).prod() - 1),
        "hold_pct": float((1 + hold).prod() - 1),
        "tilt_maxdd": _maxdd(tilt), "hold_maxdd": _maxdd(hold),
        "serie": _curve(tilt, hold),
    }


# ------------------------------------------------------------------ regime

def regime(con: sqlite3.Connection) -> dict:
    hist = [dict(r) for r in con.execute(
        "SELECT state, bar_date FROM regime_history ORDER BY bar_date")]
    if not hist:
        return {"n_cambios": 0}

    start = hist[0]["bar_date"]
    px = _prices(SPY, start)
    px = px[px.index >= pd.Timestamp(start)]
    if px.empty:
        return {"n_cambios": len(hist)}

    st = (pd.Series({pd.Timestamp(h["bar_date"]): 1.0 if h["state"] == "RISK_ON" else 0.0
                     for h in hist})
          .sort_index().reindex(px.index, method="ffill").ffill().fillna(1.0))
    ret = px["close"].pct_change().fillna(0.0)
    filtered = ret * st.shift(1).fillna(0.0)
    return {
        "n_cambios": len(hist),
        "desde": start,
        "estado_actual": hist[-1]["state"],
        "filtrado_pct": float((1 + filtered).prod() - 1),
        "spy_pct": float((1 + ret).prod() - 1),
        "filtrado_maxdd": _maxdd(filtered), "spy_maxdd": _maxdd(ret),
        "expuesto": float(st.mean()),
        "serie": _curve(filtered, ret),
    }


# ------------------------------------------------------------------ utils

def _maxdd(r: pd.Series) -> float:
    eq = (1 + r).cumprod()
    return float((eq / eq.cummax() - 1).min())


def _curve(a: pd.Series, b: pd.Series) -> list[dict]:
    ea, eb = (1 + a).cumprod() - 1, (1 + b).cumprod() - 1
    return [{"d": d.date().isoformat(), "a": round(float(x), 5), "b": round(float(y), 5)}
            for d, x, y in zip(ea.index, ea, eb)]


def build(con: sqlite3.Connection) -> dict:
    return {
        "generado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "nota_metodologica": (
            "Detectar el edge en vivo requiere ~685 operaciones (34 años). "
            "Estas métricas NO validan la estrategia: detectan si dejó de "
            "comportarse como el backtest. El contraste es contra la "
            "distribución del backtest, no contra cero."),
        "acciones": equities(con),
        "cripto": crypto(con),
        "regimen": regime(con),
    }


def write(con: sqlite3.Connection, path: Path | None = None) -> Path:
    path = path or ROOT / "docs" / "data" / "performance.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build(con), indent=2, ensure_ascii=False, default=str))
    return path
