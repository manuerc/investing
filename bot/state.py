"""SQLite state: cooldown, last crypto weight, regime, and the trade journal.

The journal is what lets the bot report its own hit rate later instead of
trusting the backtest forever.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "bot_state.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS equity_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset TEXT NOT NULL, bar_date TEXT NOT NULL,
    conditions INTEGER NOT NULL, close REAL NOT NULL,
    p_up REAL, detail TEXT, created_at TEXT NOT NULL,
    UNIQUE(asset, bar_date)
);
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset TEXT NOT NULL, signal_date TEXT NOT NULL,
    entry_price REAL, entry_date TEXT,
    bars_held INTEGER NOT NULL DEFAULT 0,
    last_bar_date TEXT, hold_bars INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    exit_price REAL, exit_date TEXT, pnl_pct REAL
);
CREATE TABLE IF NOT EXISTS crypto_weights (
    asset TEXT PRIMARY KEY, weight REAL NOT NULL,
    bar_date TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS regime_state (
    proxy TEXT PRIMARY KEY, state TEXT NOT NULL,
    bar_date TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS regime_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    proxy TEXT NOT NULL, state TEXT NOT NULL,
    bar_date TEXT NOT NULL, recorded_at TEXT NOT NULL,
    UNIQUE(proxy, bar_date)
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ran_at TEXT NOT NULL, dry_run INTEGER NOT NULL,
    n_equity INTEGER, n_crypto INTEGER, n_exits INTEGER, regime_change INTEGER
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


# ---------------- equity ----------------

def in_cooldown(con, asset: str, cooldown_bars: int, bars: list[dict]) -> bool:
    """Blocked by an open position, or by a signal fired less than N sessions ago.

    `bars` is the asset's recent session calendar, so the cooldown is measured in
    real trading days regardless of how often the job runs.
    """
    if con.execute("SELECT 1 FROM positions WHERE asset=? AND status='open'",
                   (asset,)).fetchone():
        return True
    row = con.execute(
        "SELECT bar_date FROM equity_signals WHERE asset=? ORDER BY bar_date DESC LIMIT 1",
        (asset,)).fetchone()
    if not row:
        return False
    elapsed = sum(1 for b in bars if b["date"] > row["bar_date"])
    return elapsed < cooldown_bars


def record_equity_signal(con, sig, hold_bars: int) -> bool:
    """False if this exact (asset, bar) was already recorded."""
    try:
        con.execute(
            "INSERT INTO equity_signals(asset,bar_date,conditions,close,p_up,detail,created_at)"
            " VALUES (?,?,?,?,?,?,?)",
            (sig.asset, sig.bar_date, sig.conditions, sig.close, sig.p_up,
             json.dumps(sig.detail, ensure_ascii=False), _now()))
    except sqlite3.IntegrityError:
        return False
    con.execute(
        "INSERT INTO positions(asset,signal_date,last_bar_date,hold_bars) VALUES (?,?,?,?)",
        (sig.asset, sig.bar_date, sig.bar_date, hold_bars))
    con.commit()
    return True


def advance_positions(con, latest: dict[str, dict]) -> list[dict]:
    """Update open positions from real elapsed sessions; return those due to exit.

    `latest` maps asset -> {"bar_date", "close", "bar_dates"}. The holding period
    is counted from the asset's own session calendar, so a missed run (or a
    weekend, holiday or outage) never shifts the exit date.
    """
    due = []
    for row in con.execute("SELECT * FROM positions WHERE status='open'").fetchall():
        cur = latest.get(row["asset"])
        if not cur:
            continue
        after = [b for b in cur.get("bars", []) if b["date"] > row["signal_date"]]
        held = len(after)
        if held == 0:
            continue                       # signal bar is still the latest one

        if row["entry_price"] is None:
            # entry filled at the open of the first session after the signal
            con.execute("UPDATE positions SET entry_price=?, entry_date=? WHERE id=?",
                        (after[0]["open"], after[0]["date"], row["id"]))
            entry = after[0]["open"]
        else:
            entry = row["entry_price"]

        con.execute("UPDATE positions SET bars_held=?, last_bar_date=? WHERE id=?",
                    (held, cur["bar_date"], row["id"]))

        if held >= row["hold_bars"]:
            pnl = cur["close"] / entry - 1 if entry else None
            con.execute(
                "UPDATE positions SET status='closed', exit_price=?, exit_date=?, pnl_pct=?"
                " WHERE id=?", (cur["close"], cur["bar_date"], pnl, row["id"]))
            due.append({"asset": row["asset"], "signal_date": row["signal_date"],
                        "entry_price": entry, "exit_price": cur["close"],
                        "pnl_pct": pnl, "bars_held": held})
    con.commit()
    return due


# ---------------- crypto & regime ----------------

def get_crypto_weights(con) -> dict[str, float]:
    return {r["asset"]: r["weight"] for r in con.execute("SELECT * FROM crypto_weights")}


def set_crypto_weight(con, asset: str, weight: float, bar_date: str) -> None:
    con.execute(
        "INSERT INTO crypto_weights(asset,weight,bar_date,updated_at) VALUES (?,?,?,?)"
        " ON CONFLICT(asset) DO UPDATE SET weight=excluded.weight,"
        " bar_date=excluded.bar_date, updated_at=excluded.updated_at",
        (asset, weight, bar_date, _now()))
    con.commit()


def regime_changes_this_year(con, proxy: str, year: int) -> int:
    """How many times the regime already flipped this year.

    Smoothing the rule was tested and it degrades the drawdown protection, so
    the whipsaw stays — but the alert says how choppy the year has been, which
    turns the noise into context instead of hiding it.
    """
    row = con.execute(
        "SELECT COUNT(*) c FROM regime_history WHERE proxy=? AND bar_date LIKE ?",
        (proxy, f"{year}-%")).fetchone()
    return row["c"] if row else 0


def get_regime(con, proxy: str) -> str | None:
    row = con.execute("SELECT state FROM regime_state WHERE proxy=?", (proxy,)).fetchone()
    return row["state"] if row else None


def set_regime(con, proxy: str, state: str, bar_date: str) -> None:
    con.execute("INSERT OR IGNORE INTO regime_history(proxy,state,bar_date,recorded_at)"
                " VALUES (?,?,?,?)", (proxy, state, bar_date, _now()))
    con.execute(
        "INSERT INTO regime_state(proxy,state,bar_date,updated_at) VALUES (?,?,?,?)"
        " ON CONFLICT(proxy) DO UPDATE SET state=excluded.state,"
        " bar_date=excluded.bar_date, updated_at=excluded.updated_at",
        (proxy, state, bar_date, _now()))
    con.commit()


def log_run(con, dry: bool, n_eq: int, n_cr: int, n_ex: int, regime: bool) -> None:
    con.execute("INSERT INTO runs(ran_at,dry_run,n_equity,n_crypto,n_exits,regime_change)"
                " VALUES (?,?,?,?,?,?)", (_now(), int(dry), n_eq, n_cr, n_ex, int(regime)))
    con.commit()


# ---------------- portable snapshot ----------------
# The SQLite file lives in data/, which is gitignored and does not survive an
# ephemeral CI runner. This mirrors it to readable JSON that CAN be committed,
# so the scheduler keeps its memory between runs — and the journal doubles as a
# public, diffable record of every signal the bot ever sent.

SNAPSHOT = Path(__file__).resolve().parent.parent / "state" / "state.json"
TABLES = ("equity_signals", "positions", "crypto_weights", "regime_state",
          "regime_history", "runs")


def export_json(con, path: Path = SNAPSHOT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    dump = {t: [dict(r) for r in con.execute(f"SELECT * FROM {t}")] for t in TABLES}
    path.write_text(json.dumps(dump, indent=2, ensure_ascii=False, sort_keys=True))
    return path


def import_json(con, path: Path = SNAPSHOT) -> bool:
    """Load a snapshot into an empty DB. No-op if the DB already has rows."""
    if not path.exists():
        return False
    if con.execute("SELECT COUNT(*) c FROM positions").fetchone()["c"]:
        return False
    dump = json.loads(path.read_text())
    for table in TABLES:
        for row in dump.get(table, []):
            cols = ",".join(row)
            marks = ",".join("?" * len(row))
            con.execute(f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({marks})",
                        tuple(row.values()))
    con.commit()
    return True


def journal_stats(con) -> dict:
    rows = con.execute("SELECT pnl_pct FROM positions WHERE status='closed'"
                       " AND pnl_pct IS NOT NULL").fetchall()
    pnls = [r["pnl_pct"] for r in rows]
    if not pnls:
        return {"n": 0}
    return {"n": len(pnls), "hit_rate": sum(p > 0 for p in pnls) / len(pnls),
            "avg": sum(pnls) / len(pnls), "best": max(pnls), "worst": min(pnls)}
