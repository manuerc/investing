"""Daily entry point.

    python3 -m bot.run_daily --dry-run     # print everything, touch nothing
    python3 -m bot.run_daily               # persist state and post to Discord
    python3 -m bot.run_daily --report      # also post the weekly journal report

Run it after the US close. Signals are computed on the last CLOSED bar and the
equity entry is the following session's open, so there is nothing to rush.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import yaml

from bot import engine, notifier, state

warnings.filterwarnings("ignore")
CFG_PATH = Path(__file__).resolve().parent / "config.yaml"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="no escribe estado ni manda nada a Discord")
    ap.add_argument("--report", action="store_true", help="agrega el reporte semanal")
    ap.add_argument("--as-of", help="replay de una rueda pasada, formato YYYY-MM-DD")
    args = ap.parse_args()

    cfg = yaml.safe_load(CFG_PATH.read_text())
    con = state.connect()
    if state.import_json(con):
        print(f"[BOT] estado restaurado desde {state.SNAPSHOT}")
    dry = args.dry_run
    docs = cfg.get("docs_url") or None

    print(f"[BOT] corrida {'en seco' if dry else 'real'}"
          + (f' · replay al {args.as_of}' if args.as_of else ''))

    # ---- acciones ----
    eq_signals, eq_status = engine.scan_equities(cfg, args.as_of)
    latest = {s["asset"]: {"bar_date": s["bar_date"], "close": s["close"],
                           "bars": s["bars"]} for s in eq_status}

    exits = [] if dry else state.advance_positions(con, latest)
    for pos in exits:
        notifier.send("acciones", notifier.with_docs(notifier.exit_embed(pos), docs), dry)

    sent_eq = 0
    for sig in eq_signals:
        bars = latest.get(sig.asset, {}).get("bars", [])
        if not dry and state.in_cooldown(con, sig.asset,
                                         cfg["equities"]["cooldown_bars"], bars):
            print(f"[BOT] {sig.asset}: en cooldown o con posición abierta, no se avisa")
            continue
        if dry or state.record_equity_signal(con, sig, cfg["equities"]["hold_bars"]):
            notifier.send("acciones", notifier.with_docs(notifier.equity_embed(sig), docs), dry)
            sent_eq += 1

    # ---- cripto ----
    prev = {} if dry else state.get_crypto_weights(con)
    cr_signals, cr_status = engine.scan_crypto(cfg, prev, args.as_of)
    for sig in cr_signals:
        notifier.send("cripto", notifier.with_docs(notifier.crypto_embed(sig), docs), dry)
        if not dry:
            state.set_crypto_weight(con, sig.asset, sig.target_weight, sig.bar_date)

    # ---- régimen ----
    prev_state = None if dry else state.get_regime(con, cfg["regime"]["proxy"])
    rg = engine.scan_regime(cfg, prev_state, args.as_of)
    if rg:
        year = int(rg.bar_date[:4])
        if not dry:
            state.set_regime(con, rg.proxy, rg.state, rg.bar_date)
        n_changes = state.regime_changes_this_year(con, rg.proxy, year)
        notifier.send("regimen",
                      notifier.with_docs(notifier.regime_embed(rg, n_changes), docs), dry)

    # ---- reporte ----
    if args.report:
        stats = state.journal_stats(con)
        notifier.send("reporte",
                      notifier.with_docs(notifier.report_embed(stats, eq_status, cr_status), docs),
                      dry)

    if not dry:
        state.log_run(con, dry, sent_eq, len(cr_signals), len(exits), bool(rg))
        state.export_json(con)

    near = sorted((s for s in eq_status if s["conditions"] >= 3),
                  key=lambda s: -s["conditions"])
    print(f"\n[BOT] {sent_eq} compras · {len(exits)} salidas · {len(cr_signals)} ajustes de cripto"
          f" · régimen {'cambió' if rg else 'sin cambios'}")
    if near:
        print("[BOT] cerca del disparo: " +
              ", ".join(f"{s['asset']} {s['conditions']}/6" for s in near))
    con.close()


if __name__ == "__main__":
    main()
