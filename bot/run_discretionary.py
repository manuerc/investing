"""Entry point for the experimental 'discrecional' crypto alert channel.

    python3 -m bot.run_discretionary --dry-run     # print everything, touch nothing
    python3 -m bot.run_discretionary               # persist state and post to Discord

Runs hourly, independently of bot/run_daily.py — it reads its own config
(bot/config_discrecional.yaml) and its own tiny state file, and never touches
the daily swing model's state or channels.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import yaml

from bot import notifier
from bot.discretionary import scan_discretionary

warnings.filterwarnings("ignore")
CFG_PATH = Path(__file__).resolve().parent / "config_discrecional.yaml"
STATE_PATH = Path(__file__).resolve().parent.parent / "state" / "discrecional_state.json"


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="no escribe estado ni manda nada a Discord")
    args = ap.parse_args()

    notifier.load_env()
    cfg = yaml.safe_load(CFG_PATH.read_text())
    state = {} if args.dry_run else load_state()

    results = scan_discretionary(cfg)
    sent = 0
    for r in results:
        prev_bar = state.get(r["asset"], {}).get("bar_time")
        if prev_bar == r["bar_time"]:
            continue  # already processed this closed bar
        if len(r["triggered"]) >= cfg["min_conditions_to_alert"]:
            if notifier.send("discrecional", notifier.discretionary_embed(r), args.dry_run):
                sent += 1
        if not args.dry_run:
            state[r["asset"]] = {"bar_time": r["bar_time"]}

    if not args.dry_run:
        save_state(state)

    print(f"[DISCRECIONAL] {sent} alertas enviadas de {len(results)} activos escaneados")


if __name__ == "__main__":
    main()
