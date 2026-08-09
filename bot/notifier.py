"""Discord webhook delivery, with a dry-run renderer for the console.

Webhooks are read from the environment so nothing secret lands in the repo:
    DISCORD_WEBHOOK_ACCIONES  DISCORD_WEBHOOK_CRIPTO
    DISCORD_WEBHOOK_REGIMEN   DISCORD_WEBHOOK_REPORTE
A channel with no webhook set silently falls back to console output.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

CHANNELS = {
    "acciones": "DISCORD_WEBHOOK_ACCIONES",
    "cripto": "DISCORD_WEBHOOK_CRIPTO",
    "regimen": "DISCORD_WEBHOOK_REGIMEN",
    "reporte": "DISCORD_WEBHOOK_REPORTE",
    "scorecard": "DISCORD_WEBHOOK_SCORECARD",
    "alertas-modelo": "DISCORD_WEBHOOK_DRIFT",
}

def load_env(path: Path | None = None) -> None:
    """Read .env into the environment for local runs.

    In GitHub Actions the webhooks arrive as real env vars, so this is a no-op
    there. Existing variables always win over the file.
    """
    env = path or Path(__file__).resolve().parent.parent / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if v.strip():
            os.environ.setdefault(k.strip(), v.strip())


COLOR = {"buy": 0x1BAF7A, "exit": 0x2A78D6, "up": 0x1BAF7A,
         "down": 0xEDA100, "risk_off": 0xE34948, "risk_on": 0x1BAF7A,
         "info": 0x84837C}


def _pct(v: float, digits: int = 1) -> str:
    return f"{v * 100:.{digits}f}".replace(".", ",") + "%"


def _num(v: float, digits: int = 2) -> str:
    return f"{v:,.{digits}f}".replace(",", "@").replace(".", ",").replace("@", ".")


def with_docs(embed: dict, docs_url: str | None) -> dict:
    """Append a 'cómo funciona' link so non-technical readers can self-serve."""
    if docs_url:
        embed["fields"] = embed.get("fields", []) + [
            {"name": "\u200b", "value": f"[📖 Cómo funciona esta señal]({docs_url})",
             "inline": False}]
    return embed


# ---------------- embed builders ----------------

def equity_embed(sig) -> dict:
    marcadas = [f"✅ {k}" for k, v in sig.detail.items() if v]
    faltan = [f"◽ {k}" for k, v in sig.detail.items() if not v]
    conf = ""
    if sig.p_up is not None:
        conf = (f"**{_pct(sig.p_up)}** de que suba en 10 ruedas "
                f"(base {_pct(sig.base_rate)}) · fuera de muestra {_pct(sig.p_up_oos)}")
    return {
        "title": f"🟢 COMPRA · {sig.asset} · {sig.conditions}/6 condiciones",
        "description": conf,
        "color": COLOR["buy"],
        "fields": [
            {"name": "Cierre", "value": f"${_num(sig.close)}", "inline": True},
            {"name": "Entrada", "value": "open de la próxima rueda", "inline": True},
            {"name": "Salida", "value": "cuando cierre sobre su media de 20 días", "inline": True},
            {"name": "Volatilidad", "value": f"ATR {_pct(sig.atr_pct)}", "inline": True},
            {"name": "Se cumplen", "value": "\n".join(marcadas) or "—", "inline": False},
            *([{"name": "No se cumplen", "value": "\n".join(faltan), "inline": False}]
              if faltan else []),
        ],
        "footer": {"text": f"vela cerrada {sig.bar_date} · sin stop dinámico: "
                           f"empeora el resultado"},
    }


def crypto_embed(sig) -> dict:
    arrow = {"aumentar": "🔼", "reducir": "🔽", "inicial": "🆕"}[sig.direction]
    prev = f"{_num(sig.prev_weight)}x" if sig.prev_weight is not None else "sin posición"
    return {
        "title": f"{arrow} {sig.asset} · peso objetivo {_num(sig.target_weight)}x",
        "description": f"Score de momentum **{_num(sig.score, 2)}** "
                       f"(percentil de precio vs EMA200 a 252 ruedas)",
        "color": COLOR["up"] if sig.direction != "reducir" else COLOR["down"],
        "fields": [
            {"name": "Peso anterior", "value": prev, "inline": True},
            {"name": "Peso nuevo", "value": f"{_num(sig.target_weight)}x", "inline": True},
            {"name": "Precio", "value": f"${_num(sig.close)}", "inline": True},
        ],
        "footer": {"text": f"vela cerrada {sig.bar_date} · banda de rebalanceo 0,10"},
    }


def regime_embed(sig, changes_this_year: int = 0) -> dict:
    on = sig.state == "RISK_ON"
    return {
        "title": f"{'🟩' if on else '🟥'} Régimen: {sig.state}",
        "description": (f"**{sig.proxy}** {'cruzó por encima de' if on else 'perdió'} "
                        f"su EMA200."),
        "color": COLOR["risk_on"] if on else COLOR["risk_off"],
        "fields": [
            {"name": "Cierre", "value": f"${_num(sig.close)}", "inline": True},
            {"name": "EMA200", "value": f"${_num(sig.ema200)}", "inline": True},
            {"name": "Estado previo", "value": sig.prev_state or "—", "inline": True},
            *([{"name": "⚠️ Mercado lateral",
                "value": f"Este es el **cambio n.º {changes_this_year} del año**. Cuando el "
                         "S&P oscila alrededor de su promedio, estos avisos se repiten y "
                         "muchos se dan vuelta a los pocos días. En 2022 y 2023 hubo 19 "
                         "cada uno. Conviene no sobrerreaccionar a cada uno por separado.",
                "inline": False}] if changes_this_year >= 4 else []),
        ],
        "footer": {"text": f"vela cerrada {sig.bar_date} · cambio n.º {changes_this_year} "
                           f"del año · reduce drawdown, no genera alfa"},
    }


def exit_embed(pos: dict) -> dict:
    """Exit alert.

    `pendiente` means the condition triggered on today's close, so the sale
    happens at tomorrow's open — the number shown is an estimate, not a fill.
    """
    pnl = pos["pnl_pct"]
    pend = bool(pos.get("pendiente"))
    motivo = ("volvió sobre su media de 20 días"
              if pos.get("motivo") == "vuelta a la media"
              else f"tope de plazo, {pos['bars_held']} ruedas")
    if pend:
        desc = ("**Vendé en la apertura de mañana.** Resultado estimado con el cierre "
                "de hoy: " + (_pct(pnl) if pnl is not None else "—"))
        foot = ("la condición se evalúa sobre la vela cerrada, así que la venta va "
                "al open siguiente")
    else:
        desc = "Resultado " + ("**" + _pct(pnl) + "**" if pnl is not None else "—")
        foot = "vendido en la apertura, como indica la regla"
    return {
        "title": f"🔵 SALIDA · {pos['asset']} · {motivo}",
        "description": desc,
        "color": COLOR["exit"],
        "fields": [
            {"name": "Entrada", "value": f"${_num(pos['entry_price'])}", "inline": True},
            {"name": "Cierre de hoy" if pend else "Salida",
             "value": f"${_num(pos['exit_price'])}", "inline": True},
            {"name": "Señal", "value": pos["signal_date"], "inline": True},
            {"name": "Duración", "value": f"{pos['bars_held']} ruedas", "inline": True},
        ],
        "footer": {"text": foot},
    }


def report_embed(stats: dict, eq_status: list, cr_status: list) -> dict:
    if stats.get("n"):
        line = (f"**{stats['n']}** operaciones cerradas · acierto "
                f"**{_pct(stats['hit_rate'])}** · retorno medio **{_pct(stats['avg'])}**")
    else:
        line = "Todavía no hay operaciones cerradas en el journal."
    cerca = sorted([s for s in eq_status if s["conditions"] >= 3],
                   key=lambda s: -s["conditions"])[:6]
    return {
        "title": "📊 Reporte semanal",
        "description": line,
        "color": COLOR["info"],
        "fields": [
            {"name": "Acciones cerca del disparo (3+ condiciones)",
             "value": "\n".join(f"`{s['asset']:<6}` {s['conditions']}/6" for s in cerca) or "ninguna",
             "inline": False},
            {"name": "Pesos de cripto",
             "value": "\n".join(f"`{s['asset']:<4}` {_num(s['target_weight'])}x "
                                f"(score {_num(s['score'])})" for s in cr_status) or "—",
             "inline": False},
        ],
    }


# ---------------- delivery ----------------

def _render(channel: str, embed: dict) -> None:
    print(f"\n┌─ #{channel} " + "─" * (70 - len(channel)))
    print(f"│ {embed['title']}")
    if embed.get("description"):
        print(f"│ {embed['description']}")
    for f in embed.get("fields", []):
        val = f["value"].replace("\n", "\n│     ")
        print(f"│   {f['name']}: {val}")
    if embed.get("footer"):
        print(f"│ — {embed['footer']['text']}")
    print("└" + "─" * 78)


MUTED = False   # set by --no-send: persist state but never hit Discord


def send(channel: str, embed: dict, dry_run: bool = False) -> bool:
    url = os.environ.get(CHANNELS.get(channel, ""), "")
    if MUTED:
        return True
    if dry_run or not url:
        _render(channel, embed)
        if not dry_run and not url:
            print(f"[NOTIFY] {CHANNELS.get(channel)} no está seteado — solo consola")
        return True
    payload = json.dumps({"embeds": [embed]}).encode()
    # Cloudflare rejects urllib's default agent with error 1010; Discord expects
    # the DiscordBot form.
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json",
                 "User-Agent": "DiscordBot (https://github.com/manuerc/investing, 1.0)"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return 200 <= r.status < 300
    except urllib.error.HTTPError as e:
        print(f"[NOTIFY] error {e.code} en #{channel}: {e.read()[:200]!r}")
    except Exception as e:
        print(f"[NOTIFY] fallo en #{channel}: {e}")
    return False
