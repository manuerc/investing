"""One-shot Discord server setup. Creates categories, channels, permissions,
the @señales role, and posts + pins every explanatory message.

The bot token never leaves this machine: it is read from .env or the
environment and used only against Discord's own API.

    python3 -m bot.setup_discord --dry-run     # mostrar el plan, no tocar nada
    python3 -m bot.setup_discord               # crear todo
    python3 -m bot.setup_discord --webhooks    # además crear los 4 webhooks

Idempotent: re-running skips whatever already exists, so it is safe to fix the
script and run it again.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

from bot import post_welcome

API = "https://discord.com/api/v10"
ROOT = Path(__file__).resolve().parent.parent

# --- permission bits -------------------------------------------------------
VIEW_CHANNEL = 1 << 10
SEND_MESSAGES = 1 << 11
ADD_REACTIONS = 1 << 6
READ_HISTORY = 1 << 16
CREATE_THREADS = 1 << 35
SEND_IN_THREADS = 1 << 38

# Members can read, react and discuss in threads — but not post in the channel
# itself, so a signal never gets buried under chatter.
READONLY_ALLOW = VIEW_CHANNEL | READ_HISTORY | ADD_REACTIONS | CREATE_THREADS | SEND_IN_THREADS
READONLY_DENY = SEND_MESSAGES

CHANNEL_TEXT, CHANNEL_CATEGORY = 0, 4

STRUCTURE = [
    ("📌 EMPEZÁ ACÁ", False, [
        ("reglas-y-aviso", "Qué es esto, las reglas y el aviso importante. Leelo antes de nada."),
        ("cómo-leer-las-señales", "Guía para interpretar cada tipo de alerta."),
    ]),
    ("🤖 SEÑALES", True, [
        ("regimen", "Cambios de estado del mercado general. Pocos avisos, los más importantes."),
        ("acciones", "Señales de compra y avisos de salida. ~3 mensajes por mes."),
        ("cripto", "Ajustes de peso en BTC, ETH y BNB."),
        ("reporte", "Resumen semanal de cómo viene funcionando el sistema en la vida real."),
    ]),
    ("💬 COMUNIDAD", False, [
        ("charla", "Conversación general."),
        ("preguntas", "Dudas sobre las señales o el modelo."),
        ("resultados", "Compartí cómo te fue si querés."),
    ]),
]

WEBHOOK_CHANNELS = {"acciones": "DISCORD_WEBHOOK_ACCIONES", "cripto": "DISCORD_WEBHOOK_CRIPTO",
                    "regimen": "DISCORD_WEBHOOK_REGIMEN", "reporte": "DISCORD_WEBHOOK_REPORTE"}

ROLE_NAME = "señales"


# --------------------------------------------------------------------------- api

class Discord:
    def __init__(self, token: str, dry: bool = False):
        self.token = token
        self.dry = dry

    def _call(self, method: str, path: str, body: dict | None = None, retries: int = 5):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            API + path, data=data, method=method,
            headers={"Authorization": f"Bot {self.token}",
                     "Content-Type": "application/json",
                     "User-Agent": "investing-setup (https://github.com, 1.0)"})
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(req, timeout=20) as r:
                    raw = r.read()
                    time.sleep(0.35)                      # stay well under rate limits
                    return json.loads(raw) if raw else {}
            except urllib.error.HTTPError as e:
                payload = e.read()
                if e.code == 429:
                    wait = json.loads(payload or b"{}").get("retry_after", 2)
                    print(f"    rate limit, esperando {wait}s")
                    time.sleep(float(wait) + 0.5)
                    continue
                raise RuntimeError(f"{method} {path} -> HTTP {e.code}: {payload[:300].decode()}")
        raise RuntimeError(f"{method} {path}: agotados los reintentos")

    def get(self, path):
        return self._call("GET", path)

    def post(self, path, body):
        if self.dry:
            print(f"    [dry-run] POST {path}")
            return {"id": f"DRY_{abs(hash(path + json.dumps(body, sort_keys=True))) % 10**18}"}
        return self._call("POST", path, body)

    def put(self, path):
        if self.dry:
            print(f"    [dry-run] PUT {path}")
            return {}
        return self._call("PUT", path)


# --------------------------------------------------------------------------- helpers

def load_env() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def resolve_guild(dc: Discord, given: str | None) -> tuple[str, str]:
    guilds = dc.get("/users/@me/guilds")
    if given:
        for g in guilds:
            if g["id"] == given:
                return g["id"], g["name"]
        raise SystemExit(f"El bot no está en el servidor {given}. Servidores donde sí está: "
                         + ", ".join(f'{g["name"]} ({g["id"]})' for g in guilds))
    if not guilds:
        raise SystemExit("El bot no está en ningún servidor todavía. Invitalo primero "
                         "con la URL de OAuth2 y volvé a correr esto.")
    if len(guilds) > 1:
        raise SystemExit("El bot está en varios servidores, indicá cuál con --guild-id:\n  "
                         + "\n  ".join(f'{g["name"]}: {g["id"]}' for g in guilds))
    return guilds[0]["id"], guilds[0]["name"]


def ensure_role(dc: Discord, gid: str, existing: list[dict]) -> str:
    for r in existing:
        if r["name"] == ROLE_NAME:
            print(f"  rol @{ROLE_NAME} ya existe")
            return r["id"]
    print(f"  creando rol @{ROLE_NAME}")
    r = dc.post(f"/guilds/{gid}/roles",
                {"name": ROLE_NAME, "mentionable": True, "permissions": "0",
                 "color": 0x5865F2,
                 "reason": "rol opcional para recibir ping en las señales"})
    return r["id"]


def ensure_channel(dc: Discord, gid: str, existing: dict[str, dict], name: str,
                   ctype: int, parent: str | None = None, topic: str | None = None,
                   readonly: bool = False) -> tuple[str, bool]:
    key = name.lower()
    if key in existing:
        print(f"  ya existe: {name}")
        return existing[key]["id"], False
    body: dict = {"name": name, "type": ctype}
    if parent:
        body["parent_id"] = parent
    if topic:
        body["topic"] = topic
    if readonly:
        body["permission_overwrites"] = [{
            "id": gid, "type": 0,                     # @everyone
            "allow": str(READONLY_ALLOW), "deny": str(READONLY_DENY),
        }]
    print(f"  creando: {name}" + ("  (solo lectura)" if readonly else ""))
    return dc.post(f"/guilds/{gid}/channels", body)["id"], True


def post_and_pin(dc: Discord, channel_id: str, embed: dict) -> None:
    msg = dc.post(f"/channels/{channel_id}/messages", {"embeds": [embed]})
    dc.put(f"/channels/{channel_id}/pins/{msg['id']}")


# --------------------------------------------------------------------------- content

def intro_embeds(docs_url: str | None) -> dict[str, list[dict]]:
    link = f"\n\n[📖 Leer la documentación completa del modelo]({docs_url})" if docs_url else ""
    return {
        "reglas-y-aviso": [{
            "title": "Bienvenido — leé esto antes de nada",
            "description":
                "Este servidor publica señales de compra y venta para un grupo fijo de "
                "acciones y criptomonedas. Cada regla fue medida sobre 16 años de datos "
                "reales, separando el período usado para descubrirla del usado para "
                "validarla." + link,
            "color": 0x5865F2,
            "fields": [
                {"name": "⚠️ Lo más importante",
                 "value": "**Esto no es asesoramiento financiero.** Es un proyecto personal "
                          "de investigación que comparto abiertamente. Ninguna alerta es una "
                          "orden de compra: es información para que cada uno decida con su "
                          "propia plata y su propio criterio. Si perdés plata siguiendo una "
                          "señal, la responsabilidad es tuya.",
                 "inline": False},
                {"name": "📉 Qué esperar, con honestidad",
                 "value": "La señal de acciones acierta un **64% de las veces** contra un "
                          "56% de comprar en un día cualquiera. Es una mejora real pero "
                          "moderada: **cuatro de cada diez veces se pierde**. La peor "
                          "operación del backtest fue −34%. Nadie acá va a hacerse rico "
                          "rápido, y quien lo prometa te está mintiendo.",
                 "inline": False},
                {"name": "📋 Las reglas del servidor",
                 "value": "**1.** Los canales de señales son de solo lectura. Para comentar "
                          "una alerta abrí un hilo sobre ella o escribí en <#charla>.\n"
                          "**2.** No se piden ni se dan recomendaciones personalizadas. "
                          "Nadie acá sabe tu situación financiera.\n"
                          "**3.** Prohibido promocionar otros grupos, cursos, brokers o "
                          "\"señales premium\". Expulsión directa.\n"
                          "**4.** Se puede dudar del modelo y discutirlo. La documentación "
                          "está abierta justamente para eso.",
                 "inline": False},
                {"name": "🔔 Cómo configurar las notificaciones",
                 "value": "Recomendado: **todos los mensajes** en <#regimen> y <#acciones>, "
                          "**silenciado** en <#reporte>. Click derecho en cada canal → "
                          "*Notificaciones*.",
                 "inline": False},
            ],
        }],
        "cómo-leer-las-señales": [
            {
                "title": "🟢 Cómo leer una señal de acciones",
                "description":
                    "Ejemplo de alerta:\n"
                    "```\n🟢 COMPRA · YPFD · 6/6 condiciones\n"
                    "67,9% de que suba en 10 ruedas (base 55,7%)\n```",
                "color": 0x1BAF7A,
                "fields": [
                    {"name": "«6/6 condiciones»",
                     "value": "El sistema chequea seis señales de que la acción está "
                              "castigada en el corto plazo, y solo avisa si **además** la "
                              "acción sigue en tendencia alcista de fondo. Cuantas más "
                              "coinciden, más alta fue históricamente la probabilidad de "
                              "rebote. Con 5 se avisa; con 6 la señal es más fuerte.",
                     "inline": False},
                    {"name": "«67,9% de que suba»",
                     "value": "Es lo que pasó **históricamente** en situaciones iguales, "
                              "no una predicción. El número entre paréntesis es la "
                              "comparación honesta: comprando en un día cualquiera la "
                              "probabilidad hubiera sido 55,7%. La diferencia entre esos "
                              "dos números es todo el valor del sistema.",
                     "inline": False},
                    {"name": "«Entrada: open de la próxima rueda»",
                     "value": "La señal se calcula con la vela ya cerrada, así que se opera "
                              "a la apertura del día siguiente. No hay que correr.",
                     "inline": False},
                    {"name": "La salida",
                     "value": "A las **21 ruedas**, por plazo cumplido. Va a llegar un aviso "
                              "🔵 con el resultado. Probamos stops dinámicos y todos "
                              "empeoran el resultado: la estrategia necesita aire.",
                     "inline": False},
                ],
            },
            {
                "title": "🔼 Cómo leer una señal de cripto",
                "description":
                    "En cripto la lógica es **al revés** que en acciones: conviene tener más "
                    "cuando el activo está fuerte y menos cuando está castigado. Por eso no "
                    "son señales de entrar y salir, sino de **peso**.",
                "color": 0xEDA100,
                "fields": [
                    {"name": "«peso objetivo 1,35x»",
                     "value": "Significa un 35% más de lo que tendrías normalmente en ese "
                              "activo. **0,70x** es un 30% menos. Nunca baja a cero: "
                              "siempre hay posición, lo que cambia es cuánta.",
                     "inline": False},
                    {"name": "Sin apuro",
                     "value": "A diferencia de las acciones, acá no importa el día exacto. "
                              "Podés rebalancear cuando te quede cómodo.",
                     "inline": False},
                ],
            },
            {
                "title": "🟩 Cómo leer el régimen",
                "description":
                    "Mide el S&P 500 contra su promedio de 200 días. Es el aviso más "
                    "importante del sistema y el que menos aparece.",
                "color": 0x84837C,
                "fields": [
                    {"name": "RISK_ON / RISK_OFF",
                     "value": "**RISK_ON**: el mercado está por encima de su promedio de "
                              "largo plazo.\n**RISK_OFF**: lo perdió. Históricamente esto "
                              "no mejora el retorno, pero **reduce mucho las caídas**: el "
                              "peor drawdown pasa de −39,5% a −23,8%.",
                     "inline": False},
                    {"name": "Ojo con los años laterales",
                     "value": "Cuando el mercado oscila alrededor de su promedio, estos "
                              "avisos se repiten y muchos se dan vuelta a los pocos días. "
                              "En 2022 y 2023 hubo 19 cambios cada uno. Por eso cada alerta "
                              "dice de qué número de cambio del año se trata: si vas por el "
                              "sexto, no reacciones a cada uno por separado.",
                     "inline": False},
                ],
            },
            {
                "title": "❓ Las tres confusiones más comunes",
                "color": 0x5865F2,
                "fields": [
                    {"name": "«Hace una semana que no llega nada, ¿está roto?»",
                     "value": "No. El sistema escanea todos los días pero solo habla cuando "
                              "hay algo. Son unas 20 compras al año sobre 14 acciones. "
                              "**El silencio es el estado normal y también es información.**",
                     "inline": False},
                    {"name": "«¿Por qué no avisa cuándo vender?»",
                     "value": "Porque lo probamos y **no funciona**. Los indicadores de "
                              "sobrecompra no anticipan caídas en acciones de calidad. "
                              "Antes que inventar una señal que no tiene respaldo, la salida "
                              "es mecánica: 21 ruedas y afuera. Está documentado.",
                     "inline": False},
                    {"name": "«Salió una señal y la acción siguió bajando»",
                     "value": "Va a pasar cuatro de cada diez veces. Una probabilidad del "
                              "64% no es una garantía: significa exactamente que un 36% de "
                              "las veces sale mal. El sistema se juzga sobre decenas de "
                              "operaciones, nunca sobre una.",
                     "inline": False},
                ],
            },
        ],
    }


# --------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--guild-id")
    ap.add_argument("--webhooks", action="store_true",
                    help="crear también los 4 webhooks del job diario")
    args = ap.parse_args()

    load_env()
    token = os.environ.get("DISCORD_BOT_TOKEN", "").strip()
    if not token:
        raise SystemExit("Falta DISCORD_BOT_TOKEN. Pegalo en investing/.env")

    cfg = yaml.safe_load((ROOT / "bot" / "config.yaml").read_text())
    docs_url = cfg.get("docs_url") or None
    if not docs_url:
        print("[SETUP] aviso: docs_url está vacío en bot/config.yaml, "
              "los mensajes van a salir sin link a la documentación\n")

    dc = Discord(token, args.dry_run)
    gid, gname = resolve_guild(dc, args.guild_id)
    print(f"[SETUP] servidor: {gname} ({gid}){'  · DRY RUN' if args.dry_run else ''}\n")

    existing = {c["name"].lower(): c for c in dc.get(f"/guilds/{gid}/channels")}
    roles = dc.get(f"/guilds/{gid}/roles")

    print("Rol")
    ensure_role(dc, gid, roles)

    print("\nCanales")
    created: dict[str, str] = {}
    fresh: set[str] = set()
    for cat_name, readonly, channels in STRUCTURE:
        cat_id, _ = ensure_channel(dc, gid, existing, cat_name, CHANNEL_CATEGORY)
        for name, topic in channels:
            cid, is_new = ensure_channel(dc, gid, existing, name, CHANNEL_TEXT,
                                         parent=cat_id, topic=topic, readonly=readonly)
            created[name] = cid
            if is_new:
                fresh.add(name)

    print("\nMensajes fijados")
    intros = intro_embeds(docs_url)
    welcome = post_welcome.build(docs_url)
    for name, embeds in intros.items():
        if name not in fresh:
            print(f"  {name}: ya existía, no se publica de nuevo")
            continue
        for e in embeds:
            post_and_pin(dc, created[name], e)
        print(f"  {name}: {len(embeds)} mensaje(s) publicado(s) y fijado(s)")

    for name, embed in welcome.items():
        if name not in fresh:
            print(f"  {name}: ya existía, no se publica de nuevo")
            continue
        post_and_pin(dc, created[name], notifier_ready(embed, docs_url))
        print(f"  {name}: bienvenida publicada y fijada")

    if args.webhooks:
        print("\nWebhooks — pegá estas URLs en .env y en los GitHub Secrets")
        for name, env_var in WEBHOOK_CHANNELS.items():
            wh = dc.post(f"/channels/{created[name]}/webhooks", {"name": "Señales"})
            url = wh.get("url") or (f"https://discord.com/api/webhooks/{wh['id']}/{wh['token']}"
                                    if wh.get("token") else "[dry-run]")
            print(f"  {env_var}={url}")

    print("\n[SETUP] listo.")
    if args.dry_run:
        print("[SETUP] fue una corrida en seco: no se creó nada.")


def notifier_ready(embed: dict, docs_url: str | None) -> dict:
    from bot import notifier
    return notifier.with_docs(dict(embed), docs_url)


if __name__ == "__main__":
    main()
