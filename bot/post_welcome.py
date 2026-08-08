"""Post the message that gets pinned at the top of each channel.

Discord webhooks can post but CANNOT pin — pinning needs a bot token with
Manage Messages. So: run this once, then right-click each message in Discord
and hit "Fijar mensaje". One click per channel, one time.

    python3 -m bot.post_welcome --dry-run     # ver el texto
    python3 -m bot.post_welcome               # publicar en los 4 canales
    python3 -m bot.post_welcome --channel cripto
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from bot import notifier

CFG_PATH = Path(__file__).resolve().parent / "config.yaml"

DISCLAIMER = ("Esto es un proyecto personal de investigación, no asesoramiento "
              "financiero. Ninguna alerta es una orden de compra: es información "
              "para que cada uno decida.")


def build(docs_url: str | None) -> dict[str, dict]:
    link = f"\n\n[📖 Leer cómo funciona el modelo]({docs_url})" if docs_url else ""
    return {
        "acciones": {
            "title": "📌 Qué vas a ver en este canal",
            "description":
                "Avisos de **compra** cuando una acción del watchlist cae fuerte en el "
                "corto plazo **pero sigue dentro de una tendencia alcista de fondo**. "
                "La idea es aprovechar el rebote, no adivinar un piso." + link,
            "color": notifier.COLOR["buy"],
            "fields": [
                {"name": "Cómo leer la alerta",
                 "value": "**6/6 condiciones** significa que las seis señales de sobreventa "
                          "coincidieron a la vez. Cuantas más coinciden, más alta fue "
                          "históricamente la probabilidad de que el precio suba.\n"
                          "**67,9% de que suba en 10 ruedas** es lo que pasó históricamente "
                          "en situaciones iguales, contra un 55,7% de comprar en un día "
                          "cualquiera. Es una probabilidad, no una promesa.",
                 "inline": False},
                {"name": "Cada cuánto aparecen",
                 "value": "Unas **20 compras al año** sobre 14 acciones, más sus avisos "
                          "de salida: en total 3 mensajes por mes en promedio. "
                          "Si pasan semanas sin alertas es porque no hay setup — "
                          "el silencio también es información.",
                 "inline": False},
                {"name": "La salida",
                 "value": "A las **21 ruedas** de la entrada, por plazo cumplido. "
                          "Probamos stops dinámicos y empeoran el resultado — la "
                          "estrategia necesita aire para funcionar.",
                 "inline": False},
                {"name": "Importante", "value": DISCLAIMER, "inline": False},
            ],
        },
        "cripto": {
            "title": "📌 Qué vas a ver en este canal",
            "description":
                "Ajustes de **peso** en BTC, ETH y BNB. Acá la lógica es al revés que en "
                "acciones: en cripto conviene tener más cuando el activo está fuerte y "
                "menos cuando está castigado." + link,
            "color": notifier.COLOR["up"],
            "fields": [
                {"name": "Qué significa el peso",
                 "value": "**1,35x** quiere decir un 35% más de lo que tendrías normalmente "
                          "en ese activo. **0,70x** es un 30% menos. Nunca llega a cero: "
                          "no es entrar y salir, es pesar más o menos.",
                 "inline": False},
                {"name": "Cada cuánto",
                 "value": "Unas 5 veces al año por moneda. Solo avisa cuando el peso "
                          "objetivo se mueve más de 0,10, para no marear con ajustes chicos.",
                 "inline": False},
                {"name": "Importante", "value": DISCLAIMER, "inline": False},
            ],
        },
        "regimen": {
            "title": "📌 Qué vas a ver en este canal",
            "description":
                "Avisos cuando el mercado en general cambia de estado, midiendo el S&P 500 "
                "contra su promedio de 200 días." + link,
            "color": notifier.COLOR["info"],
            "fields": [
                {"name": "RISK_ON / RISK_OFF",
                 "value": "**RISK_ON**: el mercado está por encima de su promedio de largo "
                          "plazo.\n**RISK_OFF**: lo perdió. Históricamente esto no mejora el "
                          "retorno, pero **reduce mucho las caídas**: el peor drawdown pasa "
                          "de −39,5% a −23,8%.",
                 "inline": False},
                {"name": "Cada cuánto",
                 "value": "Muy pocas veces al año. Cuando aparece un mensaje acá, "
                          "prestale atención: es el aviso más importante del sistema.",
                 "inline": False},
                {"name": "Importante", "value": DISCLAIMER, "inline": False},
            ],
        },
        "scorecard": {
            "title": "📌 Qué vas a ver en este canal",
            "description":
                "Un resumen semanal de **qué hubiera pasado siguiendo cada canal** desde que "
                "el sistema arrancó. No es el backtest: son los resultados reales de las "
                "señales que se publicaron acá." + link,
            "color": 0x2A78D6,
            "fields": [
                {"name": "⚠️ Lo que este canal NO puede hacer",
                 "value": "**No puede demostrar que el sistema funciona, y nunca va a poder.** "
                          "Detectar estadísticamente la ventaja de la señal de acciones exige "
                          "unas 685 operaciones: **34 años** a 20 por año. Si alguien te muestra "
                          "tres meses de resultados como prueba de que un sistema anda, te está "
                          "vendiendo algo.",
                 "inline": False},
                {"name": "Para qué sirve entonces",
                 "value": "Para detectar si el sistema **dejó de comportarse como el backtest**. "
                          "Cada resultado se compara contra lo que el backtest predecía para esa "
                          "cantidad de operaciones, no contra cero. Con 20 operaciones ya se "
                          "nota si algo se rompió.",
                 "inline": False},
                {"name": "Paciencia al principio",
                 "value": "Los primeros meses va a estar casi vacío: la primera operación cierra "
                          "recién 21 ruedas después de la primera señal. Es lo correcto — "
                          "inventar historia sería hacer trampa.",
                 "inline": False},
            ],
        },
        "alertas-modelo": {
            "title": "📌 Este canal casi siempre va a estar en silencio",
            "description":
                "Y eso es exactamente lo que tiene que pasar. **Solo publica cuando un "
                "resultado se sale de la banda que predice el backtest.**" + link,
            "color": 0xEDA100,
            "fields": [
                {"name": "Si no hay mensajes",
                 "value": "El sistema se está comportando dentro de lo esperado. No está roto: "
                          "no tiene nada que decir.",
                 "inline": False},
                {"name": "Si aparece un mensaje",
                 "value": "Significa que el resultado acumulado cayó por debajo del percentil 25 "
                          "(🟡) o del percentil 5 (🔴) de 20.000 carteras simuladas a partir del "
                          "backtest.\n\n**No significa que el sistema se rompió.** Con pocas "
                          "operaciones el azar explica muchísimo. Es una señal para revisar, no "
                          "para actuar.",
                 "inline": False},
            ],
        },
        "reporte": {
            "title": "📌 Qué vas a ver en este canal",
            "description":
                "Un resumen semanal con **cómo viene funcionando el sistema en la vida real**, "
                "no en el backtest: cuántas operaciones se cerraron, cuántas acertaron y el "
                "resultado promedio." + link,
            "color": notifier.COLOR["info"],
            "fields": [
                {"name": "Por qué importa",
                 "value": "El backtest dice lo que hubiera pasado. Este canal dice lo que "
                          "está pasando. Si los números reales se despegan mucho de los "
                          "históricos, hay que revisar el modelo.",
                 "inline": False},
                {"name": "Importante", "value": DISCLAIMER, "inline": False},
            ],
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--channel", choices=list(notifier.CHANNELS))
    args = ap.parse_args()

    cfg = yaml.safe_load(CFG_PATH.read_text())
    docs = cfg.get("docs_url") or None
    if not docs:
        print("[WELCOME] ojo: docs_url está vacío en bot/config.yaml, "
              "los mensajes van a salir sin link")

    msgs = build(docs)
    targets = [args.channel] if args.channel else list(msgs)
    for ch in targets:
        notifier.send(ch, msgs[ch], args.dry_run)

    if not args.dry_run:
        print("\n[WELCOME] publicado. Ahora en Discord: click derecho en cada mensaje "
              "→ 'Fijar mensaje'. Los webhooks no pueden fijar solos.")


if __name__ == "__main__":
    main()
