"""Discord digest for the performance group.

Two channels, deliberately:
  #scorecard       weekly summary of the three signals + link to the dashboard
  #alertas-modelo  speaks ONLY when a result falls outside the backtest band

The second one is the point of the whole tracker. Everything else is context.
"""

from __future__ import annotations

from bot import notifier

COLOR = {"ok": 0x1BAF7A, "vigilar": 0xEDA100, "alerta": 0xE34948, "info": 0x2A78D6}


def _p(v, d=1):
    return "—" if v is None else f"{v * 100:.{d}f}".replace(".", ",") + "%"


def scorecard_embed(perf: dict, dash_url: str | None) -> dict:
    a, c, r = perf["acciones"], perf["cripto"], perf["regimen"]
    fields = []

    if a.get("n_cerradas"):
        drift = a.get("drift", {})
        mark = {"ok": "✅", "vigilar": "🟡", "alerta": "🔴"}.get(drift.get("estado"), "")
        fields.append({
            "name": f"📈 Acciones · {a['n_cerradas']} cerradas"
                    + (f" + {a['n_abiertas']} abiertas" if a.get("n_abiertas") else ""),
            "value": (f"Acierto **{_p(a['acierto'])}** (backtest {_p(a['backtest']['acierto'])})\n"
                      f"Retorno medio **{_p(a['ret_medio'], 2)}** "
                      f"(backtest {_p(a['backtest']['ret_medio'], 2)})\n"
                      f"Cartera **{_p(a['cartera_pct'], 2)}** · banda esperada "
                      f"{_p(a['bandas']['p5'])} a {_p(a['bandas']['p95'])}\n"
                      f"{mark} {drift.get('detalle', '')}"),
            "inline": False})
    else:
        n_ab = a.get("n_abiertas", 0)
        fields.append({"name": "📈 Acciones",
                       "value": (f"{n_ab} operación(es) abierta(s), ninguna cerrada todavía."
                                 if n_ab else "Sin operaciones todavía."),
                       "inline": False})

    if c.get("serie"):
        dif = c["tilt_pct"] - c["hold_pct"]
        fields.append({
            "name": f"🔼 Cripto · {c['n_ajustes']} ajustes",
            "value": (f"Siguiendo el canal **{_p(c['tilt_pct'], 2)}** vs "
                      f"**{_p(c['hold_pct'], 2)}** manteniendo → "
                      f"**{_p(dif, 2)}** de diferencia\n"
                      f"Peor caída {_p(c['tilt_maxdd'])} vs {_p(c['hold_maxdd'])}"),
            "inline": False})

    if r.get("serie"):
        fields.append({
            "name": f"🟩 Régimen · {r['n_cambios']} cambios · hoy {r['estado_actual']}",
            "value": (f"Siguiendo el canal **{_p(r['filtrado_pct'], 2)}** vs "
                      f"**{_p(r['spy_pct'], 2)}** el SPY\n"
                      f"Peor caída **{_p(r['filtrado_maxdd'])}** vs {_p(r['spy_maxdd'])} — "
                      f"acá lo que importa es esta línea, no la de arriba\n"
                      f"Invertido el {_p(r['expuesto'], 0)} del tiempo"),
            "inline": False})

    if dash_url:
        fields.append({"name": "​",
                       "value": f"[📊 Ver el detalle con gráficos]({dash_url})", "inline": False})

    return {
        "title": "📊 Scorecard de rendimiento",
        "description": ("Qué hubiera pasado siguiendo cada canal. Los números se comparan "
                        "contra la **distribución del backtest**, no contra cero."),
        "color": COLOR["info"],
        "fields": fields,
        "footer": {"text": "Confirmar el edge exige ~685 operaciones (34 años). Esto no valida "
                           "la estrategia: detecta si dejó de comportarse como el backtest."},
    }


def drift_embed(perf: dict, dash_url: str | None) -> dict | None:
    """Only returns something when the live result leaves the expected band."""
    a = perf["acciones"]
    drift = a.get("drift")
    if not drift or drift["estado"] == "ok":
        return None
    pctl = drift["percentil"]
    grave = drift["estado"] == "alerta"
    return {
        "title": ("🔴 El modelo se salió de la banda esperada" if grave
                  else "🟡 El modelo está en la banda baja"),
        "description": (
            f"Con **{a['n_cerradas']}** operaciones cerradas, la cartera va "
            f"**{_p(a['cartera_pct'], 2)}**. El backtest predecía entre "
            f"{_p(a['bandas']['p5'])} y {_p(a['bandas']['p95'])}, con mediana "
            f"{_p(a['bandas']['p50'])}.\n\n"
            f"El resultado cae en el **percentil {pctl * 100:.0f}** de 20.000 carteras "
            f"simuladas a partir del backtest."),
        "color": COLOR["alerta" if grave else "vigilar"],
        "fields": [
            {"name": "Qué significa",
             "value": ("Estar en el percentil bajo **no prueba que el sistema se rompió** — "
                       "con pocas operaciones el azar explica muchísimo. Es una señal para "
                       "mirar, no para actuar.\n"
                       "Lo que sí conviene revisar: si los precios de entrada se parecen a "
                       "los que publicó el bot, si las señales salen a la frecuencia "
                       "esperada (~20 al año), y si cambió algo del mercado que invalide "
                       "el supuesto de reversión."), "inline": False},
            *([{"name": "​", "value": f"[📊 Ver el detalle]({dash_url})", "inline": False}]
              if dash_url else []),
        ],
    }


def publish(perf: dict, dash_url: str | None, dry_run: bool = False) -> None:
    notifier.send("scorecard", scorecard_embed(perf, dash_url), dry_run)
    d = drift_embed(perf, dash_url)
    if d:
        notifier.send("alertas-modelo", d, dry_run)
