# Señales de acciones y cripto

Sistema de señales de mediano plazo que publica avisos en Discord. Cada regla está
medida sobre datos reales, con separación entre el período usado para descubrirla y
el usado para validarla. Lo que no funciona también está documentado.

**📖 [Cómo funciona el modelo](https://TU_USUARIO.github.io/investing/)** — la explicación
completa, con los números y la metodología.

---

## Qué hace

Dos señales independientes, porque los dos mercados se comportan al revés uno del otro.
Eso no es una opinión: sale del screening de ~45 indicadores sobre 92 activos.

### Cripto — momentum

Sobrepondera cuando el precio está por encima de su EMA200, subpondera cuando está
castigado. Es una señal de **peso**, no de entrar y salir.

| | Tilt | Comprar y mantener |
|---|---|---|
| Sharpe out-of-sample | **1,03** | 0,91 |
| CAGR out-of-sample | **58,5%** | 45,3% |
| Peor caída | **−74,5%** | −79,4% |

Validado sobre 10 monedas: los 9 indicadores de momentum probados le ganan a comprar y
mantener en ambos períodos, leave-one-out da 10 de 10, y sobrevive a costos de 50 bps.

### Acciones — reversión dentro de tendencia

Compra caídas de corto plazo **solo** si la acción sigue arriba de su EMA200. La confianza
se gradúa contando cuántas de seis condiciones coinciden.

| Condiciones | P(sube en 10 ruedas) | Base rate |
|---|---|---|
| 3 | 58,1% | 55,7% |
| **5** | **63,9%** | 55,7% |
| **6** | **67,9%** | 55,7% |

338 operaciones entre 2010 y 2026, con retorno medio positivo en 15 de 17 años y en 14 de
los 15 activos. Las reglas se calibraron sobre 82 activos ajenos al watchlist y se aplican
sin reajustar nada.

**No hay señal de venta.** Los indicadores de sobrecompra no anticipan caídas en acciones
de calidad: la salida es a plazo fijo de 21 ruedas. Está explicado en la documentación.

---

## Correr el proyecto

```bash
pip install -r requirements.txt

python3 serve.py                        # documentación en localhost:8000
python3 -m bot.run_daily --dry-run      # ver las señales de hoy sin mandar nada
python3 -m bot.run_daily --as-of 2026-06-24   # replay de una rueda pasada
```

### Conectar Discord

1. Crear los canales `#acciones`, `#cripto`, `#regimen` y `#reporte`
2. Un webhook por canal → exportarlos como variables de entorno:

```bash
export DISCORD_WEBHOOK_ACCIONES="https://discord.com/api/webhooks/..."
export DISCORD_WEBHOOK_CRIPTO="..."
export DISCORD_WEBHOOK_REGIMEN="..."
export DISCORD_WEBHOOK_REPORTE="..."
```

3. Publicar y fijar los mensajes de bienvenida:

```bash
python3 -m bot.post_welcome
```

Los webhooks pueden publicar pero no fijar, así que el último paso es a mano: click derecho
en cada mensaje → *Fijar mensaje*.

### Automatizarlo

Dos opciones, **usar una sola** o salen alertas duplicadas:

- **GitHub Actions** (`.github/workflows/daily.yml`) — corre en la nube todos los días a las
  22:30 UTC, no necesita tu computadora prendida. Los webhooks van en *Settings → Secrets and
  variables → Actions*.
- **launchd** (`scripts/com.investing.senales.plist`) — corre en tu Mac. Instrucciones adentro
  del archivo.

---

## Reproducir los números

Cada tabla de la documentación sale de un script. Todos escriben en `out/`.

| Script | Qué produce |
|---|---|
| `run_screen.py` | IC de ~45 indicadores contra retornos futuros a 3, 5, 10, 21, 63 y 126 ruedas |
| `run_control.py` | Control de survivorship bias: ETFs contra acciones individuales |
| `run_meanrev.py` | Deciles del score de reversión, in-sample contra out-of-sample |
| `run_allocation.py` | Tests de tilt de exposición y de rotación |
| `run_crypto.py` | Validación de cripto: por indicador, año a año y leave-one-out |
| `run_rules.py` | Minería de reglas con intervalos de confianza |
| `run_ladder.py` | Escalera de confianza |
| `run_signal_final.py` | Comparación de reglas de salida y estabilidad |

```bash
python3 run_screen.py --force    # descarga y arma el panel (tarda unos minutos)
python3 run_ladder.py
python3 run_crypto.py
```

---

## Estructura

```
bot/            job diario: motor, estado, notificaciones a Discord
signals/        indicadores, features, backtest — compartido por research y bot
docs/           la documentación del modelo (GitHub Pages sirve esta carpeta)
state/          memoria del bot entre corridas, en JSON versionado
out/            resultados de los backtests
run_*.py        scripts de investigación
```

---

## Límites

- El edge en acciones es de **aproximadamente +1 punto porcentual por operación** sobre
  comprar en un día cualquiera. Mejora el timing, no multiplica el retorno.
- La cola izquierda es gruesa: la peor operación del backtest fue −34%. Sizing fijo,
  sin concentrar.
- El resultado de cripto se apoya en dos bull runs (2017 y 2021). El momentum
  históricamente se rompe en mercados laterales largos.
- Se probaron decenas de reglas. Lo que sobrevive lo hace por consistencia año a año y
  coherencia entre períodos, no por un p-valor aislado.

**Esto es un proyecto personal de investigación, no asesoramiento financiero.** Ninguna
alerta es una recomendación de compra. Los datos vienen de Yahoo Finance vía `yfinance`.
