# Intento de mejorar la señal de acciones — resultado negativo

Registrado para no volver a intentar lo mismo.

## Qué se probó

1. **Meta-labeling con 60 features** (`meta.py`), incluyendo features nuevos de
   contexto de mercado y transversales que no estaban en el screening original.
   Validación walk-forward con purga de 21 ruedas.
2. **Modelo mínimo** de 4 features, elegidos por poder univariado.
3. **Filtro de amplitud de mercado** sobre la regla existente.
4. **Ampliación del universo** de 14 a 82 nombres.

## Resultados

| Enfoque | Acierto OOS | AUC OOS | vs conteo |
|---|---|---|---|
| Conteo actual (baseline) | 58,6% | 0,502 | — |
| Logística, 60 features | 57,6% | 0,506 | peor |
| LightGBM, 60 features | 57,1% | 0,509 | peor |
| Ensamble | 55,8% | 0,510 | peor |
| Logística mínima, 4 features | 59,3% | 0,522 | +0,7 pp, dentro del ruido |

LightGBM da AUC 0,846 dentro de muestra y 0,523 afuera: sobreajuste sobre ruido,
no un bug del pipeline.

**Señal de que no hay información:** apretar la selección empeora el resultado.
Top 5% → 54,8%; top 50% → 58,9%. Un modelo con poder discriminante hace lo
contrario.

## El filtro de amplitud: falso positivo

Filtrar por `breadth_rsi14_median` en el peor 25% lleva el acierto de 60,5% a
69,7%. Parece enorme, pero contra el base rate **bajo la misma condición de
mercado** el lift se queda igual:

| Amplitud | Base (todos los días) | Con señal | Lift |
|---|---|---|---|
| Sin filtro | 59,1% | 60,5% | +1,4 pp |
| Peor 25% | 64,0% | 65,2% | +1,2 pp |
| Peor 15% | 65,5% | 67,4% | +1,9 pp |

Los días de mercado castigado sube todo. Es market timing, no mejora de señal.

## Por qué no hay nada que extraer

El AUC univariado máximo entre los 60 features, medido **dentro del conjunto de
eventos**, es 0,542. Condicionado a "sobrevendido dentro de tendencia alcista",
la información residual es casi nula: la regla simple ya la extrajo toda.

## Ampliar el universo

Sí aumenta las operaciones (25/año → 190/año) pero no el edge, y con capital
limitado la cartera rinde +12,2% anualizado contra +14,8% del SPY — y eso con
survivorship bias a favor.

## Conclusión

La señal de acciones es una **mejora marginal de timing (+1 a 2 pp), no una
estrategia de cartera**. Su uso correcto es el que ya tiene: avisar que una
caída es mejor punto de entrada que el promedio, sobre nombres que igual se
iban a comprar. No sirve como sistema autónomo.
