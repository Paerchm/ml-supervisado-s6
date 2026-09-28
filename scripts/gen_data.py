# -*- coding: utf-8 -*-
"""
Genera los datasets simulados requeridos por la rúbrica (mínimo 200 registros).

- data/churn_simulado.csv : Caso A, clasificación binaria (abandono Sí/No).
- data/ventas_simulado.csv : Caso B, regresión (ingreso mensual).

Semilla fija (42) para que la simulación sea reproducible.
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
OUT = Path(__file__).resolve().parents[1] / "data"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- Caso A
n = 800  # > 200 registros exigidos
compra_frecuencia = np.round(rng.gamma(2.0, 1.1, n), 1)          # compras / mes
gasto_promedio = np.round(rng.normal(120, 40, n).clip(25, None), 2)  # USD / compra
dias_ultima_compra = np.round(rng.gamma(2.5, 12, n).clip(0, 180))    # recencia
interaccion_promo = rng.binomial(10, np.clip(compra_frecuencia / 12, 0.05, 0.85))
quejas_6m = rng.poisson(0.6, n)                                    # historial de quejas
canales_digitales = np.round(rng.beta(2.2, 2.2, n), 2)             # uso de canales digitales

logit = (
    -1.10
    - 0.22 * compra_frecuencia
    - 0.006 * gasto_promedio
    + 0.016 * dias_ultima_compra
    - 0.10 * interaccion_promo
    + 0.55 * quejas_6m
    - 0.90 * canales_digitales
    + rng.normal(0, 0.4, n)
)
# Calibra el intercepto por bisección para lograr una tasa de abandono ~28 %
target = 0.28
lo, hi = -15.0, 15.0
for _ in range(60):
    mid = (lo + hi) / 2
    p = 1 / (1 + np.exp(-(logit + mid)))
    if p.mean() > target:
        hi = mid
    else:
        lo = mid
p_abandono = 1 / (1 + np.exp(-(logit + (lo + hi) / 2)))
abandono = rng.binomial(1, p_abandono)

df_churn = pd.DataFrame(
    {
        "cliente_id": np.arange(1, n + 1),
        "compra_frecuencia": compra_frecuencia,
        "gasto_promedio": gasto_promedio,
        "dias_ultima_compra": dias_ultima_compra,
        "interaccion_promo": interaccion_promo,
        "quejas_6m": quejas_6m,
        "canales_digitales": canales_digitales,
        "abandono": abandono,
    }
)
df_churn.to_csv(OUT / "churn_simulado.csv", index=False)

# ---------------------------------------------------------------- Caso B
m = 300  # 25 meses x 12 tiendas
mes = np.tile(np.arange(1, 26), 12)
fecha = pd.to_datetime(["2023-01-01", "2025-01-01"])  # ancla para generar fechas mensuales
fechas = [pd.Timestamp("2023-01-01") + pd.DateOffset(months=int(mm - 1)) for mm in mes]
tienda = np.repeat(np.arange(1, 13), 25)

gasto_marketing = np.round(rng.uniform(5, 60, m), 2)            # miles USD / mes
estacionalidad = np.round(1 + 0.25 * np.sin(2 * np.pi * (np.array(mes) - 3) / 12), 4)
precio_relativo = np.round(rng.normal(1.0, 0.08, m).clip(0.7, 1.35), 3)
actividad_competencia = np.round(rng.uniform(0, 10, m), 2)
indicador_macro = np.round(rng.normal(0, 1, m), 3)              # crecimiento económico
comportamiento_previo = np.round(rng.uniform(150, 350, m), 2)   # ventas del mes anterior

ingreso_mensual = (
    120
    + 0.55 * comportamiento_previo
    + 3.20 * gasto_marketing
    + 60 * estacionalidad
    - 110 * (precio_relativo - 1)
    - 4.5 * actividad_competencia
    + 14 * indicador_macro
    + rng.normal(0, 12, m)
)
df_ventas = pd.DataFrame(
    {
        "fecha": [f.date().isoformat() for f in fechas],
        "tienda": tienda,
        "gasto_marketing": gasto_marketing,
        "estacionalidad": estacionalidad,
        "precio_relativo": precio_relativo,
        "actividad_competencia": actividad_competencia,
        "indicador_macro": indicador_macro,
        "comportamiento_previo": comportamiento_previo,
        "ingreso_mensual": np.round(ingreso_mensual, 2),
    }
)
df_ventas.to_csv(OUT / "ventas_simulado.csv", index=False)

print("churn_simulado.csv:", df_churn.shape, "| tasa abandono =", df_churn["abandono"].mean().round(3))
print("ventas_simulado.csv:", df_ventas.shape, "| rango ingreso =",
      df_ventas["ingreso_mensual"].min(), "-", df_ventas["ingreso_mensual"].max())
