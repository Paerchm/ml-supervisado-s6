# Paso 3 — Simulación y exploración de datos

## Simulación (mínimo 200 registros) — `scripts/gen_data.py`

Semilla fija (`np.random.default_rng(42)`) → 100 % reproducible.

| Dataset | Registros | Diseño |
|---|---|---|
| `data/churn_simulado.csv` | **800** clientes | 6 predictores conductuales; Y generada con modelo logístico + ruido → tasa de abandono calibrada ~29 % |
| `data/ventas_simulado.csv` | **300** (25 meses × 12 tiendas) | Y = función lineal de marketing, estacionalidad, precio, competencia, macro + historial + ruido |

## Análisis exploratorio (matplotlib/seaborn)

- **Caso A:** boxplots de cada predictor por estado de abandono + mapa de correlación.
  Hallazgo: los clientes que abandonan compran menos, gastan menos, llevan más días sin
  comprar, tienen más quejas y usan menos canales digitales. Correlaciones moderadas →
  hay señal pero sin separadores perfectos.
- **Caso B:** barras de estacionalidad media por mes, regplots marketing↔ingreso e
  historial↔ingreso, heatmap de correlaciones. Sin nulos; relaciones casi lineales,
  coherentes con el generador.
