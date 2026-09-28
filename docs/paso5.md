# Paso 5 — Implementación y comparación de modelos (scikit-learn)

Ambos casos usan `train_test_split` (75/25, estratificado en A), escalado **solo** donde el
modelo lo requiere (logística/lineal; Random Forest no), y se comparan las métricas exigidas.

## Caso A — Clasificación (test set, 200 clientes)

| Métrica | Regresión logística | Random Forest |
|---|---|---|
| Accuracy | **0.730** | 0.695 |
| Precision | **0.643** | 0.440 |
| Recall | 0.155 | **0.190** |
| F1-score | 0.250 | **0.265** |
| AUC-ROC | **0.701** | 0.590 |

- Matrices de confusión y curvas ROC lado a lado en el notebook.
- Brecha train–test: logística ≈ **0.00** vs. bosque **0.305** (accuracy train 1.000 →
  el bosque memoriza).

## Caso B — Regresión (test set, 75 filas)

| Métrica | Regresión lineal | Random Forest |
|---|---|---|
| MAE | **9.25** | 21.72 |
| RMSE | **11.80** | 27.81 |
| R² | **0.971** | 0.839 |

- Brecha R² train–test: lineal **0.001** vs. bosque **0.125**.
- Scatter predicho vs. real en ambos modelos (la lineal se pega a la diagonal).

**Lectura:** en ambos casos el modelo interpretable no pierde — e incluso gana — frente al
modelo flexible, porque el dataset es pequeño y la relación generada es de familia "lineal
+ ruido". En datos reales con más volumen y no linealidades, el ranking puede invertirse.
