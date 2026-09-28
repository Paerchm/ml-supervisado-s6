# Paso 2 — Mapa de selección de algoritmos

La rúbrica exige 4 pasos de decisión, documentados en la Sección 2 de cada notebook:

## Paso 2.1 — Tipo de variable objetivo

| Caso | Y | Familia |
|---|---|---|
| A | `abandono` (binaria) | Clasificación |
| B | `ingreso_mensual` (continua) | Regresión |

## Paso 2.2 — Objetivo de negocio y familia de algoritmos

- **A:** rankear clientes por riesgo con probabilidad interpretable → modelos lineales
  (regresión logística) y ensambles (Random Forest).
- **B:** pronóstico descomponible en drivers económicos → regresión lineal múltiple y
  ensambles de árboles (Random Forest).

## Paso 2.3 — Restricciones del contexto

- Volumen moderado (800 y 300 registros) → descarta deep learning.
- Interpretabilidad exigida por stakeholders (marketing, finanzas, auditoría).
- Latencia y costo de entrenamiento bajos (recálculo frecuente).

## Paso 2.4 — Dos candidatos y sus limitaciones

| Candidato | Ventaja | Limitación |
|---|---|---|
| Regresión logística / lineal | Coeficientes interpretables, inferencia estadística, sin brecha train–test | Asume linealidad en logit (A) / aditividad (B); no captura interacciones |
| Random Forest | Captura no linealidad e interacciones, robusto a outliers | Caja negra; overfitting en datasets pequeños (evidencia: AUC cae a 0.590 en A; brecha R² = 0.125 en B) |
