# Documentación del proyecto — ML Supervisado, Sesión 6

> **Evaluación y comparación de dos algoritmos de aprendizaje supervisado para dos casos de
> negocio (churn y ventas), justificando la selección final con evidencia empírica.**
> Rúbrica: `Descripción.docx`. Ejemplos: `Caso_A_Churn_Prediccion.ipynb` y
> `Caso_B_Ventas_Prediccion.ipynb`.
>
> 📖 Este archivo es **Markdown puro**: ábralo en VS Code y presione `Ctrl+K` seguido de `V`
> para ver la vista previa lado a lado. Los enlaces internos funcionan en la vista previa.

---

## Tabla de contenidos

1. [Descripción del problema de negocio](#1-descripción-del-problema-de-negocio)
2. [Mapa de selección de algoritmos](#2-mapa-de-selección-de-algoritmos)
3. [Simulación y exploración de datos](#3-simulación-y-exploración-de-datos)
4. [Verificación de supuestos](#4-verificación-de-supuestos)
5. [Implementación y comparación de modelos](#5-implementación-y-comparación-de-modelos)
6. [Análisis de límites y riesgos](#6-análisis-de-límites-y-riesgos)
7. [Consideraciones éticas](#7-consideraciones-éticas)
8. [Conclusión y recomendación final](#8-conclusión-y-recomendación-final)
- [Cómo usar este proyecto en VS Code](#cómo-usar-este-proyecto-en-vs-code)
- [Estructura del proyecto](#estructura-del-proyecto)

---

## 1. Descripción del problema de negocio

### Caso A — Predicción de abandono (churn)

Una cadena de retail pierde clientes de forma silenciosa. El objetivo es **anticuar quién se
va antes de que se vaya**, para dirigir campañas de retención en lugar de descuentos masivos.

| Elemento | Definición |
|---|---|
| Variable objetivo (Y) | `abandono` → **categórica binaria** (1 = abandonó, 0 = permanece) → **clasificación** |
| Variables (X) | frecuencia de compra, gasto promedio, días desde la última compra, interacción con promociones, quejas en 6 meses, uso de canales digitales |
| Decisiones del negocio | marketing prioriza a quién contactar; gerencia monitorea la tasa de churn; atención al cliente recibe alertas |
| Error costoso | falso negativo: cliente que se va clasificado como leal → oportunidad de retención perdida |

### Caso B — Predicción de ventas futuras

La cadena debe presupuestar inventario, personal y marketing para el mes siguiente.

| Elemento | Definición |
|---|---|
| Variable objetivo (Y) | `ingreso_mensual` → **continua** en USD → **regresión** |
| Variables (X) | gasto en campañas, estacionalidad, precio relativo vs. competencia, actividad competitiva, indicador macroeconómico, comportamiento histórico |
| Decisiones | S&OP fija compras por tienda; finanzas proyecta flujo de caja; marketing evalúa retorno de campaña |

---

## 2. Mapa de selección de algoritmos

Cuatro pasos de decisión, aplicados a ambos casos:

| Paso | Decisión | Caso A | Caso B |
|---|---|---|---|
| 1 | Tipo de variable objetivo | Binaria → clasificación | Continua → regresión |
| 2 | Objetivo de negocio y familia | Rankear riesgo con probabilidad interpretable → lineal + ensambles | Pronóstico descomponible en drivers → lineal + ensambles |
| 3 | Restricciones del contexto | 800 registros, interpretabilidad, recálculo barato | 300 registros, exigencia de explicabilidad ante finanzas/auditoría |
| 4 | Dos candidatos + limitaciones | **Logística** (explicable, pero asume linealidad en el logit) vs. **Random Forest** (captura interacciones, pero caja negra y sobreajuste en datos pequeños) | **Lineal múltiple** (inferencia válida, pero aditiva) vs. **Random Forest** (mismas limitaciones) |

---

## 3. Simulación y exploración de datos

### Simulación — `scripts/gen_data.py` (semilla 42 → reproducible)

| Dataset | Registros | Diseño |
|---|---|---|
| `data/churn_simulado.csv` | **800** clientes (mínimo 200 ✅) | Y generada con modelo logístico + ruido; tasa de abandono calibrada ~29 % |
| `data/ventas_simulado.csv` | **300** (25 meses × 12 tiendas) | Y = función lineal de 6 drivers + ruido |

### EDA (matplotlib/seaborn, en los notebooks)

- **Caso A:** boxplots por estado de abandono + mapa de correlación. Los clientes que
  abandonan compran menos, gastan menos, llevan más días sin comprar, tienen más quejas y
  usan menos canales digitales. Correlaciones moderadas → señal sin separadores perfectos.
- **Caso B:** estacionalidad media por mes, regplots marketing↔ingreso e historial↔ingreso,
  heatmap. Sin nulos; relaciones casi lineales, coherentes con el generador.

---

## 4. Verificación de supuestos

### Caso A — multicolinealidad (VIF) de la regresión logística

| Variable | VIF |
|---|---|
| compra_frecuencia | 2.20 |
| interaccion_promo | 2.19 |
| gasto_promedio | 1.01 |
| dias_ultima_compra | 1.00 |
| quejas_6m | 1.00 |
| canales_digitales | 1.00 |

✅ Todos < 10 → sin multicolinealidad severa; coeficientes estables e interpretables.

### Caso B — supuestos de la regresión lineal múltiple

| Supuesto | Evidencia empírica | Veredicto |
|---|---|---|
| Normalidad de residuos | Q-Q sobre la diagonal; Shapiro-Wilk **W = 0.998, p = 0.989 > 0.05** | ✅ |
| Homocedasticidad | scatter residuo vs. ajustado sin embudo | ✅ |
| Multicolinealidad | VIF entre 1.00 y 1.04 | ✅ |

---

## 5. Implementación y comparación de modelos

División `train_test_split` 75/25 (estratificada en A); escalado solo donde se necesita
(logística/lineal; Random Forest no).

### Caso A — clasificación (métricas exigidas: accuracy, precision, recall, F1, matriz de confusión)

| Métrica | Regresión logística | Random Forest |
|---|---|---|
| Accuracy | **0.730** | 0.695 |
| Precision | **0.643** | 0.440 |
| Recall | 0.155 | **0.190** |
| F1-score | 0.250 | **0.265** |
| AUC-ROC | **0.701** | 0.590 |

Brecha train–test: logística **≈ 0.00** vs. bosque **0.305** (accuracy train 1.000 → memoriza).

### Caso B — regresión (métricas exigidas: MAE, RMSE, R²)

| Métrica | Regresión lineal | Random Forest |
|---|---|---|
| MAE | **9.25** | 21.72 |
| RMSE | **11.80** | 27.81 |
| R² | **0.971** | 0.839 |

Brecha R² train–test: lineal **0.001** vs. bosque **0.125**.

**Lectura:** en ambos casos el modelo interpretable iguala o supera al flexible porque el
dataset es pequeño y la relación generada es "lineal + ruido". Con datos reales de mayor
volumen y no linealidades, el ranking podría invertirse.

---

## 6. Análisis de límites y riesgos

| Riesgo | Evidencia / análisis | Mitigación |
|---|---|---|
| **Overfitting** | Medido como brecha train–test: A: 0.305 (bosque) vs ≈0 (logística); B: 0.125 (bosque) vs 0.001 (lineal) | Regularización, `min_samples_leaf`, validación cruzada |
| **Datos no representativos** | Base simulada: un perfil real distinto degrada el recall de la minoría; con umbral 0.5 el recall es bajo (~0.16–0.19) en ambos modelos de A | Estratificación, `class_weight`, umbral según costo, reentrenamiento out-of-time |
| **Outliers** | En lineal/logística arrastran la recta (mínimos cuadrados); en el bosque distorsionan splits pero promediar árboles amortigua | Cook's distance / boxplots de colas, winsorización, monitoreo de estabilidad |
| **Sesgo estructural** | La lineal solo captura efectos aditivos; marketing con rendimientos decrecientes quedaría subestimado | Términos polinómicos/splines manteniendo explicabilidad |

---

## 7. Consideraciones éticas

| Pregunta | Caso A (churn) | Caso B (ventas) |
|---|---|---|
| ¿Sesgos o discriminación? | Sí: puede concentrar retención en clientes rentables y excluir segmentos vulnerables vía *proxies* (aunque no use variables protegidas). Auditar FNR por segmento | Riesgo menor (nivel tienda), pero el pronóstico puede **auto-cumplirse**: menos presupuesto a tiendas ya débiles |
| ¿Transparencia a stakeholders? | Modelo explicable + razón por alerta (coeficiente dominante), no solo un score | Descomposición exacta del pronóstico en drivers, defendible ante finanzas/auditoría |
| ¿Privacidad? | Datos conductuales personales: minimización, acceso restringido, anonimización, aviso al cliente, política de retención | Solo agregados de tienda y macroindicadores → riesgo bajo; con canastas por cliente, aplicar lo de A |

---

## 8. Conclusión y recomendación final

### Caso A → **Regresión logística**

- **Técnica:** mejor AUC (0.701 vs. 0.590) y accuracy (0.730 vs. 0.695); F1 casi idéntico;
  brecha train–test nula frente a un bosque que sobreajusta.
- **Estratégica:** marketing necesita explicar *por qué* cada cliente es riesgoso para
  diseñar y auditar la acción de retención; los coeficientes lo dan directamente.
- **Ética:** la interpretabilidad permite auditar sesgos por segmento; la caja negra no.

### Caso B → **Regresión lineal múltiple**

- **Técnica:** domina MAE/RMSE/R² con supuestos verificados → intervalos de pronóstico válidos.
- **Estratégica:** S&OP y finanzas negocian presupuesto por driver; el bosque solo da
  importancias sin dirección.
- **Ética:** pronóstico auditable evita decisiones de presupuesto opacas.

### Mejoras / siguientes pasos

1. Validar con **datos reales** y ventana temporal out-of-time.
2. **Calibrar probabilidades** (Platt/Isotonic) y optimizar umbral según costo FP vs. FN.
3. Monitoreo de **drift** en producción.
4. Si el bosque gana por margen con datos reales, usarlo como motor y explicar con **SHAP**.

---

## Cómo usar este proyecto en VS Code

1. Abra la carpeta `ml-supervisado-s6` en VS Code (`Archivo → Abrir carpeta`).
2. **Documentación:** abra `Documentacion.md` y presione `Ctrl+K` `V` para la vista previa.
   (Extensión opcional recomendada: *Markdown All in One* para TOC/atajos.)
3. **Notebooks:** abra `Caso_A_Churn_Prediccion.ipynb` / `Caso_B_Ventas_Prediccion.ipynb`
   (extensión *Jupyter*). Ya vienen ejecutados con salidas; para re-ejecutar:
   `Kernel → Restart and Run All`. Requieren `data/*.csv` (génerelos con
   `python scripts/gen_data.py`).
4. **Páginas por paso** (también Markdown nativo de VS Code): `docs/paso1.md` … `docs/paso7.md`.

## Estructura del proyecto

```
ml-supervisado-s6/
├── Documentacion.md                # ← documentación completa en Markdown (VS Code)
├── README.md
├── Caso_A_Churn_Prediccion.ipynb   # Caso A: clasificación binaria
├── Caso_B_Ventas_Prediccion.ipynb  # Caso B: regresión
├── data/
│   ├── churn_simulado.csv          # 800 registros simulados
│   └── ventas_simulado.csv         # 300 registros simulados
├── docs/
│   ├── paso1.md … paso7.md         # cada paso de la rúbrica, en Markdown
│   ├── caso_a_churn.md             # notebook A como página Markdown
│   └── caso_b_ventas.md            # notebook B como página Markdown
└── scripts/
    ├── gen_data.py                 # datos reproducibles (semilla 42)
    └── build_notebooks.py          # reconstruye los notebooks
```
