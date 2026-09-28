# ML Supervisado — Sesión 6: Evaluación y comparación de algoritmos

Proyecto académico que cumple la rúbrica de `Descripción.docx`: **evaluar y comparar al menos
dos algoritmos de aprendizaje supervisado para cada uno de dos casos de negocio**, justificando
la selección final —técnica, estratégica y éticamente— con evidencia empírica.

Los dos ejemplos de la sesión (Walmart Sales y Churn) fueron actualizados con todas las
secciones sugeridas en la descripción de la actividad.

## Estructura del proyecto

```
ml-supervisado-s6/
├── Documentacion.md                # ← documentación completa en Markdown (VS Code)
├── README.md
├── Caso_A_Churn_Prediccion.ipynb   # Caso A: clasificación binaria (abandono Sí/No)
├── Caso_B_Ventas_Prediccion.ipynb  # Caso B: regresión (ingresos mensuales)
├── data/
│   ├── churn_simulado.csv          # 800 registros simulados (tasa de abandono ~29 %)
│   └── ventas_simulado.csv         # 300 registros (25 meses × 12 tiendas)
├── scripts/
│   ├── gen_data.py                 # Genera los datasets (semilla fija → reproducible)
│   └── build_notebooks.py          # Construye los notebooks
└── docs/                           # Páginas Markdown por paso (opcionales)
    ├── paso1.md … paso7.md
    └── caso_a_churn.md / caso_b_ventas.md
```

## Cómo reproducir

```bash
# 1. Entorno
pip install pandas numpy scipy matplotlib seaborn statsmodels scikit-learn nbformat nbclient ipykernel

# 2. Datos simulados (semilla 42 → mismo resultado)
python scripts/gen_data.py

# 3. Los notebooks ya vienen ejecutados con sus salidas; para re-ejecutarlos:
jupyter nbconvert --to notebook --execute --inplace Caso_A_Churn_Prediccion.ipynb
jupyter nbconvert --to notebook --execute --inplace Caso_B_Ventas_Prediccion.ipynb

# 4. Documentación (Markdown para VS Code)
#    → abrir Documentacion.md y presionar Ctrl+K V para vista previa
```

## Los dos casos

| | Caso A — Churn | Caso B — Ventas |
|---|---|---|
| Problema | Clasificación binaria | Regresión |
| Variable objetivo (Y) | `abandono` (Sí/No) | `ingreso_mensual` (USD) |
| Algoritmo 1 | Regresión logística | Regresión lineal múltiple |
| Algoritmo 2 | Random Forest | Random Forest |
| Supuesto verificado | Multicolinealidad (VIF < 10) | Normalidad (Shapiro-Wilk), homocedasticidad, VIF |
| Métricas | Accuracy, Precision, Recall, F1, matriz de confusión, AUC-ROC | MAE, RMSE, R² |
| Modelo seleccionado | **Regresión logística** (AUC 0.701 vs. 0.590; sin brecha train–test) | **Regresión lineal múltiple** (R² 0.971 vs. 0.839; brecha 0.001) |

## Documentación de los pasos

La documentación completa paso a paso está en **[Documentacion.md](Documentacion.md)** —
Markdown puro para la vista previa de VS Code (`Ctrl+K` `V`). Cada paso también tiene su
página individual en `docs/`: [Paso 1](docs/paso1.md) · [Paso 2](docs/paso2.md) ·
[Paso 3](docs/paso3.md) · [Paso 4](docs/paso4.md) · [Paso 5](docs/paso5.md) ·
[Paso 6](docs/paso6.md) · [Paso 7](docs/paso7.md)
