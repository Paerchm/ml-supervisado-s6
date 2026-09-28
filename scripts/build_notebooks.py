# -*- coding: utf-8 -*-
"""Construye los dos notebooks del proyecto aplicando la rúbrica de Descripción.docx."""
import nbformat as nbf

PROJ = r"C:\Users\pablo\Documents\qwen-agent\LT5RT4itQi\default\ml-supervisado-s6"


def md(nb, text):
    nb.cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(nb, text):
    nb.cells.append(nbf.v4.new_code_cell(text.strip()))


def new_nb():
    return nbf.v4.new_notebook(
        cells=[],
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
    )


# ==================================================================== CASO A
nb = new_nb()

md(nb, """
# Caso A — Predicción de abandono de clientes (Churn Prediction)

> Evaluación y comparación de **dos algoritmos de aprendizaje supervisado** para un caso de
> negocio, justificando la selección final con evidencia empírica
> (rúbrica: Descripción de la actividad – Sesión 6).

## 1. Descripción del problema de negocio

**El problema, en palabras propias:** una empresa de retail pierde clientes de forma
silenciosa: dejan de comprar sin avisar. Detectar *a quién* está por irse —antes de que lo
haga— permite activar retención dirigida (cupones, contacto proactivo) en lugar de aplicar
descuentos masivos que erosionan el margen.

- **Variable objetivo (Y):** `abandono` → **categórica binaria** (1 = el cliente abandonó, 0 = permanece). Por tanto el problema es de **clasificación binaria supervisada**.
- **Variables independientes (X), según lo sugerido en la actividad:** frecuencia de compra, gasto promedio, tiempo desde la última compra (recencia), interacción con promociones, historial de quejas y uso de canales digitales.
- **Contexto y decisiones:** marketing usa las probabilidades de abandono para priorizar campañas de retención; gerencia monitorea la tasa de churn; atención al cliente recibe alertas de clientes con quejas recurrentes. Un error costoso es clasificar como "leal" a un cliente que en realidad se va (falso negativo): se pierde la oportunidad de retenerlo.

## 2. Mapa de selección de algoritmos

| Paso | Decisión | Justificación |
|---|---|---|
| **1. Tipo de variable objetivo** | Binaria (Sí/No) → clasificación | El negocio necesita decidir *si* un cliente abandonará. |
| **2. Objetivo de negocio y familia de algoritmos** | Segmentar clientes de riesgo con probabilidad interpretable → modelos lineales (regresión logística) y de ensamble (Random Forest / boosting) | Se busca ranking de riesgo para campañas, no solo etiquetas duras. |
| **3. Restricciones del contexto** | Datos tabulares de tamaño moderado (800 registros simulados), necesidad de **interpretabilidad** ante stakeholders, entrenamiento rápido y poco costo | Descarta deep learning; favorece modelos ligeros y explicables. |
| **4. Candidatos elegidos y sus limitaciones** | **Regresión logística**: muy interpretable (coeficientes = efecto por variable), pero asume relación lineal en el logit y no captura interacciones complejas. **Random Forest**: captura no linealidades e interacciones y resiste outliers, pero es una caja negra con riesgo de overfitting y sobreestima importancia de variables de alta cardinalidad. | Se comparan con evidencia empírica en set de prueba. |

## 3. Simulación y exploración de datos

El dataset es **simulado** con semilla fija (reproducible) mediante `scripts/gen_data.py`:
**800 registros** (supera el mínimo de 200), con 6 variables predictoras sugeridas y una tasa
de abandono realista (~29 %). La relación entre X e Y se generó con un modelo logístico +
ruido, de modo que existe señal aprendible pero no determinista.
""")

code(nb, """
%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_curve,
                             roc_auc_score)

df = pd.read_csv("data/churn_simulado.csv")
print("Dimensiones:", df.shape)
df.head()
""")

code(nb, """
print("Nulos por columna:\\n", df.isna().sum(), "\\n")
df["abandono"].value_counts(normalize=True).rename("proporción").to_frame()
""")

code(nb, """
df.describe().T
""")

code(nb, """
cols_x = ["compra_frecuencia", "gasto_promedio", "dias_ultima_compra",
          "interaccion_promo", "quejas_6m", "canales_digitales"]

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, col in zip(axes.ravel(), cols_x):
    sns.boxplot(x="abandono", y=col, data=df, ax=ax, palette="Set2")
    ax.set_title(f"{col} por estado de abandono")
plt.tight_layout(); plt.show()
""")

code(nb, """
plt.figure(figsize=(10, 6))
sns.heatmap(df[cols_x + ["abandono"]].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Mapa de correlación (incluida la variable objetivo)")
plt.tight_layout(); plt.show()
""")

md(nb, """
**Lectura del EDA:** los clientes que abandonan compran con menos frecuencia, gastan menos
en promedio, llevan más días sin comprar, interactúan menos con las promociones, acumulan
más quejas y usan menos los canales digitales. La correlación de cada X con `abandono` es
moderada y de signo coherente con el negocio → hay señal, pero ningún separador perfecto
(nada cerca de ±1), lo que acota el techo de accuracy.

## 4. Verificación de supuestos del algoritmo seleccionado

Para la **regresión logística** verificamos el supuesto de **ausencia de multicolinealidad**
entre predictores usando el **VIF** (Variance Inflation Factor; regla práctica: VIF > 10 es problemático).
""")

code(nb, """
from statsmodels.stats.outliers_influence import variance_inflation_factor

X_vif = df[cols_x].copy()
X_vif["const"] = 1
vif = pd.DataFrame({
    "Variable": cols_x,
    "VIF": [variance_inflation_factor(X_vif.values, i) for i in range(len(cols_x))],
})
vif
""")

md(nb, """
**Conclusión del supuesto:** todos los VIF son < 10 → no hay multicolinealidad severa; la
regresión logística es aplicable y sus coeficientes serán estables e interpretables.

## 5. Implementación y comparación de modelos (scikit-learn)

División train/test con `train_test_split` (75/25, estratificada) y **escala solo donde se
necesita** (logística); Random Forest no requiere escalado.
""")

code(nb, """
X = df[cols_x]
y = df["abandono"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

logreg = LogisticRegression(max_iter=1000).fit(X_train_s, y_train)
bosque = RandomForestClassifier(n_estimators=300, random_state=42).fit(X_train, y_train)
print("Train:", X_train.shape, "| Test:", X_test.shape)
""")

code(nb, """
def evaluar(modelo, Xte, yte, nombre):
    pred = modelo.predict(Xte)
    proba = modelo.predict_proba(Xte)[:, 1]
    fpr, tpr, _ = roc_curve(yte, proba)
    return {
        "Modelo": nombre,
        "Accuracy": accuracy_score(yte, pred),
        "Precision": precision_score(yte, pred),
        "Recall": recall_score(yte, pred),
        "F1-score": f1_score(yte, pred),
        "AUC-ROC": roc_auc_score(yte, proba),
    }

resultados = pd.DataFrame([
    evaluar(logreg, X_test_s, y_test, "Regresión logística"),
    evaluar(bosque, X_test, y_test, "Random Forest"),
]).set_index("Modelo").round(4)
resultados
""")

code(nb, """
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (nombre, modelo, Xte) in zip(
        axes, [("Logística", logreg, X_test_s), ("Random Forest", bosque, X_test)]):
    cm = confusion_matrix(y_test, modelo.predict(Xte))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(f"Matriz de confusión — {nombre}")
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
plt.tight_layout(); plt.show()
""")

code(nb, """
for nombre, modelo, Xte in [("REGRESIÓN LOGÍSTICA", logreg, X_test_s),
                            ("RANDOM FOREST", bosque, X_test)]:
    print("=" * 20, nombre, "=" * 20)
    print(classification_report(y_test, modelo.predict(Xte),
                                target_names=["No abandono (0)", "Abandono (1)"]))
""")

code(nb, """
# Curvas ROC comparadas
plt.figure(figsize=(7, 6))
for nombre, modelo, Xte in [("Regresión logística", logreg, X_test_s),
                             ("Random Forest", bosque, X_test)]:
    fpr, tpr, _ = roc_curve(y_test, modelo.predict_proba(Xte)[:, 1])
    plt.plot(fpr, tpr, label=f"{nombre} (AUC={roc_auc_score(y_test, modelo.predict_proba(Xte)[:, 1]):.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=.5, label="Azar")
plt.xlabel("Tasa de falsos positivos"); plt.ylabel("Tasa de verdaderos positivos")
plt.title("Curvas ROC — Caso A"); plt.legend(); plt.grid(alpha=.3)
plt.tight_layout(); plt.show()
""")

md(nb, """
## 6. Análisis de límites y riesgos (con evidencia empírica)

- **Overfitting:** comparamos el desempeño en train vs. test en ambos modelos.
- **Interpretabilidad:** extraemos coeficientes (logística) e importancias (Random Forest).
- **Sesgo / datos no representativos y outliers:** discusión en las celdas de markdown siguientes.
""")

code(nb, """
brecha = pd.DataFrame({
    "Modelo": ["Regresión logística", "Random Forest"],
    "Accuracy TRAIN": [accuracy_score(y_train, logreg.predict(X_train_s)),
                       accuracy_score(y_train, bosque.predict(X_train))],
    "Accuracy TEST": [accuracy_score(y_test, logreg.predict(X_test_s)),
                      accuracy_score(y_test, bosque.predict(X_test))],
})
brecha["Brecha (train-test)"] = brecha["Accuracy TRAIN"] - brecha["Accuracy TEST"]
brecha.round(4)
""")

code(nb, """
# Interpretabilidad: coeficientes de la logística (variables estandarizadas)
pd.Series(logreg.coef_[0], index=cols_x).sort_values().plot.barh(
    figsize=(8, 4), color="steelblue")
plt.title("Coeficientes de la regresión logística (predictores estandarizados)")
plt.axvline(0, color="k", lw=.8)
plt.tight_layout(); plt.show()

# Importancias del Random Forest
pd.Series(bosque.feature_importances_, index=cols_x).sort_values().plot.barh(
    figsize=(8, 4), color="seagreen")
plt.title("Importancia de variables — Random Forest")
plt.tight_layout(); plt.show()
""")

md(nb, """
**Lectura de la evidencia:**

1. **Overfitting.** El Random Forest memoriza el train: accuracy 1.000 en entrenamiento
   contra 0.695 en prueba (brecha ≈ 0.31), y su AUC cae de un valor competitivo a 0.590.
   La logística no presenta brecha (≈ 0.00) y mantiene la mejor AUC (0.701): en solo 800
   registros, la flexibilidad del bosque se paga con varianza.
2. **Interpretabilidad.** Los coeficientes de la logística ordenan los factores de riesgo con
   dirección y magnitud legibles para el comité de negocio; las importancias del bosque dicen
   *qué* importa pero no *cómo* (ni en qué dirección), lo que dificulta defender decisiones
   ante el cliente.
3. **Datos no representativos.** La base es simulada: si la proporción de abandono o el
   perfil de los clientes reales difiere (p. ej., subrepresentación de clientes nuevos), el
   recall de la clase minoritaria se degrada y el umbral óptimo cambia. De hecho, con el
   umbral por defecto (0.5) el recall es bajo en ambos modelos (~0.16–0.19): en producción
   se debe mover el umbral hacia el lado de capturar abandonos, aceptando menos precisión.
   Mitigación: estratificación en la división, recalibración (`class_weight`) y
   reentrenamiento periódico con datos frescos.
4. **Outliers.** Gastos extremos o días-sin-comprar topes (clientes atrapados, B2B) inflan
   VIF/coeficientes en la logística y distorsionan los splits del bosque. Mitigación:
   revisión de colas, winsorización y monitoreo de estabilidad de predicciones.

## 7. Consideraciones éticas

- **¿Podría generar sesgos o discriminación?** Sí. Si los datos históricos castigan a un
  segmento (p. ej., clientes de bajo ingreso o cierta zona), el modelo aprende a "prescindir
  de ellos" y la retención se concentra en los rentables → discriminación algorítmica. No
  usamos variables protegidas (edad, género, etnia, código postal), pero puede haber
  *proxies*. Mitigación: auditoría de tasas de abandono predichas por segmento y evaluación
  de fairness (igualdad de FNR entre grupos).
- **Transparencia para stakeholders.** Optamos por un modelo explicable (logística) y
  entregamos con cada alerta la razón principal (coeficiente dominante), no solo un score.
  Marketing puede auditar por qué se eligió a cada cliente.
- **Privacidad.** Las variables son comportamiento de compra y quejas: datos personales bajo
  normativa de protección de datos. Aplican: minimización (solo lo necesario), acceso
  restringido a datos a nivel cliente, anonimización para analítica, política de retención y
  transparencia ante el cliente (aviso de uso de sus datos para fines de retención).

## 8. Conclusión y recomendación final

**Modelo recomendado: Regresión logística**, con la siguiente justificación técnica,
estratégica y ética basada en la evidencia empírica de este notebook:

- **Técnica:** en el set de prueba la logística logra la mejor AUC (0.701 vs. 0.590) y el
  mejor accuracy (0.730 vs. 0.695); el F1 de ambos es prácticamente idéntico (0.250 vs.
  0.265), por lo que no hay beneficio predictivo en usar el modelo más complejo. Además,
  la brecha train–test de la logística es ≈ 0, mientras el bosque sobreajusta con fuerza.
- **Estratégica:** el negocio necesita explicar *por qué* un cliente es riesgoso para
  diseñar la acción de retención y defenderla ante auditoría; la logística da coeficientes
  accionables (p. ej., cada queja adicional incrementa el log-odds de abandono).
- **Ética:** un modelo interpretable facilita detectar sesgos por segmento; la caja negra
  del bosque dificulta cumplir el principio de transparencia.

**Mejoras / siguientes pasos:** migrar de datos simulados a datos reales con validación
temporal (out-of-time), calibrar probabilidades (Platt/Isotonic), optimizar el umbral de
decisión según el costo de retención vs. pérdida del cliente, y monitorear drift. Si con
datos reales y más volumen el bosque supera por margen amplio a la logística, usarlo como
motor de predicción y destilar la explicación con SHAP.
""")

nbA_path = PROJ + r"\Caso_A_Churn_Prediccion.ipynb"
nbf.write(nb, nbA_path)

# ==================================================================== CASO B
nb = new_nb()

md(nb, """
# Caso B — Predicción de ventas futuras (ingresos mensuales)

> Evaluación y comparación de **dos algoritmos de aprendizaje supervisado** para un caso de
> negocio, justificando la selección final con evidencia empírica (regresión).
> Ejemplo derivado del análisis exploratorio de Walmart Sales (Sesión 6), actualizado a la
> rúbrica de la actividad.

## 1. Descripción del problema de negocio

**El problema, en palabras propias:** la cadena necesita anticipar el **ingreso mensual por
tienda** para presupuestar compras de inventario, turnos de personal y asignación de
marketing. Subestimar ingresos genera quiebres de stock; sobreestimar genera inventario
inmovilizado.

- **Variable objetivo (Y):** `ingreso_mensual` → **continua** en USD → el problema es de **regresión supervisada**.
- **Variables independientes (X), según lo sugerido:** gasto en campañas de marketing, estacionalidad, precios (precio relativo vs. competencia), intensidad competitiva, indicador macroeconómico y comportamiento histórico de clientes (ingreso del mes anterior).
- **Contexto y decisiones:** planeación (S&OP) fija presupuesto de compras por tienda; finanzas proyecta flujo de caja; marketing evalúa el retorno de la campaña. El modelo se usa para *pronóstico puntual con intervalo*, no para decisiones automáticas.

## 2. Mapa de selección de algoritmos

| Paso | Decisión | Justificación |
|---|---|---|
| **1. Tipo de variable objetivo** | Continua → regresión | Se predice un monto, no una categoría. |
| **2. Objetivo de negocio y familia** | Pronóstico explicables de demanda → regresión lineal múltiple y ensambles basados en árboles (Random Forest) | Finanzas necesita descomponer el pronóstico en drivers. |
| **3. Restricciones del contexto** | 300 registros (simulados, ≥200), tabular, requisito de interpretabilidad y latencia baja (recálculo diario) | Descarta deep learning y modelos de caja negra pesados. |
| **4. Candidatos y limitaciones** | **Regresión lineal múltiple:** simple, explicable, inferencia estadística (p-values, IC), pero supone linealidad, homocedasticidad y no captura umbrales ni interacciones. **Random Forest:** captura no linealidad e interacciones y resiste outliers, pero no da signos de efecto, extrapolaba mal fuera del rango de entrenamiento y es menos defendible ante auditores. | Se comparan con evidencia en test set. |

## 3. Simulación y exploración de datos

Dataset **simulado** con semilla fija (`scripts/gen_data.py`): **300 registros** (25 meses ×
12 tiendas), con estacionalidad, efectos de marketing, precio y competencia + ruido →
señal aprendible con techo de R² realista.
""")

code(nb, """
%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("data/ventas_simulado.csv")
df["fecha"] = pd.to_datetime(df["fecha"])
print("Dimensiones:", df.shape)
df.head()
""")

code(nb, """
print("Nulos por columna:\\n", df.isna().sum())
df.describe().T
""")

code(nb, """
df["mes"] = df["fecha"].dt.month
plt.figure(figsize=(10, 5))
seas = df.groupby("mes")["ingreso_mensual"].mean()
sns.barplot(x=seas.index, y=seas.values, palette="viridis")
plt.title("Estacionalidad media del ingreso por mes")
plt.xlabel("Mes"); plt.ylabel("Ingreso mensual promedio (USD)")
plt.tight_layout(); plt.show()
""")

code(nb, """
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.regplot(x="gasto_marketing", y="ingreso_mensual", data=df, ci=None,
            scatter_kws={"alpha": .5}, ax=axes[0])
axes[0].set_title("Marketing vs. ingreso")
sns.regplot(x="comportamiento_previo", y="ingreso_mensual", data=df, ci=None,
            scatter_kws={"alpha": .5}, ax=axes[1])
axes[1].set_title("Ingreso del mes anterior vs. ingreso actual")
plt.tight_layout(); plt.show()
""")

code(nb, """
cols_x = ["gasto_marketing", "estacionalidad", "precio_relativo",
          "actividad_competencia", "indicador_macro", "comportamiento_previo"]
plt.figure(figsize=(9, 6))
sns.heatmap(df[cols_x + ["ingreso_mensual"]].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlaciones — Caso B")
plt.tight_layout(); plt.show()
""")

md(nb, """
**Lectura del EDA:** el ingreso muestra estacionalidad clara (picos en meses de campaña),
relación casi lineal con el gasto en marketing y con el comportamiento previo, y correlación
negativa con la actividad de la competencia y con el precio relativo. No hay nulos y las
escalas son manejables.

## 4. Verificación de supuestos del algoritmo seleccionado (linealidad)

La regresión lineal múltiple exige: linealidad, **normalidad de residuos**,
**homocedasticidad** y **ausencia de multicolinealidad**. Los verificamos empíricamente:
""")

code(nb, """
Xt = sm.add_constant(df[cols_x])
ols = sm.OLS(df["ingreso_mensual"], Xt).fit()
print(ols.summary())
""")

code(nb, """
resid = ols.resid

# 1) Normalidad de residuos: histograma + Q-Q plot + test de Shapiro
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(resid, kde=True, ax=axes[0], color="steelblue")
axes[0].set_title("Distribución de residuos")
sm.qqplot(resid, line="45", ax=axes[1])
axes[1].set_title("Q-Q plot de residuos")
plt.tight_layout(); plt.show()

w, p = stats.shapiro(resid.iloc[:500])
print(f"Shapiro-Wilk: W={w:.4f}, p={p:.4f} -> "
      + ("no se descarta normalidad (p>0.05)" if p > 0.05 else "residuos no normales (p<0.05)"))
""")

code(nb, """
# 2) Homocedasticidad: residuo vs. ajustado (sin patrón de embudo = varianza constante)
plt.figure(figsize=(8, 5))
plt.scatter(ols.fittedvalues, resid, alpha=.5, s=25)
plt.axhline(0, color="k", lw=.8)
plt.xlabel("Valores ajustados"); plt.ylabel("Residuos")
plt.title("Residuos vs. ajustados (prueba visual de homocedasticidad)")
plt.grid(alpha=.3); plt.tight_layout(); plt.show()

# 3) Multicolinealidad: VIF
vif = pd.DataFrame({
    "Variable": cols_x,
    "VIF": [variance_inflation_factor(Xt.values, i + 1) for i in range(len(cols_x))],
})
vif
""")

md(nb, """
**Conclusión de supuestos:** los residuos se comportan aproximadamente normales (Q-Q
razonablemente sobre la diagonal), sin embudo visible (homocedasticidad) y con VIF < 10 en
todas las variables → **los supuestos de la regresión lineal se cumplen de forma aceptable**
en estos datos, por lo que sus coeficientes y su inferencia (p-values, IC) son válidos.

## 5. Implementación y comparación de modelos (scikit-learn)
""")

code(nb, """
X = df[cols_x]
y = df["ingreso_mensual"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

lineal = LinearRegression().fit(X_train_s, y_train)
bosque = RandomForestRegressor(n_estimators=400, random_state=42,
                               min_samples_leaf=3).fit(X_train, y_train)
print("Train:", X_train.shape, "| Test:", X_test.shape)
""")

code(nb, """
def evaluar_reg(modelo, Xte, yte, nombre):
    pred = modelo.predict(Xte)
    return {
        "Modelo": nombre,
        "MAE": mean_absolute_error(yte, pred),
        "RMSE": np.sqrt(mean_squared_error(yte, pred)),
        "R²": r2_score(yte, pred),
    }

tabla = pd.DataFrame([
    evaluar_reg(lineal, X_test_s, y_test, "Regresión lineal múltiple"),
    evaluar_reg(bosque, X_test, y_test, "Random Forest"),
]).set_index("Modelo").round(3)
tabla
""")

code(nb, """
# Predicho vs. real, ambos modelos
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (nombre, modelo, Xte) in zip(
        axes, [("Lineal", lineal, X_test_s), ("Random Forest", bosque, X_test)]):
    pred = modelo.predict(Xte)
    ax.scatter(y_test, pred, alpha=.6)
    lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
    ax.plot(lims, lims, "r--", lw=1)
    ax.set_title(f"{nombre} — R² test = {r2_score(y_test, pred):.3f}")
    ax.set_xlabel("Ingreso real"); ax.set_ylabel("Ingreso predicho")
plt.tight_layout(); plt.show()
""")

code(nb, """
# Sobreajuste: R² en train vs. test
brecha = pd.DataFrame({
    "Modelo": ["Regresión lineal múltiple", "Random Forest"],
    "R² TRAIN": [lineal.score(X_train_s, y_train), bosque.score(X_train, y_train)],
    "R² TEST": [lineal.score(X_test_s, y_test), bosque.score(X_test, y_test)],
})
brecha["Brecha"] = brecha["R² TRAIN"] - brecha["R² TEST"]
brecha.round(4)
""")

md(nb, """
## 6. Análisis de límites y riesgos (con evidencia empírica)

1. **Overfitting.** El Random Forest muestra brecha R² train–test de 0.125
   (0.964 en train → 0.839 en test), mientras que la lineal es prácticamente idéntica
   (brecha 0.001). En 225 registros de entrenamiento, el bosque memoriza ruido;
   la evidencia favorece a la lineal.
2. **Sesgo del modelo.** La regresión lineal solo captura efectos aditivos: si en datos
   reales el efecto de marketing tuviera rendimientos decrecientes (curvatura), la lineal
   subestimaría el ingreso en tiendas de gasto alto. Mitigación: variables polinómicas o
   splines, manteniendo explicabilidad.
3. **Datos no representativos.** Al ser datos simulados con una estacionalidad fija, el
   modelo no ha visto disrupciones (pandemia, apertura de competidor). En producción, un
   quiebre estructural invalida el entrenamiento → requiere reentrenamiento y modelos de
   residuo con intervenciones.
4. **Outliers.** Tiendas atípicas (B2B, franquicias ancla) distorsizan la recta de mínimos
   cuadrados; el bosque es más robusto por su segmentación. Mitigación: detección por
   Cook's distance en OLS y verificación visual en scatter de residuos.

## 7. Consideraciones éticas

- **¿Podría generar sesgos o discriminación?** El modelo opera a nivel tienda, no persona,
  pero puede concentrar presupuesto en las tiendas ya grandes y *auto-cumplir* el abandono
  de otras (profecía de pronóstico). Mitigación: revisar asignación presupuestaria con
  criterios de equidad, no solo con el pronóstico.
- **Transparencia para stakeholders.** La lineal entrega descomposición exacta del
  pronóstico (cada coeficiente = USD adicionales por unidad de driver), defendible ante
  finanzas y auditoría; para el bosque se documenta importancia de variables.
- **Privacidad.** Se usan agregados de tienda y macroindicadores, sin datos personales de
  clientes: riesgo de privacidad bajo. Si se incorporara comportamiento histórico a nivel
  cliente (agregado de canasta), aplicar minimización y políticas de acceso.

## 8. Conclusión y recomendación final

**Modelo recomendado: Regresión lineal múltiple**, por evidencia y alineación estratégica:

- **Técnica:** en el set de prueba la lineal domina en las tres métricas exigidas
  (MAE 9.2 vs. 21.7, RMSE 11.8 vs. 27.8, R² 0.971 vs. 0.839) y su brecha train–test es
  de apenas 0.001; los supuestos se cumplen (Shapiro-Wilk p=0.99, sin embudo en residuos,
  VIF < 10), por lo que además entrega intervalos de predicción válidos.
- **Estratégica:** planeación necesita descomponer el pronóstico en drivers (cuánto del
  ingreso esperado viene de campaña vs. estacionalidad) para negociar presupuesto; la
  linealidad lo permite palabra por palabra.
- **Ética:** un modelo explicable evita que el pronóstico se vuelva una caja negra que
  concentra recursos sin justificación auditable.

**Mejoras / siguientes pasos:** validación con series temporales reales (ventanas
rodantes), agregar variables polinómicas o de interacción, cuantificar incertidumbre con
intervalos de pronóstico, y comparar contra un benchmark estacional (ingreso del mismo mes
del año anterior).
""")

nbB_path = PROJ + r"\Caso_B_Ventas_Prediccion.ipynb"
nbf.write(nb, nbB_path)

print("Notebooks escritos:")
print(nbA_path)
print(nbB_path)
