# Paso 7 — Conclusión y recomendación final

## Caso A — Recomendado: **Regresión logística**

- **Técnica:** mejor AUC (0.701 vs. 0.590) y accuracy (0.730 vs. 0.695) en test; F1
  prácticamente idéntico (0.250 vs. 0.265); brecha train–test nula frente a un bosque que
  sobreajusta.
- **Estratégica:** marketing necesita explicar *por qué* cada cliente es riesgoso para
  diseñar la acción y auditarla; los coeficientes dan eso palabra por palabra.
- **Ética:** la interpretabilidad permite auditar sesgos por segmento; un score de caja
  negra no cumple el principio de transparencia.

## Caso B — Recomendado: **Regresión lineal múltiple**

- **Técnica:** domina MAE (9.2 vs. 21.7), RMSE (11.8 vs. 27.8) y R² (0.971 vs. 0.839), con
  supuestos verificados (Shapiro-Wilk p = 0.99, homocedasticidad, VIF < 10) → intervalos de
  pronóstico válidos.
- **Estratégica:** S&OP y finanzas negocian presupuesto por driver (cuánto ingreso aporta
  cada USD de campaña); el bosque solo da importancias sin dirección.
- **Ética:** pronóstico auditable evita decisiones de presupuesto opacas.

## Mejoras / siguientes pasos (ambos casos)

1. Validar con **datos reales** y ventana temporal out-of-time (no solo aleatoria).
2. **Calibrar probabilidades** (Platt/Isotonic) y optimizar el umbral según el costo
   negocio de FP vs. FN.
3. Monitoreo de **drift** de datos y de rendimiento en producción.
4. Si con datos reales el bosque supera por margen, usarlo como motor de predicción y
   explicar con **SHAP** (predicción + explicación post-hoc).
