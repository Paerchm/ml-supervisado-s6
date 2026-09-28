# Paso 6 — Análisis de límites y riesgos + consideraciones éticas

## Límites y riesgos (con evidencia empírica en los notebooks)

### Overfitting
Medido directamente: **brecha train–test**.
- Caso A: logística −0.007 (nula) vs. Random Forest **0.305** (accuracy 1.000 → 0.695).
- Caso B: lineal 0.001 vs. Random Forest **0.125** (R² 0.964 → 0.839).
Conclusión: con datasets pequeños, la flexibilidad del bosque se paga con varianza.

### Datos no representativos
- Los datos son **simulados**: si el perfil real difiere (p. ej., subrepresentación de
  clientes nuevos o un quiebre estructural tipo pandemia), el rendimiento declarado no se
  sostiene y el umbral óptimo cambia.
- Evidencia visible: recall bajo (~0.16–0.19) con umbral 0.5 en ambos modelos de A → hay
  que recalibrar umbral/class_weight al costo real de retención.
- Mitigación: estratificación, reentrenamiento periódico, validación out-of-time.

### Efecto de outliers
- En la lineal/logística: un gasto o recencia extremo arrastra la recta (minimizar errores
  cuadrados) e infla los coeficientes → detectar con Cook's distance / boxplots de colas,
  winsorizar.
- En el bosque: los outliers distorsionan los splits pero el modelo es más robusto por
  promediar árboles; su riesgo opuesto es extrapolar mal fuera del rango visto.

## Consideraciones éticas

| Pregunta de la rúbrica | Caso A (churn) | Caso B (ventas) |
|---|---|---|
| ¿Sesgos o discriminación? | Sí: puede concentrar retención en clientes rentables y excluir segmentos vulnerables (*proxies* aunque no usemos variables protegidas). Auditar FNR por segmento. | Riesgo menor (nivel tienda), pero el pronóstico puede **auto-cumplirse**: menos presupuesto a tiendas ya débiles. Revisar equidad de la asignación. |
| ¿Transparencia a stakeholders? | Modelo explicable + razón por alerta (coeficiente dominante). No entregar solo un score. | Descomposición exacta del pronóstico en drivers → defendible ante finanzas/auditoría. |
| ¿Privacidad? | Datos conductuales personales: minimización, acceso restringido, anonimización para analítica, aviso al cliente, retención definida. | Solo agregados de tienda e indicadores macro → riesgo bajo; si se usan canastas a nivel cliente, aplicar lo mismo que en A. |
