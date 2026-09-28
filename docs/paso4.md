# Paso 4 — Verificación de supuestos

La rúbrica exige verificar al menos un supuesto del algoritmo seleccionado. Se verifican más.

## Caso A — Regresión logística: ausencia de multicolinealidad (VIF)

VIF de los 6 predictores (resultados reales del notebook):

| Variable | VIF |
|---|---|
| compra_frecuencia | 2.20 |
| interaccion_promo | 2.19 |
| gasto_promedio | 1.01 |
| dias_ultima_compra | 1.00 |
| quejas_6m | 1.00 |
| canales_digitales | 1.00 |

**Conclusión:** todos < 10 → sin multicolinealidad severa; los coeficientes son estables e
interpretables. (Nota: `interaccion_promo` y `compra_frecuencia` comparten leve correlación
constructual, dentro de rango seguro.)

## Caso B — Regresión lineal múltiple: tres supuestos

1. **Normalidad de residuos:** Q-Q plot sobre la diagonal + Shapiro-Wilk **W = 0.998,
   p = 0.989 > 0.05** → no se rechaza normalidad.
2. **Homocedasticidad:** scatter residuo vs. ajustado sin patrón de embudo → varianza
   constante.
3. **Multicolinealidad:** VIF entre 1.00 y 1.04 para las 6 variables → sin colinealidad.

**Conclusión:** los supuestos se cumplen de forma aceptable → inferencia (p-values, IC)
válida y el modelo lineal es una elección legítima para la comparación.
