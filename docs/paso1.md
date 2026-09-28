# Paso 1 — Descripción del problema de negocio

## Caso A: Predicción de abandono (churn)

Una cadena de retail pierde clientes de forma silenciosa. El objetivo es **anticuar quién se
va antes de que se vaya**, para dirigir campañas de retención en lugar de descuentos masivos.

- **Variable objetivo (Y):** `abandono` → **categórica binaria** (1 = abandonó, 0 = permanece)
  → problema de **clasificación**.
- **Variables independientes (X):** frecuencia de compra, gasto promedio, días desde la última
  compra, interacción con promociones, quejas en 6 meses, uso de canales digitales.
- **Decisiones del negocio:** marketing prioriza a quién contactar; gerencia monitorea la tasa
  de churn; atención al cliente recibe alertas de clientes con quejas recurrentes.
- **Error costoso:** falso negativo (cliente que se va y se clasifica como leal) = oportunidad
  de retención perdida.

## Caso B: Predicción de ventas futuras

La cadena debe presupuestar inventario, personal y marketing para el mes siguiente.

- **Variable objetivo (Y):** `ingreso_mensual` → **continua** en USD → problema de **regresión**.
- **Variables independientes (X):** gasto en campañas, estacionalidad, precio relativo vs.
  competencia, actividad de la competencia, indicador macroeconómico y comportamiento histórico.
- **Decisiones:** S&OP fija compras por tienda; finanzas proyecta flujo de caja; marketing
  evalúa retorno de campaña.
