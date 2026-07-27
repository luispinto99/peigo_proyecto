# Piloto de tarjetas físicas — Pipeline de datos y modelo de segmentación

El presente proyecto describe un pipeline de punta a punta (ingesta → estandarización → enriquecimiento → análisis/modelo → resultado accionable) para evaluar el piloto de entrega de tarjetas físicas de un banco ecuatoriano, y construir un criterio replicable de priorización para la siguiente ola.

## Resumen ejecutivo

**¿Cómo le fue al piloto?**
El piloto tuvo un efecto causal positivo y estadísticamente significativo tanto en frecuencia (**+7.3 transacciones por cliente en una ventana de 60 días**) como en monto transaccional (**+$62 por cliente en una ventana de 60 días**). De esta manera, se confirma la hipótesis de que la tarjeta física incrementa la actividad transaccional. El resultado es robusto a tres pruebas estadísticas distintas (t-test, Wilcoxon, bootstrap), lo que descarta que esté siendo distorsionado por outliers o supuestos de normalidad.

**¿A quién priorizar en la siguiente ola?**
Se construyó una segmentación (K-means) sobre toda la base de clientes elegibles (quienes nunca han tenido tarjeta física). El segmento prioritario es aquel con un nivel **moderado/alto de frecuencia y monto transaccional**, combinado con **alta tasa de respuesta a marketing**. La razón: alguien con actividad moderada y buena receptividad a campañas es quien más probablemente active y aproveche la tarjeta, sin desperdiciar presupuesto en clientes que no responderían a la oferta.

**Recomendación concreta:** asignar el presupuesto disponible de tarjetas físicas priorizando el segmento `alto_potencial` resultante del modelo (ver sección de Segmentación), disponible consultable vía API.

## Estructura del repositorio

```
├── data/
│   ├── raw/              # datos crudos
│   ├── processed/        # datos limpios y estandarizados (salida del ETL), con flags de calidad
│   └── for_modelling/     # bases construidas para los modelos (inferencia causal y segmentación)
├── models/
│   └── modelo_segmentacion.pkl   # KMeans + preprocesador + mapeo cluster->nombre de prioridad
├── reports/
│   ├── reporte_limpieza.md          # resumen de cada función de limpieza ejecutada
│   └── resumen_flags_negocio.md     # conteo y % de cada flag de regla de negocio
├── scripts/
│   ├── local/            # scripts probados originalmente en local
│   └── aws/              # versiones adaptadas para correr en Glue/Lambda (I/O a S3)
└── exploracion_inicial.ipynb   # notebook de chequeo/perfilado inicial, previo a los scripts productivos
```

## Arquitectura en AWS

```
S3 (bucket: peigo)
├── data/raw/<tabla>/<tabla>.parquet              -- zona raw, catalogada con Glue Crawler
├── data/trusted/<tabla>/<tabla>.parquet          -- zona limpia, salida del Glue ETL Job
├── data/trusted/_reports/                        -- reportes de limpieza y flags
├── data/for_modelling/<base>/<base>.parquet       -- bases para los modelos
└── models/modelo_segmentacion.pkl                -- modelo entrenado

Glue Data Catalog: piloto_tarjetas_raw, piloto_tarjetas_trusted (consultables vía Athena)

Glue Jobs (Python Shell):
  - job_limpieza_piloto_tarjetas           raw -> trusted (limpieza + flags)
  - glue_job_modelo_segmentacion           for_modelling -> models + resultado final

Disponibilización: API Gateway (HTTP API) + Lambda
  - GET  /priorizacion?cliente_id=X   -> resultado precomputado (segmento/score de un cliente ya en la base)
  - POST /score                       -> inferencia en vivo con el modelo real (cliente nuevo, fuera del batch)
```

## Cómo correr el proyecto

### Local (exploración y desarrollo)
```bash
pip install pandas numpy scikit-learn scipy pyarrow

# 1. Perfilado inicial
jupyter notebook exploracion_inicial.ipynb

# 2. Limpieza y estandarización (lee data/raw/, escribe data/processed/)
python scripts/local/01_pipeline.py

# 3. Construcción de base + modelo causal (evalúa el piloto)
python scripts/local/03_base_modelo.py
python scripts/local/04_modelo_causal.py

# 4. Construcción de base + modelo de segmentación (prioriza la siguiente ola)
python scripts/local/05_base_segmentacion.py
python scripts/local/06_segmentacion.py
```

### AWS (producción)
1. Sube `data/raw/` a `s3://peigo/data/raw/<tabla>/<tabla>.parquet` (una subcarpeta por tabla).
2. Corre el Glue Crawler sobre `raw/` para catalogarla.
3. Sube `scripts/aws/etl_utils.py` y `scripts/aws/glue_job_limpieza.py` a `s3://peigo/scripts/`.
4. Crea y corre el Glue Job de limpieza (Python Shell, `--additional-python-modules pyarrow,s3fs,boto3`, `--extra-py-files` apuntando a `etl_utils.py`) → escribe `data/trusted/`.
5. Corre un segundo Glue Crawler sobre `trusted/`.
6. Corre el Glue Job de construcción de base (`--additional-python-modules pyarrow,s3fs,boto3`) → escribe `data/for_modelling/`.
7. Corre el Glue Job del modelo de segmentación (agrega `scikit-learn,scipy` a los módulos) → escribe `models/modelo_segmentacion.pkl` y el resultado final segmentado.
8. Despliega la Lambda (`scripts/aws/lambda_function.py`) con el layer administrado `AWSSDKPandas-Python<version>`, permisos de `s3:GetObject` sobre `data/for_modelling/*` y `models/*`.
9. Conecta la Lambda a un API Gateway (HTTP API) con las rutas `GET /priorizacion` y `POST /score`.

## Problemas de calidad de datos encontrados y cómo se resolvieron

Como diagnóstico general: todas las columnas llegaron como texto (`object`), salvo `estado_codigo` de tarjetas — se requirió parseo/estandarización de tipos en prácticamente toda la base, además de una fuerte inconsistencia de formatos en las columnas de fecha.

### Tabla `clientes` (12,048 filas - 12,000 tras limpieza)
- **48 filas 100% duplicadas** — eliminadas.
- `cliente_id` tiene 12,000 valores únicos (llave correcta); `cedula` tiene 11,940 (60 cédulas duplicadas en 120 registros). Se investigó cada par: los demás campos son completamente distintos entre sí, lo que descarta que sean la misma persona registrada dos veces. es más consistente con cédulas mal capturadas o inválidas. Por eso **se mantiene `cliente_id` como único nivel de granularidad**, no se intentó deduplicar por cédula.
- `cedula` se estandarizó (se quitaron guiones, se completó el cero inicial faltante en casos de 9 dígitos) y se validó con el algoritmo de módulo 10 para persona natural ecuatoriana.
- `canal_adquisicion` y `estado_cuenta` requirieron normalización de mayúsculas/espacios y tratamiento de placeholders de nulo (`"n/a"`, `"null"`, etc.) como NA real.

### Tabla `tarjetas` (16,634 filas - 16,585 tras limpieza)
- **49 filas 100% duplicadas** — eliminadas.
- `tipo` tenía múltiples variantes de escritura para las mismas dos categorías (física/virtual), se estandarizó clasificando por prefijo (`f*` = física, `v*` = virtual), tolerante a acentos y mayúsculas.
- `estado_codigo` es numérico (1/2/3) sin diccionario de datos provisto, **queda pendiente decodificar su significado real** (se intentó inferir cruzando con `fecha_activacion` y actividad transaccional, sin llegar a confirmación definitiva en el alcance de esta prueba).

### Tabla `transacciones` (194,173 filas - 193,015 tras limpieza)
- **1,158 filas 100% duplicadas** - eliminadas.
- `tipo_transaccion` requirió normalización de mayúsculas/guiones bajos (ej. `"CASH_IN"`, `"Cash In"`, `"cash_in"` - una sola categoría).
- `comercio_codigo` solo está poblado para transacciones de tipo `compra_tarjeta` — es un campo condicionalmente aplicable, no un problema de calidad (el resto de tipos de transacción no tienen comercio asociado por diseño).
- `es_devolucion` se estandarizó a booleano real (de valores mixtos `'0'/'1'/'Si'/'No'/'True'/'False'`).
- Se descubrió, cruzando con `tarjetas`, que **`compra_tarjeta` es prácticamente exclusiva de clientes con física** (0.3% de las transacciones de clientes solo-virtuales vs. 30.4% en clientes con física) - mientras que `cash_in`/`cash_out` no dependen de tener tarjeta física (proporciones similares en ambos grupos). Este hallazgo fue clave para descartar un supuesto inicial equivocado sobre qué variable debía usarse como proxy de "necesidad de tarjeta física" en la segmentación.

### Tabla `interacciones_marketing` (36,000 filas, sin duplicados)
- `campana` y `canal` requirieron normalización de mayúsculas/espacios.
- `respondio` se estandarizó a booleano real.

### Tabla `catalogo_comercios` (42 filas, sin duplicados)
- Sin hallazgos de calidad relevantes.

### Flags de reglas de negocio (cruces entre columnas/tablas)
Se agregaron como columnas explícitas, sin anular ni eliminar ningún dato.

1. `flag_menor_edad_al_registro`: por definición, un menor de edad no puede recibir una tarjeta de forma independiente.
2. `flag_cedula_invalida`: cédula que no pasa el checksum de módulo 10.
3. `flag_activacion_antes_emision`: tarjetas activadas antes de haber sido emitidas (fecha ilógica).
4. `flag_fecha_futura` / `flag_fecha_contacto_futura`: transacciones o contactos de marketing con fecha posterior al presente.
5. `flag_cliente_id_huerfano` / `flag_comercio_codigo_huerfano`: IDs foráneos que no existen en su tabla maestra correspondiente.

## Enfoque de modelo y decisiones tomadas

Se separaron **dos modelos con propósitos distintos**, entrenados sobre poblaciones distintas a propósito:

### 1. Modelo de inferencia causal (¿funcionó el piloto?)
Se comparó la actividad transaccional de quienes **activaron** la tarjeta física durante la campaña del piloto contra quienes fueron contactados pero **no la activaron**. Se excluyó a quien ya tenía física activa **antes** de ser contactado pues esta tarjeta no estaría afectada por la campaña.

Para la base del modelo:
- De `interacciones_marketing` se mantuvieron solo registros sin flags, tomando el **primer contacto** por cliente como referencia (ancla temporal).
- De `tarjetas` se tomó, por cliente, la **primera activación válida** (sin flags) de tarjeta física.
- De `transacciones` se excluyeron registros con flags, y se **invirtió el signo de las devoluciones** (una devolución revierte el efecto económico de la transacción original).
- La ventana de análisis pre/post activación se fijó en **60 días** — es el período más largo que no reduce la cobertura de clientes válidos (con suficiente antigüedad para el "pre" y suficiente tiempo transcurrido para el "post" dentro del rango de datos disponible); ventanas mayores (90-120 días) hacían caer la cobertura de forma pronunciada por la recencia del piloto.

Técnica: T-learner (dos regresiones Ridge, una entrenada solo en tratamiento y otra solo en control), lo que permite tanto estimar el efecto promedio (contra el contrafactual correcto).

**Resultado:** efecto positivo y significativo en frecuencia (+7.3 transacciones/60 días) y monto (+$62/60 días, tras controlar outliers), confirmado con t-test, Wilcoxon y bootstrap.

**Limitación reconocida:** el piloto compara activadores vs. no-activadores dentro de la misma cohorte contactada, lo cual controla la exposición a marketing como confusor, pero la activación en sí fue una decisión del cliente, no aleatorizada. El efecto estimado puede estar parcialmente inflado por autoselección (clientes más comprometidos tienden tanto a activar la tarjeta como a transaccionar más). Se recomienda, para futuros pilotos, asignar la tarjeta física de forma aleatoria dentro de la cohorte elegible para poder aislar el efecto causal con mayor confianza.

### 2. Modelo de segmentación (¿a quién priorizar?)
Este modelo fue entrenado sobre toda la base de clientes sin tarjeta física:

- Ventana de observación de comportamiento: hasta 365 días, contados desde la fecha del último registro de transacciones.
- Técnica: K-means, con k elegido por silhouette score (probado entre 3 y 6 clusters).
- **Criterio de priorización** (aplicado sobre los clusters ya formados): el cluster prioritario es el que combina un nivel **moderado/alto de frecuencia y monto**, junto con **alta tasa de respuesta a marketing**. La lógica de negocio:
  - Un nivel moderado/alto de actividad: se busca a quien tiene actividad real pero también espacio para crecer con la fricción reducida que da la tarjeta física (ej. poder comprar en comercios físicos, categoría que con tarjeta virtual está limitada).
  - Alta tasa de respuesta a marketing, para no desperdiciar presupuesto en clientes que típicamente no responden a campañas y por tanto no activarían/usarían la tarjeta asignada.
- El nombre del cluster prioritario (`alto_potencial`, etc.) se deriva **dinámicamente** en cada entrenamiento a partir de un ranking sobre los centroides.

## Disponibilización

El resultado final (`clientes_segmentados.parquet`) y el modelo entrenado (`modelo_segmentacion.pkl`) se guardan en S3 y se exponen vía API Gateway + Lambda:

```bash
# Resultado precomputado de un cliente ya en la base
curl "https://vlz8zpvdya.execute-api.us-east-2.amazonaws.com/default/priorizacion?cliente_id=CHK-000123"

# Inferencia en vivo para un cliente nuevo (fuera del batch), con el modelo real
curl -X POST "https://vlz8zpvdya.execute-api.us-east-2.amazonaws.com/default/score" \
  -H "Content-Type: application/json" \
  -d '{"edad": 35, "antiguedad_cliente": 400, "frecuencia_mensual": 9.0, "monto_mensual": 450.0, "diversidad_tipo": 4, "contactos_mensuales": 1.2, "tasa_respuesta": 0.85, "tiene_virtual_activa": 1.0, "antiguedad_tarjeta_virtual": 400, "ciudad": "quito", "canal_adquisicion": "app"}'
```

*(Ver capturas de la respuesta real en la sección de evidencia del entregable.)*
