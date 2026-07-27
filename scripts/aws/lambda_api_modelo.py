"""
lambda_function.py

Dos formas de consumir el trabajo de segmentación:

  GET  /priorizacion?cliente_id=X
      Sirve el RESULTADO precomputado (batch) -- consulta directa a
      clientes_segmentados.parquet. Cubre la decisión de negocio actual:
      priorizar sobre la base ya existente.

  POST /score  (body JSON con las features crudas de un cliente)
      Sirve el MODELO real -- carga modelo_segmentacion.pkl (KMeans +
      preprocesador ya entrenados) y calcula el segmento EN VIVO. Cubre el
      caso de un cliente nuevo que todavía no pasó por ningún batch.

Requiere el layer administrado "AWSSDKPandas-Python<version>" (pandas +
pyarrow + numpy + scikit-learn NO vienen en ese layer -- si se usa POST
/score, agregar también un layer o capa con scikit-learn, o empaquetar la
función como contenedor Docker en vez de zip).
"""

import json
import pickle
from io import BytesIO

import boto3
import pandas as pd

BUCKET = "peigo"
KEY_RESULTADO = "data/for_modelling/clientes_segmentados/clientes_segmentados.parquet"
KEY_MODELO = "models/modelo_segmentacion.pkl"

# Cache a nivel de contenedor: mientras la Lambda siga "caliente", no vuelve
# a descargar el parquet/pickle en cada invocación.
_cache = {"df": None, "modelo": None}


def _cargar_resultado() -> pd.DataFrame:
    if _cache["df"] is None:
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=BUCKET, Key=KEY_RESULTADO)
        _cache["df"] = pd.read_parquet(BytesIO(obj["Body"].read()))
        print(f"Resultado cargado en memoria: {len(_cache['df']):,} clientes")
    return _cache["df"]


def _cargar_modelo() -> dict:
    if _cache["modelo"] is None:
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=BUCKET, Key=KEY_MODELO)
        _cache["modelo"] = pickle.loads(obj["Body"].read())
        print("Modelo cargado en memoria")
    return _cache["modelo"]


def _respuesta(status: int, body: dict):
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body, default=str),
    }


def _manejar_get(event) -> dict:
    """Sirve el resultado precomputado por cliente_id."""
    params = event.get("queryStringParameters") or {}
    cliente_id = params.get("cliente_id")

    if not cliente_id:
        return _respuesta(400, {"error": "falta el parámetro cliente_id"})

    try:
        df = _cargar_resultado()
    except Exception as e:
        return _respuesta(500, {"error": f"no se pudo cargar el resultado: {e}"})

    fila = df[df["cliente_id"].astype(str) == str(cliente_id)]
    if fila.empty:
        return _respuesta(404, {"error": f"cliente_id {cliente_id} no encontrado en la priorización"})

    return _respuesta(200, fila.iloc[0].to_dict())


def _manejar_post(event) -> dict:
    """Calcula el segmento EN VIVO para un cliente con features crudas (no necesita estar en el batch)."""
    try:
        body = json.loads(event.get("body") or "{}")
    except json.JSONDecodeError:
        return _respuesta(400, {"error": "body no es JSON válido"})

    try:
        artefactos = _cargar_modelo()
    except Exception as e:
        return _respuesta(500, {"error": f"no se pudo cargar el modelo: {e}"})

    columnas_x = artefactos["columnas_x"]
    faltantes = [c for c in columnas_x if c not in body]
    if faltantes:
        return _respuesta(400, {"error": f"faltan variables requeridas: {faltantes}"})

    fila = pd.DataFrame([{c: body[c] for c in columnas_x}])

    try:
        X_transformado = artefactos["preprocesador"].transform(fila)
        if hasattr(X_transformado, "toarray"):
            X_transformado = X_transformado.toarray()
        cluster_id = int(artefactos["kmeans"].predict(X_transformado)[0])
    except Exception as e:
        return _respuesta(400, {"error": f"no se pudo calcular el cluster: {e}"})

    segmento = artefactos["mapa_cluster_a_nombre"].get(cluster_id, "desconocido")

    # puntaje de prioridad, con las medias/stds de la POBLACIÓN DE ENTRENAMIENTO
    # (guardadas en el pickle) -- nunca recalcular la media/std sobre un solo
    # cliente nuevo, eso no tendría sentido estadístico.
    puntaje = 0.0
    for col in artefactos["variables_prioridad"]:
        media = artefactos["medias_poblacion"][col]
        std = artefactos["stds_poblacion"][col]
        puntaje += (body[col] - media) / std if std else 0.0

    return _respuesta(200, {
        "cluster_id": cluster_id,
        "segmento_prioridad": segmento,
        "puntaje_prioridad_individual": round(puntaje, 4),
    })


def lambda_handler(event, context):
    metodo = event.get("requestContext", {}).get("http", {}).get("method", "GET")
    if metodo == "POST":
        return _manejar_post(event)
    return _manejar_get(event)