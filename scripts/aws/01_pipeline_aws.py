"""
pipeline limpieza para aws
"""

import sys

import boto3
import pandas as pd
from awsglue.utils import getResolvedOptions

from etl_utils import (
    ReporteLimpieza,
    quitar_duplicados_exactos,
    normalizar_texto,
    estandarizar_fecha,
    rango_plausible_para_columna,
    estandarizar_tipo_fisica_virtual,
    estandarizar_binario,
    convertir_a_numerico,
    estandarizar_formato_cedula,
    validar_cedula_modulo10,
)

# ---------------------------------------------------------------------------
# 0. Argumentos del job y rutas S3
# ---------------------------------------------------------------------------
args = getResolvedOptions(sys.argv, ["bucket", "raw_prefix", "trusted_prefix", "reports_prefix"])
BUCKET = args["bucket"]
RAW_PREFIX = args.get("raw_prefix", "data/raw")
TRUSTED_PREFIX = args.get("trusted_prefix", "data/trusted")
REPORTS_PREFIX = args.get("reports_prefix", "data/trusted/_reports")

s3 = boto3.client("s3")


def ruta_raw(tabla: str) -> str:
    return f"s3://{BUCKET}/{RAW_PREFIX}/{tabla}/{tabla}.parquet"


def ruta_trusted(tabla: str) -> str:
    return f"s3://{BUCKET}/{TRUSTED_PREFIX}/{tabla}/{tabla}.parquet"


def subir_texto_a_s3(contenido: str, key: str):
    s3.put_object(Bucket=BUCKET, Key=key, Body=contenido.encode("utf-8"))
    print(f"Reporte guardado en: s3://{BUCKET}/{key}")


reportes: list[ReporteLimpieza] = []
flags_negocio: list[dict] = []


def aplicar(func, serie_o_df, nombre_columna, *args_f, **kwargs_f):
    """Wrapper delgado: corre la función, guarda su reporte, devuelve solo el dato limpio."""
    salida = func(serie_o_df, nombre_columna, *args_f, **kwargs_f)
    *valores, reporte = salida
    reportes.append(reporte)
    return valores[0]


def agregar_flag(df: pd.DataFrame, nombre_flag: str, mask: pd.Series, nombre_tabla: str = "") -> pd.Series:
    """Asigna un flag booleano y lo deja registrado en flags_negocio para el resumen final."""
    mask_bool = mask.fillna(False).astype(bool)
    df[nombre_flag] = mask_bool
    n = int(mask_bool.sum())
    total = len(df)
    pct = (n / total * 100) if total else 0.0
    flags_negocio.append({"tabla": nombre_tabla, "flag": nombre_flag, "n": n, "total": total, "pct": pct})
    print(f"[flag] {nombre_tabla}.{nombre_flag}: {n} de {total} filas ({pct:.2f}%)")
    return df[nombre_flag]


# ---------------------------------------------------------------------------
# 1. Carga de datos desde la zona raw
# ---------------------------------------------------------------------------
FILES = ["clientes", "tarjetas", "transacciones", "interacciones_marketing", "catalogo_comercios"]

dfs = {}
for nombre in FILES:
    dfs[nombre] = pd.read_parquet(ruta_raw(nombre))
    print(f"{nombre}: {dfs[nombre].shape[0]:,} filas, {dfs[nombre].shape[1]} columnas")


# ---------------------------------------------------------------------------
# 2. Quitar duplicados 100% exactos en las 5 tablas
# ---------------------------------------------------------------------------
for nombre in list(dfs.keys()):
    dfs[nombre] = aplicar(quitar_duplicados_exactos, dfs[nombre], nombre)

clientes = dfs.get("clientes")
tarjetas = dfs.get("tarjetas")
transacciones = dfs.get("transacciones")
marketing = dfs.get("interacciones_marketing")
comercios = dfs.get("catalogo_comercios")


# ---------------------------------------------------------------------------
# 3. Estandarización columna por columna (mismo orden que el pipeline local)
# ---------------------------------------------------------------------------
# --- fechas ---
clientes["fecha_nacimiento"] = aplicar(estandarizar_fecha, clientes["fecha_nacimiento"], "fecha_nacimiento")
clientes["fecha_registro"] = aplicar(estandarizar_fecha, clientes["fecha_registro"], "fecha_registro")  # bug de copy-paste corregido
tarjetas["fecha_emision"] = aplicar(estandarizar_fecha, tarjetas["fecha_emision"], "fecha_emision")
tarjetas["fecha_activacion"] = aplicar(estandarizar_fecha, tarjetas["fecha_activacion"], "fecha_activacion")
transacciones["fecha"] = aplicar(estandarizar_fecha, transacciones["fecha"], "fecha")
marketing["fecha_contacto"] = aplicar(estandarizar_fecha, marketing["fecha_contacto"], "fecha_contacto")

# --- texto ---
clientes["canal_adquisicion"] = aplicar(normalizar_texto, clientes["canal_adquisicion"], "canal_adquisicion")
clientes["ciudad"] = aplicar(normalizar_texto, clientes["ciudad"], "ciudad")
clientes["estado_cuenta"] = aplicar(normalizar_texto, clientes["estado_cuenta"], "estado_cuenta")

tarjetas["tipo"] = aplicar(normalizar_texto, tarjetas["tipo"], "tipo")
tarjetas["tipo"] = aplicar(estandarizar_tipo_fisica_virtual, tarjetas["tipo"], "tipo")

transacciones["es_devolucion"] = aplicar(normalizar_texto, transacciones["es_devolucion"], "es_devolucion")
transacciones["es_devolucion"] = aplicar(estandarizar_binario, transacciones["es_devolucion"], "es_devolucion")
transacciones["tipo_transaccion"] = aplicar(normalizar_texto, transacciones["tipo_transaccion"], "tipo_transaccion")

marketing["campana"] = aplicar(normalizar_texto, marketing["campana"], "campana")
marketing["canal"] = aplicar(normalizar_texto, marketing["canal"], "canal")
marketing["respondio"] = aplicar(normalizar_texto, marketing["respondio"], "respondio")
marketing["respondio"] = aplicar(estandarizar_binario, marketing["respondio"], "respondio")

comercios["categoria"] = aplicar(normalizar_texto, comercios["categoria"], "categoria")

# --- numérico ---
transacciones["monto"] = aplicar(convertir_a_numerico, transacciones["monto"], "monto")

# --- cédula ---
clientes["cedula"] = aplicar(estandarizar_formato_cedula, clientes["cedula"], "cedula")


# ---------------------------------------------------------------------------
# 4. Flags de reglas de negocio
# ---------------------------------------------------------------------------
agregar_flag(
    clientes, "flag_menor_edad_al_registro",
    (clientes["fecha_registro"] - clientes["fecha_nacimiento"]).dt.days / 365.25 < 18,
    "clientes",
)
agregar_flag(
    clientes, "flag_cedula_invalida",
    ~clientes["cedula"].apply(validar_cedula_modulo10),
    "clientes",
)

agregar_flag(
    tarjetas, "flag_activacion_antes_emision",
    (tarjetas["fecha_activacion"] < tarjetas["fecha_emision"])
    & tarjetas["fecha_activacion"].notna() & tarjetas["fecha_emision"].notna(),
    "tarjetas",
)
agregar_flag(
    tarjetas, "flag_cliente_id_huerfano",
    ~tarjetas["cliente_id"].isin(clientes["cliente_id"]),
    "tarjetas",
)

agregar_flag(
    transacciones, "flag_cliente_id_huerfano",
    ~transacciones["cliente_id"].isin(clientes["cliente_id"]),
    "transacciones",
)
agregar_flag(
    transacciones, "flag_fecha_futura",
    transacciones["fecha"] > pd.Timestamp.now(),
    "transacciones",
)
agregar_flag(
    transacciones, "flag_comercio_codigo_huerfano",
    transacciones["comercio_codigo"].notna() & ~transacciones["comercio_codigo"].isin(comercios["comercio_codigo"]),
    "transacciones",
)

agregar_flag(
    marketing, "flag_fecha_contacto_futura",
    marketing["fecha_contacto"] > pd.Timestamp.now(),
    "interacciones_marketing",
)
agregar_flag(
    marketing, "flag_cliente_id_huerfano",
    ~marketing["cliente_id"].isin(clientes["cliente_id"]),
    "interacciones_marketing",
)


# ---------------------------------------------------------------------------
# 5. Resumen de reportes y flags -> se suben a S3 (no hay filesystem persistente en Glue)
# ---------------------------------------------------------------------------
def imprimir_resumen_reportes(reportes: list[ReporteLimpieza]):
    print(f"Total de funciones corridas: {len(reportes)}")
    con_errores = [r for r in reportes if not r.ok]
    con_advertencias = [r for r in reportes if r.advertencias and r.ok]
    print(f"  con errores inesperados: {len(con_errores)}")
    print(f"  con advertencias (esperable): {len(con_advertencias)}")


imprimir_resumen_reportes(reportes)

texto_reportes = "# Reporte de limpieza y estandarización\n\n"
for r in reportes:
    texto_reportes += "```\n" + r.resumen() + "\n```\n\n"
subir_texto_a_s3(texto_reportes, f"{REPORTS_PREFIX}/reporte_limpieza.md")

texto_flags = "# Resumen de flags de reglas de negocio\n\n"
texto_flags += "| Tabla | Flag | Filas marcadas | Total | % |\n|---|---|---|---|---|\n"
for item in flags_negocio:
    texto_flags += f"| {item['tabla']} | {item['flag']} | {item['n']} | {item['total']} | {item['pct']:.2f}% |\n"
subir_texto_a_s3(texto_flags, f"{REPORTS_PREFIX}/resumen_flags_negocio.md")


# ---------------------------------------------------------------------------
# 6. Guardar tablas procesadas en la zona trusted
# ---------------------------------------------------------------------------
dfs_finales = {
    "clientes": clientes, "tarjetas": tarjetas, "transacciones": transacciones,
    "interacciones_marketing": marketing, "catalogo_comercios": comercios,
}
for nombre, df in dfs_finales.items():
    destino = ruta_trusted(nombre)
    df.to_parquet(destino, index=False)
    print(f"{nombre}: {df.shape[0]:,} filas, {df.shape[1]} columnas -> {destino}")

print("\nJob de limpieza completado.")