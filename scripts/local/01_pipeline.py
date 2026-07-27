# # Pipeline de limpieza — Piloto de tarjetas físicas
#
# Orquesta las funciones de `etl_functions.py` sobre las 5 tablas.

from pathlib import Path
 
import pandas as pd
 
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
    validar_cedula_modulo10
)


# ## 1. Carga de datos 

DATA_DIR = Path("./data/raw")
REPORT_DIR = Path("./reports")
REPORT_DIR.mkdir(exist_ok=True)
PROCESSED_DIR = Path("./data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
 
FILES = {
    "clientes": DATA_DIR / "clientes.parquet",
    "tarjetas": DATA_DIR / "tarjetas.parquet",
    "transacciones": DATA_DIR / "transacciones.parquet",
    "interacciones_marketing": DATA_DIR / "interacciones_marketing.parquet",
    "catalogo_comercios": DATA_DIR / "catalogo_comercios.parquet",
}
 
# Acumulador de reportes de TODAS las funciones que se corran, en orden.
reportes: list[ReporteLimpieza] = []
 
 
def aplicar(func, serie_o_df, nombre_columna, *args, **kwargs):
    """
    Wrapper delgado: corre la función, guarda su reporte en `reportes`,
    y devuelve solo el dato limpio (para no repetir el manejo de tupla
    en cada línea de abajo).
    """
    salida = func(serie_o_df, nombre_columna, *args, **kwargs)
    *valores, reporte = salida
    reportes.append(reporte)
 
    dato_principal = valores[0]
 
    return dato_principal

dfs = {}
for nombre, path in FILES.items():
    if not path.exists():
        print(f"No se encontró el archivo esperado: {path}")
        continue
    dfs[nombre] = pd.read_parquet(path)
    print(f"{nombre}: {dfs[nombre].shape[0]:,} filas, {dfs[nombre].shape[1]} columnas")


# ## 2. Quitar duplicados 100% exactos en las 5 tablas

for nombre in list(dfs.keys()):
    dfs[nombre] = aplicar(quitar_duplicados_exactos, dfs[nombre], nombre)
 
clientes = dfs.get("clientes")
tarjetas = dfs.get("tarjetas")
transacciones = dfs.get("transacciones")
marketing = dfs.get("interacciones_marketing")
comercios = dfs.get("catalogo_comercios")


# ## 3. Estandarización columna por columna

# ESTANDARIZACIÓN DE FECHAS

clientes["fecha_nacimiento"] = aplicar(
     estandarizar_fecha, clientes["fecha_nacimiento"], "fecha_nacimiento"
)

clientes["fecha_registro"] = aplicar(
     estandarizar_fecha, clientes["fecha_registro"], "fecha_registro"
)

tarjetas["fecha_emision"] = aplicar(
     estandarizar_fecha, tarjetas["fecha_emision"], "fecha_emision"
)

tarjetas["fecha_activacion"] = aplicar(
     estandarizar_fecha, tarjetas["fecha_activacion"], "fecha_activacion"
)

transacciones["fecha"] = aplicar(
     estandarizar_fecha, transacciones["fecha"], "fecha"
)

marketing["fecha_contacto"] = aplicar(
     estandarizar_fecha, marketing["fecha_contacto"], "fecha_contacto"
)


# ESTANDARIZACIÓN DE TEXTO

clientes["canal_adquisicion"] = aplicar(
     normalizar_texto, clientes["canal_adquisicion"], "canal_adquisicion"
)

clientes["ciudad"] = aplicar(
     normalizar_texto, clientes["ciudad"], "ciudad"
)

clientes["estado_cuenta"] = aplicar(
     normalizar_texto, clientes["estado_cuenta"], "estado_cuenta"
)

tarjetas["tipo"] = aplicar(
     normalizar_texto, tarjetas["tipo"], "tipo"
)

tarjetas["tipo"] = aplicar(
     estandarizar_tipo_fisica_virtual, tarjetas["tipo"], "tipo"
)

transacciones["es_devolucion"] = aplicar(
     normalizar_texto, transacciones["es_devolucion"], "es_devolucion"
)

transacciones["es_devolucion"] = aplicar(
     estandarizar_binario, transacciones["es_devolucion"], "es_devolucion"
)

transacciones["tipo_transaccion"] = aplicar(
     normalizar_texto, transacciones["tipo_transaccion"], "tipo_transaccion"
)

marketing["campana"] = aplicar(
     normalizar_texto, marketing["campana"], "campana"
)

marketing["canal"] = aplicar(
     normalizar_texto, marketing["canal"], "canal"
)

marketing["respondio"] = aplicar(
     normalizar_texto, marketing["respondio"], "respondio"
)

marketing["respondio"] = aplicar(
     estandarizar_binario, marketing["respondio"], "respondio"
)

comercios["categoria"] = aplicar(
     normalizar_texto, comercios["categoria"], "categoria"
)


# Transformar strings a numericos  

transacciones["monto"] = aplicar(
     convertir_a_numerico, transacciones["monto"], "monto"
)


# ESTANDARIZACION DE CÉDULA

clientes["cedula"] = aplicar(
     estandarizar_formato_cedula, clientes["cedula"], "cedula"
)


# CREACIÓN DE FLAGS
flags_negocio: list[dict] = [] 
 
def agregar_flag(df: pd.DataFrame, nombre_flag: str, mask: pd.Series, nombre_tabla: str = "") -> pd.Series:
    """
    Asigna un flag booleano a df[nombre_flag] y lo deja registrado en
    `flags_negocio` para el resumen final.
    """
    mask_bool = mask.fillna(False).astype(bool)
    df[nombre_flag] = mask_bool
    n = int(mask_bool.sum())
    total = len(df)
    pct = (n / total * 100) if total else 0.0
    flags_negocio.append({"tabla": nombre_tabla, "flag": nombre_flag, "n": n, "total": total, "pct": pct})
    print(f"[flag] {nombre_tabla}.{nombre_flag}: {n} de {total} filas ({pct:.2f}%)")
    return df[nombre_flag]
 
 
# --- clientes ---
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
 
# --- tarjetas ---
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
 
# --- transacciones ---
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
 
# --- interacciones_marketing ---
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


#====================================================================#
# imprimir resumen

def imprimir_resumen_reportes(reportes: list[ReporteLimpieza]):
    print(f"Total de funciones corridas: {len(reportes)}")
    con_errores = [r for r in reportes if not r.ok]
    con_advertencias = [r for r in reportes if r.advertencias and r.ok]
    print(f"  con errores inesperados: {len(con_errores)}")
    print(f"  con advertencias (esperable): {len(con_advertencias)}")
    for r in reportes:
        print("\n" + r.resumen())
 
 
imprimir_resumen_reportes(reportes)

def guardar_reportes_md(reportes: list[ReporteLimpieza], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Reporte de limpieza y estandarización\n\n")
        for r in reportes:
            f.write("```\n" + r.resumen() + "\n```\n\n")
    print(f"Reportes guardados en: {path}")
 
 
guardar_reportes_md(reportes, REPORT_DIR / "reporte_limpieza.md")


def guardar_flags_md(flags_negocio: list[dict], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Resumen de flags de reglas de negocio\n\n")
        f.write("| Tabla | Flag | Filas marcadas | Total | % |\n")
        f.write("|---|---|---|---|---|\n")
        for item in flags_negocio:
            f.write(f"| {item['tabla']} | {item['flag']} | {item['n']} | {item['total']} | {item['pct']:.2f}% |\n")
    print(f"Resumen de flags guardado en: {path}")
 
 
guardar_flags_md(flags_negocio, REPORT_DIR / "resumen_flags_negocio.md")


# GUARDAR TABLAS PROCESADAS
 
for nombre, df in dfs.items():
    destino = PROCESSED_DIR / f"{nombre}.parquet"
    df.to_parquet(destino, index=False)
    print(f"{nombre}: {df.shape[0]:,} filas, {df.shape[1]} columnas -> {destino}")