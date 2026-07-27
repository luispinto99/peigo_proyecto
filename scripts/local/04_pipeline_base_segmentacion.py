# procesmiento base segmentacion

from pathlib import Path
 
import numpy as np
import pandas as pd
 
 
PROCESSED_DIR = Path("./data/processed")
FOR_MODELLING_DIR = Path("./data/for_modelling")
clientes = pd.read_parquet(PROCESSED_DIR / "clientes.parquet")
tarjetas = pd.read_parquet(PROCESSED_DIR / "tarjetas.parquet")
transacciones = pd.read_parquet(PROCESSED_DIR / "transacciones.parquet")
marketing = pd.read_parquet(PROCESSED_DIR / "interacciones_marketing.parquet")

VENTANA_MAX_DIAS = 365       # ventana de observación (o antigüedad, si es menor)
DIAS_POR_MES = 30.44         # para convertir conteos a tasas mensuales
MIN_DIAS_ANTIGUEDAD = 30     # bajo esto, la tasa mensual es demasiado inestable -- se excluye
VALORES_CASH = ["cash in", "cash out"]

# ## 1. Limpiar cada tabla según sus flags -- con conteo explícito de exclusiones
 
def reportar_exclusion(df_original: pd.DataFrame, df_limpio: pd.DataFrame, nombre_tabla: str, motivo: str):
    n_excluidas = len(df_original) - len(df_limpio)
    pct = n_excluidas / len(df_original) * 100 if len(df_original) else 0
    print(f"  [{nombre_tabla}] excluidas por {motivo}: {n_excluidas:,} ({pct:.2f}%)")
 
 
print("Limpieza de tarjetas:")
n0 = len(tarjetas)
tarjetas_limpias = tarjetas[~tarjetas["flag_cliente_id_huerfano"].astype(bool)]
reportar_exclusion(tarjetas, tarjetas_limpias, "tarjetas", "cliente_id huérfano")
tmp = tarjetas_limpias[~tarjetas_limpias["flag_activacion_antes_emision"].astype(bool)]
reportar_exclusion(tarjetas_limpias, tmp, "tarjetas", "activación antes de emisión")
tarjetas_limpias = tmp
print(f"  Total tarjetas: {n0:,} -> {len(tarjetas_limpias):,}\n")
 
print("Limpieza de transacciones:") 
n0 = len(transacciones)
transacciones_limpias = transacciones[~transacciones["flag_cliente_id_huerfano"].astype(bool)]
reportar_exclusion(transacciones, transacciones_limpias, "transacciones", "cliente_id huérfano")
tmp = transacciones_limpias[~transacciones_limpias["flag_fecha_futura"].astype(bool)]
reportar_exclusion(transacciones_limpias, tmp, "transacciones", "fecha futura")
transacciones_limpias = tmp.copy()
print(f"  Total transacciones: {n0:,} -> {len(transacciones_limpias):,}\n")
 
transacciones_limpias["monto_neto"] = np.where(
    transacciones_limpias["es_devolucion"].astype(bool),
    -transacciones_limpias["monto"],
    transacciones_limpias["monto"],
)
 
print("Limpieza de marketing:")
n0 = len(marketing)
marketing_limpia = marketing[~marketing["flag_cliente_id_huerfano"].astype(bool)]
reportar_exclusion(marketing, marketing_limpia, "marketing", "cliente_id huérfano")
tmp = marketing_limpia[~marketing_limpia["flag_fecha_contacto_futura"].astype(bool)]
reportar_exclusion(marketing_limpia, tmp, "marketing", "fecha de contacto futura")
marketing_limpia = tmp.copy()
print(f"  Total marketing: {n0:,} -> {len(marketing_limpia):,}\n")
 
fecha_max_datos = transacciones_limpias["fecha"].max()
print(f"Fecha de referencia 'hoy': {fecha_max_datos.date()}")


# ## 2. Definir población elegible

print("\nDefinición de población elegible:")
n0 = len(clientes)
 
fisicas_activadas = tarjetas_limpias[
    (tarjetas_limpias["tipo"] == "fisica") & tarjetas_limpias["fecha_activacion"].notna()
]
ids_con_fisica_alguna_vez = set(fisicas_activadas["cliente_id"])
poblacion = clientes[~clientes["cliente_id"].isin(ids_con_fisica_alguna_vez)].copy()
print(f"  Excluidos por ya tener/haber tenido física: {n0 - len(poblacion):,}")
 
n1 = len(poblacion) 
poblacion = poblacion[~poblacion["flag_menor_edad_al_registro"].astype(bool)]
print(f"  Excluidos por flag_menor_edad_al_registro: {n1 - len(poblacion):,}")
 
n2 = len(poblacion)
antiguedad_dias = (fecha_max_datos - poblacion["fecha_registro"]).dt.days
poblacion = poblacion[antiguedad_dias >= MIN_DIAS_ANTIGUEDAD]
print(f"  Excluidos por antigüedad < {MIN_DIAS_ANTIGUEDAD} días (tasa mensual inestable): {n2 - len(poblacion):,}")
 
print(f"  Población elegible final: {len(poblacion):,} de {n0:,} clientes totales")



# ## 3. Ventana de observación por cliente (365 días, o desde el registro si es más nuevo)
 
limite_inferior = fecha_max_datos - pd.Timedelta(days=VENTANA_MAX_DIAS)

poblacion["fecha_inicio_ventana"] = np.maximum(
    poblacion["fecha_registro"], 
    pd.Series(limite_inferior, index=poblacion.index)
)

poblacion["dias_observados"] = (fecha_max_datos - poblacion["fecha_inicio_ventana"]).dt.days
poblacion["meses_observados"] = poblacion["dias_observados"] / DIAS_POR_MES
 
print(f"Distribución de meses observados:\n{poblacion['meses_observados'].describe()}")
print(f"Clientes con ventana completa (365 días): "
      f"{(poblacion['dias_observados'] >= VENTANA_MAX_DIAS).sum():,} de {len(poblacion):,}")
 
# ## 4. Comportamiento transaccional en la ventana, como tasas mensuales
 
tx = transacciones_limpias[transacciones_limpias["cliente_id"].isin(poblacion["cliente_id"])].copy()
tx = tx.merge(poblacion[["cliente_id", "fecha_inicio_ventana"]], on="cliente_id", how="inner")
tx_en_ventana = tx[tx["fecha"] >= tx["fecha_inicio_ventana"]]
 
agg_tx = tx_en_ventana.groupby("cliente_id").agg(
    n_transacciones=("tipo_transaccion", "size"),
    monto_total=("monto_neto", "sum"),
    diversidad_tipo=("tipo_transaccion", "nunique"),
    n_cash=("tipo_transaccion", lambda s: s.isin(VALORES_CASH).sum()),
)
poblacion = poblacion.merge(agg_tx, on="cliente_id", how="left")
for col in ["n_transacciones", "monto_total", "diversidad_tipo", "n_cash"]:
    poblacion[col] = poblacion[col].fillna(0)
 
poblacion["frecuencia_mensual"] = poblacion["n_transacciones"] / poblacion["meses_observados"]
poblacion["monto_mensual"] = poblacion["monto_total"] / poblacion["meses_observados"]
# diversidad_tipo se deja como conteo (no como tasa) -- no tiene sentido de "por mes"


# ## 5. Marketing en la ventana, como tasa mensual de contactos
 
mkt = marketing_limpia[marketing_limpia["cliente_id"].isin(poblacion["cliente_id"])].copy()
mkt = mkt.merge(poblacion[["cliente_id", "fecha_inicio_ventana"]], on="cliente_id", how="inner")
mkt_en_ventana = mkt[mkt["fecha_contacto"] >= mkt["fecha_inicio_ventana"]]
 
agg_mkt = mkt_en_ventana.groupby("cliente_id").agg(
    n_contactos=("respondio", "size"),
    tasa_respuesta=("respondio", "mean"),
)
poblacion = poblacion.merge(agg_mkt, on="cliente_id", how="left")
poblacion["n_contactos"] = poblacion["n_contactos"].fillna(0)
poblacion["tasa_respuesta"] = poblacion["tasa_respuesta"].fillna(0)
poblacion["contactos_mensuales"] = poblacion["n_contactos"] / poblacion["meses_observados"]


# ## 6. Estado actual de tarjeta virtual y demográficas (sin ventana -- es "hoy")
 
virtual = tarjetas_limpias[tarjetas_limpias["tipo"] == "virtual"]
virtual = virtual[virtual["cliente_id"].isin(poblacion["cliente_id"])].copy()
virtual_vigente = virtual[virtual["fecha_emision"] < fecha_max_datos]
agg_virtual = virtual_vigente.groupby("cliente_id").agg(
    tiene_virtual_activa=("fecha_activacion", lambda s: s.notna().any()),
    fecha_emision_virtual_min=("fecha_emision", "min"),
)
poblacion = poblacion.merge(agg_virtual, on="cliente_id", how="left")
poblacion["tiene_virtual_activa"] = poblacion["tiene_virtual_activa"].fillna(False)
poblacion["antiguedad_tarjeta_virtual"] = (fecha_max_datos - poblacion["fecha_emision_virtual_min"]).dt.days
poblacion["antiguedad_tarjeta_virtual"] = poblacion["antiguedad_tarjeta_virtual"].fillna(0)
 
poblacion["edad"] = (fecha_max_datos - poblacion["fecha_nacimiento"]).dt.days / 365.25
poblacion["antiguedad_cliente"] = (fecha_max_datos - poblacion["fecha_registro"]).dt.days
 
# bool -> float, por compatibilidad con el clustering/imputación más adelante
poblacion["tiene_virtual_activa"] = poblacion["tiene_virtual_activa"].astype(float)


COLUMNAS_FINALES = [
    "cliente_id",
    "edad", "antiguedad_cliente",
    "frecuencia_mensual", "monto_mensual", "diversidad_tipo",
    "contactos_mensuales", "tasa_respuesta",
    "tiene_virtual_activa", "antiguedad_tarjeta_virtual",
    "ciudad", "canal_adquisicion"
]
base_final = poblacion[COLUMNAS_FINALES].copy()
 
destino = FOR_MODELLING_DIR / "base_segmentacion.parquet"
base_final.to_parquet(destino, index=False)
print(f"\nGuardado en: {destino}")
print(f"Filas: {len(base_final):,}")