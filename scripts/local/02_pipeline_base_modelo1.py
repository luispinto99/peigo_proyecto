# PIPELINE PARA CONSTRUIR LA BASE PARA EL MODELO DE INFERENCIA CAUSAL

from pathlib import Path
 
import numpy as np
import pandas as pd

PROCESSED_DIR = Path("./data/processed")
FOR_MODELLING_DIR = Path("./data/for_modelling")
VALOR_CAMPANA_PILOTO = "piloto tarjeta fisica q2"
VENTANA_DIAS = 60
VALORES_CASH = ["cash in", "cash out"]
 
clientes = pd.read_parquet(PROCESSED_DIR / "clientes.parquet")
tarjetas = pd.read_parquet(PROCESSED_DIR / "tarjetas.parquet")
transacciones = pd.read_parquet(PROCESSED_DIR / "transacciones.parquet")
marketing = pd.read_parquet(PROCESSED_DIR / "interacciones_marketing.parquet")
comercios = pd.read_parquet(PROCESSED_DIR / "catalogo_comercios.parquet")

for nombre, df in [("clientes", clientes), ("tarjetas", tarjetas), ("transacciones", transacciones),
                    ("marketing", marketing), ("comercios", comercios)]:
    print(f"{nombre}: {df.shape[0]:,} filas, {df.shape[1]} columnas -> {df.columns.tolist()}")


marketing_limpia_temp = marketing[
    (~marketing["flag_cliente_id_huerfano"].astype(bool))
    & (~marketing["flag_fecha_contacto_futura"].astype(bool))
]

cohorte_piloto = (
    marketing_limpia_temp[marketing_limpia_temp["campana"] == VALOR_CAMPANA_PILOTO]
    .groupby("cliente_id")["fecha_contacto"]
    .min()  # si hubo más de un contacto de esta campaña, nos quedamos con el primero
    .rename("fecha_contacto_piloto")
)
print(f"Clientes contactados en la campaña del piloto: {len(cohorte_piloto):,}")

clientes = clientes[clientes["cliente_id"].isin(cohorte_piloto.index)].copy()
clientes = clientes.merge(cohorte_piloto, left_on="cliente_id", right_index=True, how="left")
print(f"Población del análisis causal (cohorte del piloto): {len(clientes):,}")


fisicas_activadas = tarjetas[
    (tarjetas["tipo"] == "fisica")
    & tarjetas["fecha_activacion"].notna()
    & (~tarjetas["flag_activacion_antes_emision"].astype(bool))   # descarta fechas ilógicas
    & (~tarjetas["flag_cliente_id_huerfano"].astype(bool))           # descarta cliente_id que no existe en clientes
]
# si un cliente tiene más de una tarjeta física, usamos la PRIMERA activación
t0_tratamiento = fisicas_activadas.groupby("cliente_id")["fecha_activacion"].min()
clientes["fecha_activacion_fisica"] = clientes["cliente_id"].map(t0_tratamiento)

# excluir a quien YA tenía física activa ANTES de ser contactado -- esa
# física no la causó el piloto, y no puede contar ni como tratamiento ni
# como control limpio
clientes["ya_tenia_fisica_antes_del_piloto"] = (
    clientes["fecha_activacion_fisica"].notna()
    & (clientes["fecha_activacion_fisica"] < clientes["fecha_contacto_piloto"])
)

n_excluidos_previos = int(clientes["ya_tenia_fisica_antes_del_piloto"].sum())
print(f"Excluidos: ya tenían física activa ANTES del contacto del piloto: {n_excluidos_previos:,}")
clientes = clientes[~clientes["ya_tenia_fisica_antes_del_piloto"]].copy()
 
clientes["T"] = clientes["fecha_activacion_fisica"].notna().astype(int)
print(f"Tratamiento (T=1): {clientes['T'].sum():,} clientes")
print(f"Control (T=0): {(clientes['T'] == 0).sum():,} clientes")
 
clientes["t0"] = clientes["fecha_activacion_fisica"]
mask_control = clientes["t0"].isna()
 
mediana_dias_activacion = (
    clientes.loc[clientes["T"] == 1, "t0"] - clientes.loc[clientes["T"] == 1, "fecha_contacto_piloto"]
).dt.days.median()
print(f"Mediana días contacto->activación (grupo tratamiento): {mediana_dias_activacion:.0f}")
 
clientes.loc[mask_control, "t0"] = (
    clientes.loc[mask_control, "fecha_contacto_piloto"] + pd.Timedelta(days=mediana_dias_activacion)
)


# ## Limpiar transacciones de filas huérfanas/con fecha futura

transacciones_limpias = transacciones[
    (~transacciones["flag_cliente_id_huerfano"].astype(bool))
    & (~transacciones["flag_fecha_futura"].astype(bool))
].copy()
 
# poner negativo el valor de la devolución, puesto que es una transacción no válida
transacciones_limpias["monto_neto"] = np.where(
    transacciones_limpias["es_devolucion"].astype(bool),
    -transacciones_limpias["monto"],
    transacciones_limpias["monto"],
)
 
print(f"Transacciones excluidas por huérfano/fecha futura: "
      f"{len(transacciones) - len(transacciones_limpias):,} de {len(transacciones):,}")

fecha_min_datos = transacciones_limpias["fecha"].min()
fecha_max_datos = transacciones_limpias["fecha"].max()
print(f"Rango de fechas disponible en transacciones: {fecha_min_datos.date()} a {fecha_max_datos.date()}")

for candidato in [30, 45, 60, 90, 120]:
    pre_ok = (clientes["t0"] - pd.Timedelta(days=candidato)) >= clientes["fecha_registro"]
    post_ok = (clientes["t0"] + pd.Timedelta(days=candidato)) <= fecha_max_datos
    validos = (pre_ok & post_ok & clientes["t0"].notna()).sum()
    print(f"VENTANA_DIAS={candidato}: {validos:,} de {len(clientes):,} clientes con ventana válida "
          f"({validos/len(clientes)*100:.1f}%)")


# ## 2. Validar que la ventana pre/post esté completa dentro del rango de datos
#
# Si t0 es muy reciente (cerca del final de los datos disponibles), el
# período "post" queda cortado y el delta sale artificialmente bajo -- mejor
# excluir esos clientes de la base de modelo que usar un delta incompleto.
 
fecha_min_datos = transacciones_limpias["fecha"].min()
fecha_max_datos = transacciones_limpias["fecha"].max()
print(f"Rango de fechas disponible en transacciones: {fecha_min_datos.date()} a {fecha_max_datos.date()}")
 
clientes["ventana_pre_completa"] = (
    clientes["t0"] - pd.Timedelta(days=VENTANA_DIAS)
) >= clientes["fecha_registro"]
 
clientes["ventana_post_completa"] = (
    clientes["t0"] + pd.Timedelta(days=VENTANA_DIAS)
) <= fecha_max_datos
 
clientes["ventana_valida"] = (
    clientes["ventana_pre_completa"] & clientes["ventana_post_completa"] & clientes["t0"].notna()
)
print(f"Clientes con ventana pre/post válida: {clientes['ventana_valida'].sum():,} de {len(clientes):,} "
      f"({clientes['ventana_valida'].mean()*100:.1f}%)")


tx = transacciones_limpias.merge(clientes[["cliente_id", "t0"]], on="cliente_id", how="inner")
tx["dias_relativos"] = (tx["fecha"] - tx["t0"]).dt.days
 
pre = tx[(tx["dias_relativos"] >= -VENTANA_DIAS) & (tx["dias_relativos"] < 0)]
post = tx[(tx["dias_relativos"] >= 0) & (tx["dias_relativos"] < VENTANA_DIAS)]
 
agg_pre = pre.groupby("cliente_id").agg(
    frecuencia_pre=("tipo_transaccion", "size"),
    monto_pre=("monto_neto", "sum"),
    diversidad_tipo_pre=("tipo_transaccion", "nunique"),
    n_cash_pre=("tipo_transaccion", lambda s: s.isin(VALORES_CASH).sum()),
)
agg_post = post.groupby("cliente_id").agg(
    frecuencia_post=("tipo_transaccion", "size"),
    monto_post=("monto_neto", "sum"),
)
 
base = clientes.merge(agg_pre, on="cliente_id", how="left").merge(agg_post, on="cliente_id", how="left")
 
for col in ["frecuencia_pre", "monto_pre", "diversidad_tipo_pre", "n_cash_pre", "frecuencia_post", "monto_post"]:
    base[col] = base[col].fillna(0)
 
base["Y_frecuencia"] = base["frecuencia_post"] - base["frecuencia_pre"]
base["Y_monto"] = base["monto_post"] - base["monto_pre"]
base["uso_cash_in_out_pre"] = base["n_cash_pre"] / base["frecuencia_pre"].replace(0, np.nan)
base["uso_cash_in_out_pre"] = base["uso_cash_in_out_pre"].fillna(0)  # sin transacciones pre = sin uso de cash



# ## 4. Covariables (X): demográficas, marketing y tarjeta actual — todas pre-t0
# --- demográficas ---
base["edad"] = (base["t0"] - base["fecha_nacimiento"]).dt.days / 365.25
base["antiguedad_cliente"] = (base["t0"] - base["fecha_registro"]).dt.days
base["cedula_valida"] = ~base["flag_cedula_invalida"].astype(bool)  # invertimos el flag existente
 
# --- marketing, contactos ANTES de t0 (limpio de huérfanos/fecha futura) ---
marketing_limpia = marketing[
    (~marketing["flag_cliente_id_huerfano"].astype(bool))
    & (~marketing["flag_fecha_contacto_futura"].astype(bool))
]
mkt = marketing_limpia.merge(clientes[["cliente_id", "t0"]], on="cliente_id", how="inner")
mkt_pre = mkt[mkt["fecha_contacto"] < mkt["t0"]]
agg_mkt = mkt_pre.groupby("cliente_id").agg(
    n_contactos_marketing=("respondio", "size"),
    tasa_respuesta_marketing=("respondio", "mean"),  # ya boolean -> mean = proporción de True
)
base = base.merge(agg_mkt, on="cliente_id", how="left")
base["n_contactos_marketing"] = base["n_contactos_marketing"].fillna(0)
base["tasa_respuesta_marketing"] = base["tasa_respuesta_marketing"].fillna(0)
 
# --- tarjeta virtual, emitida ANTES de t0 ---
virtual = tarjetas[tarjetas["tipo"] == "virtual"].merge(
    clientes[["cliente_id", "t0"]], on="cliente_id", how="inner"
)
virtual_pre = virtual[virtual["fecha_emision"] < virtual["t0"]]
agg_virtual = virtual_pre.groupby("cliente_id").agg(
    tiene_virtual_activa=("fecha_activacion", lambda s: s.notna().any()),
    fecha_emision_virtual_min=("fecha_emision", "min"),
)
base = base.merge(agg_virtual, on="cliente_id", how="left")
base["tiene_virtual_activa"] = base["tiene_virtual_activa"].fillna(False)
base["antiguedad_tarjeta_virtual"] = (base["t0"] - base["fecha_emision_virtual_min"]).dt.days
base["antiguedad_tarjeta_virtual"] = base["antiguedad_tarjeta_virtual"].fillna(0)


# ## 5. Filtrar a ventana válida y armar la tabla final
base_modelo = base[base["ventana_valida"]].copy()
print(f"\nBase final: {len(base_modelo):,} clientes de {len(base):,} ({len(base_modelo)/len(base)*100:.1f}%)")
print(base_modelo["T"].value_counts())
 
COLUMNAS_FINALES = [
    "cliente_id", "T", "t0",
    "Y_frecuencia", "Y_monto",
    "edad", "ciudad", "canal_adquisicion", "antiguedad_cliente",
    "frecuencia_pre", "monto_pre", "diversidad_tipo_pre", "uso_cash_in_out_pre",
    "n_contactos_marketing", "tasa_respuesta_marketing",
    "tiene_virtual_activa", "antiguedad_tarjeta_virtual",
    "cedula_valida",
]
base_final = base_modelo[COLUMNAS_FINALES].copy()
 
destino = FOR_MODELLING_DIR / "base_modelo.parquet"
base_final.to_parquet(destino, index=False)
print(f"\nGuardado en: {destino}")
print(base_final.head())