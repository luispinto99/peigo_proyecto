# MODELO SEGMENTACION

import pickle
from pathlib import Path
 
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
 
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)
 
FOR_MODELLING_DIR = Path("./data/for_modelling")
MODELS_DIR = Path("./models")
MODELS_DIR.mkdir(exist_ok=True)

base = pd.read_parquet(FOR_MODELLING_DIR / "base_segmentacion.parquet")
print(f"Base de segmentación: {len(base):,} clientes")
 
COLUMNAS_NUMERICAS = [
    "edad", "antiguedad_cliente", "frecuencia_mensual", "monto_mensual",
    "diversidad_tipo", "contactos_mensuales", "tasa_respuesta", "antiguedad_tarjeta_virtual",
]
COLUMNAS_CATEGORICAS = ["ciudad", "canal_adquisicion"]
COLUMNAS_BOOLEANAS = ["tiene_virtual_activa"]
COLUMNAS_X = COLUMNAS_NUMERICAS + COLUMNAS_CATEGORICAS + COLUMNAS_BOOLEANAS

VARIABLES_PRIORIDAD = ["frecuencia_mensual", "monto_mensual", "tasa_respuesta"]


# ## 2. Preprocesamiento (mismo patrón que los modelos anteriores)
 
# %%
preprocesador = ColumnTransformer(transformers=[
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), COLUMNAS_NUMERICAS),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="desconocido")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
    ]), COLUMNAS_CATEGORICAS),
    ("bool", SimpleImputer(strategy="most_frequent"), COLUMNAS_BOOLEANAS),
])
 
X = base[COLUMNAS_X]
X_transformado = preprocesador.fit_transform(X)
if hasattr(X_transformado, "toarray"):
    X_transformado = X_transformado.toarray()


# ## 3. Elegir k probando varias opciones (silhouette score)
 
resultados_k = []
for k in [3, 4, 5, 6]:
    km_prueba = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_transformado)
    score = silhouette_score(X_transformado, km_prueba.labels_)
    resultados_k.append({"k": k, "silhouette": score})
    print(f"k={k}: silhouette={score:.3f}")
 
tabla_k = pd.DataFrame(resultados_k)
K_ELEGIDO = int(tabla_k.loc[tabla_k["silhouette"].idxmax(), "k"])
print(f"\nk elegido (mayor silhouette): {K_ELEGIDO}")

# ## 4. Entrenar el modelo final con el k elegido

kmeans = KMeans(n_clusters=K_ELEGIDO, random_state=42, n_init=10)
base["cluster_id"] = kmeans.fit_predict(X_transformado)
print(f"Distribución de clientes por cluster:\n{base['cluster_id'].value_counts().sort_index()}")


# ## 5. Perfilar cada cluster (medias en unidades originales, no escaladas)

perfil = base.groupby("cluster_id")[COLUMNAS_NUMERICAS].mean()
print("\nPerfil de clusters (medias en unidades originales):")
print(perfil)
 
print("\nComposición categórica por cluster (moda):")
for col in COLUMNAS_CATEGORICAS:
    print(f"\n{col}:")
    print(base.groupby("cluster_id")[col].agg(lambda s: s.value_counts().idxmax()))


# ## 6. Ranking de prioridad -- SOLO con las 3 variables de negocio justificadas
#
# Se calcula el z-score de cada variable de prioridad a nivel CLIENTE
# (usando media/std de toda la población, no del cluster) y se promedia por
# cluster -- así el ranking usa una escala consistente entre clusters.

for col in VARIABLES_PRIORIDAD:
    base[f"z_{col}"] = (base[col] - base[col].mean()) / base[col].std()
 
base["puntaje_prioridad_individual"] = base[[f"z_{c}" for c in VARIABLES_PRIORIDAD]].sum(axis=1)
 
puntaje_por_cluster = base.groupby("cluster_id")["puntaje_prioridad_individual"].mean().sort_values(ascending=False)
print("\nPuntaje de prioridad por cluster (mayor = más prioritario):")
print(puntaje_por_cluster)
 
# nombres genéricos ordinales -- funcionan para cualquier K_ELEGIDO
if K_ELEGIDO <= 5:
    etiquetas_disponibles = ["alto_potencial", "medio_alto", "medio", "medio_bajo", "bajo_potencial"]
else:
    etiquetas_disponibles = [f"prioridad_{i+1}" for i in range(K_ELEGIDO)]
 
mapa_cluster_a_nombre = {
    cluster_id: etiquetas_disponibles[rank]
    for rank, cluster_id in enumerate(puntaje_por_cluster.index)
}
print(f"\nMapeo cluster_id -> nombre de prioridad: {mapa_cluster_a_nombre}")
 
base["segmento_prioridad"] = base["cluster_id"].map(mapa_cluster_a_nombre)


# ## 7. Guardar modelo, preprocesador, mapeo y base final

artefactos = {
    "kmeans": kmeans,
    "preprocesador": preprocesador,
    "mapa_cluster_a_nombre": mapa_cluster_a_nombre,
    "columnas_x": COLUMNAS_X,
    "variables_prioridad": VARIABLES_PRIORIDAD,
    "medias_poblacion": {col: base[col].mean() for col in VARIABLES_PRIORIDAD},
    "stds_poblacion": {col: base[col].std() for col in VARIABLES_PRIORIDAD},
}
with open(MODELS_DIR / "modelo_segmentacion.pkl", "wb") as f:
    pickle.dump(artefactos, f)
print(f"\nModelo y mapeo guardados en: {MODELS_DIR / 'modelo_segmentacion.pkl'}")
 
COLUMNAS_SALIDA = ["cliente_id", "cluster_id", "segmento_prioridad", "puntaje_prioridad_individual"] + COLUMNAS_X
resultado = base[COLUMNAS_SALIDA].sort_values("puntaje_prioridad_individual", ascending=False)
 
destino = FOR_MODELLING_DIR / "clientes_segmentados.parquet"
resultado.to_parquet(destino, index=False)
print(f"Base segmentada guardada en: {destino}")
print(f"\nDistribución de segmento_prioridad:\n{resultado['segmento_prioridad'].value_counts()}")