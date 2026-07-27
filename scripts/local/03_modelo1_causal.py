# # Modelo causal (T-learner) — efecto de la tarjeta física
#

from pathlib import Path
 
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FOR_MODELLING_DIR = Path("./data/for_modelling")
base = pd.read_parquet(FOR_MODELLING_DIR / "base_modelo.parquet")
print(f"Base de modelo: {base.shape[0]:,} filas, T=1: {(base['T']==1).sum():,}, T=0: {(base['T']==0).sum():,}")



COLUMNAS_NUMERICAS = [
    "edad", "antiguedad_cliente", "frecuencia_pre", "monto_pre",
    "diversidad_tipo_pre", "uso_cash_in_out_pre",
    "n_contactos_marketing", "tasa_respuesta_marketing", "antiguedad_tarjeta_virtual",
]
COLUMNAS_CATEGORICAS = ["ciudad", "canal_adquisicion"]
COLUMNAS_BOOLEANAS = ["tiene_virtual_activa", "cedula_valida"]
COLUMNAS_X = COLUMNAS_NUMERICAS + COLUMNAS_CATEGORICAS + COLUMNAS_BOOLEANAS

base[COLUMNAS_BOOLEANAS] = base[COLUMNAS_BOOLEANAS].astype(float)


# ## 1. Balance de covariables entre tratamiento y control
#

 
# %%
def smd_numerica(x_tratamiento: pd.Series, x_control: pd.Series) -> float:
    """Diferencia de medias estandarizada para una variable numérica."""
    var_pool = (x_tratamiento.var() + x_control.var()) / 2
    if var_pool == 0 or pd.isna(var_pool):
        return 0.0
    return (x_tratamiento.mean() - x_control.mean()) / np.sqrt(var_pool)
 
 
tratamiento = base[base["T"] == 1]
control = base[base["T"] == 0]
 
resultados_balance = []
for col in COLUMNAS_NUMERICAS + COLUMNAS_BOOLEANAS:
    x_t = tratamiento[col].astype(float)
    x_c = control[col].astype(float)
    valor_smd = smd_numerica(x_t, x_c)
    resultados_balance.append({
        "variable": col, "media_tratamiento": x_t.mean(), "media_control": x_c.mean(),
        "smd": valor_smd, "desbalanceada": abs(valor_smd) > 0.1,
    })
 
tabla_balance = pd.DataFrame(resultados_balance).sort_values("smd", key=abs, ascending=False)
print(tabla_balance.to_string(index=False))
 
n_desbalanceadas = tabla_balance["desbalanceada"].sum()
print(f"\nVariables desbalanceadas (|SMD| > 0.1): {n_desbalanceadas} de {len(tabla_balance)}")
if n_desbalanceadas > 0:
    print("ADVERTENCIA: hay covariables desbalanceadas.")
 
# categóricas: comparar distribución de proporciones
for col in COLUMNAS_CATEGORICAS:
    print(f"\nDistribución de '{col}' — tratamiento vs. control:")
    comparacion = pd.concat([
        tratamiento[col].value_counts(normalize=True).rename("tratamiento"),
        control[col].value_counts(normalize=True).rename("control"),
    ], axis=1).fillna(0)
    print((comparacion * 100).round(1))


    # ## 2. Preparar el preprocesador de features (reutilizable en ambos modelos)
 
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
 
 
def construir_modelo():
    return Pipeline([
        ("prep", preprocesador),
        ("reg", Ridge(alpha=1.0)),
    ])
 
 
# ## 3. Entrenar el T-learner: un modelo en tratamiento, otro en control
#
# Separamos train/test DENTRO de cada grupo, para poder reportar un R² honesto
# (no inflado por evaluar sobre los mismos datos de entrenamiento).
 
def entrenar_y_validar(df_grupo: pd.DataFrame, nombre_y: str, nombre_grupo: str):
    X = df_grupo[COLUMNAS_X]
    y = df_grupo[nombre_y]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
 
    modelo = construir_modelo()
    modelo.fit(X_tr, y_tr)
    r2 = r2_score(y_te, modelo.predict(X_te))
    print(f"[{nombre_grupo} -> {nombre_y}] R² en holdout: {r2:.3f} (n_train={len(X_tr)}, n_test={len(X_te)})")
 
    # reentrenar con TODO el grupo para el modelo final que se usará en producción
    modelo_final = construir_modelo()
    modelo_final.fit(X, y)
    return modelo_final, r2
 
 
modelo_control_freq, r2_control_freq = entrenar_y_validar(control, "Y_frecuencia", "control")
modelo_tratamiento_freq, r2_tratamiento_freq = entrenar_y_validar(tratamiento, "Y_frecuencia", "tratamiento")
modelo_control_monto, r2_control_monto = entrenar_y_validar(control, "Y_monto", "control")
modelo_tratamiento_monto, r2_tratamiento_monto = entrenar_y_validar(tratamiento, "Y_monto", "tratamiento")



# ## 4. Validar el efecto: ¿el tratamiento realmente cambió algo?

def validar_efecto(df_tratamiento: pd.DataFrame, modelo_control, nombre_y: str, etiqueta: str):
    X_tratamiento = df_tratamiento[COLUMNAS_X]
    y_real = df_tratamiento[nombre_y]
    y_esperado_sin_tarjeta = modelo_control.predict(X_tratamiento)
 
    efecto_individual = y_real.values - y_esperado_sin_tarjeta
    t_stat, p_valor = stats.ttest_1samp(efecto_individual, 0)
 
    print(f"\n--- Validación del efecto: {etiqueta} ---")
    print(f"Distribución del efecto individual:\n{pd.Series(efecto_individual).describe()}")
    print(f"Efecto promedio estimado (ATE, media): {efecto_individual.mean():.3f}")
    print(f"Efecto mediano estimado (más robusto a outliers): {np.median(efecto_individual):.3f}")
    print(f"% de personas con efecto individual positivo: {(efecto_individual > 0).mean()*100:.1f}%")
    print(f"p-valor t-test (asume normalidad, sensible a colas pesadas): {p_valor:.4f}")
 
    # Wilcoxon: no asume normalidad, se basa en rangos -- más confiable si la
    # variable tiene outliers grandes (típico en montos monetarios)
    try:
        w_stat, p_wilcoxon = stats.wilcoxon(efecto_individual)
        print(f"p-valor Wilcoxon (robusto a outliers): {p_wilcoxon:.4f}")
    except ValueError as e:
        p_wilcoxon = None
        print(f"Wilcoxon no se pudo calcular: {e}")
 
    # Bootstrap del promedio, para ver qué tan estable es el ATE sin asumir
    # distribución normal
    np.random.seed(42)
    medias_bootstrap = [
        np.mean(np.random.choice(efecto_individual, size=len(efecto_individual), replace=True))
        for _ in range(2000)
    ]
    ic_95 = np.percentile(medias_bootstrap, [2.5, 97.5])
    print(f"IC 95% del ATE (bootstrap): [{ic_95[0]:.3f}, {ic_95[1]:.3f}]")
 
    if p_valor < 0.05 or (p_wilcoxon is not None and p_wilcoxon < 0.05):
        print("-> Hay evidencia de efecto distinto de cero")
    else:
        print("-> Ninguna prueba encuentra evidencia suficiente de efecto distinto de cero.")
    if ic_95[0] > 0 or ic_95[1] < 0:
        print("   (el IC bootstrap no cruza el cero, lo cual sugiere que sí hay señal real ")
 
    return efecto_individual

p1, p99 = tratamiento["Y_monto"].quantile([0.01, 0.99])
tratamiento["Y_monto_winsor"] = tratamiento["Y_monto"].clip(p1, p99)
 
efecto_frecuencia = validar_efecto(tratamiento, modelo_control_freq, "Y_frecuencia", "frecuencia de transacciones")
efecto_monto = validar_efecto(tratamiento, modelo_control_monto, "Y_monto_winsor", "monto transaccional")