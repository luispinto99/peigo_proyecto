"""
etl_utils

Funciones puras y reutilizables de limpieza/estandarización para el pipeline
del piloto de tarjetas físicas. 


Uso típico:
    from etl_functions import estandarizar_fecha, normalizar_texto, ...

    serie_limpia, reporte = normalizar_texto(clientes["ciudad"])
    clientes["ciudad_std"] = serie_limpia
    print(reporte.resumen())
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging: en Glue esto se ve en CloudWatch: en local, se ve en la consola.
# ---------------------------------------------------------------------------
logger = logging.getLogger("etl_pipeline")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

MAX_VALORES_NO_RECONOCIDOS = 50  # tope para no inflar el reporte con datasets gigantes


# ---------------------------------------------------------------------------
# Reporte estandarizado
# ---------------------------------------------------------------------------
@dataclass
class ReporteLimpieza:
    """Reporte estandarizado que devuelve cada función de limpieza del pipeline."""
    funcion: str
    columna: Optional[str] = None
    filas_entrada: int = 0
    filas_salida: int = 0
    valores_nulos_entrada: int = 0
    valores_nulos_salida: int = 0
    valores_no_reconocidos: dict = field(default_factory=dict)   # valor_original -> conteo
    formatos_detectados: dict = field(default_factory=dict)      # solo lo usa estandarizar_fecha
    advertencias: list = field(default_factory=list)
    errores: list = field(default_factory=list)

    def registrar_no_reconocido(self, valor):
        """Acumula un valor no reconocido, respetando el tope MAX_VALORES_NO_RECONOCIDOS."""
        if len(self.valores_no_reconocidos) >= MAX_VALORES_NO_RECONOCIDOS and valor not in self.valores_no_reconocidos:
            return
        self.valores_no_reconocidos[valor] = self.valores_no_reconocidos.get(valor, 0) + 1

    @property
    def ok(self) -> bool:
        """True si la función corrió sin errores inesperados (puede tener advertencias igual)."""
        return len(self.errores) == 0

    def resumen(self) -> str:
        lineas = [f"[{self.funcion}]" + (f" columna='{self.columna}'" if self.columna else "")]
        lineas.append(f"  filas: {self.filas_entrada} -> {self.filas_salida}")
        lineas.append(f"  nulos: {self.valores_nulos_entrada} -> {self.valores_nulos_salida}")
        if self.formatos_detectados:
            lineas.append(f"  formatos detectados: {self.formatos_detectados}")
        if self.valores_no_reconocidos:
            n_total = sum(self.valores_no_reconocidos.values())
            lineas.append(f"  valores no reconocidos ({n_total} filas): {self.valores_no_reconocidos}")
        for w in self.advertencias:
            lineas.append(f"  ADVERTENCIA: {w}")
        for e in self.errores:
            lineas.append(f"  ERROR: {e}")
        return "\n".join(lineas)


def _log_reporte(reporte: ReporteLimpieza):
    if reporte.errores:
        logger.error(reporte.resumen())
    elif reporte.advertencias:
        logger.warning(reporte.resumen())
    else:
        logger.info(reporte.resumen())


# ---------------------------------------------------------------------------
# 1. Quitar filas 100% duplicadas
# ---------------------------------------------------------------------------
def quitar_duplicados_exactos(df: pd.DataFrame, nombre_tabla: str = "tabla") -> tuple[pd.DataFrame, ReporteLimpieza]:
    """
    Elimina filas donde TODAS las columnas son idénticas a otra fila
    (mismo transaccion_id/cliente_id incluido). No toca duplicados parciales
    (esos requieren una regla de negocio propia, no una limpieza genérica).
    """
    reporte = ReporteLimpieza(funcion="quitar_duplicados_exactos", columna=nombre_tabla)
    reporte.filas_entrada = len(df)
    try:
        df_limpio = df.drop_duplicates(keep="first").reset_index(drop=True)
        reporte.filas_salida = len(df_limpio)
        n_removidas = reporte.filas_entrada - reporte.filas_salida
        if n_removidas > 0:
            pct = n_removidas / reporte.filas_entrada * 100 if reporte.filas_entrada else 0
            reporte.advertencias.append(f"se eliminaron {n_removidas} filas 100% duplicadas ({pct:.2f}%)")
        _log_reporte(reporte)
        return df_limpio, reporte
    except Exception as e:
        reporte.errores.append(f"error inesperado eliminando duplicados: {e}")
        _log_reporte(reporte)
        return df, reporte  # fail-safe: no perder datos si algo falla


# ---------------------------------------------------------------------------
# 2. Normalizar texto genérico (minúsculas, guiones bajos -> espacio, placeholders de nulo)
# ---------------------------------------------------------------------------
VALORES_NULOS_TEXTO = {"n/a", "na", "n\\a", "null", "none", "nan", "-", "s/d", "sin dato", ""}


def normalizar_texto(serie: pd.Series, nombre_columna: str = "columna") -> tuple[pd.Series, ReporteLimpieza]:
    """
    - Pasa a minúsculas y quita espacios extremos.
    - Reemplaza guiones bajos por espacios (ej. "cash_in" -> "cash in").
    - Colapsa espacios múltiples.
    - Convierte placeholders de nulo ("n/a", "null", "none", etc., en
      cualquier variante de mayúsculas) a NA real, no a un string "null".
    """
    reporte = ReporteLimpieza(funcion="normalizar_texto", columna=nombre_columna)
    reporte.filas_entrada = len(serie)
    reporte.valores_nulos_entrada = int(serie.isna().sum())

    def limpiar(v):
        if pd.isna(v):
            return np.nan
        try:
            s = str(v).strip().lower().replace("_", " ")
            s = re.sub(r"\s+", " ", s).strip()
            if s in VALORES_NULOS_TEXTO:
                return np.nan
            return s
        except Exception as e:
            reporte.errores.append(f"no se pudo normalizar el valor {v!r}: {e}")
            return np.nan

    try:
        serie_limpia = serie.apply(limpiar)
        reporte.filas_salida = len(serie_limpia)
        reporte.valores_nulos_salida = int(serie_limpia.isna().sum())
        nuevos_nulos = reporte.valores_nulos_salida - reporte.valores_nulos_entrada
        if nuevos_nulos > 0:
            reporte.advertencias.append(
                f"{nuevos_nulos} valores se convirtieron a NA por matchear un placeholder (ej. 'n/a', 'null')"
            )
        _log_reporte(reporte)
        return serie_limpia, reporte
    except Exception as e:
        reporte.errores.append(f"error inesperado normalizando texto: {e}")
        _log_reporte(reporte)
        return serie, reporte


# ---------------------------------------------------------------------------
# 3. Estandarizar fechas en múltiples formatos (string, excel serial, epoch)
# ---------------------------------------------------------------------------
FORMATOS_CANDIDATOS = [
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
    "%d/%m/%Y %H:%M",
    "%d/%m/%Y",     # día-primero: default para Ecuador/Latam
    "%d-%m-%Y",
    "%d.%m.%Y",
    "%m/%d/%Y",     # mes-primero: solo como fallback
    "%Y%m%d",
]

LIMITE_PARSEO_MIN = pd.Timestamp("1900-01-01")
LIMITE_PARSEO_MAX = pd.Timestamp("2100-01-01")

def estandarizar_fecha(
    serie: pd.Series,
    nombre_columna: str = "fecha",
    fecha_min: Optional[pd.Timestamp] = None,
    fecha_max: Optional[pd.Timestamp] = None,
) -> tuple[pd.Series, ReporteLimpieza]:
    """
    Estandariza una columna de fecha en formatos mixtos: strings en varios
    patrones, serial de Excel, y epoch Unix (segundos o milisegundos,
    incluyendo negativos si fecha_min es anterior a 1970 — ej. para
    fecha_nacimiento).

    fecha_min / fecha_max acotan qué se considera un valor NUMÉRICO
    plausible como fecha (importa para no confundir un ID con un epoch).
    Default: 1900-01-01 a hoy+1 día si no se especifica.
    """
    reporte = ReporteLimpieza(funcion="estandarizar_fecha", columna=nombre_columna)
    reporte.filas_entrada = len(serie)
    reporte.valores_nulos_entrada = int(serie.isna().sum())
 
    try:
        epoch_ref = pd.Timestamp("1970-01-01")
        epoch_min_s = (LIMITE_PARSEO_MIN - epoch_ref).total_seconds()
        epoch_max_s = (LIMITE_PARSEO_MAX - epoch_ref).total_seconds()
        epoch_min_ms = epoch_min_s * 1000.0
        epoch_max_ms = epoch_max_s * 1000.0
        excel_ref = pd.Timestamp("1899-12-30")
        serial_min = (LIMITE_PARSEO_MIN - excel_ref).days
        serial_max = (LIMITE_PARSEO_MAX - excel_ref).days
 
        serie_str = serie.astype(str).str.strip()
        resultado = pd.Series(pd.NaT, index=serie.index, dtype="datetime64[ns]")
        formato_detectado = pd.Series(pd.NA, index=serie.index, dtype="object")
 
        pendiente = resultado.isna() & serie.notna()
        for fmt in FORMATOS_CANDIDATOS:
            if not pendiente.any():
                break
            try:
                parseado = pd.to_datetime(serie_str[pendiente], format=fmt, errors="coerce")
            except Exception as e:
                reporte.advertencias.append(f"formato '{fmt}' no se pudo evaluar: {e}")
                continue
            exito = parseado.notna()
            idx_exito = parseado[exito].index
            resultado.loc[idx_exito] = parseado[exito]
            formato_detectado.loc[idx_exito] = fmt
            pendiente = resultado.isna() & serie.notna()
 
        if pendiente.any():
            try:
                numericos = pd.to_numeric(serie[pendiente], errors="coerce")
                es_serial = numericos.notna() & numericos.between(serial_min, serial_max)
                if es_serial.any():
                    idx_serial = numericos[es_serial].index
                    resultado.loc[idx_serial] = pd.to_datetime(numericos[es_serial], unit="D", origin="1899-12-30")
                    formato_detectado.loc[idx_serial] = "excel_serial"
                pendiente = resultado.isna() & serie.notna()
            except Exception as e:
                reporte.advertencias.append(f"no se pudo evaluar serial de Excel: {e}")
 
        if pendiente.any():
            try:
                numericos = pd.to_numeric(serie[pendiente], errors="coerce")
 
                es_epoch_s = numericos.notna() & numericos.between(epoch_min_s, epoch_max_s)
                if es_epoch_s.any():
                    idx_s = numericos[es_epoch_s].index
                    resultado.loc[idx_s] = pd.to_datetime(numericos[es_epoch_s], unit="s")
                    formato_detectado.loc[idx_s] = "unix_epoch_segundos"
                pendiente = resultado.isna() & serie.notna()
 
                es_epoch_ms = numericos.notna() & numericos.between(epoch_min_ms, epoch_max_ms)
                if es_epoch_ms.any():
                    idx_ms = numericos[es_epoch_ms].index
                    resultado.loc[idx_ms] = pd.to_datetime(numericos[es_epoch_ms], unit="ms")
                    formato_detectado.loc[idx_ms] = "unix_epoch_milisegundos"
            except Exception as e:
                reporte.advertencias.append(f"no se pudo evaluar epoch Unix: {e}")
 
        reporte.formatos_detectados = formato_detectado.value_counts(dropna=False).to_dict()
        reporte.filas_salida = len(resultado)
        reporte.valores_nulos_salida = int(resultado.isna().sum())
 
        no_parseadas = (~resultado.notna()) & serie.notna()
        if no_parseadas.any():
            for v in serie[no_parseadas].unique():
                reporte.registrar_no_reconocido(v)
            reporte.advertencias.append(
                f"{int(no_parseadas.sum())} valores no calzaron con ningún formato/rango de parseo (1900-2100)"
            )
 
        # --- flag de plausibilidad de negocio: NO afecta el parseo, solo señala ---
        f_min = fecha_min if fecha_min is not None else LIMITE_PARSEO_MIN
        f_max = fecha_max if fecha_max is not None else LIMITE_PARSEO_MAX
        fuera_de_rango = resultado.notna() & ((resultado < f_min) | (resultado > f_max))
        if fuera_de_rango.any():
            reporte.advertencias.append(
                f"{int(fuera_de_rango.sum())} valores se parsearon OK pero caen fuera del rango plausible "
                f"de negocio [{f_min.date()}, {f_max.date()}] — quedan marcados en el flag, NO se anulan"
            )
 
        _log_reporte(reporte)
        return resultado, fuera_de_rango, reporte
    except Exception as e:
        reporte.errores.append(f"error inesperado estandarizando fechas: {e}")
        _log_reporte(reporte)
        flag_vacio = pd.Series(False, index=serie.index)
        return serie, flag_vacio, reporte


def rango_plausible_para_columna(nombre_columna: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Rango de fecha plausible según el significado de la columna. Ajusta los
    umbrales (18 años, 2015-01-01, etc.) a la realidad real de tu negocio.
    """
    col = nombre_columna.lower()
    if "nacimiento" in col:
        return (
            pd.Timestamp.now() - pd.Timedelta(days=110 * 365),
            pd.Timestamp.now() - pd.Timedelta(days=18 * 365),
        )
    if "vencimiento" in col or "expira" in col:
        return pd.Timestamp("1900-01-01"), pd.Timestamp.now() + pd.Timedelta(days=10 * 365)
    return pd.Timestamp("1900-01-01"), pd.Timestamp.now() + pd.Timedelta(days=1)


# ---------------------------------------------------------------------------
# 4. Estandarizar 'tipo' de tarjeta: f* -> fisica, v* -> virtual
# ---------------------------------------------------------------------------
def _quitar_acentos(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")


def estandarizar_tipo_fisica_virtual(serie: pd.Series, nombre_columna: str = "tipo") -> tuple[pd.Series, ReporteLimpieza]:
    """
    Clasifica por PREFIJO (tolerante a mayúsculas/acentos/espacios):
      - empieza con 'f' -> 'fisica'   (cubre "f", "fisica", "FÍSICA", ...)
      - empieza con 'v' -> 'virtual'  (cubre "v", "virtual", "VIRTUAL", ...)
      - cualquier otro valor -> NA, y se reporta para revisión manual.
    """
    reporte = ReporteLimpieza(funcion="estandarizar_tipo_fisica_virtual", columna=nombre_columna)
    reporte.filas_entrada = len(serie)
    reporte.valores_nulos_entrada = int(serie.isna().sum())

    def clasificar(v):
        if pd.isna(v):
            return np.nan
        try:
            s = _quitar_acentos(str(v).strip().lower())
            if s.startswith("f"):
                return "fisica"
            if s.startswith("v"):
                return "virtual"
            reporte.registrar_no_reconocido(v)
            return np.nan
        except Exception as e:
            reporte.errores.append(f"no se pudo clasificar el valor {v!r}: {e}")
            return np.nan

    try:
        serie_std = serie.apply(clasificar)
        reporte.filas_salida = len(serie_std)
        reporte.valores_nulos_salida = int(serie_std.isna().sum())
        if reporte.valores_no_reconocidos:
            n = sum(reporte.valores_no_reconocidos.values())
            reporte.advertencias.append(f"{n} valores no empezaban con 'f' ni 'v', quedaron NA")
        _log_reporte(reporte)
        return serie_std, reporte
    except Exception as e:
        reporte.errores.append(f"error inesperado: {e}")
        _log_reporte(reporte)
        return serie, reporte


# ---------------------------------------------------------------------------
# 5. Estandarizar binarios: '0'/'1', True/False, Sí/No, mayúsc/minúsc, texto "nan"
# ---------------------------------------------------------------------------
VALORES_BINARIO_TRUE = {"1", "true", "si", "sí", "yes", "y", "verdadero", "s", "S"}
VALORES_BINARIO_FALSE = {"0", "false", "no", "n", "falso", "N"}
VALORES_BINARIO_NULO = {"nan", "none", "null", "n/a", "na", ""}


def estandarizar_binario(serie: pd.Series, nombre_columna: str = "columna") -> tuple[pd.Series, "ReporteLimpieza"]:
    """
    Estandariza una columna con codificación binaria mixta a booleano real
    (dtype nullable 'boolean' de pandas: True / False / <NA>).
    Cubre, entre otros: '0'/'1', 0/1, True/False, 'true'/'false' (cualquier
    capitalización), 'Sí'/'No' (con o sin tilde), y strings "nan"/"null"
    como nulo real (no como texto).
    """
    reporte = ReporteLimpieza(funcion="estandarizar_binario", columna=nombre_columna)
    reporte.filas_entrada = len(serie)
    reporte.valores_nulos_entrada = int(serie.isna().sum())

    def clasificar(v):
        if pd.isna(v):
            return pd.NA
        try:
            if isinstance(v, (bool, np.bool_)):
                return bool(v)
            if isinstance(v, (int, float, np.integer, np.floating)):
                if v == 0:
                    return False
                if v == 1:
                    return True
                reporte.registrar_no_reconocido(v)
                return pd.NA
            s = _quitar_acentos(str(v).strip().lower())
            if s in VALORES_BINARIO_TRUE:
                return True
            if s in VALORES_BINARIO_FALSE:
                return False
            if s in VALORES_BINARIO_NULO:
                return pd.NA
            reporte.registrar_no_reconocido(v)
            return pd.NA
        except Exception as e:
            reporte.errores.append(f"no se pudo estandarizar el valor {v!r}: {e}")
            return pd.NA

    try:
        serie_std = serie.apply(clasificar).astype("boolean")  # dtype nullable, no object
        reporte.filas_salida = len(serie_std)
        reporte.valores_nulos_salida = int(serie_std.isna().sum())
        if reporte.valores_no_reconocidos:
            n = sum(reporte.valores_no_reconocidos.values())
            reporte.advertencias.append(f"{n} valores no se reconocieron como binarios, quedaron NA")
        _log_reporte(reporte)
        return serie_std, reporte
    except Exception as e:
        reporte.errores.append(f"error inesperado: {e}")
        _log_reporte(reporte)
        return serie, reporte


# ---------------------------------------------------------------------------
# 6. Convertir a numérico de forma robusta
# ---------------------------------------------------------------------------
def convertir_a_numerico(serie: pd.Series, nombre_columna: str = "columna") -> tuple[pd.Series, ReporteLimpieza]:
    """
    Convierte a numérico (float) tolerando:
      - símbolos de moneda / espacios / texto pegado (ej. "$1,234.50", "1234.50 USD")
      - separador de miles y decimal mixtos (coma o punto en cualquier orden)
      - valores que ya son numéricos (se dejan tal cual)
    Lo que no se pueda convertir queda como NaN y se reporta explícitamente
    (nunca lanza ValueError hacia afuera).
    """
    reporte = ReporteLimpieza(funcion="convertir_a_numerico", columna=nombre_columna)
    reporte.filas_entrada = len(serie)
    reporte.valores_nulos_entrada = int(serie.isna().sum())

    def limpiar_y_convertir(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            return float(v)
        try:
            s = str(v).strip()
            s = re.sub(r"[^\d,.\-]", "", s)  # quita símbolos de moneda, letras, espacios
            if s in ("", "-", ".", ","):
                reporte.registrar_no_reconocido(v)
                return np.nan
            if "," in s and "." in s:
                # el símbolo que aparece último es el separador decimal real
                if s.rfind(",") > s.rfind("."):
                    s = s.replace(".", "").replace(",", ".")
                else:
                    s = s.replace(",", "")
            elif "," in s:
                partes = s.split(",")
                # coma con <=2 dígitos después: es decimal (convención latam). Si no, es miles.
                s = s.replace(",", ".") if len(partes[-1]) <= 2 else s.replace(",", "")
            return float(s)
        except (ValueError, TypeError):
            reporte.registrar_no_reconocido(v)
            return np.nan
        except Exception as e:
            reporte.errores.append(f"error inesperado convirtiendo {v!r}: {e}")
            return np.nan

    try:
        serie_num = serie.apply(limpiar_y_convertir)
        reporte.filas_salida = len(serie_num)
        reporte.valores_nulos_salida = int(serie_num.isna().sum())
        if reporte.valores_no_reconocidos:
            n = sum(reporte.valores_no_reconocidos.values())
            reporte.advertencias.append(f"{n} valores no se pudieron convertir a numérico, quedaron NA")
        _log_reporte(reporte)
        return serie_num, reporte
    except Exception as e:
        reporte.errores.append(f"error inesperado: {e}")
        _log_reporte(reporte)
        return serie, reporte

# ---------------------------------------------------------------------------
# 7. Cédula: estandarizar formato, y validar módulo 10 (Ecuador, persona natural)
# ---------------------------------------------------------------------------
def estandarizar_formato_cedula(serie: pd.Series, nombre_columna: str = "cedula") -> tuple[pd.Series, ReporteLimpieza]:
    """
    Estandariza el FORMATO de una cédula:
      - Quita guiones, puntos y espacios.
      - Si el resultado tiene 9 dígitos y no empieza por cero, se asume que
        se perdió el cero inicial y se le antepone un '0'.
      - Si después de esto no queda un string de exactamente 10 dígitos,
        se reporta como no reconocido.
    """
    reporte = ReporteLimpieza(funcion="estandarizar_formato_cedula", columna=nombre_columna)
    reporte.filas_entrada = len(serie)
    reporte.valores_nulos_entrada = int(serie.isna().sum())
 
    def limpiar(v):
        if pd.isna(v):
            return np.nan
        try:
            s = str(v).strip()
            if s.endswith(".0"):                   # típico si la columna venía como float
                s = s[:-2]
            s = re.sub(r"[-.\s]", "", s)          # quita guiones, puntos, espacios (después del .0)
            if s.isdigit() and len(s) == 9 and not s.startswith("0"):
                s = "0" + s
            if not (s.isdigit() and len(s) == 10):
                reporte.registrar_no_reconocido(v)
            return s
        except Exception as e:
            reporte.errores.append(f"no se pudo estandarizar el valor {v!r}: {e}")
            return np.nan
 
    try:
        serie_std = serie.apply(limpiar)
        reporte.filas_salida = len(serie_std)
        reporte.valores_nulos_salida = int(serie_std.isna().sum())
        if reporte.valores_no_reconocidos:
            n = sum(reporte.valores_no_reconocidos.values())
            reporte.advertencias.append(f"{n} valores no quedaron en formato de 10 dígitos tras la limpieza")
        _log_reporte(reporte)
        return serie_std, reporte
    except Exception as e:
        reporte.errores.append(f"error inesperado: {e}")
        _log_reporte(reporte)
        return serie, reporte


def validar_cedula_modulo10(cedula) -> bool:
    """
    Valida el dígito verificador de una cédula ecuatoriana de persona
    natural (algoritmo módulo 10). Recibe UN valor (no una Series) y
    devuelve SOLO True/False — sin reporte, sin side effects — pensada
    para usarse directo como flag:
 
 
    Cualquier problema (nulo, formato incorrecto, tipo inesperado,
    excepción) devuelve False.
 
    Nota: asume persona natural (tercer dígito 0-6). No valida RUC de
    persona jurídica (tercer dígito 9) ni de sector público (tercer
    dígito 6 con estructura distinta) ni pasaporte.
    """
    try:
        if pd.isna(cedula):
            return False
        s = str(cedula).strip()
        if not s.isdigit() or len(s) != 10:
            return False
 
        provincia = int(s[0:2])
        if provincia < 1 or provincia > 24:
            return False
 
        tercer_digito = int(s[2])
        if tercer_digito > 6:
            return False  # persona natural: tercer dígito debe ser 0-6
 
        coeficientes = [2, 1, 2, 1, 2, 1, 2, 1, 2]
        suma = 0
        for i, coef in enumerate(coeficientes):
            valor = int(s[i]) * coef
            if valor >= 10:
                valor -= 9
            suma += valor
 
        digito_verificador = int(s[9])
        return (10 - (suma % 10)) % 10 == digito_verificador
    except Exception:
        return False