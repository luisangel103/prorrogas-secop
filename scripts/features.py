"""
Variables del modelo: SOLO informacion conocida el dia de la firma.

No se usan (fuga de informacion, se actualizan despues de firmar):
  duracion del contrato, fecha de fin, valor del contrato, valor pagado/facturado, dias adicionados,
  montos por fuente de recursos, `el_contrato_puede_ser_prorrogado`, fecha de inicio (se modifica con actas).
Este modulo lo usan el entrenamiento (04_modelo.py) y la app.
"""
import numpy as np
import pandas as pd

NUM = ["log_valor_pactado", "log_plazo_pactado", "log_valor_por_dia", "descuento_oferta", "log_oferentes",
       "hist_entidad_n", "hist_entidad_tasa", "hist_contratista_n", "hist_contratista_tasa"]
CAT = ["modalidad_de_contratacion", "orden", "sector", "rama", "entidad_centralizada", "departamento",
       "familia_unspsc", "es_grupo", "es_pyme", "habilita_pago_adelantado", "destino_gasto",
       "origen_de_los_recursos", "mes_firma"]
FEATURES = NUM + CAT

NOMBRES = {  # nombres legibles para la app y las graficas
    "log_valor_pactado": "Valor pactado", "log_plazo_pactado": "Plazo pactado", "log_valor_por_dia": "Valor por día de plazo",
    "descuento_oferta": "Descuento de la oferta frente al presupuesto", "log_oferentes": "Número de oferentes",
    "hist_entidad_n": "Contratos previos de la entidad", "hist_entidad_tasa": "% de prórroga histórico de la entidad",
    "hist_contratista_n": "Contratos previos del contratista", "hist_contratista_tasa": "% de prórroga histórico del contratista",
    "modalidad_de_contratacion": "Modalidad de contratación", "orden": "Orden de la entidad", "sector": "Sector",
    "rama": "Rama", "entidad_centralizada": "Entidad centralizada", "departamento": "Departamento",
    "familia_unspsc": "Tipo de obra (código UNSPSC)", "es_grupo": "Consorcio o unión temporal", "es_pyme": "Contratista pyme",
    "habilita_pago_adelantado": "Pago adelantado", "destino_gasto": "Destino del gasto",
    "origen_de_los_recursos": "Origen de los recursos", "mes_firma": "Mes de firma",
}

SUAVIZADO = 5  # contratos "virtuales" con la tasa global, para no confiar en historiales de 1 o 2 contratos


def variables_basicas(df: pd.DataFrame) -> pd.DataFrame:
    """Variables que salen directamente del contrato y su proceso."""
    out = pd.DataFrame(index=df.index)
    valor = df["valor_pactado"].astype(float)
    plazo = df["plazo_pactado_dias"].astype(float).clip(lower=1)
    out["log_valor_pactado"] = np.log10(valor.clip(lower=1e5))
    out["log_plazo_pactado"] = np.log10(plazo)
    out["log_valor_por_dia"] = np.log10((valor / plazo).clip(lower=1e3))
    pb = df["precio_base"].astype(float)
    out["descuento_oferta"] = (1 - valor / pb).where(pb > 0).clip(-0.5, 0.9)
    out["log_oferentes"] = np.log1p(df["proveedores_unicos_con"].astype(float).fillna(0))
    for c in ["modalidad_de_contratacion", "orden", "sector", "rama", "entidad_centralizada", "departamento",
              "es_grupo", "es_pyme", "habilita_pago_adelantado", "destino_gasto", "origen_de_los_recursos"]:
        out[c] = df[c].fillna("No Definido").astype(str)
    out["familia_unspsc"] = df["codigo_de_categoria_principal"].fillna("").str.extract(r"(\d{4})")[0].fillna("0000")
    out["mes_firma"] = pd.to_datetime(df["fecha_de_firma"]).dt.month.fillna(0).astype(int).astype(str)
    return out


def grupos_historial(referencia: pd.DataFrame, clave: str) -> dict:
    """Indexa la referencia por `clave`: fechas de fin pactado ordenadas y prorrogas acumuladas.
    Se calcula una vez (la app lo guarda en cache) y se reutiliza para cada consulta."""
    ref = referencia[[clave, "fin_pactado", "prorroga"]].dropna(subset=[clave, "fin_pactado"])
    ref = ref.sort_values([clave, "fin_pactado"])
    return {k: (g["fin_pactado"].values.astype("datetime64[ns]"), np.cumsum(g["prorroga"].values.astype(float)))
            for k, g in ref.groupby(clave)}


def historial(consulta: pd.DataFrame, referencia: pd.DataFrame, clave: str, prefijo: str, tasa_global: float,
              grupos: dict | None = None) -> pd.DataFrame:
    """Para cada contrato de `consulta`, cuenta los contratos de `referencia` con la misma `clave`
    cuyo plazo pactado YA HABIA VENCIDO el dia de la firma, y su tasa de prorroga (suavizada).
    Asi el modelo nunca mira resultados que no se conocian al firmar."""
    if grupos is None:
        grupos = grupos_historial(referencia, clave)
    n = np.zeros(len(consulta)); s = np.zeros(len(consulta))
    fechas = pd.to_datetime(consulta["fecha_de_firma"]).values.astype("datetime64[ns]")
    claves = consulta[clave].values
    for i, (k, t) in enumerate(zip(claves, fechas)):
        if k in grupos:
            fin, cum = grupos[k]
            j = np.searchsorted(fin, t, side="left")
            n[i] = j; s[i] = cum[j - 1] if j > 0 else 0
    out = pd.DataFrame(index=consulta.index)
    out[f"hist_{prefijo}_n"] = np.log1p(n)
    out[f"hist_{prefijo}_tasa"] = (s + SUAVIZADO * tasa_global) / (n + SUAVIZADO)
    return out


def construir(consulta: pd.DataFrame, referencia: pd.DataFrame, tasa_global: float,
              grupos: dict | None = None) -> pd.DataFrame:
    """`grupos` (opcional): {"nit_entidad": ..., "documento_proveedor": ...} de grupos_historial(), para la app."""
    grupos = grupos or {}
    X = variables_basicas(consulta)
    X = X.join(historial(consulta, referencia, "nit_entidad", "entidad", tasa_global, grupos.get("nit_entidad")))
    X = X.join(historial(consulta, referencia, "documento_proveedor", "contratista", tasa_global,
                         grupos.get("documento_proveedor")))
    return X[FEATURES]
