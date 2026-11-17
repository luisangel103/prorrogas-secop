"""
Motor del sistema (sin interfaz): lo usan la app (app.py) y la evaluación de punta a punta (09_eval_flujo.py),
así que la evaluación prueba exactamente el mismo código que ve el usuario.

  contrato -> variables al firmar -> ML (probabilidad + aporte de cada variable) -> reglas (alertas)
           -> hechos de la ficha (historial y causas de la entidad) -> ficha (LLM o plantilla)
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import features as F
import ficha as FI
import reglas as R
from causas import clasificar_reglas

ROOT = Path(__file__).resolve().parents[1]
PROC, MOD = ROOT / "data" / "processed", ROOT / "models"
CORTE_TEST = pd.Timestamp("2024-07-01")
MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
FAMILIAS = {  # familias UNSPSC más frecuentes en contratos de obra
    "7210": "Mantenimiento y reparación de construcciones", "7211": "Edificaciones residenciales",
    "7212": "Edificaciones no residenciales", "7214": "Construcción pesada (vías, puentes, redes)",
    "7215": "Obras especializadas (eléctricas, hidráulicas, acabados)", "8110": "Servicios de ingeniería",
    "9511": "Vías y caminos", "9512": "Edificios y estructuras permanentes",
    "8310": "Servicios públicos (agua, alcantarillado, energía)", "0000": "Sin código",
}
GRUPO_HIST = {"hist_entidad_n": "Historial de la entidad", "hist_entidad_tasa": "Historial de la entidad",
              "hist_contratista_n": "Historial del contratista", "hist_contratista_tasa": "Historial del contratista"}
COLS_BASE = ["id_contrato", "nombre_entidad", "nit_entidad", "departamento", "ciudad", "orden", "sector", "rama",
             "entidad_centralizada", "estado_contrato", "codigo_de_categoria_principal", "objeto_del_contrato",
             "modalidad_de_contratacion", "fecha_de_firma", "fecha_de_fin_del_contrato", "documento_proveedor",
             "proveedor_adjudicado", "es_grupo", "es_pyme", "habilita_pago_adelantado", "origen_de_los_recursos",
             "destino_gasto", "urlproceso", "valor_del_contrato", "valor_pactado", "plazo_pactado_dias", "precio_base",
             "proveedores_unicos_con", "fin_pactado", "prorroga", "universo_modelo", "en_ejecucion", "tiene_proceso",
             "pct_adicion_valor", "n_mod_prorroga_txt", "dias_extra"]
CATEGORICAS_FORM = ["departamento", "modalidad_de_contratacion", "es_grupo", "es_pyme", "habilita_pago_adelantado",
                    "destino_gasto", "origen_de_los_recursos"]


# ----------------------------------------------------------------------------------------------- formato
def miles(x: float, dec: int = 0) -> str:
    return f"{x:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pesos(x: float) -> str:
    if x is None or pd.isna(x):
        return "sin dato"
    if abs(x) >= 1e9:
        return f"${miles(x / 1e9, 1)} mil M"
    if abs(x) >= 1e6:
        return f"${miles(x / 1e6)} M"
    return f"${miles(x / 1e3)} mil"


def pct(x: float, dec: int = 0) -> str:
    return "sin dato" if x is None or pd.isna(x) else f"{miles(x * 100, dec)} %"


def nombre_familia(f: str) -> str:
    return f"{f} · {FAMILIAS[f]}" if f in FAMILIAS else f"{f} · otro código UNSPSC"


def nivel_de(p: float) -> str:  # mismos cortes que 04_modelo.py (pd.cut con [0, 0.4, 0.7, 1])
    return "Alto" if p > 0.7 else ("Medio" if p > 0.4 else "Bajo")


def url_de(v) -> str | None:
    if isinstance(v, dict):
        return v.get("url")
    return v if isinstance(v, str) and v.startswith("http") else None


# ----------------------------------------------------------------------------------------------- motor
class Motor:
    def __init__(self):
        x = pd.read_parquet(PROC / "base_contratos.parquet", columns=COLS_BASE)
        x["nombre_entidad"] = x["nombre_entidad"].str.replace("\x96", "–", regex=False)  # guion dañado en SECOP
        x["url"] = x["urlproceso"].map(url_de)
        x = x.drop(columns="urlproceso")
        x["familia"] = x["codigo_de_categoria_principal"].fillna("").str.extract(r"(\d{4})")[0].fillna("0000")
        for c in ["nit_entidad", "documento_proveedor"]:
            x[c] = x[c].fillna("").astype(str)
        self.base = x
        self.por_id = x.drop_duplicates("id_contrato").set_index("id_contrato", drop=False)

        b = joblib.load(MOD / "modelo_prorroga.joblib")
        grupo = []
        for n in b["nombres_transformados"]:
            tipo, resto = n.split("__", 1)
            f = resto if tipo == "num" else next(c for c in sorted(F.CAT, key=len, reverse=True)
                                                if resto.startswith(c + "_"))
            grupo.append(GRUPO_HIST.get(f, F.NOMBRES[f]))
        b["grupo"] = np.array(grupo)
        b["logit_base"] = b["intercepto"] + float(np.dot(b["coeficientes"], b["media_transformada"]))
        b["tasa_nivel"] = {n["nivel"]: n["tasa_real"] for n in b["niveles"]}
        b["modelo_explicacion"] = (b["modelo"] if b["familia"] == "Regresión logística"
                                   else joblib.load(MOD / "modelo_logistico.joblib"))
        self.b = b
        self.p_base = float(1 / (1 + np.exp(-b["logit_base"])))

        self.u = x[x["universo_modelo"]].copy()
        self.grupos = {k: F.grupos_historial(self.u, k) for k in ["nit_entidad", "documento_proveedor"]}
        self.rangos = R.rangos_normales(self.u)
        self._causas = None

    # --- datos de apoyo
    def nombre_entidad(self, nit: str) -> str:
        s = self.base.loc[self.base["nit_entidad"] == nit, "nombre_entidad"]
        return str(s.mode().iloc[0]).title() if len(s) else "Entidad no registrada"

    def causas(self):
        """Justificaciones de prórroga con su causa (LLM si ya se clasificaron; si no, reglas de palabras clave)."""
        if self._causas is None:
            m = pd.read_parquet(PROC / "modificaciones_validas.parquet",
                                columns=["id_contrato", "fecha_aprob", "texto", "proposito_modificacion",
                                         "es_prorroga_txt"])
            m = m[m["es_prorroga_txt"]].drop(columns="es_prorroga_txt")
            archivo_llm = PROC / "causas_llm.parquet"
            if archivo_llm.exists():
                c = pd.read_parquet(archivo_llm)[["texto", "causa"]].drop_duplicates("texto")
                m = m.merge(c, on="texto", how="left")
                fuente = "llm"
            else:
                m["causa"] = pd.Series(None, index=m.index, dtype=object)
                fuente = "reglas"
            m["fuente_causa"] = np.where(m["causa"].notna(), "llm", "reglas")
            falta = m["causa"].isna()
            if falta.any():
                uniq = pd.Series(m.loc[falta, "texto"].unique())
                m["causa"] = m["causa"].fillna(m["texto"].map(dict(zip(uniq, uniq.map(clasificar_reglas)))))
            m = m.merge(self.base[["id_contrato", "nit_entidad", "nombre_entidad", "departamento", "familia"]],
                        on="id_contrato", how="left")
            m["anio"] = m["fecha_aprob"].dt.year
            m["mes"] = m["fecha_aprob"].dt.month
            self._causas = (m, fuente)
        return self._causas

    def causas_de(self, nit: str, depto: str) -> dict | None:
        m, _ = self.causas()
        s, ambito = m[m["nit_entidad"] == nit], "la entidad"
        if len(s) < 10:
            s, ambito = m[m["departamento"] == depto], f"{depto} (la entidad tiene pocas)"
        if s.empty:
            return None
        vc = s["causa"].value_counts(normalize=True)
        con = vc.drop("SIN_CAUSA", errors="ignore")
        return {"ambito": ambito, "n_justificaciones": int(len(s)), "pct_sin_causa": float(vc.get("SIN_CAUSA", 0)),
                "principales": [{"causa": k, "porcentaje": float(v)} for k, v in con.head(3).items()]}

    # --- un contrato
    def fila_de_contrato(self, r: pd.Series) -> dict:
        """Variables al firmar de un contrato de la base (mismas columnas que arma el formulario de la app)."""
        def txt(v):
            return "No Definido" if pd.isna(v) else str(v)
        return {
            "nit_entidad": r["nit_entidad"], "documento_proveedor": r["documento_proveedor"],
            "valor_pactado": float(r["valor_pactado"]), "plazo_pactado_dias": float(r["plazo_pactado_dias"]),
            "precio_base": float(r["precio_base"]) if pd.notna(r["precio_base"]) and r["precio_base"] > 0 else np.nan,
            "proveedores_unicos_con": float(r["proveedores_unicos_con"]) if pd.notna(r["proveedores_unicos_con"]) else 0.0,
            "modalidad_de_contratacion": txt(r["modalidad_de_contratacion"]), "departamento": txt(r["departamento"]),
            "orden": txt(r["orden"]), "sector": txt(r["sector"]), "rama": txt(r["rama"]),
            "entidad_centralizada": txt(r["entidad_centralizada"]), "es_grupo": txt(r["es_grupo"]),
            "es_pyme": txt(r["es_pyme"]), "habilita_pago_adelantado": txt(r["habilita_pago_adelantado"]),
            "destino_gasto": txt(r["destino_gasto"]), "origen_de_los_recursos": txt(r["origen_de_los_recursos"]),
            "codigo_de_categoria_principal": r["familia"], "fecha_de_firma": pd.Timestamp(r["fecha_de_firma"]),
        }

    def evaluar(self, fila: dict):
        b = self.b
        X = F.construir(pd.DataFrame([fila]), self.u, b["tasa_global"], self.grupos)
        p = float(b["modelo"].predict_proba(X)[0, 1])
        t = np.asarray(b["modelo_explicacion"].named_steps["prep"].transform(X)).ravel()
        aporte = pd.Series(b["coeficientes"] * (t - b["media_transformada"])).groupby(b["grupo"]).sum()
        return p, X.iloc[0], aporte

    def describir(self, grupo: str, fila: dict, xr: pd.Series) -> str:
        v, pl = fila["valor_pactado"], fila["plazo_pactado_dias"]
        tg = self.b["tasa_global"]

        def hist(pref):
            n = float(np.expm1(xr[f"hist_{pref}_n"]))
            if n < 0.5:
                return "sin contratos vencidos previos"
            k = xr[f"hist_{pref}_tasa"] * (n + F.SUAVIZADO) - F.SUAVIZADO * tg
            return f"{miles(k)} de {miles(n)} contratos vencidos se prorrogaron"

        fijos = {
            "Valor pactado": pesos(v), "Plazo pactado": f"{miles(pl)} días",
            "Valor por día de plazo": f"{pesos(v / max(pl, 1))} por día",
            "Descuento de la oferta frente al presupuesto": (
                "sin dato" if pd.isna(xr["descuento_oferta"]) else
                f"{pct(xr['descuento_oferta'])} por debajo del presupuesto" if xr["descuento_oferta"] >= 0 else
                f"{pct(-xr['descuento_oferta'])} por encima del presupuesto"),
            "Número de oferentes": f"{miles(np.expm1(xr['log_oferentes']))} oferentes",
            "Historial de la entidad": hist("entidad"), "Historial del contratista": hist("contratista"),
        }
        if grupo in fijos:
            return fijos[grupo]
        var = next(k for k, n in F.NOMBRES.items() if n == grupo)
        val = str(xr[var])
        if var == "familia_unspsc":
            return nombre_familia(val)
        if var == "mes_firma":
            return MESES[int(val) - 1] if val.isdigit() and 1 <= int(val) <= 12 else val
        return val

    def analizar(self, fila: dict, r: pd.Series | None = None) -> dict:
        """Todo lo que la app muestra de un contrato: probabilidad, aportes, alertas y hechos de la ficha."""
        p, xr, aporte = self.evaluar(fila)
        niv = nivel_de(p)
        top = aporte.reindex(aporte.abs().sort_values(ascending=False).index).head(8)
        d = pd.DataFrame({"variable": top.index, "aporte": top.values})
        d["efecto"] = np.where(d["aporte"] > 0, "Sube el riesgo", "Baja el riesgo")
        d["valor"] = [self.describir(v, fila, xr) for v in d["variable"]]
        alertas = R.todas(
            fila["valor_pactado"], fila["plazo_pactado_dias"], self.rangos,
            pct_adicion=float(r["pct_adicion_valor"]) if r is not None and pd.notna(r["pct_adicion_valor"]) else None,
            ya_prorrogado=bool(r["prorroga"]) if r is not None and r["en_ejecucion"] else None,
            n_prorrogas=int(r["n_mod_prorroga_txt"]) if r is not None else None,
            dias_extra=float(r["dias_extra"]) if r is not None and pd.notna(r["dias_extra"]) else None)
        hist_ent = None
        n_ent = float(np.expm1(xr["hist_entidad_n"]))
        if n_ent >= 0.5:
            k = xr["hist_entidad_tasa"] * (n_ent + F.SUAVIZADO) - F.SUAVIZADO * self.b["tasa_global"]
            hist_ent = {"contratos_vencidos": int(round(n_ent)), "tasa_prorroga": float(k / n_ent)}
        hechos = {
            "contrato": {
                "entidad": (str(r["nombre_entidad"]).title() if r is not None else self.nombre_entidad(fila["nit_entidad"])),
                "objeto": str(r["objeto_del_contrato"])[:300] if r is not None else None,
                "departamento": fila["departamento"], "modalidad": fila["modalidad_de_contratacion"],
                "tipo_de_obra": nombre_familia(str(fila["codigo_de_categoria_principal"])[:4]),
                "valor_pactado": pesos(fila["valor_pactado"]), "plazo_pactado": f"{miles(fila['plazo_pactado_dias'])} días",
            },
            "riesgo": {"probabilidad": p, "nivel": niv, "tasa_real_nivel": self.b["tasa_nivel"][niv],
                       "prob_contrato_promedio": self.p_base},
            "factores_sube": [{"variable": v, "valor": val} for v, val, a in zip(d["variable"], d["valor"], d["aporte"]) if a > 0],
            "factores_baja": [{"variable": v, "valor": val} for v, val, a in zip(d["variable"], d["valor"], d["aporte"]) if a < 0],
            "historial_entidad": hist_ent,
            "causas": self.causas_de(fila["nit_entidad"], fila["departamento"]),
            "alertas": [{"titulo": a["titulo"], "detalle": a["detalle"]} for a in alertas],
        }
        return {"p": p, "nivel": niv, "xr": xr, "aportes": d, "alertas": alertas, "hechos": hechos}
