"""
¿Esta obra se va a prorrogar? - App web local (Streamlit)

Abrir con:   streamlit run app.py
Requiere haber corrido scripts/01 a 05 (data/processed, models/ y reports/).
Pestañas: 1) Evaluar un contrato  2) Causas de prórroga  3) Tablero de seguimiento  4) Cómo sabemos que funciona
La lógica (modelo, reglas, ficha) está en scripts/motor.py; aquí solo la interfaz.
"""
import json
import os
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass
import ficha as FI  # noqa: E402
import reglas as R  # noqa: E402
from causas import NOMBRES_CAUSA  # noqa: E402
from motor import (CATEGORICAS_FORM, CORTE_TEST, MESES, Motor, miles, nivel_de, nombre_familia, pct,  # noqa: E402
                   url_de)

PROC, MOD, REP, FIGS = ROOT / "data" / "processed", ROOT / "models", ROOT / "reports", ROOT / "figs"

# Paleta del anteproyecto (señalización vial de obra). Validada: contraste >= 4,3:1 sobre el asfalto y pares
# distinguibles con protanopia, deuteranopia y tritanopia (Delta E OKLab >= 10).
ASFALTO, AMARILLO, NARANJA, AZUL, VERDE = "#1F2326", "#F5B800", "#E4572E", "#5AA9E6", "#7FB77E"
TINTA, TINTA2, BARRA, PANEL, GRILLA = "#EEEFEC", "#C9CDD1", "#D5D7D3", "#2B3036", "#3A3F44"
COLOR_NIVEL = {"Bajo": VERDE, "Medio": AMARILLO, "Alto": NARANJA}
ICONO_NIVEL = {"Bajo": "▼", "Medio": "●", "Alto": "▲"}

st.set_page_config(page_title="¿Esta obra se va a prorrogar?", page_icon="🚧", layout="wide")
st.markdown(f"""<style>
h1, h2, h3, h4, h5, h6, p, li, label, input, textarea, .stMarkdown, [data-testid="stMetricValue"],
[data-testid="stMetricLabel"] p, .nivel, .alerta, .contrato {{ font-family: 'Bahnschrift', 'Segoe UI', sans-serif; }}
.senal {{ display:inline-block; background:{AMARILLO}; color:{ASFALTO}; font-weight:700; padding:2px 10px;
          border-radius:4px; letter-spacing:.06em; font-size:.8rem; }}
.nivel {{ border-radius:8px; padding:14px 18px; color:{ASFALTO}; margin-bottom:8px; }}
.nivel .p {{ font-size:2.8rem; font-weight:700; line-height:1.05; }}
.nivel .t {{ font-size:1rem; font-weight:700; letter-spacing:.06em; }}
.alerta {{ border-left:4px solid; padding:8px 12px; margin:6px 0; background:{PANEL}; border-radius:4px; }}
.contrato {{ background:{PANEL}; border-radius:8px; padding:10px 14px; margin:6px 0 12px 0; font-size:.92rem; }}
</style>""", unsafe_allow_html=True)

FALTAN = [p for p in [PROC / "base_contratos.parquet", PROC / "modificaciones_validas.parquet",
                      PROC / "predicciones_en_ejecucion.parquet", PROC / "predicciones_test.parquet",
                      MOD / "modelo_prorroga.joblib", REP / "metricas_modelo.json"] if not p.exists()]
if FALTAN:
    st.error("Faltan archivos para abrir la app. Corra en orden `scripts/01_descarga.py` a `05_prueba_fuga.py`.\n\n"
             + "\n".join(f"- `{p.relative_to(ROOT)}`" for p in FALTAN))
    st.stop()


# ----------------------------------------------------------------------------------------------- datos (en cache)
@st.cache_resource(show_spinner="Cargando contratos, modelo e historial de entidades…")
def motor() -> Motor:
    return Motor()


@st.cache_data
def opciones() -> dict:
    x = motor().base
    xp = x[x["tiene_proceso"] & x["valor_pactado"].notna()]
    ent = (xp.groupby(["nit_entidad", "nombre_entidad"]).size().reset_index(name="n")
           .sort_values("n", ascending=False).drop_duplicates("nit_entidad"))
    ultimo = xp.sort_values("fecha_de_firma").drop_duplicates("nit_entidad", keep="last").set_index("nit_entidad")
    return {
        "entidades": ent["nit_entidad"].tolist(),
        "etiqueta_entidad": dict(zip(ent["nit_entidad"], ent["nombre_entidad"].str.title() + " · NIT " + ent["nit_entidad"])),
        "datos_entidad": ultimo[["orden", "sector", "rama", "entidad_centralizada", "departamento"]].fillna(
            "No Definido").astype(str).to_dict("index"),
        **{c: xp[c].fillna("No Definido").astype(str).value_counts().index.tolist() for c in CATEGORICAS_FORM},
        "familias": xp["familia"].value_counts().index.tolist(),
    }


@st.cache_data(show_spinner="Cargando contratos en ejecución…")
def cargar_ejecucion() -> pd.DataFrame:
    e = pd.read_parquet(PROC / "predicciones_en_ejecucion.parquet")
    e["nombre_entidad"] = e["nombre_entidad"].str.replace("\x96", "–", regex=False)
    e["url"] = e["urlproceso"].map(url_de)
    e["nivel"] = e["prob_prorroga"].map(nivel_de)
    if "ya_prorrogado" not in e:
        e["ya_prorrogado"] = e["prorroga"].astype(bool)
    return e.drop(columns=["urlproceso"])


@st.cache_data
def cargar_causas():
    return motor().causas()


@st.cache_data
def ejemplos_demo() -> list[tuple[str, str]]:
    """Contratos reales para la demo, elegidos con reglas fijas (no a mano)."""
    mo = motor()
    x, e, t = mo.base, cargar_ejecucion(), pd.read_parquet(PROC / "predicciones_test.parquet")
    t = t.merge(x[["id_contrato", "valor_pactado"]], on="id_contrato")
    out = []

    def agrega(df, etiqueta):
        if len(df):
            idc = df.iloc[0]["id_contrato"]
            if idc not in [i for _, i in out]:
                out.append((etiqueta, idc))

    def cerca_de(df, col, objetivo):  # ejemplos representativos, no los extremos (99 % se ve sospechoso)
        return df.assign(_d=(df[col] - objetivo).abs()).sort_values("_d")

    agrega(cerca_de(e[e["valor_pactado"].between(2e9, 5e10) & ~e["ya_prorrogado"]], "prob_prorroga", 0.85),
           "Contrato grande en ejecución, riesgo alto")
    agrega(cerca_de(t[(t["y"] == 1) & t["valor_pactado"].between(5e8, 5e9)], "p", 0.85),
           "Periodo de prueba: el modelo dijo alto y se prorrogó")
    agrega(cerca_de(t[(t["y"] == 0) & (t["valor_pactado"] > 5e7)], "p", 0.20),
           "Periodo de prueba: el modelo dijo bajo y no se prorrogó")
    cerca = e[e["pct_adicion_valor"].between(0.45, 0.50)]
    if len(cerca):
        agrega(cerca_de(cerca, "prob_prorroga", 0.75), "Adiciones cerca del tope de la Ley 80")
    vpd = x["valor_pactado"] / x["plazo_pactado_dias"].clip(lower=1)
    agrega(x[x["universo_modelo"] & (vpd > mo.rangos["valor_por_dia"][1])].sort_values("valor_pactado", ascending=False),
           "Datos atípicos (valor por día fuera de rango)")
    return out


def llm_disponible() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY", "").startswith("sk-ant-"))


# ----------------------------------------------------------------------------------------------- gráficos
def grafico_aportes(d: pd.DataFrame) -> alt.Chart:
    orden = d["variable"].tolist()
    barras = alt.Chart(d).mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y("variable:N", sort=orden, title=None, scale=alt.Scale(paddingInner=0.35),
                axis=alt.Axis(labelLimit=280, labelColor=TINTA, labelFontSize=12, ticks=False, domain=False)),
        x=alt.X("aporte:Q", title="←  baja el riesgo        sube el riesgo  →",
                axis=alt.Axis(labels=False, ticks=False, grid=False, domain=False, titleColor=TINTA2)),
        color=alt.Color("efecto:N", title=None, legend=alt.Legend(orient="top", labelColor=TINTA),
                        scale=alt.Scale(domain=["Sube el riesgo", "Baja el riesgo"], range=[NARANJA, AZUL])),
        tooltip=[alt.Tooltip("variable:N", title="Variable"), alt.Tooltip("valor:N", title="En este contrato"),
                 alt.Tooltip("efecto:N", title="Efecto")],
    )
    cero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=TINTA2, strokeWidth=1).encode(x="x:Q")
    return (barras + cero).properties(height=32 * len(d) + 30).configure_view(stroke=None)


def mostrar_alertas(alertas: list[dict]):
    borde = {"alta": NARANJA, "media": AMARILLO, "info": TINTA2}
    icono = {"alta": "⛔", "media": "⚠️", "info": "ℹ️"}
    for a in alertas:
        st.markdown(f"<div class='alerta' style='border-color:{borde[a['nivel']]}'><b>{icono[a['nivel']]} "
                    f"{a['titulo']}</b><br>{a['detalle']}</div>", unsafe_allow_html=True)


# ----------------------------------------------------------------------------------------------- estado del formulario
def poner_entidad(nit: str):
    d = opciones()["datos_entidad"].get(nit, {})
    st.session_state["f_departamento"] = d.get("departamento", "No Definido")
    st.session_state["f_extra"] = {k: d.get(k, "No Definido") for k in ["orden", "sector", "rama", "entidad_centralizada"]}
    st.session_state["f_extra_nit"] = nit


def al_cambiar_entidad():
    poner_entidad(st.session_state["f_entidad"])


def fila_formulario() -> dict:
    ss = st.session_state
    nit = ss["f_entidad"]
    extra = ss["f_extra"] if ss.get("f_extra_nit") == nit else opciones()["datos_entidad"].get(nit, {})
    return {
        "nit_entidad": nit, "documento_proveedor": (ss.get("f_contratista") or "").strip(),
        "valor_pactado": ss["f_valor_m"] * 1e6, "plazo_pactado_dias": float(ss["f_plazo"]),
        "precio_base": ss["f_presupuesto_m"] * 1e6 if ss["f_presupuesto_m"] > 0 else np.nan,
        "proveedores_unicos_con": float(ss["f_oferentes"]),
        "modalidad_de_contratacion": ss["f_modalidad"], "departamento": ss["f_departamento"],
        "orden": extra.get("orden", "No Definido"), "sector": extra.get("sector", "No Definido"),
        "rama": extra.get("rama", "No Definido"), "entidad_centralizada": extra.get("entidad_centralizada", "No Definido"),
        "es_grupo": ss["f_consorcio"], "es_pyme": ss["f_pyme"], "habilita_pago_adelantado": ss["f_anticipo"],
        "destino_gasto": ss["f_destino"], "origen_de_los_recursos": ss["f_origen"],
        "codigo_de_categoria_principal": ss["f_familia"], "fecha_de_firma": pd.Timestamp(ss["f_fecha"]),
    }


def cargar_contrato(idc: str):
    ss = st.session_state
    idc = (idc or "").strip()
    if not idc:
        ss["msg_carga"] = ("warning", "Escriba el ID del contrato en SECOP II (empieza por CO1.PCCNTR).")
        return
    mo = motor()
    if idc not in mo.por_id.index:
        ss["msg_carga"] = ("error", f"No encontré el contrato **{idc}** entre los contratos de obra de SECOP II.")
        return
    r = mo.por_id.loc[idc]
    if not r["tiene_proceso"] or pd.isna(r["valor_pactado"]) or pd.isna(r["plazo_pactado_dias"]):
        ss["msg_carga"] = ("warning", f"El contrato **{idc}** no tiene proceso de contratación enlazado (p. ej. "
                                      f"contratación directa), así que no se conocen su plazo y valor pactados. "
                                      f"El modelo aplica a modalidades competitivas.")
        return
    op = opciones()
    fila = mo.fila_de_contrato(r)

    def en(lista, v):
        return v if v in op[lista] else op[lista][0]

    ss["f_entidad"] = r["nit_entidad"]
    ss["f_departamento"] = en("departamento", fila["departamento"])
    ss["f_modalidad"] = en("modalidad_de_contratacion", fila["modalidad_de_contratacion"])
    ss["f_valor_m"] = fila["valor_pactado"] / 1e6
    ss["f_plazo"] = int(round(fila["plazo_pactado_dias"]))
    ss["f_presupuesto_m"] = fila["precio_base"] / 1e6 if pd.notna(fila["precio_base"]) else 0.0
    ss["f_oferentes"] = int(fila["proveedores_unicos_con"])
    ss["f_fecha"] = fila["fecha_de_firma"].date()
    ss["f_familia"] = r["familia"] if r["familia"] in op["familias"] else op["familias"][0]
    ss["f_consorcio"] = en("es_grupo", fila["es_grupo"])
    ss["f_pyme"] = en("es_pyme", fila["es_pyme"])
    ss["f_anticipo"] = en("habilita_pago_adelantado", fila["habilita_pago_adelantado"])
    ss["f_destino"] = en("destino_gasto", fila["destino_gasto"])
    ss["f_origen"] = en("origen_de_los_recursos", fila["origen_de_los_recursos"])
    ss["f_contratista"] = r["documento_proveedor"]
    ss["f_extra"] = {k: fila[k] for k in ["orden", "sector", "rama", "entidad_centralizada"]}
    ss["f_extra_nit"] = r["nit_entidad"]
    ss["contrato"] = idc
    ss["msg_carga"] = None
    ss["fila_original"] = fila_formulario()  # como quedó en el formulario (para detectar si el usuario cambió algo)
    ss["fila_exacta"] = fila  # valores exactos del contrato (los mismos del tablero y del test)


def al_elegir_ejemplo():
    sel = st.session_state.get("f_ejemplo")
    if sel:
        cargar_contrato(sel)


def contrato_nuevo():
    ss = st.session_state
    ss["contrato"] = None
    ss["f_contratista"] = ""
    ss["msg_carga"] = None


if "f_entidad" not in st.session_state:  # primera carga: el primer ejemplo de la demo
    ej0 = ejemplos_demo()
    if ej0:
        cargar_contrato(ej0[0][1])
        st.session_state["f_ejemplo"] = ej0[0][1]

# ----------------------------------------------------------------------------------------------- encabezado
mo = motor()
b = mo.b
fecha_datos = pd.Timestamp(os.path.getmtime(PROC / "base_contratos.parquet"), unit="s").strftime("%d-%m-%Y")
st.markdown("<span class='senal'>OBRA PÚBLICA · SECOP II</span>", unsafe_allow_html=True)
st.title("¿Esta obra se va a prorrogar?")
st.caption(f"Riesgo de prórroga de contratos de obra pública estimado el día de la firma · reglas + machine learning "
           f"+ IA generativa · datos abiertos de datos.gov.co procesados el {fecha_datos}")

with st.sidebar:
    st.markdown("### Cómo leer el riesgo")
    for niv, rango in [("Alto", "más de 70 %"), ("Medio", "40–70 %"), ("Bajo", "40 % o menos")]:
        st.markdown(f"<span style='color:{COLOR_NIVEL[niv]}'>{ICONO_NIVEL[niv]}</span> **{niv}** ({rango}): "
                    f"se prorrogó el {pct(b['tasa_nivel'][niv])} de los contratos de prueba", unsafe_allow_html=True)
    st.divider()
    st.markdown("**Modelo de lenguaje (Claude Haiku 5.5):** " + ("conectado ✅" if llm_disponible() else
                "las fichas de los ejemplos ya las redactó y están guardadas; las de otros contratos usan una "
                "plantilla"))
    st.caption("Las prórrogas se detectan cruzando la columna oficial, las modificaciones y las fechas de SECOP II "
               "(la columna oficial sola deja por fuera casi la mitad).")

tab1, tab2, tab3, tab4 = st.tabs(["Evaluar un contrato", "Causas de prórroga", "Tablero de seguimiento",
                                  "¿Cómo sabemos que funciona?"])

# ----------------------------------------------------------------------------------------------- 1. evaluar
with tab1:
    op = opciones()
    izq, der = st.columns([5, 6], gap="large")
    with izq:
        st.markdown("#### Contrato")
        ej = ejemplos_demo()
        c1, c2 = st.columns([3, 2])
        c1.selectbox("Ejemplos reales", [i for _, i in ej], key="f_ejemplo",
                     format_func=lambda i: next(e for e, j in ej if j == i), on_change=al_elegir_ejemplo,
                     placeholder="Elegir un ejemplo…")
        c2.button("Contrato nuevo", on_click=contrato_nuevo, width="stretch",
                  help="Llenar el formulario para un contrato que aún no está en SECOP")
        c3, c4 = st.columns([3, 2], vertical_alignment="bottom")
        c3.text_input("Buscar por ID de SECOP", placeholder="CO1.PCCNTR.1234567", key="f_buscar")
        c4.button("Cargar", on_click=lambda: cargar_contrato(st.session_state.get("f_buscar")), width="stretch",
                  type="primary")
        msg = st.session_state.get("msg_carga")
        if msg:
            getattr(st, msg[0])(msg[1])

        st.selectbox("Entidad contratante", op["entidades"], key="f_entidad",
                     format_func=lambda n: op["etiqueta_entidad"].get(n, n), on_change=al_cambiar_entidad)
        if st.session_state.get("f_extra_nit") != st.session_state["f_entidad"]:
            poner_entidad(st.session_state["f_entidad"])
        a, bcol = st.columns(2)
        a.number_input("Valor pactado (millones de pesos)", min_value=1.0, step=50.0, format="%.0f", key="f_valor_m")
        bcol.number_input("Plazo pactado (días)", min_value=1, max_value=3650, step=15, key="f_plazo")
        a, bcol = st.columns(2)
        a.selectbox("Modalidad", op["modalidad_de_contratacion"], key="f_modalidad")
        bcol.selectbox("Departamento", op["departamento"], key="f_departamento")
        a, bcol = st.columns(2)
        a.number_input("Presupuesto oficial (millones; 0 = sin dato)", min_value=0.0, step=50.0, format="%.0f",
                       key="f_presupuesto_m")
        bcol.number_input("Número de oferentes", min_value=0, max_value=1000, step=1, key="f_oferentes")
        a, bcol = st.columns(2)
        a.date_input("Fecha de firma", key="f_fecha", format="DD/MM/YYYY")
        bcol.selectbox("Tipo de obra", op["familias"], key="f_familia", format_func=nombre_familia)
        a, bcol, c = st.columns(3)
        a.radio("Consorcio o unión temporal", op["es_grupo"], key="f_consorcio", horizontal=True)
        bcol.radio("Contratista pyme", op["es_pyme"], key="f_pyme", horizontal=True)
        c.radio("Pago adelantado", op["habilita_pago_adelantado"], key="f_anticipo", horizontal=True)
        with st.expander("Más datos (opcionales)"):
            st.selectbox("Destino del gasto", op["destino_gasto"], key="f_destino")
            st.selectbox("Origen de los recursos", op["origen_de_los_recursos"], key="f_origen")
            st.text_input("Documento del contratista (para su historial)", key="f_contratista")

    with der:
        fila = fila_formulario()
        idc_actual = st.session_state.get("contrato")
        r = mo.por_id.loc[idc_actual] if idc_actual else None
        fila_orig = st.session_state.get("fila_original")
        modificado = r is not None and fila_orig is not None and any(
            not (pd.isna(fila_orig[k]) and pd.isna(fila[k])) and fila_orig[k] != fila[k] for k in fila)
        if r is not None and not modificado and st.session_state.get("fila_exacta"):
            fila = st.session_state["fila_exacta"]  # contrato sin cambios: valores exactos (sin redondeo del formulario)
        res = mo.analizar(fila, r)
        p, niv, d, alertas, hechos = res["p"], res["nivel"], res["aportes"], res["alertas"], res["hechos"]

        delta_txt = ""
        if modificado:
            p_orig = mo.evaluar(st.session_state.get("fila_exacta") or fila_orig)[0]
            if abs(p - p_orig) >= 0.005:
                delta_txt = (f"<div style='font-size:.95rem;font-weight:600'>Escenario modificado: el contrato original "
                             f"tenía {pct(p_orig)} ({'+' if p > p_orig else '−'}{miles(abs(p - p_orig) * 100)} puntos)</div>")
        st.markdown(f"<div class='nivel' style='background:{COLOR_NIVEL[niv]}'><div class='t'>{ICONO_NIVEL[niv]} "
                    f"RIESGO {niv.upper()} DE PRÓRROGA</div><div class='p'>{pct(p)}</div>{delta_txt}</div>",
                    unsafe_allow_html=True)
        st.markdown(f"En los contratos de prueba con riesgo {niv.lower()}, se prorrogó el **{pct(b['tasa_nivel'][niv])}**. "
                    f"Un contrato promedio tiene {pct(mo.p_base)}.")

        if r is not None:
            periodo = ("del **periodo de prueba** (firmado desde jul-2024): el modelo nunca lo vio" if
                       r["fecha_de_firma"] >= CORTE_TEST else "del periodo de entrenamiento")
            if r["universo_modelo"]:
                real = "**se prorrogó**" if r["prorroga"] else "**no se prorrogó**"
                resultado = f"Resultado real: {real}. Es un contrato {periodo}."
            elif r["en_ejecucion"]:
                resultado = ("En ejecución: " + ("SECOP **ya registra una prórroga**." if r["prorroga"] else
                                                 "todavía sin prórroga registrada."))
            else:
                resultado = "Contrato sin resultado observable todavía."
            link = f" · [ver en SECOP]({r['url']})" if r["url"] else ""
            st.markdown(f"<div class='contrato'><b>{r['id_contrato']}</b> · {str(r['nombre_entidad']).title()} · "
                        f"{r['estado_contrato']}<br>{str(r['objeto_del_contrato'])[:260]}</div>", unsafe_allow_html=True)
            st.markdown(resultado + link)

        st.markdown("##### Qué empuja el riesgo")
        st.altair_chart(grafico_aportes(d), width="stretch")
        if b["familia"] != "Regresión logística":
            st.caption(f"La probabilidad sale del modelo final ({b['familia']}); la explicación, de la regresión "
                       f"logística de referencia.")
        with st.expander("Ver como tabla"):
            st.dataframe(d[["variable", "valor", "efecto"]].rename(columns={"variable": "Variable", "valor":
                         "En este contrato", "efecto": "Efecto"}), hide_index=True, width="stretch")

        st.markdown("##### Alertas por reglas")
        if alertas:
            mostrar_alertas(alertas)
        else:
            st.markdown("Sin alertas: adiciones bajo el 40 % del valor inicial y valores dentro del rango normal.")

        st.markdown("##### Ficha de riesgo")
        hl = FI.hechos_para_llm(hechos)
        k = FI.clave(hl)
        guardada = FI.leer_cache().get(k)
        if guardada is None and llm_disponible():
            if st.button("Redactar la ficha con Claude Haiku", icon="✨", help="Cuesta menos de un centavo de dólar"):
                with st.spinner("Redactando con Claude Haiku 5.5…"):
                    out = FI.ficha_llm(hl)
                if out["texto"]:
                    FI.guardar_en_cache(k, out, etiqueta=idc_actual or "formulario")
                    guardada = FI.leer_cache().get(k)
                else:
                    st.warning(f"No se pudo usar el modelo de lenguaje ({out['error']}). Se muestra la ficha con plantilla.")
        with st.container(border=True):
            st.markdown(guardada["texto"] if guardada else FI.ficha_plantilla(hechos))
        if guardada:
            ok = ("todas las cifras están en los datos de arriba ✅" if not guardada["no_respaldadas"] else
                  f"⚠️ cifras que no están en los datos: {', '.join(guardada['no_respaldadas'])}")
            st.caption(f"Redactada por {guardada['modelo']} solo con los datos de arriba · verificación automática: {ok}")
        else:
            st.caption("Ficha con plantilla (sin IA) a partir de los datos de arriba." +
                       ("" if llm_disponible() else " Sin API key no se puede redactar con el LLM."))

# ----------------------------------------------------------------------------------------------- 2. causas
with tab2:
    m, fuente = cargar_causas()
    if fuente == "reglas":
        st.info("Clasificación provisional con **reglas de palabras clave** (el baseline sin IA). Cuando se corra el "
                "modelo de lenguaje, esta pestaña muestra su clasificación.")
    else:
        st.caption(f"Causas clasificadas por Claude Haiku 5.5 ({pct((m['fuente_causa'] == 'llm').mean())} de las "
                   f"justificaciones; el resto, textos nuevos aún sin clasificar, con reglas).")
    f1, f2, f3 = st.columns([2, 2, 2])
    deps = f1.multiselect("Departamento", sorted(m["departamento"].dropna().unique()), placeholder="Todos")
    anios = sorted(m["anio"].dropna().astype(int).unique())
    rango = f2.select_slider("Año de la prórroga", options=anios, value=(anios[0], anios[-1])) if len(anios) > 1 else None
    fams = f3.multiselect("Tipo de obra", m["familia"].dropna().value_counts().index.tolist(),
                          format_func=nombre_familia, placeholder="Todos")
    s = m
    if deps:
        s = s[s["departamento"].isin(deps)]
    if rango:
        s = s[s["anio"].between(*rango)]
    if fams:
        s = s[s["familia"].isin(fams)]
    if s.empty or (s["causa"] != "SIN_CAUSA").sum() == 0:
        st.warning("No hay prórrogas con causa explícita para esos filtros.")
    else:
        vc = s["causa"].value_counts()
        con = vc.drop("SIN_CAUSA", errors="ignore")
        k1, k2, k3 = st.columns(3)
        k1.metric("Prórrogas con justificación escrita", miles(len(s)))
        k2.metric("No dicen la causa", pct(vc.get("SIN_CAUSA", 0) / len(s)))
        k3.metric(f"Causa más común ({pct(con.iloc[0] / con.sum())} de las que dicen la causa)",
                  NOMBRES_CAUSA[con.index[0]])
        g1, g2 = st.columns(2, gap="large")
        with g1:
            st.markdown("##### ¿Por qué se prorrogan?")
            st.caption("Porcentaje entre las justificaciones que sí dicen la causa")
            dc = pd.DataFrame({"causa": [NOMBRES_CAUSA[c] for c in con.index], "n": con.values})
            dc["pct"] = dc["n"] / dc["n"].sum()
            barras = alt.Chart(dc).mark_bar(cornerRadiusEnd=4, color=BARRA).encode(
                y=alt.Y("causa:N", sort="-x", title=None, scale=alt.Scale(paddingInner=0.3),
                        axis=alt.Axis(labelLimit=240, labelColor=TINTA, ticks=False, domain=False)),
                x=alt.X("pct:Q", title=None, axis=alt.Axis(format="%", gridColor=GRILLA, domain=False, tickCount=5)),
                tooltip=[alt.Tooltip("causa:N", title="Causa"), alt.Tooltip("pct:Q", format=".0%", title="%"),
                         alt.Tooltip("n:Q", format=",", title="Prórrogas")])
            etiqueta = alt.Chart(dc.head(1)).mark_text(align="left", dx=6, color=TINTA, fontWeight="bold").encode(
                y=alt.Y("causa:N", sort="-x"), x="pct:Q", text=alt.Text("pct:Q", format=".0%"))
            st.altair_chart((barras + etiqueta).properties(height=30 * len(dc) + 20).configure_view(stroke=None),
                            width="stretch")
        with g2:
            st.markdown("##### ¿En qué meses?")
            opciones_causa = list(con.index)
            elegida = st.selectbox("Causa", opciones_causa, format_func=lambda c: NOMBRES_CAUSA[c],
                                   index=opciones_causa.index("CLIMA") if "CLIMA" in opciones_causa else 0,
                                   label_visibility="collapsed")
            sc = s[s["causa"] != "SIN_CAUSA"]
            mm = sc.groupby("mes").agg(n=("causa", "size"), k=("causa", lambda z: (z == elegida).sum())).reindex(
                range(1, 13), fill_value=0).reset_index()
            mm["pct"] = np.where(mm["n"] > 0, mm["k"] / mm["n"].clip(lower=1), 0)
            mm["mes_txt"] = [MESES[i - 1] for i in mm["mes"]]
            st.caption(f"% de prórrogas por «{NOMBRES_CAUSA[elegida].lower()}», según el mes en que se aprobó la prórroga")
            ch = alt.Chart(mm).mark_bar(cornerRadiusEnd=4, color=BARRA).encode(
                x=alt.X("mes_txt:N", sort=MESES, title=None, axis=alt.Axis(labelColor=TINTA, ticks=False, labelAngle=0)),
                y=alt.Y("pct:Q", title=None, axis=alt.Axis(format="%", gridColor=GRILLA, domain=False, tickCount=4)),
                tooltip=[alt.Tooltip("mes_txt:N", title="Mes"), alt.Tooltip("pct:Q", format=".1%", title="%"),
                         alt.Tooltip("k:Q", title="Prórrogas por esta causa"), alt.Tooltip("n:Q", title="Con causa")])
            st.altair_chart(ch.properties(height=260).configure_view(stroke=None), width="stretch")
        st.markdown(f"##### Ejemplos de justificaciones: {NOMBRES_CAUSA[elegida].lower()}")
        ejs = s[s["causa"] == elegida]
        ejs = ejs.sample(min(6, len(ejs)), random_state=1) if len(ejs) else ejs
        st.dataframe(ejs[["nombre_entidad", "departamento", "fecha_aprob", "proposito_modificacion"]].rename(columns={
            "nombre_entidad": "Entidad", "departamento": "Departamento", "fecha_aprob": "Aprobada",
            "proposito_modificacion": "Texto de la modificación"}), hide_index=True, width="stretch",
            column_config={"Aprobada": st.column_config.DateColumn(format="DD/MM/YYYY")})

# ----------------------------------------------------------------------------------------------- 3. tablero
with tab3:
    e = cargar_ejecucion()
    f1, f2, f3, f4 = st.columns([2, 2, 2, 1.4], vertical_alignment="bottom")
    deps = f1.multiselect("Departamento ", sorted(e["departamento"].dropna().unique()), placeholder="Todos")
    texto = f2.text_input("Entidad (buscar)", placeholder="p. ej. INVIAS, alcaldía de…")
    nivs = f3.multiselect("Nivel de riesgo", ["Alto", "Medio", "Bajo"], placeholder="Todos")
    solo80 = f4.toggle("Solo alertas Ley 80")
    s = e
    if deps:
        s = s[s["departamento"].isin(deps)]
    if texto:
        s = s[s["nombre_entidad"].fillna("").str.contains(texto, case=False, regex=False)]
    if nivs:
        s = s[s["nivel"].isin(nivs)]
    if solo80:
        s = s[s["pct_adicion_valor"] >= R.ZONA_ALERTA_LEY80]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Contratos en ejecución", miles(len(s)))
    k2.metric(f"Riesgo alto ({pct((s['nivel'] == 'Alto').mean()) if len(s) else '–'})",
              miles((s["nivel"] == "Alto").sum()))
    k3.metric("Ya registran prórroga", pct(s["ya_prorrogado"].mean()) if len(s) else "–")
    k4.metric("Alertas Ley 80 (≥ 40 % adicionado)", miles((s["pct_adicion_valor"] >= R.ZONA_ALERTA_LEY80).sum()))
    nv = s[~s["universo_modelo"].astype(bool)]  # solo contratos que el modelo no usó para entrenar ni probar
    if len(nv) and nv["nivel"].nunique() > 1:
        ya = nv.groupby("nivel")["ya_prorrogado"].mean()
        partes = [f"riesgo {n.lower()}: **{pct(ya[n])}**" for n in ["Alto", "Medio", "Bajo"] if n in ya]
        st.markdown("Ya registran prórroga hoy, según el riesgo que les dio el modelo — " + " · ".join(partes))
        st.caption(f"Calculado con {miles(len(nv))} contratos que el modelo nunca vio; como siguen en ejecución, "
                   f"estas tasas solo pueden subir.")
    t = s.sort_values("prob_prorroga", ascending=False).assign(
        prob=lambda z: z["prob_prorroga"] * 100, adic=lambda z: z["pct_adicion_valor"] * 100,
        valor_m=lambda z: z["valor_pactado"] / 1e6, objeto=lambda z: z["objeto_del_contrato"].fillna("").str[:140])
    cols = {"prob": "Prob. prórroga", "nivel": "Nivel", "ya_prorrogado": "Ya prorrogado", "adic": "Adiciones",
            "nombre_entidad": "Entidad", "departamento": "Departamento", "objeto": "Objeto", "valor_m": "Valor pactado (M)",
            "plazo_pactado_dias": "Plazo (días)", "fecha_de_firma": "Firma", "fecha_de_fin_del_contrato": "Fin actual",
            "url": "SECOP", "id_contrato": "ID"}
    st.dataframe(t[list(cols)].rename(columns=cols), hide_index=True, width="stretch", height=520, column_config={
        "Prob. prórroga": st.column_config.ProgressColumn(format="%.0f %%", min_value=0, max_value=100),
        "Ya prorrogado": st.column_config.CheckboxColumn(),
        "Adiciones": st.column_config.NumberColumn(format="%.0f %%", help="Adiciones en valor sobre el valor pactado"),
        "Valor pactado (M)": st.column_config.NumberColumn(format="%.0f"),
        "Plazo (días)": st.column_config.NumberColumn(format="%.0f"),
        "Firma": st.column_config.DateColumn(format="DD/MM/YYYY"),
        "Fin actual": st.column_config.DateColumn(format="DD/MM/YYYY"),
        "SECOP": st.column_config.LinkColumn(display_text="Abrir"),
    })
    st.caption("Ordenados de mayor a menor riesgo. Para ver por qué, copie el ID en la pestaña «Evaluar un contrato».")
    st.download_button("Descargar CSV", t[list(cols)].rename(columns=cols).to_csv(index=False).encode("utf-8-sig"),
                       "contratos_en_ejecucion_riesgo.csv", "text/csv")

# ----------------------------------------------------------------------------------------------- 4. validacion
with tab4:
    met = json.loads((REP / "metricas_modelo.json").read_text(encoding="utf-8"))
    st.markdown("#### Machine learning: probado con contratos que el modelo nunca vio")
    st.markdown(f"Entrenado con contratos firmados hasta jun-2024 y probado **una sola vez** con "
                f"{miles(met.get('n_test', 0))} contratos firmados desde jul-2024 (partición temporal).")
    filas = [{"Modelo": nom, "ROC-AUC": m_["roc_auc"], "PR-AUC": m_["pr_auc"],
              "Exactitud balanceada": m_["exactitud_balanceada"], "MCC": m_["mcc"], "Brier": m_["brier"]}
             for nom, m_ in met["test"].items()]
    st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in
                                ["ROC-AUC", "PR-AUC", "Exactitud balanceada", "MCC", "Brier"]})
    fuga = REP / "prueba_fuga.json"
    if fuga.exists():
        auc_f = json.loads(fuga.read_text())["roc_auc_con_fuga"]
        st.markdown(f"**Prueba de fuga:** con los campos que SECOP actualiza después de firmar (duración y valor "
                    f"actuales) el ROC-AUC «subiría» a {miles(auc_f, 3)}. Es falso: esos campos ya contienen la prórroga. "
                    f"El modelo no los usa.")
    c1, c2 = st.columns(2)
    for col, fig, cap in [(c1, "11_comparacion_modelos.png", "Comparación con los baselines"),
                          (c2, "13_calibracion.png", "Calibración: la probabilidad predicha coincide con la real"),
                          (c1, "12_sobreajuste.png", "Por qué no el random forest: memoriza el entrenamiento"),
                          (c2, "14_importancia.png", "Variables que más pesan")]:
        if (FIGS / fig).exists():
            col.image(str(FIGS / fig), caption=cap, width="stretch")
    st.markdown("#### Datos")
    c1, c2 = st.columns(2)
    for col, fig, cap in [(c1, "01_subregistro.png", "La columna oficial no ve casi la mitad de las prórrogas"),
                          (c2, "10_adiciones.png", "Las adiciones se acumulan justo antes del tope de la Ley 80")]:
        if (FIGS / fig).exists():
            col.image(str(FIGS / fig), caption=cap, width="stretch")
    st.markdown("#### Modelo de lenguaje: clasificación de causas")
    ev = REP / "eval_llm.json"
    if ev.exists():
        el = json.loads(ev.read_text(encoding="utf-8"))
        if el["fuente_respuesta"].startswith("PROVISIONAL"):
            st.warning("Resultados provisionales: la respuesta correcta todavía no es la del etiquetado a ciegas del equipo.")
        else:
            st.markdown(f"Respuesta correcta: {el['fuente_respuesta'].lower()}.")
        filas = [{"Clasificador": v["variante"], "Exactitud": v["exactitud"] * 100,
                  "Con causa explícita": (v["exactitud_con_causa_explicita"] or 0) * 100,
                  "Sin causa": (v["exactitud_sin_causa"] or 0) * 100, "F1 macro": v["f1_macro"],
                  "JSON válido": v["pct_lotes_json_valido"] * 100 if "pct_lotes_json_valido" in v else np.nan,
                  "USD por 1000 textos": v.get("usd_por_1000_textos", np.nan)}
                 for v in el["variantes"]]
        st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch", column_config={
            **{c: st.column_config.NumberColumn(format="%.0f %%") for c in
               ["Exactitud", "Con causa explícita", "Sin causa", "JSON válido"]},
            "F1 macro": st.column_config.NumberColumn(format="%.3f"),
            "USD por 1000 textos": st.column_config.NumberColumn(format="%.3f")})
        st.markdown(f"**Consistencia:** con el prompt completo, el {pct(el['consistencia_3_repeticiones'])} de los casos "
                    f"recibió la misma causa en 3 repeticiones con los textos en distinto orden. Modelo: {el['modelo']} "
                    f"(effort {el['effort']}), lotes de {el['lote']} textos con salida estructurada.")
        if el.get("casos_borde"):
            cb = el["casos_borde"]
            st.markdown(f"**Casos borde** (texto vacío, intento de manipular el prompt, nombres de entidades, otro "
                        f"idioma…): {cb['aciertos']} de {cb['n']} correctos.")
        if el.get("errores_prompt_completo"):
            with st.expander(f"Ver los {len(el['errores_prompt_completo'])} errores del prompt completo"):
                st.dataframe(pd.DataFrame(el["errores_prompt_completo"]), hide_index=True, width="stretch")
    else:
        st.info("Pendiente: evaluación del LLM con los 60 casos etiquetados por el equipo.")
    fl = REP / "eval_flujo.json"
    st.markdown("#### Flujo completo de punta a punta")
    if fl.exists():
        ef = json.loads(fl.read_text(encoding="utf-8"))
        st.markdown(ef["resumen"])
        st.dataframe(pd.DataFrame(ef["contratos"]), hide_index=True, width="stretch")
    else:
        st.info("Pendiente: evaluación de punta a punta (scripts/09_eval_flujo.py).")
