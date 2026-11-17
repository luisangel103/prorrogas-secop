"""
Reglas de contratacion (la parte simbolica de la IA): dan certeza donde no hace falta predecir.

  1. Ley 80 de 1993, art. 40, paragrafo: los contratos no pueden adicionarse en mas del 50 % de su valor inicial.
     Alerta media desde el 40 %, alta por encima del 50 %.
     (La norma lo expresa en salarios minimos; aqui se compara en pesos nominales, una aproximacion.)
  2. Valores atipicos: valor, plazo o valor por dia fuera del rango del 99 % de los contratos historicos
     (p. ej. 418 mil millones con plazo de 1 mes). Suelen ser errores de registro y el modelo es menos confiable.
  3. Prorroga ya registrada (contratos en ejecucion): SECOP ya muestra una prorroga.
Lo usan la app y el flujo de Prefect.
"""
import numpy as np
import pandas as pd

TOPE_LEY80 = 0.50
ZONA_ALERTA_LEY80 = 0.40
PCT_BAJO, PCT_ALTO = 0.005, 0.995


def rangos_normales(universo: pd.DataFrame) -> dict:
    """Percentiles 0,5 % y 99,5 % del universo historico para valor, plazo y valor por dia."""
    v = universo["valor_pactado"].astype(float)
    p = universo["plazo_pactado_dias"].astype(float)
    vpd = v / p.clip(lower=1)
    return {nombre: (float(s.quantile(PCT_BAJO)), float(s.quantile(PCT_ALTO)))
            for nombre, s in [("valor", v), ("plazo", p), ("valor_por_dia", vpd)]}


def _pesos(x: float) -> str:
    if x >= 1e9:
        return f"${x / 1e9:,.1f} mil M".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"${x / 1e6:,.0f} M".replace(",", ".")


def alerta_ley80(pct_adicion: float | None, valor_pactado: float) -> dict | None:
    if pct_adicion is None or not np.isfinite(pct_adicion) or pct_adicion < ZONA_ALERTA_LEY80:
        return None
    if pct_adicion > TOPE_LEY80:
        return {"nivel": "alta", "titulo": "Supera el tope de adiciones de la Ley 80",
                "detalle": f"Las adiciones suman {pct_adicion:.0%} del valor inicial ({_pesos(valor_pactado)}), por encima "
                           f"del 50 % que permite el art. 40 de la Ley 80 de 1993. Puede ser un incumplimiento o un "
                           f"error de registro en SECOP: hay que revisarlo."}
    margen = (TOPE_LEY80 - pct_adicion) * valor_pactado
    return {"nivel": "media", "titulo": "Cerca del tope de adiciones de la Ley 80",
            "detalle": f"Las adiciones ya suman {pct_adicion:.0%} del valor inicial. Solo quedan {_pesos(margen)} antes del "
                       f"tope del 50 % (Ley 80 de 1993, art. 40): si hace falta más, hay que revisar el alcance, "
                       f"no adicionar."}


def alertas_atipicos(valor: float, plazo_dias: float, rangos: dict) -> list[dict]:
    out = []
    vpd = valor / max(plazo_dias, 1)
    for nombre, x, fmt, texto in [
        ("valor_por_dia", vpd, _pesos, "Valor por día de plazo"),
        ("plazo", plazo_dias, lambda d: f"{d:,.0f} días".replace(",", "."), "Plazo pactado"),
        ("valor", valor, _pesos, "Valor pactado"),
    ]:
        lo, hi = rangos[nombre]
        if x < lo or x > hi:
            out.append({"nivel": "media", "titulo": f"{texto} atípico",
                        "detalle": f"{texto}: {fmt(x)}. El 99 % de los contratos históricos está entre {fmt(lo)} y "
                                   f"{fmt(hi)}. Revise si el valor y el plazo están bien registrados; el modelo es "
                                   f"menos confiable con datos fuera de ese rango."})
    return out


def alerta_ya_prorrogado(ya_prorrogado: bool | None, n_prorrogas: int | None, dias_extra: float | None) -> dict | None:
    if not ya_prorrogado:
        return None
    partes = []
    if n_prorrogas:
        partes.append(f"{int(n_prorrogas)} modificación(es) de prórroga publicadas")
    if dias_extra and np.isfinite(dias_extra) and dias_extra > 0:
        partes.append(f"unos {dias_extra:,.0f} días más que el plazo pactado".replace(",", "."))
    return {"nivel": "info", "titulo": "SECOP ya registra una prórroga",
            "detalle": "Este contrato ya se prorrogó" + (": " + "; ".join(partes) if partes else "") + "."}


def todas(valor: float, plazo_dias: float, rangos: dict, pct_adicion: float | None = None,
          ya_prorrogado: bool | None = None, n_prorrogas: int | None = None, dias_extra: float | None = None) -> list[dict]:
    out = [alerta_ley80(pct_adicion, valor), *alertas_atipicos(valor, plazo_dias, rangos),
           alerta_ya_prorrogado(ya_prorrogado, n_prorrogas, dias_extra)]
    return [a for a in out if a]
