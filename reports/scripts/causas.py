"""
Categorias de causa de prorroga y clasificador por REGLAS de palabras clave (baseline sin IA).
Lo usan: 06_causas_reglas.py, la evaluacion del LLM y la app.
"""
import re

CATEGORIAS = {
    "CLIMA": "Lluvias, ola invernal, fenómeno de La Niña, condiciones climáticas o del terreno por clima.",
    "DISENO": "Ajustes o cambios en diseños, estudios, planos o especificaciones técnicas; rediseños.",
    "MAYORES_CANTIDADES": "Mayores cantidades de obra, obras adicionales o complementarias, ítems no previstos.",
    "PREDIOS_PERMISOS": "Predios, servidumbres, permisos, licencias, autorizaciones o interferencias con redes.",
    "SUMINISTROS": "Escasez o demora de materiales, insumos, equipos o proveedores.",
    "ORDEN_PUBLICO": "Orden público, paros, bloqueos, seguridad o problemas con la comunidad.",
    "FINANCIERO": "Recursos, presupuesto, disponibilidad presupuestal, pagos, anticipo o vigencias.",
    "ADMINISTRATIVO": "Trámites, aprobaciones, cambios de supervisión o interventoría, cesión, actas pendientes.",
    "SUSPENSION": "La prórroga se debe a una suspensión o reinicio del contrato.",
    "SIN_CAUSA": "El texto no dice por qué se prorroga (p. ej. 'prórroga 1', 'ver documento adjunto').",
}

NOMBRES_CAUSA = {  # para mostrar en la app y en la ficha
    "CLIMA": "Clima y lluvias", "DISENO": "Diseños y estudios", "MAYORES_CANTIDADES": "Mayores cantidades de obra",
    "PREDIOS_PERMISOS": "Predios y permisos", "SUMINISTROS": "Suministros y materiales",
    "ORDEN_PUBLICO": "Orden público y comunidad", "FINANCIERO": "Recursos y pagos",
    "ADMINISTRATIVO": "Trámites administrativos", "SUSPENSION": "Ligada a una suspensión",
    "SIN_CAUSA": "Sin causa explícita",
}

# Orden = prioridad cuando el texto menciona varias causas (de la mas especifica a la mas general)
REGLAS = [
    ("CLIMA", r"lluvi|invern|precipitac|nino|nina|clima|temporal|creciente|inundac|aguacero"),
    ("ORDEN_PUBLICO", r"orden publico|paro\b|paros\b|bloqueo|protesta|comunidad|seguridad|grupos armados|amenaza"),
    ("PREDIOS_PERMISOS", r"predio|servidumbre|permiso|licencia|autorizacion|interferencia|redes de servicio|redes electric|redes de acueducto"),
    ("SUMINISTROS", r"material|insumo|suministr|escasez|desabastec|importac|maquinaria"),
    ("DISENO", r"disen|planos|especificac|rediseno|topograf|geotecn|estudios? (?:tecnicos?|de suelos?|hidraulic)"),
    ("MAYORES_CANTIDADES", r"mayores cantidades|cantidades de obra|obras? adicional|obras? complementari|items? no previst"
                           r"|items? nuevos|balance de obra|obra extra|no previst"),
    ("FINANCIERO", r"recurso|presupuest|cdp|disponibilidad|pago|anticipo|vigencia|financ|desembolso"),
    ("SUSPENSION", r"suspen|reinici|reanud"),
    ("ADMINISTRATIVO", r"tramite|aprobac|supervis|interventor|cesion|acta|firma|liquidac|polizas?|garantia"),
]
_REGLAS = [(c, re.compile(p)) for c, p in REGLAS]


def clasificar_reglas(texto_normalizado: str) -> str:
    """Devuelve la primera categoria cuya regla aparece en el texto; si ninguna, SIN_CAUSA."""
    t = texto_normalizado or ""
    for cat, rx in _REGLAS:
        if rx.search(t):
            return cat
    return "SIN_CAUSA"
