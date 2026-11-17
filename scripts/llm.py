"""
Modelo de lenguaje (Claude Haiku 5.5): clasifica la causa escrita en cada prórroga.

Lo usan 07_llm_causas.py (todos los textos), 08_eval_llm.py (evaluación con los 60 casos) y la app.
Dos versiones del prompt, para compararlas en la evaluación:
  * simple:   solo la lista de categorías.
  * completo: definiciones, reglas para casos ambiguos y ejemplos. Los ejemplos son redactados por el equipo;
              ninguno sale de los 60 casos de prueba (para no evaluar con lo mismo que se le enseñó).
La respuesta se pide con salida estructurada (JSON Schema): el JSON es válido por construcción y la causa
solo puede ser una de las 10 categorías. Igual se verifica que devuelva cada id una sola vez.
"""
import json
import os
import time
from pathlib import Path

import anthropic

from causas import CATEGORIAS

ROOT = Path(__file__).resolve().parents[1]
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

MODELO = "claude-haiku-5-5"
PRECIO_ENTRADA, PRECIO_SALIDA = 0.10 / 1e6, 0.50 / 1e6  # USD por token (prompts de hasta 100 mil tokens)
MAX_CARACTERES = 2000  # justificaciones más largas se recortan (la causa suele estar al inicio); se reporta cuántas
EFFORT = "low"

ESQUEMA = {
    "type": "object",
    "properties": {
        "resultados": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "causa": {"type": "string", "enum": list(CATEGORIAS)},
                    "confianza": {"type": "string", "enum": ["alta", "media", "baja"]},
                },
                "required": ["id", "causa", "confianza"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["resultados"],
    "additionalProperties": False,
}

PROMPT_SIMPLE = (
    "Clasifica la causa de cada prórroga de contrato de obra pública en una de estas categorías: "
    + ", ".join(CATEGORIAS) + ". Devuelve un resultado por cada texto, con su id, la causa y tu confianza."
)

_definiciones = "\n".join(f"- {k}: {v}" for k, v in CATEGORIAS.items())
PROMPT_COMPLETO = f"""Eres un analista de contratación pública en Colombia. Vas a leer justificaciones de prórrogas \
de contratos de obra pública publicadas en SECOP II y a clasificar POR QUÉ se prorrogó cada contrato.

Categorías:
{_definiciones}

Reglas:
1. Elige la causa PRINCIPAL: el hecho que impidió terminar a tiempo. Si menciona varias, elige la que el texto \
presenta como motivo del atraso.
2. Es SIN_CAUSA si el texto solo dice quién pidió o avaló la prórroga (contratista, supervisor, interventoría, \
comité), que se anexa una justificación o documento, el número de la prórroga u otrosí, la nueva fecha o el nuevo \
plazo, o una finalidad genérica como "culminar las actividades" o "cumplir el objeto", sin decir por qué no se \
alcanzó a terminar.
3. Los nombres de entidades, dependencias, contratistas o del objeto del contrato no son causas (por ejemplo, \
"Secretaría de Seguridad" no es ORDEN_PUBLICO y "construcción de redes" no es PREDIOS_PERMISOS).
4. Si la prórroga acompaña una adición por ítems no previstos, obras adicionales o mayores cantidades, es \
MAYORES_CANTIDADES.
5. Usa SUSPENSION cuando la prórroga se explica por una suspensión, reanudación o reinicio y no se da otra causa \
de fondo.
6. Confianza: alta si la causa es explícita; media si hay que inferirla; baja si el texto es ambiguo o está cortado.

Ejemplos (inventados para ilustrar):
- "Se prorroga el contrato 30 días debido a las fuertes lluvias que impidieron la colocación de concreto" -> CLIMA
- "Prórroga No. 2 por solicitud del contratista, avalada por la interventoría" -> SIN_CAUSA
- "Se amplía el plazo por cuanto la empresa de energía no ha expedido la factibilidad de conexión" -> PREDIOS_PERMISOS
- "Se requiere ajustar los diseños hidráulicos aprobados, lo que impide continuar la obra" -> DISENO
- "Adición y prórroga para ejecutar ítems no previstos necesarios para la funcionalidad de la obra" -> MAYORES_CANTIDADES
- "Por el paro de transportadores no fue posible el acceso de volquetas al frente de obra" -> ORDEN_PUBLICO
- "Demora en la entrega de la tubería por parte del proveedor" -> SUMINISTROS
- "No se ha realizado el desembolso de los recursos del convenio" -> FINANCIERO
- "Se prorroga mientras se surte el trámite de cesión del contrato" -> ADMINISTRATIVO
- "Se reanuda el contrato y se prorroga por el tiempo que estuvo suspendido" -> SUSPENSION

Devuelve un resultado por cada texto, con el mismo id."""

PROMPTS = {"simple": PROMPT_SIMPLE, "completo": PROMPT_COMPLETO}

_cliente = None


def cliente() -> anthropic.Anthropic:
    global _cliente
    if _cliente is None:
        if not os.environ.get("ANTHROPIC_API_KEY", "").startswith("sk-ant-"):
            raise RuntimeError("Falta ANTHROPIC_API_KEY en el archivo .env")
        _cliente = anthropic.Anthropic(max_retries=4)
    return _cliente


def mensaje_usuario(items: list[tuple[str, str]]) -> str:
    partes = [f'<texto id="{i}">\n{t[:MAX_CARACTERES]}\n</texto>' for i, t in items]
    return "Clasifica estas justificaciones de prórroga:\n\n" + "\n\n".join(partes)


def clasificar_lote(items: list[tuple[str, str]], prompt: str = "completo", con_esquema: bool = True) -> dict:
    """items = [(id, texto original)]. Devuelve {'resultados': {id: (causa, confianza)}, 'ok_formato': bool,
    'ids_completos': bool, 'tokens_entrada', 'tokens_salida', 'usd', 'segundos', 'stop_reason', 'error'}."""
    sistema = PROMPTS[prompt]
    output_config = {"effort": EFFORT}
    if con_esquema:
        output_config["format"] = {"type": "json_schema", "schema": ESQUEMA}
    else:  # solo para medir el formato sin la garantía del esquema
        sistema += ('\n\nResponde SOLO con JSON con esta forma: {"resultados": [{"id": "...", "causa": "...", '
                    '"confianza": "alta|media|baja"}]}')
    out = {"resultados": {}, "ok_formato": False, "ids_completos": False, "tokens_entrada": 0, "tokens_salida": 0,
           "usd": 0.0, "segundos": 0.0, "stop_reason": None, "error": None}
    t0 = time.time()
    try:
        r = cliente().messages.create(model=MODELO, max_tokens=8000, system=sistema, output_config=output_config,
                                      messages=[{"role": "user", "content": mensaje_usuario(items)}])
    except anthropic.APIStatusError as e:
        out["error"] = f"HTTP {e.status_code}: {e.message}"
        out["segundos"] = time.time() - t0
        return out
    except anthropic.APIConnectionError as e:
        out["error"] = f"conexión: {e}"
        out["segundos"] = time.time() - t0
        return out
    out["segundos"] = time.time() - t0
    out["stop_reason"] = r.stop_reason
    out["tokens_entrada"], out["tokens_salida"] = r.usage.input_tokens, r.usage.output_tokens
    out["usd"] = r.usage.input_tokens * PRECIO_ENTRADA + r.usage.output_tokens * PRECIO_SALIDA
    if r.stop_reason in ("refusal", "max_tokens"):
        out["error"] = f"stop_reason={r.stop_reason}"
        return out
    texto = next((b.text for b in r.content if b.type == "text"), "")
    if not con_esquema:  # quitar cercas de código si las hay
        texto = texto.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(texto)
        res = data["resultados"]
        validos = {x["id"]: (x["causa"], x.get("confianza", "media")) for x in res
                   if isinstance(x, dict) and x.get("causa") in CATEGORIAS}
        out["ok_formato"] = len(validos) == len(res)
        out["resultados"] = validos
        ids = [x.get("id") for x in res if isinstance(x, dict)]
        out["ids_completos"] = sorted(ids) == sorted(i for i, _ in items)
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        out["error"] = f"JSON inválido: {e.__class__.__name__}"
    return out
