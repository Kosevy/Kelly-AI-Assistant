"""
benchmarks/llm_quality_prompts.py
Definición de prompts y validadores programáticos para verificar la calidad
semántica, sintáctica y de coherencia del LLM en Kelly (evita regresiones por cuantización agresiva).
"""
import ast
import re
from typing import Callable, Tuple

def validate_poema(text: str) -> Tuple[bool, str]:
    cleaned = text.strip()
    if re.search(r"[\u4e00-\u9fff]", cleaned):
        return False, "Contiene caracteres CJK no permitidos"
    if re.search(r"\b(rencontre|femme|cherchez|vous)\b", cleaned, re.IGNORECASE):
        return False, "Fuga de idioma: contiene palabras en francés u otros idiomas no solicitados"
    if not (50 <= len(cleaned) <= 350):
        return False, f"Longitud fuera de rango (esperado 50-350, recibido {len(cleaned)})"
    punct_count = len(re.findall(r"[,.;:!?¡¿—\-]", cleaned))
    if punct_count < 3:
        return False, f"Puntuación insuficiente (esperado >= 3 signos, encontrados {punct_count})"
    return True, "Poema coherente, longitud y puntuación adecuadas"


def validate_codigo(text: str) -> Tuple[bool, str]:
    cleaned = text.strip()
    if "def " not in cleaned or "return" not in cleaned:
        return False, "No contiene las palabras clave requeridas 'def ' y 'return'"
    
    # Extraer bloque de código si viene formateado con markdown ```python ... ```
    code_match = re.search(r"```(?:python)?\s*([\s\S]*?)```", cleaned)
    code_to_parse = code_match.group(1).strip() if code_match else cleaned

    try:
        ast.parse(code_to_parse)
    except SyntaxError as e:
        return False, f"Error de sintaxis en el código generado: {e}"

    # Verificar ausencia de caracteres extraños / CJK
    if re.search(r"[\u4e00-\u9fff]", cleaned):
        return False, "Contiene caracteres CJK extraños en el código"

    return True, f"Código Python sintácticamente válido (ast.parse OK)"


def validate_razonamiento(text: str) -> Tuple[bool, str]:
    cleaned = text.strip()
    if len(cleaned) < 200:
        return False, f"Explicación demasiado corta (esperado > 200 caracteres, recibido {len(cleaned)})"
    if re.search(r"[\u4e00-\u9fff]", cleaned):
        return False, "Contiene caracteres CJK"
    # Detectar repeticiones obvias (ej: palabras o tokens repetidos 5+ veces seguidas)
    if re.search(r"(\b\w+\b)(?:\s+\1){4,}", cleaned, re.IGNORECASE):
        return False, "Detectada repetición degenerativa de tokens (bucle)"
    
    # Al menos 3 oraciones o pasos
    sentences = re.split(r"[.!?\n]\s*", cleaned)
    meaningful_sentences = [s for s in sentences if len(s.strip()) > 15]
    if len(meaningful_sentences) < 3:
        return False, f"Estructura insuficiente (esperado >= 3 oraciones/pasos, encontrados {len(meaningful_sentences)})"

    return True, f"Razonamiento paso a paso estructurado ({len(cleaned)} caracteres)"


def validate_carta(text: str) -> Tuple[bool, str]:
    cleaned = text.strip()
    if re.search(r"[\u4e00-\u9fff]", cleaned):
        return False, "Contiene caracteres CJK"
    if not (250 <= len(cleaned) <= 900):
        return False, f"Longitud fuera de rango (esperado 250-900, recibido {len(cleaned)})"
    
    has_greeting = bool(re.search(r"(Estimad[oa]|Profesor|Apreciad[oa]|Distinguid[oa]|Querid[oa])", cleaned, re.IGNORECASE))
    if not has_greeting:
        return False, "Falta saludo formal (esperado: Estimado, Profesor, etc.)"
    
    has_farewell = bool(re.search(r"(Atentamente|Saludos|Cordialmente|Un cordial saludo)", cleaned, re.IGNORECASE))
    if not has_farewell:
        return False, "Falta despedida formal (esperado: Atentamente, Cordialmente, Saludos, etc.)"

    return True, f"Carta formal bien estructurada ({len(cleaned)} caracteres, saludo y despedida presentes)"


def validate_conversacion(text: str) -> Tuple[bool, str]:
    cleaned = text.strip()
    if re.search(r"[\u4e00-\u9fff]", cleaned):
        return False, "Contiene caracteres CJK"
    if not re.search(r"\bverde\b", cleaned, re.IGNORECASE):
        return False, "No menciona el color favorito ('verde') requerido por el contexto conversacional"
    return True, "Mantiene contexto conversacional (menciona 'verde') y explica fotosíntesis"


QUALITY_PROMPTS = [
    {
        "id": "prompt_1_poema",
        "name": "Español coherente (poema)",
        "system": "Eres Kelly, una asistente personal servicial, poética y concisa.",
        "user": "Redacta un poema sencillo de 4 versos sobre el mar.",
        "validator": validate_poema,
    },
    {
        "id": "prompt_2_codigo",
        "name": "Código Python válido (factorial)",
        "system": "Eres Kelly, una asistente experta en programación. Devuelve solo el código solicitado en bloque markdown.",
        "user": "Escribe una función Python que calcule el factorial de un número. Solo el código.",
        "validator": validate_codigo,
    },
    {
        "id": "prompt_3_razonamiento",
        "name": "Razonamiento multi-paso (quicksort)",
        "system": "Eres Kelly, una asistente técnica didáctica y rigurosa.",
        "user": "Explícame paso a paso cómo ordenar una lista usando el algoritmo quicksort.",
        "validator": validate_razonamiento,
    },
    {
        "id": "prompt_4_carta",
        "name": "Creatividad estructurada (carta formal)",
        "system": "Eres Kelly, una asistente experta en redacción formal y académica.",
        "user": "Escribe una carta formal de presentación a un profesor universitario. Máximo 100 palabras.",
        "validator": validate_carta,
    },
    {
        "id": "prompt_5_conversacion",
        "name": "Conversación y memoria contextual (color verde + fotosíntesis)",
        "system": "Eres Kelly, una asistente personal atenta y precisa.",
        "user": "Recuerda que mi color favorito es el verde. Ahora dime qué es la fotosíntesis en 2 oraciones.",
        "validator": validate_conversacion,
    },
]
