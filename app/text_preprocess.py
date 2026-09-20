"""
app/text_preprocess.py
Text normalization pipeline for Coqui XTTS-v2 speech synthesis.

Handles:
- Isolated letter sequences synthesized as mumbles/fade-outs.
- Acronyms expanded phonetically or semantically.
- Numeric expressions converted to Spanish spoken words via num2words.
- Stripping non-printable characters, emojis, and markdown artifacts.

Usage:
    from app.text_preprocess import normalize_for_tts
    clean = normalize_for_tts("La IA generó 3 resultados: A, B, C.")
"""
from __future__ import annotations

import re
import unicodedata

from num2words import num2words

# ─── Tabla fonética del alfabeto español ─────────────────────────────────────
# Cada letra → su nombre fonético en español
_LETTER_NAMES: dict[str, str] = {
    "A": "a", "B": "be", "C": "ce", "D": "de", "E": "e",
    "F": "efe", "G": "ge", "H": "hache", "I": "i", "J": "jota",
    "K": "ka", "L": "ele", "M": "eme", "N": "ene", "Ñ": "eñe",
    "O": "o", "P": "pe", "Q": "cu", "R": "erre", "S": "ese",
    "T": "te", "U": "u", "V": "uve", "W": "uve doble",
    "X": "equis", "Y": "ye", "Z": "zeta",
}

# ─── Siglas comunes con expansión semántica ───────────────────────────────────
# Editable: ampliar según necesidades del usuario
_ACRONYM_MAP: dict[str, str] = {
    "IA":   "inteligencia artificial",
    "AI":   "inteligencia artificial",
    "PC":   "computadora",
    "URL":  "u erre ele",
    "HTTP": "hache te te pe",
    "HTTPS":"hache te te pe ese",
    "API":  "a pe i",
    "CPU":  "ce pe u",
    "GPU":  "ge pe u",
    "RAM":  "erre a eme",
    "SSD":  "ese ese de",
    "USB":  "u ese be",
    "PDF":  "pe de efe",
    "FAQ":  "preguntas frecuentes",
    "GPS":  "ge pe ese",
    "LED":  "ele e de",
    "OK":   "okay",
    "TV":   "televisión",
    "VPN":  "ve pe ene",
    "SQL":  "ese cu ele",
    "UI":   "interfaz de usuario",
    "UX":   "experiencia de usuario",
    "LLM":  "modelo de lenguaje grande",
    "TTS":  "síntesis de voz",
    "STT":  "reconocimiento de voz",
    "VAD":  "detección de actividad de voz",
}

# Regex para detectar secuencias de letras separadas por coma/espacio/punto
# Ejemplo: "A, B, C" | "A B C" | "a, b, c"
_LETTER_SEQ_RE = re.compile(
    r"\b([A-ZÑa-zñ])[,\s]+([A-ZÑa-zñ])(?:[,\s]+([A-ZÑa-zñ]))*\b"
)

# Regex para detectar números (enteros y decimales)
_NUMBER_RE = re.compile(r"\b(\d+(?:[.,]\d+)?)\b")

# Regex para emojis y caracteres no imprimibles (excluye ASCII + español)
_EMOJI_RE = re.compile(
    "["
    "\U00010000-\U0010FFFF"   # supplementary planes (emojis, símbolos)
    "\U00002600-\U000027BF"   # misceláneos: ☀ ★ etc.
    "\U0001F300-\U0001F9FF"   # emojis populares
    "]+",
    flags=re.UNICODE,
)


def _expand_acronym(text: str) -> str:
    """Reemplaza siglas conocidas por su expansión fonética/semántica."""
    for acronym, expansion in _ACRONYM_MAP.items():
        # Palabra completa, mayúsculas, no parte de otra palabra
        text = re.sub(rf"\b{re.escape(acronym)}\b", expansion, text)
    return text


def _expand_letter_sequences(text: str) -> str:
    """
    Convierte secuencias de letras sueltas separadas por coma/espacio
    en sus nombres fonéticos.
    Ejemplo: "A, B, C" → "a, be, ce"
    """
    # Buscamos patrones: letra, coma/espacio, letra (al menos 2 letras sueltas)
    def _replace_seq(m: re.Match) -> str:
        full = m.group(0)
        # Extraer las letras individuales
        letters = re.findall(r"[A-ZÑa-zñ]", full)
        if len(letters) < 2:
            return full
        named = [_LETTER_NAMES.get(l.upper(), l.lower()) for l in letters]
        return ", ".join(named)

    # Patrón: letra seguida de coma+espacio o solo espacio, repite
    pattern = re.compile(r"\b([A-ZÑa-zñ](?:[,\s]+[A-ZÑa-zñ])+)\b")
    return pattern.sub(_replace_seq, text)


def _expand_numbers(text: str) -> str:
    """Convierte números a su forma en palabras en español."""
    def _replace_num(m: re.Match) -> str:
        raw = m.group(1).replace(",", ".")
        try:
            if "." in raw:
                # Número decimal
                return num2words(float(raw), lang="es")
            else:
                return num2words(int(raw), lang="es")
        except (ValueError, OverflowError):
            return raw  # Si falla, dejar el número original

    return _NUMBER_RE.sub(_replace_num, text)


def _normalize_punctuation(text: str) -> str:
    """
    - Colapsa espacios múltiples.
    - Normaliza puntos suspensivos.
    - Elimina emojis y caracteres no imprimibles.
    - Preserva ¿ ¡ ñ á-ú.
    """
    # Emojis
    text = _EMOJI_RE.sub("", text)

    # Caracteres de control (excepto newline)
    text = "".join(
        ch for ch in text
        if unicodedata.category(ch)[0] != "C" or ch == "\n"
    )

    # Puntos suspensivos → punto simple
    text = re.sub(r"\.{2,}", ".", text)

    # Markdown básico: asteriscos, guiones bajos, backticks
    text = re.sub(r"[*_`#]", "", text)

    # Colapsar espacios múltiples
    text = re.sub(r"[ \t]+", " ", text)

    # Limpiar espacios antes de puntuación
    text = re.sub(r" ([.,;:!?])", r"\1", text)

    return text.strip()


def normalize_for_tts(text: str) -> str:
    """
    Complete text normalization pipeline for XTTS-v2.

    Stages:
    1. Expand known acronyms phonetically/semantically.
    2. Expand isolated letter sequences to spoken letter names.
    3. Expand digits and decimals to Spanish spoken words.
    4. Normalize punctuation and strip unsupported symbols/emojis.
    """
    if not text or not text.strip():
        return text

    text = _expand_acronym(text)
    text = _expand_letter_sequences(text)
    text = _expand_numbers(text)
    text = _normalize_punctuation(text)

    return text

