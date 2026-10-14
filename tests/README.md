# Kelly — Suite de Tests

Kelly implementa una estrategia de pruebas en dos niveles: tests unitarios puros (aislados, sin modelos ni servidor) y tests de integración/protocolo E2E (requieren servidor FastAPI y LLM en vivo).

---

## 1. Tests rápidos (unitarios)

Ejecuta todas las pruebas unitarias de procesamiento de texto, segmentación de oraciones, prompts, protocolo y configuración sin dependencias externas:

```powershell
pytest -v
```

> **Nota:** Si el servidor no está corriendo, los tests marcados con `@pytest.mark.e2e` se saltarán automáticamente (`SKIPPED`) sin fallar la ejecución.

---

## 2. Tests E2E (requieren servidor vivo)

Prueban el flujo completo WebSocket, streaming token a token (`llm_chunk`), generación de bloques Markdown con historial, síntesis de audio streaming con XTTS-v2 (`audio_chunk_start/end`) y servido de librerías vendorizadas.

### Paso 1: Arrancar Kelly en otra terminal
```powershell
.\start.ps1
```

### Paso 2: Ejecutar la suite E2E
```powershell
pytest -v -m e2e
```

---

## 3. Tests completos (unitarios + E2E)

Para ejecutar toda la batería combinada cuando el servidor está activo:

```powershell
pytest -v -m ""
```

---

## Estructura de Tests

- `tests/test_config.py`: Validación de variables de entorno y defaults de `Settings`.
- `tests/test_prompt.py`: Modos de instrucción (voz vs texto) y placeholders.
- `tests/test_protocol.py`: Parsing robusto de payloads WebSocket (`text_message`, `audio_with_history`).
- `tests/test_sentence_split.py`: Segmentador de oraciones con `pysbd` y fallback por longitud.
- `tests/test_stt_prompt.py`: Construcción de prompt inicial para faster-whisper.
- `tests/test_text_preprocess.py`: Normalización de texto para XTTS-v2 (números, siglas, puntuación).
- `tests/test_tts_streaming_protocol.py`: Empaquetado y protocolo de streaming PCM16.
- `tests/test_e2e_protocol.py`: Tests E2E cliente-servidor vía WebSockets y HTTP assets (`@pytest.mark.e2e`).
