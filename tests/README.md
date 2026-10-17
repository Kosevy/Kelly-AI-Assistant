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

## 4. Pruebas manuales post-cambio (Calendario)

Tras realizar cambios en tools, formatos y razonamiento temporal del calendario, ejecutar las siguientes 5 pruebas manuales en el chat conversacional:

### E1. Próximo examen sin rango
- **Pregunta:** *"¿Cuál es mi próximo examen?"*
- **Comportamiento esperado:** Kelly invoca `calendar_next_deadline(types=["exam"])` y devuelve la fecha y días restantes directamente sin pedir rango de fechas ni asumir la semana en curso.

### E2. Tiempo libre realista
- **Pregunta:** *"¿Tengo tiempo para estudiar el lunes?"*
- **Comportamiento esperado:** Kelly invoca `calendar_free_slots` y devuelve bloques disponibles acotados y contextualizados (ej. *"entre clase y almuerzo"*), sin calcular jornadas absurdas como "17 horas libres".

### E3. Semana completa sin mezcla
- **Pregunta:** *"¿Qué tengo la próxima semana?"*
- **Comportamiento esperado:**
  - El tool devuelve formato semi-compacto por día con encabezado en negrita, duraciones explícitas y nombre de cursos completos (`**Lunes 19 de octubre**` con viñetas `- 08:00-10:00 Transporte`).
  - No ocurre filtrado ni mezcla de bloques entre días distintos.
  - La respuesta en el frontend se muestra con viñetas en texto plano/`plain-output`, sin encerrarse en bloques de código `<pre>` con scroll horizontal.

### E4. Próxima entrega genérica
- **Pregunta:** *"¿Qué entrega viene?"*
- **Comportamiento esperado:** Kelly usa `calendar_next_deadline` con filtro abierto o sin types específicos y devuelve el entregable evaluativo más próximo en el tiempo.

### E5. Range acotado
- **Pregunta:** *"Muéstrame todo el mes de noviembre"*
- **Comportamiento esperado:** `calendar_read_range` aplica el límite configurable de seguridad (7 días) y Kelly presenta los primeros 7 días acompañados del aviso explicativo para pedir rangos más específicos.

---

## Estructura de Tests

- `tests/test_calendar_parser.py`: Carga, validación Pydantic y detección de errores en `calendar.yaml`.
- `tests/test_calendar_ergonomics.py`: Diagnóstico ergonómico de errores manuales en YAML (línea, bloque, sugerencia).
- `tests/test_calendar_expander.py`: Expansión de recurrencias semanales y eventos puntuales en rangos de fechas.
- `tests/test_calendar_query.py`: Consultas temporales puras (`read_range`, `free_slots` literal/realista, `next_deadline`, `list_courses`).
- `tests/test_calendar_tools.py`: Declaración OpenAI JSON Schema y ejecución de tools de calendario.
- `tests/test_calendar_e2e.py`: Flujos E2E de interacción con calendario (`calendar_add_event`, `calendar_delete_event`, etc.).
- `tests/test_config.py`: Validación de variables de entorno y defaults de `Settings`.
- `tests/test_prompt.py`: Modos de instrucción, reglas de calendario, renderizado de frontend y temperatura de tools.
- `tests/test_protocol.py`: Parsing robusto de payloads WebSocket (`text_message`, `audio_with_history`).
- `tests/test_sentence_split.py`: Segmentador de oraciones con `pysbd` y fallback por longitud.
- `tests/test_stt_prompt.py`: Construcción de prompt inicial para faster-whisper.
- `tests/test_text_preprocess.py`: Normalización de texto para XTTS-v2 (números, siglas, puntuación).
- `tests/test_tts_streaming_protocol.py`: Empaquetado y protocolo de streaming PCM16.
- `tests/test_tools.py`: Sandboxing y ejecución de herramientas de sistema de archivos.
- `tests/test_e2e_protocol.py`: Tests E2E cliente-servidor vía WebSockets y HTTP assets (`@pytest.mark.e2e`).
