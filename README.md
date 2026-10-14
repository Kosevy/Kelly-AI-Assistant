# Kelly — 100% Local Real-Time Voice Assistant

Kelly is an open-source, fully local, private, and zero-subscription virtual voice assistant designed for Windows with low-latency bidirectional streaming. It runs speech-to-text (STT), a large language model (LLM), and zero-shot voice cloning text-to-speech (TTS) entirely on consumer hardware, with simultaneous desktop and mobile access over private encrypted networks (Tailscale).

**Zero external API dependencies. Zero cloud subscriptions. Zero telemetry.** All voice data and conversation logs remain strictly on your local machine.

---

## Architecture Overview

```
User Voice (PWA / Mic)
        │ (Opus / WebM over WebSocket)
        ▼
[STT: faster-whisper small (int8 / CPU) + VAD Gating]
        │
        ▼ (Transcribed User Text)
[LLM: llama-server — Qwen2.5-7B-Instruct (Q4_K_M)]
  ├── Conversational History (Injected via ChatML)
  └── Sentence Segmentation (pysbd)
        │
        ▼ (Sentence Stream)
[Text Normalization: app/text_preprocess.py]
  ├── Acronym & Phonetic Expansion
  └── Number-to-Words Conversion (es)
        │
        ▼ (Normalized Spanish Sentences)
[TTS: Coqui XTTS-v2 (GPU / CUDA 12.6)]
  └── Cached Voice Conditioning Latents (voice_sample.wav)
        │
        ▼ (WAV PCM_16 @ 24 kHz)
Browser Audio Queue (FIFO + Real-Time Barge-In Support)
```

### Key Technical Highlights

- **Sentence-Level Pipeline Overlap:** Streams tokens from the LLM, splits sentences on punctuation boundaries with `pysbd`, and feeds individual sentences directly to the TTS engine before the LLM has finished generating the full response.
- **Cached Voice Conditioning Latents:** Pre-computes and caches speaker latents (`gpt_cond_latent`, `speaker_embedding`) from a single 6–10 second voice sample at server startup, eliminating redundant recalculations on every phrase.
- **Client-Side Conversational Memory:** Maintains multi-turn context (up to 6 turns / 2,048 estimated tokens) in frontend session memory formatted as ChatML. Memory resets on page refresh to ensure session privacy.
- **Instant Barge-In (Interruption):** Holding or clicking the mic button halts current audio playback in the browser, purges the playback queue, and sends a control event (`{"type": "cancel"}`) to immediately abort active LLM generation and TTS synthesis in the backend.
- **Phonetic & Text Preprocessing:** Custom normalizer resolves common XTTS audio artifacts (mumbling, letter fade-outs, raw numeric strings, unpronounceable symbols, and markdown).
- **Non-Blocking Async Backend:** Built with FastAPI and `httpx.AsyncClient` streaming, ensuring GPU inference and CPU STT operations execute via worker threads (`asyncio.to_thread`) without blocking the event loop.
- **Privacy by Design:** Voice samples, audio debug recordings, and session logs are strictly gitignored to prevent accidental leakage of biometric audio or conversation history.

---

## Hardware & System Requirements

### Tested Reference Hardware

- **GPU:** NVIDIA GeForce RTX 3060 (12 GB VRAM) — driver 610.88+, CUDA 12.6
- **CPU:** AMD Ryzen 5 7600X (6 physical cores, 12 logical threads)
- **RAM:** 16 GB DDR5 @ 5200 MHz
- **OS:** Windows 10 / 11 (64-bit) with PowerShell

### Software Requirements

- **Python:** Exactly **3.10.x** (3.11+ is unstable with Coqui XTTS-v2 Windows dependencies)
- **CUDA Toolkit:** CUDA 12.1+ / 12.6 driver support
- **CMake & Visual Studio Build Tools:** For building `llama.cpp`
- **Tailscale:** For secure HTTPS access and mobile microphone permissions

---

## Step-by-Step Installation Guide

### 1. Clone the Repository

```powershell
git clone https://github.com/<your-username>/AssistantPrototype.git
cd AssistantPrototype
```

### 2. Set Up Python 3.10 Virtual Environment

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install PyTorch with CUDA 12.6

Install the matched PyTorch and torchaudio wheels for CUDA 12.6, then install the remaining project dependencies:

```powershell
pip install torch==2.11.0+cu126 torchaudio==2.11.0+cu126 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

### 4. Clone and Build `llama.cpp`

`llama.cpp` is decoupled from the main repository. Clone and compile it with CUDA acceleration enabled:

```powershell
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j
cd ..
```

Verify that `llama.cpp\build\bin\Release\llama-server.exe` was generated successfully.

### 5. Download the LLM Weights

Download the Qwen2.5-7B Instruct model in GGUF format (Q4_K_M quantization recommended):

- Create the models directory if needed: `mkdir llama.cpp\models`
- Download `qwen2.5-7b-instruct-q4_k_m.gguf` from Hugging Face into `llama.cpp\models\`.

_Note: XTTS-v2 and faster-whisper small weights download automatically on first launch._

### 6. Prepare Voice Sample for Cloning

Provide a clean reference WAV file in the project root:

- **Path:** `voice_sample.wav`
- **Duration:** 6–10 seconds of clear, natural speech
- **Format:** Mono WAV, 22,050 Hz or 24,000 Hz, 16-bit PCM

> **Privacy Note:** `voice_sample.wav` is automatically ignored by `.gitignore` to protect personal biometric voice data.

### 7. Configure Environment Variables

Copy the template and adjust settings as needed:

```powershell
Copy-Item .env.example .env
```

_(All variables are documented with sensible defaults in `app/config.py` and `.env.example`.)_

---

## Running Kelly

### Start All Services

Launch both `llama-server` and the FastAPI application in synchronized windows:

```powershell
.\start.ps1
```

The script will:

1. Verify the virtual environment and compiled binaries.
2. Launch `llama-server` with `--threads 5 --threads-batch 5 -ngl 8` (offloading 8 layers to GPU while leaving headroom for XTTS-v2).
3. Launch the FastAPI server on port 8000.
4. Display local LAN and Tailscale access URLs.

### Stop All Services

```powershell
.\stop.ps1
```

Gracefully terminates `llama-server` and `uvicorn` processes.

---

## Accessing the Assistant

- **Local PC:** Open `http://localhost:8000` in Chrome or Edge.
- **Mobile Access:** Mobile browsers require secure contexts (HTTPS) to enable microphone permissions (`getUserMedia`). Use Tailscale:
  ```powershell
  tailscale serve --https=443 8000
  ```
  Then navigate to `https://<your-device-name>.<your-tailnet>.ts.net` on your mobile phone.

---

## Benchmarking & Performance Tuning

### Run LLM Benchmark

To evaluate throughput (tokens per second) and Time-To-First-Token (TTFT):

```powershell
.venv\Scripts\python.exe benchmarks\llm_bench.py --label "threads5_ngl8"
```

Results are saved to `benchmarks/<label>.json`.

### Tunable Parameters in `start.ps1`

- `--threads <N>`: Set to physical CPU cores minus 1 (e.g., `5` on a 6-core Ryzen 5 7600X).
- `-ngl <layers>`: GPU layers offloaded to VRAM (e.g., `8` uses ~1.5 GB extra VRAM, fitting comfortably within the ~2.8 GB free VRAM alongside XTTS-v2 on a 12 GB GPU).

---

## Health Checks & Monitoring

Check the status of all components at runtime:

```powershell
curl http://localhost:8000/health
```

Returns JSON status (`200 OK` or `503 Service Unavailable`):

```json
{
  "status": "ok",
  "components": {
    "stt": { "status": "ok", "model": "small", "device": "cpu" },
    "llm": { "status": "ok", "url": "http://127.0.0.1:8085/completion" },
    "tts": { "status": "ok", "device": "cuda", "voice_cached": true }
  }
}
```

---

## Running Automated Tests

Run the test suite covering text normalization, Spanish phonetics, sentence segmentation, and configuration overrides:

```powershell
pip install -r requirements-dev.txt
pytest
```

---

## Project Structure

```
AssistantPrototype/
├── app/
│   ├── __init__.py
│   ├── config.py             # Pydantic Settings management (KELLY_ prefix)
│   ├── logging_config.py     # Loguru colored console & rotating file logging
│   └── text_preprocess.py    # Spanish phonetic normalizer, acronyms, number-to-words
├── patches/
│   ├── __init__.py
│   └── torch_compat.py       # PyTorch 2.6+ & torchaudio Windows monkey-patches
├── prompts/
│   └── system.txt            # System prompt with dynamic datetime injection
├── tests/
│   ├── test_config.py        # Settings and environment variable tests
│   ├── test_sentence_split.py# pysbd sentence boundary edge case tests
│   └── test_text_preprocess.py# Phonetic and normalization unit tests
├── benchmarks/
│   └── llm_bench.py          # Reproducible LLM latency & tok/s benchmark
├── .env.example              # Documented environment variable template
├── .gitignore                # Strict exclusion for secrets, audio, models, and caches
├── index.html                # Responsive PWA frontend with history panel & barge-in
├── pyproject.toml            # Pytest configuration
├── README.md                 # English documentation
├── requirements.txt          # Pinned production dependencies
├── requirements-dev.txt      # Development & testing dependencies
├── server.py                 # Core FastAPI WebSocket streaming server
├── start.ps1                 # Windows PowerShell startup automation
└── stop.ps1                  # Process termination script
```

---

## Troubleshooting

### CUDA Out of Memory (OOM)

XTTS-v2 consumes ~2.5–3.0 GB of VRAM. If llama-server with `-ngl` consumes too much memory:

1. Lower `-ngl` in `start.ps1` from `8` to `4` or `0`.
2. Check active GPU usage via `nvidia-smi`.
3. Set `KELLY_TTS_DEVICE=cpu` in `.env` as a temporary CPU-only fallback.

### Microphone Permission Denied on Mobile

Browsers restrict microphone capture over insecure HTTP on non-localhost origins.

- Configure Tailscale HTTPS: `tailscale serve --https=443 8000`.
- Access the app via `https://...` instead of `http://...`.

### Speech-to-Text Recognition Quality

If transcriptions cut off early or miss speech:

- Lower the minimum silence duration: `KELLY_STT_VAD_MIN_SILENCE_MS=300`
- Enable audio debugging: `KELLY_DEBUG_SAVE_AUDIO=true` → inspect files in `debug/`

---

## STT: Configuration and Design Decisions

### Dynamic `initial_prompt` from Conversation History

**Problem observed:** Whisper `small` systematically corrupted the first 1–2 words of utterances ("Hola Kelly" → "con lo que le comestas").

**Solution:** [`app/stt_prompt.py`](app/stt_prompt.py) builds a dynamic `initial_prompt` from the last 2 user messages in the conversation history, passed to Whisper's decoder on every transcription call.

**Why this approach (not alternatives):**

- **Not a static vocabulary list:** Would require manual maintenance every time topics change. Breaks when the user discusses anything not pre-listed.
- **Not a larger model:** Valid long-term option but doubles latency and RAM. Not warranted until dynamic prompting is benchmarked. See "Pending Decision" below.
- **Dynamic history prompt:** Whisper's `initial_prompt` biases beam search toward vocabulary the user has already used in the current conversation. Effect is strongest at utterance boundaries (where errors concentrated). Automatically adapts to any topic. Zero maintenance.

Only **user messages** are used — Kelly's responses are TTS-synthesized audio the model has never processed, so including them would add irrelevant decoder bias.

### Decoding Parameters

All parameters are externalized via `.env` (prefix `KELLY_STT_`). No tuning requires code changes.

| Parameter                        | Default               | Rationale                                                                                                                                                                                                                               |
| -------------------------------- | --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `STT_BEAM_SIZE`                  | `5`                   | Whisper's original default. `1` = greedy (faster but less accurate). `5` = beam search with measurably better first-word accuracy on short conversational utterances.                                                                   |
| `STT_TEMPERATURE_FALLBACK`       | `0.0,0.2,0.4,...,1.0` | Whisper's documented progressive fallback strategy. Starts deterministic (T=0), retries with increasing randomness only when the decoder's confidence score falls below threshold. Not a custom hack — this is the published algorithm. |
| `STT_NO_SPEECH_THRESHOLD`        | `0.6`                 | Whisper default. Controls silence detection sensitivity. Lower = more likely to discard real speech; higher = more likely to transcribe background noise.                                                                               |
| `STT_CONDITION_ON_PREVIOUS_TEXT` | `true`                | Enables intra-utterance context from Whisper's internal decoded text. Consistent with `initial_prompt` usage.                                                                                                                           |
| `STT_INITIAL_PROMPT_MAX_CHARS`   | `224`                 | Chosen to stay within Whisper's ~224-token context window (Spanish averages ~4 chars/token, so 224 chars ≈ 56 tokens — well within budget).                                                                                             |

### Benchmark Tool

Run `benchmarks/stt_bench.py` against saved debug audio files to measure transcription accuracy before and after configuration changes:

```powershell
# Capture audio first
# Set KELLY_DEBUG_SAVE_AUDIO=true in .env, speak to Kelly, then:
.venv\Scripts\python.exe benchmarks\stt_bench.py --output benchmarks\stt_baseline.json

# Compare with and without initial_prompt on a specific file:
.venv\Scripts\python.exe benchmarks\stt_bench.py --audio debug\stt_001.raw --with-prompt "reglas del ajedrez"
```

### Pending Decision: Model Upgrade

If accuracy remains insufficient after these improvements, evaluate `faster-whisper medium`:

- Latency impact: ~2x STT latency (1,200–1,900 ms → ~2,500–3,500 ms)
- RAM impact: +~1 GB RAM
- Accuracy: significantly better on Spanish conversational speech

**Do not upgrade the model without first running the STT benchmark** — the dynamic `initial_prompt` + beam search changes may already be sufficient.

---

## Lectura de Archivos Locales & Function Calling

Kelly cuenta con capacidades nativas para interactuar con tus archivos locales de estudio y desarrollo, ya sea mediante subida directa (Drag & Drop en la interfaz web) o mediante **Function Calling** cuando el modelo decide consultar el disco.

### 1. Formatos Soportados

La extracción de texto se realiza de forma 100% local sin dependencias en servicios externos:
- **Documentos PDF**: Extraídos con `PyMuPDF` (`fitz`), preservando el layout y texto estructurado con alta velocidad.
- **Archivos de Código y Texto**: `.py`, `.md`, `.txt`, `.cpp`, `.c`, `.h`, `.hpp`, `.ts`, `.js`, `.java`, `.json`, `.yaml`, `.yml`, `.html`, `.css`, `.sql`. Decodificados como UTF-8 con fallback automático a Latin-1.

### 2. Sandboxing de Seguridad en Windows

El acceso al disco está restringido estrictamente a las rutas declaradas en la variable `KELLY_ALLOWED_PATHS`:
- **Rutas permitidas por defecto:** `D:\Git projects` y `D:\Ciclo 6`.
- **Protección contra Path Traversal:** Se resuelven symlinks, componentes relativos (`..`) y nombres de unidad en Windows de forma case-insensitive antes de validar la pertenencia al sandbox.
- **Manejo de errores:** Intentos de lectura fuera del sandbox lanzan un `PermissionError` interno capturado por el ejecutor de herramientas, permitiendo que Kelly explique cortésmente al usuario por qué no puede acceder sin exponer archivos del sistema (como `C:\Windows\System32`).

### 3. Presupuesto de Tokens y Truncado

- **Límite por archivo:** `KELLY_MAX_FILE_TOKENS=5500` (~22,000 caracteres con ratio 0.25 tokens/char).
- Si un archivo excede el presupuesto, se truncan las páginas/líneas posteriores preservando el inicio del documento y notificando al LLM y a la interfaz con la etiqueta `(truncado)`.
- **Límite de subida HTTP:** `KELLY_MAX_UPLOAD_SIZE_MB=50`.
- **TTL de archivos temporales:** `KELLY_UPLOAD_TTL_MINUTES=60` con limpieza periódica en segundo plano cada 10 minutos.

### 4. Function Calling con Qwen 2.5 y llama.cpp

El backend utiliza el endpoint estándar de OpenAI `/v1/chat/completions` en `llama-server`:
- **Herramientas disponibles:**
  - `read_file(path)`: Lee el contenido de un archivo en las rutas permitidas.
  - `list_directory(path)`: Lista archivos y subcarpetas dentro de las rutas permitidas.
- **Bucle de ejecución:** Si el LLM emite `finish_reason: "tool_calls"`, Kelly ejecuta la función solicitada de manera asíncrona (timeout de 10s), inyecta el resultado con rol `tool`, y reanuda el streaming hasta generar la respuesta final (con un límite de seguridad de 3 iteraciones).

---

## Calendario Local con YAML y Tools

Kelly cuenta con un sistema de calendario 100% local donde la fuente de verdad es un archivo YAML editable a mano (diseñado para actualizarse una vez por semestre académico).

### 1. Formato YAML y Campos

El archivo se organiza en cuatro secciones principales:
- **`metadata`**: Datos globales del estudiante y semestre (`student_name`, `semester`, `semester_start`, `semester_end`, `exam_weeks` como semanas 8 para parciales y 15 para finales).
- **`courses`**: Cursos matriculados con `id`, `name`, `difficulty` (escala 1 a 5), `sessions_per_week` y lista de `topics`.
- **`recurring_blocks`**: Horarios fijos semanales (`class`, `transport`, `meal`, `exercise`, `study`, `other`) con días de la semana (`day_of_week`), hora de inicio (`start_time`), fin (`end_time`) y vigencia de fechas (`from_date`, `to_date`).
- **`events`**: Hitos puntuales y evaluaciones (`exam`, `entregable`, `presentacion`, `practica`, `personal`) con `id`, `name`, `date`, `course_id`, `start_time`, `duration_minutes`, `weight`, `topics_covered` y `notes`.

### 2. Cómo Usar el Template

1. Copia el archivo de ejemplo para inicializar tu calendario:
   ```powershell
   Copy-Item data/calendar.yaml.example data/calendar.yaml
   ```
2. Abre `data/calendar.yaml` en tu editor de texto favorito y ajusta tus cursos, horarios de clase y fechas de exámenes del semestre.
3. El archivo `data/calendar.yaml` se encuentra excluido de git por seguridad y privacidad de tus datos personales.

### 3. Herramientas de Function Calling Expuestas

Kelly dispone de 5 herramientas nativas para interactuar con tu calendario:
- **`calendar_read_range(start_date, end_date, types)`**: Consulta los eventos y clases programadas en un rango de fechas. Permite responder preguntas como *¿qué tengo esta semana?* o *¿qué hay el 12 de mayo?*.
- **`calendar_free_slots(start_date, end_date, min_block_minutes)`**: Calcula franjas horarias disponibles para estudio concentrado entre las 06:00 y las 23:00. **Ignora automáticamente bloques no bloqueantes** como transporte, comidas o ejercicio.
- **`calendar_add_event(name, date, type, course_id, start_time, duration_minutes, ...)`**: Registra un nuevo evento o evaluación en el YAML generando un slug único e identificable.
- **`calendar_list_courses()`**: Lista los cursos del ciclo con su dificultad, sesiones semanales y temas.
- **`calendar_delete_event(event_id)`**: Elimina un evento puntual del calendario a petición explícita del usuario.

### 4. Backup Automático y Escritura Atómica

- **Respaldo rotativo previo:** Antes de que Kelly modifique el archivo `calendar.yaml`, se genera automáticamente una copia de seguridad en `data/calendar_backups/calendar_{YYYYMMDD_HHMMSS}.yaml`.
- **Rotación:** Se conservan únicamente las últimas N versiones (configurable mediante `KELLY_CALENDAR_BACKUP_COUNT=10`).
- **Escritura atómica:** Para evitar corrupción del YAML si el proceso se interrumpe durante el guardado, los datos se vuelcan primero a un archivo temporal `calendar.yaml.tmp` y luego se reemplazan de forma atómica mediante `os.replace()`.

### 5. Ejemplos de Conversación

- **Consulta semanal:**
  > *Usuario:* "¿Qué tengo esta semana?"
  > *Kelly:* Invoca `calendar_read_range` con las fechas de lunes a domingo de la semana en curso y resume tus clases y entregas.

- **Próximo examen o deadline:**
  > *Usuario:* "¿Cuándo es mi próximo examen?"
  > *Kelly:* Consulta las evaluaciones programadas y te informa fecha, hora y curso de la evaluación más cercana.

- **Añadir un evento:**
  > *Usuario:* "Kelly, añade una práctica de Cálculo el 20 de abril a las 8am."
  > *Kelly:* Invoca `calendar_add_event(name='Práctica de Cálculo', date='2026-04-20', type='practica', course_id='calc3', start_time='08:00')` y confirma la adición indicando el ID asignado.

- **Disponibilidad para estudio:**
  > *Usuario:* "¿Tengo 90 minutos libres el miércoles?"
  > *Kelly:* Invoca `calendar_free_slots` con `min_block_minutes=90` para la fecha solicitada y te muestra los intervalos en los que puedes concentrarte.

- **Revisión de cursos:**
  > *Usuario:* "¿Cuáles son mis cursos de este ciclo y qué dificultad tienen?"
  > *Kelly:* Invoca `calendar_list_courses` y presenta la lista ordenada con sus temas y nivel de dificultad.

---

## Model Licenses & Attributions

| Component     | Model / Engine                  | License                                                                      |
| ------------- | ------------------------------- | ---------------------------------------------------------------------------- |
| **STT**       | faster-whisper (OpenAI Whisper) | [MIT License](https://github.com/SYSTRAN/faster-whisper/blob/master/LICENSE) |
| **LLM**       | Qwen 2.5 7B Instruct            | [Apache 2.0](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)                |
| **Inference** | llama.cpp                       | [MIT License](https://github.com/ggerganov/llama.cpp/blob/master/LICENSE)    |
| **TTS**       | Coqui XTTS-v2                   | [Coqui Public Model License (CPML)](https://coqui.ai/cpml)                   |

> **Note on Commercial Use:** Coqui XTTS-v2 is distributed under CPML (non-commercial personal/research use). For commercial deployments, consider replacing XTTS-v2 with an Apache/MIT-licensed TTS engine such as Piper.
