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

*Note: XTTS-v2 and faster-whisper small weights download automatically on first launch.*

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
*(All variables are documented with sensible defaults in `app/config.py` and `.env.example`.)*

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
    "stt": {"status": "ok", "model": "small", "device": "cpu"},
    "llm": {"status": "ok", "url": "http://127.0.0.1:8085/completion"},
    "tts": {"status": "ok", "device": "cuda", "voice_cached": true}
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
- Lower the minimum silence duration in `.env`: `KELLY_STT_VAD_MIN_SILENCE_MS=300`.
- Enable audio debugging in `.env`: `KELLY_DEBUG_SAVE_AUDIO=true` and inspect captured files in `debug/`.

---

## Model Licenses & Attributions

| Component | Model / Engine | License |
|---|---|---|
| **STT** | faster-whisper (OpenAI Whisper) | [MIT License](https://github.com/SYSTRAN/faster-whisper/blob/master/LICENSE) |
| **LLM** | Qwen 2.5 7B Instruct | [Apache 2.0](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) |
| **Inference** | llama.cpp | [MIT License](https://github.com/ggerganov/llama.cpp/blob/master/LICENSE) |
| **TTS** | Coqui XTTS-v2 | [Coqui Public Model License (CPML)](https://coqui.ai/cpml) |

> **Note on Commercial Use:** Coqui XTTS-v2 is distributed under CPML (non-commercial personal/research use). For commercial deployments, consider replacing XTTS-v2 with an Apache/MIT-licensed TTS engine such as Piper.
