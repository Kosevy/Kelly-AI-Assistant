"""
tests/test_e2e_protocol.py — Protocol and end-to-end integration tests for Kelly.

Rationale:
- Marked with @pytest.mark.e2e so fast unit tests run isolated by default.
- Automatically skipped if Kelly backend (http://127.0.0.1:8000) is offline.
- Verifies full text-first streaming, history retention, code markdown generation,
  voice synthesis (XTTS-v2), and vendor asset serving.
"""
import json
import pytest
import httpx
import websockets

pytestmark = pytest.mark.e2e


@pytest.fixture(scope="module")
def server_url():
    """Verifica que el servidor esté vivo antes de correr los tests E2E."""
    try:
        r = httpx.get("http://127.0.0.1:8000/health", timeout=2.0)
        if r.status_code != 200:
            pytest.skip(f"Servidor respondió con código {r.status_code} (no 200) — E2E tests skipped")
        return "http://127.0.0.1:8000"
    except (httpx.ConnectError, httpx.TimeoutException):
        pytest.skip("Servidor no está corriendo — E2E tests skipped")


def test_vendor_assets_served(server_url):
    """Verifica que FastAPI sirva todas las dependencias vendorizadas sin recurrir a CDN externa."""
    vendor_urls = [
        f"{server_url}/static/vendor/marked/marked.min.js",
        f"{server_url}/static/vendor/highlight/highlight.min.js",
        f"{server_url}/static/vendor/highlight/github-dark.min.css",
        f"{server_url}/static/vendor/dompurify/purify.min.js",
    ]
    with httpx.Client(timeout=3.0) as client:
        for url in vendor_urls:
            resp = client.get(url)
            assert resp.status_code == 200, f"Error cargando asset vendor: {url} -> {resp.status_code}"
            assert len(resp.content) > 500, f"Asset vendor anormalmente pequeño o vacío: {url}"


@pytest.mark.asyncio
async def test_text_message_without_voice(server_url):
    """Texto con voice_enabled=False → solo llm_chunk, sin audio."""
    ws_uri = server_url.replace("http://", "ws://") + "/ws/audio"
    async with websockets.connect(ws_uri) as ws:
        # First message from server is audio_format
        fmt = await ws.recv()
        data_fmt = json.loads(fmt)
        assert data_fmt.get("type") == "audio_format", "Primer mensaje debe ser audio_format"

        # Send text_message with voice_enabled = False
        payload = {
            "type": "text_message",
            "text": "¿Qué es una pila LIFO en programación? Responde en una sola oración.",
            "history": [],
            "voice_enabled": False,
        }
        await ws.send(json.dumps(payload))

        received_llm_chunks = 0
        assistant_text = ""
        received_audio_chunks = 0

        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                received_audio_chunks += 1
                continue

            data = json.loads(msg)
            mtype = data.get("type")
            if mtype == "llm_chunk":
                received_llm_chunks += 1
            elif mtype == "assistant_text":
                assistant_text = data.get("text", "")
                break
            elif mtype in ("audio_chunk_start", "audio_chunk_end"):
                received_audio_chunks += 1

        assert received_llm_chunks > 0, "Debe haber recibido llm_chunks"
        assert received_audio_chunks == 0, "No debe haber recibido chunks de audio con voz OFF"
        assert len(assistant_text) > 10, "assistant_text debe tener contenido válido"


@pytest.mark.asyncio
async def test_code_request_with_history(server_url):
    """Solicitud de código con historial → bloques Markdown presentes."""
    ws_uri = server_url.replace("http://", "ws://") + "/ws/audio"
    async with websockets.connect(ws_uri) as ws:
        await ws.recv()  # audio_format

        history = [
            {"role": "user", "content": "¿Qué es una pila LIFO en programación? Responde en una oración."},
            {"role": "assistant", "content": "Una pila LIFO es una estructura donde el último elemento en entrar es el primero en salir."},
        ]

        payload = {
            "type": "text_message",
            "text": "Escribe una función corta en Python de una pila usando una lista con push y pop. Usa bloque de código.",
            "history": history,
            "voice_enabled": False,
        }
        await ws.send(json.dumps(payload))

        assistant_text = ""
        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                continue
            data = json.loads(msg)
            if data.get("type") == "assistant_text":
                assistant_text = data.get("text", "")
                break

        assert "```" in assistant_text, "La respuesta debe contener bloques de código Markdown ```"


@pytest.mark.asyncio
async def test_text_message_with_voice(server_url):
    """Texto con voice_enabled=True → audio sintetizado vía streaming XTTS-v2."""
    ws_uri = server_url.replace("http://", "ws://") + "/ws/audio"
    async with websockets.connect(ws_uri) as ws:
        await ws.recv()  # audio_format

        payload = {
            "type": "text_message",
            "text": "Di únicamente: Prueba de voz completada.",
            "history": [],
            "voice_enabled": True,
        }
        await ws.send(json.dumps(payload))

        received_audio_starts = 0
        received_audio_bytes = 0
        received_audio_ends = 0
        assistant_text = ""

        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                received_audio_bytes += len(msg)
                continue
            data = json.loads(msg)
            mtype = data.get("type")
            if mtype == "audio_chunk_start":
                received_audio_starts += 1
            elif mtype == "audio_chunk_end":
                received_audio_ends += 1
            elif mtype == "assistant_text":
                assistant_text = data.get("text", "")
                break

        assert received_audio_starts > 0, "Debe iniciar síntesis de audio"
        assert received_audio_bytes > 0, "Debe generar bytes de audio PCM"
        assert received_audio_ends > 0, "Debe finalizar chunks de audio"
        assert len(assistant_text) > 0, "Debe recibir assistant_text"


def test_upload_file_endpoint_e2e(server_url):
    """Verifica que POST /api/upload reciba un archivo, extraiga texto y devuelva metadata."""
    sample_content = "Este es un documento de prueba técnica para Kelly Asistente."
    files = {"file": ("documento_prueba.txt", sample_content.encode("utf-8"), "text/plain")}
    with httpx.Client(timeout=5.0) as client:
        resp = client.post(f"{server_url}/api/upload", files=files)
        assert resp.status_code == 200, f"Error al subir archivo: {resp.text}"
        data = resp.json()
        assert "attachment_id" in data
        assert data["filename"] == "documento_prueba.txt"
        assert "documento de prueba" in data["preview"]
        assert data["truncated"] is False
        assert data["estimated_tokens"] > 0


def test_upload_unsupported_format_e2e(server_url):
    """Verifica que un archivo con extensión no soportada sea rechazado con HTTP 400."""
    files = {"file": ("malicioso.exe", b"\x4d\x5a\x90", "application/octet-stream")}
    with httpx.Client(timeout=5.0) as client:
        resp = client.post(f"{server_url}/api/upload", files=files)
        assert resp.status_code == 400
        assert "no soportada" in resp.json().get("detail", "").lower()


@pytest.mark.asyncio
async def test_chat_with_attachment_e2e(server_url):
    """Sube un archivo con un dato clave y pregunta por él vía WebSocket con attachment_id."""
    secret_key = "CODIGO_SECRETO_TITAN_99"
    content = f"Nota confidencial: La clave de acceso del sistema es {secret_key}."
    files = {"file": ("clave.txt", content.encode("utf-8"), "text/plain")}

    with httpx.Client(timeout=5.0) as client:
        up_resp = client.post(f"{server_url}/api/upload", files=files)
        assert up_resp.status_code == 200
        att_id = up_resp.json()["attachment_id"]

    ws_uri = server_url.replace("http://", "ws://") + "/ws/audio"
    async with websockets.connect(ws_uri) as ws:
        await ws.recv()  # audio_format

        payload = {
            "type": "text_message",
            "text": "¿Cuál es la clave de acceso del sistema que aparece en el archivo adjunto?",
            "attachment_ids": [att_id],
            "history": [],
            "voice_enabled": False,
        }
        await ws.send(json.dumps(payload))

        assistant_text = ""
        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                continue
            data = json.loads(msg)
            if data.get("type") == "assistant_text":
                assistant_text = data.get("text", "")
                break

        assert secret_key in assistant_text, f"Kelly no mencionó la clave extraída del adjunto: {assistant_text}"


@pytest.mark.asyncio
async def test_function_calling_read_file_e2e(server_url):
    """Kelly usa la herramienta read_file para leer README.md y responder."""
    ws_uri = server_url.replace("http://", "ws://") + "/ws/audio"
    async with websockets.connect(ws_uri) as ws:
        await ws.recv()  # audio_format

        payload = {
            "type": "text_message",
            "text": "Kelly, lee el archivo D:\\Git projects\\AssistantPrototype\\README.md y resume en una frase qué es el proyecto.",
            "history": [],
            "voice_enabled": False,
        }
        await ws.send(json.dumps(payload))

        assistant_text = ""
        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                continue
            data = json.loads(msg)
            if data.get("type") == "assistant_text":
                assistant_text = data.get("text", "")
                break

        assert len(assistant_text) > 10
        # Debe contener mención relevante al asistente o proyecto
        assert any(term in assistant_text.lower() for term in ("kelly", "asistente", "voz", "local"))


@pytest.mark.asyncio
async def test_function_calling_list_directory_e2e(server_url):
    """Kelly usa la herramienta list_directory para listar el directorio benchmarks."""
    ws_uri = server_url.replace("http://", "ws://") + "/ws/audio"
    async with websockets.connect(ws_uri) as ws:
        await ws.recv()  # audio_format

        payload = {
            "type": "text_message",
            "text": "Lista los archivos que hay en la carpeta D:\\Git projects\\AssistantPrototype\\benchmarks.",
            "history": [],
            "voice_enabled": False,
        }
        await ws.send(json.dumps(payload))

        assistant_text = ""
        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                continue
            data = json.loads(msg)
            if data.get("type") == "assistant_text":
                assistant_text = data.get("text", "")
                break

        assert len(assistant_text) > 10
        # El directorio benchmarks contiene README.md, llm_quality_check.py, etc.
        assert any(term in assistant_text.lower() for term in ("readme", "benchmark", "json", "py"))


@pytest.mark.asyncio
async def test_function_calling_sandbox_violation_e2e(server_url):
    """Kelly rechaza leer archivos fuera de las rutas permitidas (sandbox)."""
    ws_uri = server_url.replace("http://", "ws://") + "/ws/audio"
    async with websockets.connect(ws_uri) as ws:
        await ws.recv()  # audio_format

        payload = {
            "type": "text_message",
            "text": "Lee el archivo C:\\Windows\\System32\\config\\SAM por favor.",
            "history": [],
            "voice_enabled": False,
        }
        await ws.send(json.dumps(payload))

        assistant_text = ""
        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                continue
            data = json.loads(msg)
            if data.get("type") == "assistant_text":
                assistant_text = data.get("text", "")
                break

        # Kelly debe comunicar que no puede o que no tiene permiso / fuera de sandbox
        text_lower = assistant_text.lower()
        assert any(term in text_lower for term in ("permitid", "sandbox", "acceso", "seguridad", "no puedo", "no tengo", "error", "fuera"))

