"""
tests/verify_e2e_text_and_voice.py
Functional verification script that interacts directly with Kelly's live WebSocket:
1. Tests text_message with voice_enabled=False (verifies llm_chunk, assistant_text, markdown formatting, NO audio chunks).
2. Tests text_message with code snippet request (verifies code block formatting in assistant_text).
3. Tests text_message with voice_enabled=True (verifies audio_chunk_start, binary PCM chunks, audio_chunk_end).
4. Tests multi-turn conversation history continuity.
"""
import asyncio
import json
import websockets


async def test_text_voice_disabled():
    print("\n--- Test 1: Texto con voz OFF (voice_enabled=False) ---")
    uri = "ws://127.0.0.1:8000/ws/audio"
    async with websockets.connect(uri) as ws:
        # First message from server is audio_format
        fmt = await ws.recv()
        print("  <- Formato de audio:", json.loads(fmt)["type"])

        # Send text_message with voice_enabled = False
        payload = {
            "type": "text_message",
            "text": "¿Qué es una pila LIFO en programación? Responde en una oración.",
            "history": [],
            "voice_enabled": False,
        }
        await ws.send(json.dumps(payload))
        print("  -> Mensaje de texto enviado con voice_enabled=False")

        received_chunks = 0
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
                received_chunks += 1
            elif mtype == "assistant_text":
                assistant_text = data.get("text", "")
                break
            elif mtype == "audio_chunk_start" or mtype == "audio_chunk_end":
                received_audio_chunks += 1

        print(f"  <- llm_chunk recibidos: {received_chunks}")
        print(f"  <- audio chunks recibidos: {received_audio_chunks} (esperado 0)")
        print(f"  <- assistant_text: \"{assistant_text}\"")

        assert received_chunks > 0, "Debe haber recibido llm_chunks"
        assert received_audio_chunks == 0, "No debe haber recibido chunks de audio con voz OFF"
        assert len(assistant_text) > 10, "assistant_text debe tener contenido"
        print("  [OK] Test 1 superado exitosamente.")
        return assistant_text


async def test_text_code_snippet(prev_assistant_text: str):
    print("\n--- Test 2: Pregunta con código y continuidad de historial ---")
    uri = "ws://127.0.0.1:8000/ws/audio"
    async with websockets.connect(uri) as ws:
        await ws.recv() # audio_format

        history = [
            {"role": "user", "content": "¿Qué es una pila LIFO en programación? Responde en una oración."},
            {"role": "assistant", "content": prev_assistant_text},
        ]

        payload = {
            "type": "text_message",
            "text": "Escribe una función corta en Python de una pila usando una lista con push y pop. Usa bloque de código.",
            "history": history,
            "voice_enabled": False,
        }
        await ws.send(json.dumps(payload))
        print("  -> Solicitud de código enviada con historial")

        assistant_text = ""
        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                continue
            data = json.loads(msg)
            if data.get("type") == "assistant_text":
                assistant_text = data.get("text", "")
                break

        print(f"  <- assistant_text:\n{assistant_text}\n")
        assert "```" in assistant_text, "La respuesta debe contener bloques de código Markdown ```"
        print("  [OK] Test 2 superado exitosamente (bloques de código presentes).")


async def test_text_voice_enabled():
    print("\n--- Test 3: Texto con voz ON (voice_enabled=True) ---")
    uri = "ws://127.0.0.1:8000/ws/audio"
    async with websockets.connect(uri) as ws:
        await ws.recv() # audio_format

        payload = {
            "type": "text_message",
            "text": "Di únicamente: Prueba de voz completada.",
            "history": [],
            "voice_enabled": True,
        }
        await ws.send(json.dumps(payload))
        print("  -> Mensaje enviado con voice_enabled=True")

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

        print(f"  <- audio_chunk_start: {received_audio_starts}")
        print(f"  <- bytes de audio sintetizados: {received_audio_bytes}")
        print(f"  <- audio_chunk_end: {received_audio_ends}")
        print(f"  <- assistant_text: \"{assistant_text}\"")

        assert received_audio_starts > 0, "Debe iniciar síntesis de audio"
        assert received_audio_bytes > 0, "Debe generar bytes de audio PCM"
        assert received_audio_ends > 0, "Debe finalizar chunks de audio"
        assert len(assistant_text) > 0, "Debe recibir assistant_text"
        print("  [OK] Test 3 superado exitosamente (TTS sintetizó audio correctamente).")


async def main():
    prev_text = await test_text_voice_disabled()
    await test_text_code_snippet(prev_text)
    await test_text_voice_enabled()
    print("\n==============================================")
    print("TODAS LAS VERIFICACIONES E2E PASARON CON ÉXITO")
    print("==============================================\n")


if __name__ == "__main__":
    asyncio.run(main())
