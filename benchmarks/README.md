# Kelly — Benchmarks y Control de Calidad LLM

Este directorio contiene suites automatizadas para medir el rendimiento cuantitativo (`tok/s`, TTFT) y la calidad cualitativa/semántica de las respuestas generadas por `llama-server`.

---

## 1. Verificación de Calidad LLM (`llm_quality_check.py`)

### Rationale: ¿Por qué existe este benchmark?
La velocidad (`tok/s`) por sí sola es un indicador incompleto y potencialmente engañoso. Cuantizaciones agresivas como `q4_0` en el KV cache (`-ctk q4_0 -ctv q4_0`) aumentan la velocidad o reducen marginalmente la VRAM (~115 MiB en 8k contexto), pero introducen degradaciones severas e intolerables en modelos de 7B+:
- Fugas de idiomas no solicitados (francés, CJK).
- Bucles de repetición de tokens degenerativos (ej. "7 7 7 7 7").
- Incoherencia sintáctica y semántica en respuestas multiturno.

Esta suite actúa como **Quality Gate**: valida de forma determinista y programática que el LLM genera texto coherente en español, código Python válido analizable con `ast.parse()`, razonamiento estructurado paso a paso y retención de memoria contextual.

---

### Cuándo ejecutarlo
- **Obligatorio** después de modificar cualquier flag en `start.ps1` (`-ctk`, `-ctv`, `-c`, `-ngl`, `--threads`, `--threads-batch`).
- **Obligatorio** antes de actualizar la versión binaria de `llama-server.exe` o cambiar el archivo GGUF.
- Como regresión periódica antes de nuevos releases.

---

### Cómo ejecutarlo

Asegúrate de que `llama-server` esté activo en el puerto 8085:

```powershell
.venv\Scripts\python.exe benchmarks\llm_quality_check.py
```

O apuntando a una URL personalizada:

```powershell
.venv\Scripts\python.exe benchmarks\llm_quality_check.py --url "http://127.0.0.1:8085/completion"
```

---

### Criterios y Validación Programática

| Prompt | Dominio Evaluado | Criterios de Aceptación Programáticos |
| :--- | :--- | :--- |
| **1. Poema** | Español coherente | Sin CJK (`\u4e00-\u9fff`), sin fugas de idiomas latinos, longitud 50-350 caracteres, >= 3 signos de puntuación. |
| **2. Código** | Código Python válido | Presencia de `def ` y `return`, parseable con `ast.parse()`, sin caracteres CJK. |
| **3. Razonamiento** | Estructura paso a paso | Longitud > 200 chars, >= 3 oraciones significativas, sin bucles repetitivos de tokens. |
| **4. Carta formal** | Creatividad estructurada | Longitud 250-900 chars, sin CJK, saludo formal (`Estimado/Profesor/...`) y despedida (`Atentamente/...`). |
| **5. Conversación** | Memoria y contexto | Recuerda contexto previo (menciona color `"verde"` al explicar fotosíntesis), sin CJK. |

---

### Cómo interpretar PASS / FAIL
- **5/5 PASS (Exit code 0)**: Configuración aprobada. La inferencia es rápida y cualitativamente coherente.
- **Cualquier FAIL (Exit code 1)**: Configuración rechazada. El script guarda un reporte JSON en `benchmarks/quality_{timestamp}.json` con el texto completo recibido y la causa del fallo.
  - *Acción correctiva*: Si ocurre tras cuantizar KV cache, revertir a `-ctk q8_0 -ctv q8_0` o `-ctk f16 -ctv f16`. No reducir `-ngl` si la GPU tiene VRAM suficiente.

---

## 2. Benchmark de Rendimiento (`llm_bench.py`)

Mide el throughput de generación sostenida (`tok/s`) y el tiempo hasta el primer token (`TTFT`):

```powershell
.venv\Scripts\python.exe benchmarks\llm_bench.py --label "q8_0_ctx8k"
```

Guarda los resultados en `benchmarks/{label}.json`.
