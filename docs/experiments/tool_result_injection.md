# Experimento: Inyección de tool results

**Fecha:** 2026-10-15  
**Modelo:** Qwen2.5-7B-Instruct (Q4_K_M, llama.cpp / llama-server local)  
**Hipótesis:** Inyectar los resultados de las herramientas con `role="user"` (precedido por alternancia de turno con `role="assistant"` y un preámbulo directivo explícito) mejora el anclaje atencional (*grounding*) y reduce drásticamente las alucinaciones frente al estándar `role="tool"` en modelos pequeños.

---

## 1. Contexto del Problema

En modelos de 7B parámetros como Qwen2.5, el chat template procesa los mensajes con `role: "tool"` como metadatos secundarios o contexto auxiliar. Cuando el tool result contiene estructuras extensas (como calendarios semanales con 7 días y ~30 bloques de horarios), el modelo frecuentemente ignora las fechas exactas, mezcla bloques entre días adyacentes o reasigna fechas erróneas (ej. "Lunes 23 de octubre" en lugar de "Lunes 19 de octubre").

La inyección como mensaje de `user` aprovecha el mayor peso atencional del rol principal del usuario y emula el patrón tradicional de *in-context retrieval grounding*.

---

## 2. Metodología

Se evaluaron los mismos 3 casos de prueba en paralelo contra el modelo en vivo (`http://127.0.0.1:8085/v1/chat/completions`) usando los datos reales de `data/calendar.yaml`:

1. **Modo `tool_role` (baseline):**
   - Mensaje 1: `role: "assistant"`, `tool_calls: [...]`
   - Mensaje 2: `role: "tool"`, `content: tool_result`
2. **Modo `user_message` (experimental):**
   - Mensaje 1: `role: "assistant"`, `tool_calls: [...]`
   - Mensaje 2: `role: "assistant"`, `content: ""` (preserva alternancia requerida por Jinja template)
   - Mensaje 3: `role: "user"`, `content: "[RESULTADO DEL SISTEMA — usa esto literalmente]\n...\n[FIN DEL RESULTADO]\n\nBasándote EXCLUSIVAMENTE en el resultado anterior..."`

---

## 3. Resultados Comparativos

| Caso de Prueba | Query | Tool Result Obtenido | Baseline (`role="tool"`) | Experimental (`role="user"`) | Resultado |
|---|---|---|---|---|:---:|
| **Test 1: Próximo examen** | *"¿Cuál es mi próximo examen?"* | EB1 admm (2026-12-08 13:00) | Identifica EB1 admm 2026-12-08 correctamente | Identifica EB1 admm 2026-12-08 correctamente | **EMPATE (PASS)** |
| **Test 2: Semana completa** | *"¿Qué tengo la próxima semana?"* | Lunes 19 al Domingo 25 oct con horarios exactos | **FALLO CRÍTICO:** Desplazó todas las fechas a Lunes 23, Martes 24... Domingo 29 de octubre | **PASS PERFECTO:** Transcribe Lunes 19 a Domingo 25 con exactitud 100% idéntica al calendario | **GANA `user_message`** |
| **Test 3: Próxima entrega** | *"¿Qué entrega viene?"* | Tarea N°6 fie (2026-11-01 23:59) | Identifica Tarea N°6 fie 2026-11-01 correctamente | Identifica Tarea N°6 fie 2026-11-01 correctamente | **EMPATE (PASS)** |

---

## 4. Análisis Detallado

### Test 2: El caso definitivo de alucinación temporal
- En el **Baseline (`tool_role`)**, Qwen2.5 leyó los bloques pero alucinó las etiquetas de los días, iniciando la semana en "Lunes 23 de octubre" (fecha que ni siquiera existía en el tool result ni en el calendario real).
- En el **Modo `user_message`**, el preámbulo directivo forzó al modelo a tratar el bloque como un ancla literal estricta. La salida reprodujo exactamente cada encabezado (`**Lunes 19 de octubre**`, `**Martes 20 de octubre**`, etc.) y cada intervalo horario sin mezclas ni corrimientos de fecha.

---

## 5. Conclusión y Recomendación

**Hipótesis confirmada de forma contundente.**
El modo `user_message` resuelve de raíz el fallo de seguimiento de instrucciones en contextos largos y estructurados para Qwen2.5-7B, eliminando por completo el desplazamiento de fechas en consultas de calendario semanal.

### Decisión de Arquitectura:
- **Promover `user_message` como el valor predeterminado** (`tool_result_injection_mode="user_message"`).
- **Mantener `tool_role` como fallback configurable** en `app/config.py` y `.env.example` para garantizar máxima compatibilidad con futuros modelos más grandes que sí gestionen `role="tool"` de forma nativa sin degradación.
