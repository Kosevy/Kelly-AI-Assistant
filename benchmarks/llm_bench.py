"""
benchmarks/llm_bench.py
Reproducible LLM throughput benchmark (tok/s and TTFT).

Usage:
    .venv\Scripts\python.exe benchmarks\llm_bench.py --label "baseline"
    .venv\Scripts\python.exe benchmarks\llm_bench.py --label "threads_5"
    .venv\Scripts\python.exe benchmarks\llm_bench.py --label "ngl_8"
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import httpx

LLM_URL = "http://127.0.0.1:8085/completion"
PROMPT = (
    "<|im_start|>system\n"
    "Eres Kelly, una asistente personal concisa.<|im_end|>\n"
    "<|im_start|>user\n"
    "Escribe un parrafo de 100 palabras sobre el mar, en espanol."
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)
PAYLOAD = {
    "prompt": PROMPT,
    "n_predict": 150,
    "temperature": 0,
    "stream": True,
    "stop": ["<|im_end|>", "<|endoftext|>"],
}
RUNS = 3


def run_once(label: str) -> dict:
    t_start = time.perf_counter()
    t_first_token = None
    token_count = 0

    with httpx.Client(timeout=60.0) as client:
        with client.stream("POST", LLM_URL, json=PAYLOAD) as response:
            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                token = chunk.get("content", "")
                if not token:
                    continue
                if t_first_token is None:
                    t_first_token = time.perf_counter()
                token_count += 1

    t_end = time.perf_counter()
    total_s = t_end - t_start
    ttft_ms = (t_first_token - t_start) * 1000 if t_first_token else None
    # tok/s excluyendo TTFT (solo velocidad de generacion)
    gen_s = t_end - (t_first_token or t_start)
    toks_per_sec = (token_count - 1) / gen_s if gen_s > 0 and token_count > 1 else 0

    return {
        "label": label,
        "ttft_ms": round(ttft_ms, 1) if ttft_ms else None,
        "total_ms": round(total_s * 1000, 1),
        "tokens": token_count,
        "toks_per_sec": round(toks_per_sec, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="bench", help="Etiqueta para el archivo de salida")
    parser.add_argument("--url", default=LLM_URL, help="URL del endpoint LLM")
    args = parser.parse_args()

    results = []
    print(f"\n=== Benchmark LLM: {args.label} ({RUNS} runs) ===")
    for i in range(RUNS):
        print(f"  Run {i+1}/{RUNS}...", end=" ", flush=True)
        r = run_once(args.label)
        results.append(r)
        print(f"TTFT={r['ttft_ms']}ms | total={r['total_ms']}ms | "
              f"{r['tokens']} tokens | {r['toks_per_sec']} tok/s")

    avg = {
        "label": args.label,
        "runs": RUNS,
        "avg_ttft_ms": round(sum(r["ttft_ms"] for r in results if r["ttft_ms"]) / RUNS, 1),
        "avg_total_ms": round(sum(r["total_ms"] for r in results) / RUNS, 1),
        "avg_tokens": round(sum(r["tokens"] for r in results) / RUNS, 1),
        "avg_toks_per_sec": round(sum(r["toks_per_sec"] for r in results) / RUNS, 2),
        "runs_detail": results,
    }

    print(f"\n  PROMEDIO: TTFT={avg['avg_ttft_ms']}ms | "
          f"{avg['avg_tokens']} tokens | {avg['avg_toks_per_sec']} tok/s")

    out_dir = Path("benchmarks")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"{args.label}.json"
    out_file.write_text(json.dumps(avg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Guardado: {out_file}\n")


if __name__ == "__main__":
    main()

