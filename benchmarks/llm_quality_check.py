"""
benchmarks/llm_quality_check.py
Ejecuta la suite de verificación de calidad cualitativa y estructural del LLM.

Uso:
    python benchmarks/llm_quality_check.py
    python benchmarks/llm_quality_check.py --url "http://127.0.0.1:8085/completion"
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root and benchmarks folder to sys.path
_bench_dir = Path(__file__).resolve().parent
_root_dir = _bench_dir.parent
for p in (_root_dir, _bench_dir):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from benchmarks.llm_quality_prompts import QUALITY_PROMPTS
except ImportError:
    from llm_quality_prompts import QUALITY_PROMPTS
import httpx


def build_chatml_prompt(system_text: str, user_text: str) -> str:
    return (
        f"<|im_start|>system\n{system_text}<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def run_quality_check(server_url: str = "http://127.0.0.1:8085/completion") -> int:
    print("=" * 70)
    print("  Kelly Voice Assistant — Suite de Verificación de Calidad LLM")
    print("=" * 70)
    print(f"Target URL: {server_url}")
    print(f"Timestamp:  {datetime.now().isoformat()}\n")

    results = []
    all_passed = True
    total_tokens = 0
    t_start_total = time.perf_counter()

    with httpx.Client(timeout=60.0) as client:
        for idx, item in enumerate(QUALITY_PROMPTS, 1):
            prompt_id = item["id"]
            name = item["name"]
            system_prompt = item["system"]
            user_prompt = item["user"]
            validator = item["validator"]

            chatml_prompt = build_chatml_prompt(system_prompt, user_prompt)
            payload = {
                "prompt": chatml_prompt,
                "temperature": 0.4,
                "n_predict": 512,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "stop": ["<|im_end|>", "<|endoftext|>"],
                "stream": False,
            }

            print(f"[{idx}/{len(QUALITY_PROMPTS)}] Evaluando: {name}...")
            t0 = time.perf_counter()
            try:
                resp = client.post(server_url, json=payload)
                elapsed = time.perf_counter() - t0
                if resp.status_code != 200:
                    print(f"  [ERROR] HTTP {resp.status_code}: {resp.text}")
                    results.append({
                        "id": prompt_id,
                        "name": name,
                        "status": "FAIL",
                        "error": f"HTTP {resp.status_code}",
                        "content": "",
                        "elapsed_s": elapsed,
                    })
                    all_passed = False
                    continue

                data = resp.json()
                content = data.get("content", "")
                tokens = data.get("tokens_predicted", len(content) // 4)
                total_tokens += tokens

                passed, reason = validator(content)
                status_str = "PASS" if passed else "FAIL"
                if not passed:
                    all_passed = False

                print(f"  -> Resultado: [{status_str}] ({reason}) | {elapsed:.2f}s | ~{tokens} tokens")
                results.append({
                    "id": prompt_id,
                    "name": name,
                    "prompt_user": user_prompt,
                    "status": status_str,
                    "reason": reason,
                    "elapsed_s": round(elapsed, 3),
                    "tokens_predicted": tokens,
                    "content": content,
                })

            except Exception as e:
                elapsed = time.perf_counter() - t0
                print(f"  [EXCEPTION]: {e}")
                results.append({
                    "id": prompt_id,
                    "name": name,
                    "status": "FAIL",
                    "error": str(e),
                    "content": "",
                    "elapsed_s": elapsed,
                })
                all_passed = False

            print("-" * 70)

    total_time = time.perf_counter() - t_start_total
    passed_count = sum(1 for r in results if r.get("status") == "PASS")
    total_count = len(QUALITY_PROMPTS)

    print("\n" + "=" * 70)
    print(f"RESUMEN FINAL: {passed_count}/{total_count} PASSED  ({total_time:.2f}s total)")
    print("=" * 70)

    # Guardar reporte detallado
    bench_dir = Path(__file__).parent
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = bench_dir / f"quality_{timestamp_str}.json"
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "server_url": server_url,
        "summary": f"{passed_count}/{total_count} PASS",
        "total_time_s": round(total_time, 2),
        "total_tokens": total_tokens,
        "results": results,
    }
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    print(f"Reporte completo guardado en: {report_file}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verificador de calidad del LLM")
    parser.add_argument("--url", default="http://127.0.0.1:8085/completion", help="URL del endpoint /completion")
    args = parser.parse_args()

    sys.exit(run_quality_check(server_url=args.url))
