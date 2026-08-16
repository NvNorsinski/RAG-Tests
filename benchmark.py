#!/usr/bin/env python3

import requests
import time
import statistics
import sys

OLLAMA_URL = "http://localhost:11434"
OLLAMA_API_URL = f"{OLLAMA_URL.rstrip('/')}/api/generate"

MODEL = "gemma4:e2b"

PROMPT = """
Erkläre ausführlich, aber verständlich, wie ein moderner Verbrennungsmotor
funktioniert. Beschreibe Ansaugung, Verdichtung, Verbrennung, Expansion und
Ausstoß sowie die Rolle von Kurbelwelle, Kolben, Ventilen und Einspritzung.
"""

RUNS = 1


def run_benchmark():
    results = []

    print("=" * 70)
    print("Ollama LLM Benchmark")
    print("=" * 70)
    print(f"Modell:  {MODEL}")
    print(f"Runs:    {RUNS}")
    print()

    for i in range(RUNS):
        print(f"Run {i + 1}/{RUNS} ...", end="", flush=True)

        payload = {
            "model": MODEL,
            "prompt": PROMPT,
            "stream": False,
            "options": {"temperature": 0}
        }

        start = time.perf_counter()

        response = requests.post(
            OLLAMA_API_URL,
            json=payload,
            timeout=600
        )

        wall_time = time.perf_counter() - start

        response.raise_for_status()
        data = response.json()

        eval_count = data.get("eval_count", 0)
        eval_duration = data.get("eval_duration", 0)

        prompt_eval_count = data.get("prompt_eval_count", 0)
        prompt_eval_duration = data.get("prompt_eval_duration", 0)

        # Ollama liefert Zeiten in Nanosekunden -> umrechnen in Sekunden
        eval_seconds = eval_duration / 1e9
        prompt_eval_seconds = prompt_eval_duration / 1e9

        if eval_seconds > 0:
            generation_tps = eval_count / eval_seconds
        else:
            generation_tps = 0


        if prompt_eval_seconds > 0:
            prompt_tps = prompt_eval_count / prompt_eval_seconds
        else:
            prompt_tps = 0



        results.append({
            "generation_tps": generation_tps,
            "prompt_tps": prompt_tps,
            "eval_count": eval_count,
            "prompt_eval_count": prompt_eval_count,
            "wall_time": wall_time
        })

        print(f" {generation_tps:.2f} tok/s")

    print()
    print("=" * 70)
    print("ERGEBNIS")
    print("=" * 70)

    generation = [r["generation_tps"] for r in results]
    prompt = [r["prompt_tps"] for r in results]

    print(f"Generation:        {statistics.mean(generation):.2f} tok/s")
    print(f"Prompt processing: {statistics.mean(prompt):.2f} tok/s")

    print()
    print("Einzelmessungen:")

    for i, r in enumerate(results, 1):
        print(
            f"  Run {i}: "
            f"{r['generation_tps']:.2f} tok/s"
        )

    print()
    print(f"Generierte Tokens: {results[-1]['eval_count']}")
    print(f"Prompt Tokens:     {results[-1]['prompt_eval_count']}")


if __name__ == "__main__":
    try:
        run_benchmark()

    except requests.exceptions.ConnectionError:
        print("Fehler: Ollama ist nicht erreichbar.")
        sys.exit(1)

    except requests.exceptions.HTTPError as e:
        print(f"Ollama API Fehler: {e}")
        sys.exit(1)