#!/usr/bin/env python3
"""Fixed launcher for the hard-25 two-arm Qwen3 4B Thinking benchmark.

The original preflight accidentally capped generation at 2048 tokens even though the
benchmark itself allowed 6000. Dedicated thinking models can spend that entire budget
in reasoning and therefore return a valid thinking trace but no final answer.

This wrapper keeps the exact two benchmark arms and scoring/retrieval code unchanged,
but gives the preflight the same generation budget as the real run and raises the
default budget to 8192 tokens.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = ROOT / "scripts" / "benchmark_qwen3_4b_thinking_two_arm_25.py"

spec = importlib.util.spec_from_file_location("thinking_two_arm", BASE_SCRIPT)
bench = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(bench)

bench.VERSION = "thinking-two-arm-hard25-v1.1"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", default=str(bench.DEFAULT_CASES))
    p.add_argument("--db", default=str(bench.DEFAULT_DB))
    p.add_argument("--model", default="qwen3:4b-thinking")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--out", default=str(ROOT / "qwen3_4b_thinking_two_arm_25_results_v2.json"))
    p.add_argument("--checkpoint", default=str(ROOT / "qwen3_4b_thinking_two_arm_25_checkpoint_v2.json"))
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--num-ctx", type=int, default=16384)
    p.add_argument("--num-predict", type=int, default=8192)
    return p.parse_args()


def preflight(args) -> None:
    probe = {
        "model": args.model,
        "prompt": "Think briefly about whether macrophages are immune cells. Return exactly JSON: {\"ok\": true}",
        "stream": False,
        "format": "json",
        "think": True,
        "options": {
            "temperature": args.temperature,
            "top_k": bench.TOP_K,
            "top_p": bench.TOP_P,
            "num_ctx": args.num_ctx,
            "num_predict": args.num_predict,
        },
    }
    response = bench.app._http_json(
        f"{bench.app.ollama_url()}/api/generate",
        headers={"Content-Type": "application/json"},
        data=json.dumps(probe).encode(),
        timeout=600,
    )
    thinking = str(response.get("thinking") or "").strip()
    final = str(response.get("response") or "").strip()
    if not thinking:
        raise RuntimeError(
            f"PREFLIGHT FAILED: {args.model} returned an empty thinking field. Benchmark was NOT started."
        )
    if not final:
        raise RuntimeError(
            "PREFLIGHT FAILED: thinking was produced but the generation budget ended before a final answer. "
            f"eval_count={response.get('eval_count')} done_reason={response.get('done_reason')}. "
            "Increase --num-predict before running the benchmark."
        )
    bench.app._parse_json_object(final)
    print(
        f"Thinking preflight passed: {args.model} returned {len(thinking)} thinking characters "
        f"and a final JSON response (eval_count={response.get('eval_count')})."
    )


bench.parse_args = parse_args
bench.preflight = preflight

if __name__ == "__main__":
    bench.main()
