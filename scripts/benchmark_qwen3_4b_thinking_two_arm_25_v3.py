#!/usr/bin/env python3
"""Two-arm hard-25 benchmark for qwen3:4b-thinking using Ollama /api/chat.

Why v3 exists:
- qwen3:4b-thinking did produce a thinking trace in v1/v2.
- With /api/generate + format='json', Ollama returned an empty final response even though done_reason='stop'.
- Ollama documents /api/chat for qwen3:4b-thinking, and structured-output + reasoning interactions are known to be problematic.

This runner therefore:
1) uses /api/chat,
2) keeps think=true,
3) removes the format=json constraint,
4) still requires a non-empty thinking trace,
5) parses the final content as JSON using the repo's existing parser,
6) runs only thinking_base and thinking_arm2.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = ROOT / "scripts" / "benchmark_qwen3_4b_thinking_two_arm_25.py"

spec = importlib.util.spec_from_file_location("thinking_two_arm", BASE_SCRIPT)
bench = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(bench)

bench.VERSION = "thinking-two-arm-hard25-v1.2-chat"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", default=str(bench.DEFAULT_CASES))
    p.add_argument("--db", default=str(bench.DEFAULT_DB))
    p.add_argument("--model", default="qwen3:4b-thinking")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--out", default=str(ROOT / "qwen3_4b_thinking_two_arm_25_results_v3.json"))
    p.add_argument("--checkpoint", default=str(ROOT / "qwen3_4b_thinking_two_arm_25_checkpoint_v3.json"))
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--num-ctx", type=int, default=16384)
    p.add_argument("--num-predict", type=int, default=8192)
    return p.parse_args()


def chat_payload(prompt: str, *, model: str, temperature: float, num_ctx: int, num_predict: int) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": True,
        "options": {
            "temperature": temperature,
            "top_k": bench.TOP_K,
            "top_p": bench.TOP_P,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
    }


def strict_chat_call(prompt: str, question: str, **cfg):
    attempts = []
    last = None
    for attempt in range(2):
        retry = "\nPrevious final answer was not valid JSON. Return one complete JSON object only, with no commentary outside it." if attempt else ""
        started = time.monotonic()
        try:
            response = bench.app._http_json(
                f"{bench.app.ollama_url()}/api/chat",
                headers={"Content-Type": "application/json"},
                data=json.dumps(chat_payload(prompt + retry, **cfg)).encode(),
                timeout=600,
            )
            message = response.get("message") or {}
            thinking = str(message.get("thinking") or "").strip()
            final = str(message.get("content") or "").strip()
            if not thinking:
                raise RuntimeError(
                    f"Model {cfg['model']} returned no thinking trace. Aborting: not a valid thinking-mode run."
                )
            if not final:
                raise RuntimeError(
                    f"Model {cfg['model']} produced thinking but no final content via /api/chat. "
                    f"done_reason={response.get('done_reason')} eval_count={response.get('eval_count')}"
                )
            plan = bench.base.clean(bench.app._parse_json_object(final), question)
            attempts.append({
                "attempt": attempt + 1,
                "thinking": thinking,
                "final_response": final,
                "seconds": round(time.monotonic() - started, 3),
                "prompt_eval_count": response.get("prompt_eval_count"),
                "eval_count": response.get("eval_count"),
                "done_reason": response.get("done_reason"),
            })
            return plan, {"attempts": attempts, "thinking_present": True, "endpoint": "/api/chat"}
        except Exception as exc:
            last = exc
            attempts.append({"attempt": attempt + 1, "error": str(exc), "seconds": round(time.monotonic() - started, 3)})
            if "returned no thinking trace" in str(exc) or "produced thinking but no final content" in str(exc):
                raise
    raise RuntimeError(f"Thinking Ollama chat call failed after one retry. ({last})") from last


# The existing two-arm runner delegates model calls to the imported four-arm module.
bench.four.call_thinking_qwen = strict_chat_call


def preflight(args) -> None:
    cfg = {
        "model": args.model,
        "temperature": args.temperature,
        "num_ctx": args.num_ctx,
        "num_predict": args.num_predict,
    }

    # First check the transport itself without structured-output constraints.
    probe = chat_payload(
        "Think briefly, then answer exactly: OK",
        **cfg,
    )
    response = bench.app._http_json(
        f"{bench.app.ollama_url()}/api/chat",
        headers={"Content-Type": "application/json"},
        data=json.dumps(probe).encode(),
        timeout=600,
    )
    message = response.get("message") or {}
    thinking = str(message.get("thinking") or "").strip()
    final = str(message.get("content") or "").strip()
    if not thinking or not final:
        raise RuntimeError(
            "PREFLIGHT FAILED on /api/chat: "
            f"thinking_chars={len(thinking)} final_chars={len(final)} "
            f"done_reason={response.get('done_reason')} eval_count={response.get('eval_count')}"
        )

    # Then check the actual benchmark JSON pathway using the real base prompt.
    test_question = "Which proteins mark zebrafish macrophages?"
    _, trace = strict_chat_call(bench.base.base_prompt(test_question), test_question, **cfg)
    print(
        f"Thinking preflight passed via /api/chat: transport thinking={len(thinking)} chars, "
        f"final={len(final)} chars; benchmark-style JSON parse passed."
    )


bench.parse_args = parse_args
bench.preflight = preflight

if __name__ == "__main__":
    bench.main()
