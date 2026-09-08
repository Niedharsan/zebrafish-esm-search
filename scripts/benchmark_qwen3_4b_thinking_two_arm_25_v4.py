#!/usr/bin/env python3
"""Hard-25 v4: real Qwen3 4B Thinking + ZFIN nomenclature repair + QuickGO annotations.

Changes relative to v3:
- keeps the verified /api/chat + think=true transport;
- normalizes zebrafish candidate symbols with ZFIN Previous Names;
- adds ZFIN curated human/mouse -> zebrafish ortholog candidates;
- expands retrieved QuickGO GO terms into actual Danio rerio annotation candidates;
- normalizes the final plan again before deterministic validation/scoring.

The v3 files/checkpoint remain untouched so the 88% baseline stays reproducible.
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

import scientific_candidate_repair as repair

bench.VERSION = "thinking-two-arm-hard25-v1.3-chat-zfin-quickgo"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", default=str(bench.DEFAULT_CASES))
    p.add_argument("--db", default=str(bench.DEFAULT_DB))
    p.add_argument("--model", default="qwen3:4b-thinking")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--out", default=str(ROOT / "qwen3_4b_thinking_two_arm_25_results_v4.json"))
    p.add_argument("--checkpoint", default=str(ROOT / "qwen3_4b_thinking_two_arm_25_checkpoint_v4.json"))
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
        retry = (
            "\nPrevious final answer was not valid JSON. Return one complete JSON object only, "
            "with no commentary outside it."
            if attempt
            else ""
        )
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
            attempts.append(
                {
                    "attempt": attempt + 1,
                    "thinking": thinking,
                    "final_response": final,
                    "seconds": round(time.monotonic() - started, 3),
                    "prompt_eval_count": response.get("prompt_eval_count"),
                    "eval_count": response.get("eval_count"),
                    "done_reason": response.get("done_reason"),
                }
            )
            return plan, {"attempts": attempts, "thinking_present": True, "endpoint": "/api/chat"}
        except Exception as exc:
            last = exc
            attempts.append(
                {"attempt": attempt + 1, "error": str(exc), "seconds": round(time.monotonic() - started, 3)}
            )
            if "returned no thinking trace" in str(exc) or "produced thinking but no final content" in str(exc):
                raise
    raise RuntimeError(f"Thinking Ollama chat call failed after one retry. ({last})") from last


bench.four.call_thinking_qwen = strict_chat_call
_original_retrieve_tools = bench.four.retrieve_tools


def retrieve_tools_with_repairs(question: str, plan: dict) -> dict:
    tools = _original_retrieve_tools(question, plan)
    annotation_evidence, annotation_errors = repair.quickgo_annotation_evidence(
        tools.get("evidence") or [],
        bench.app._http_json,
    )
    if annotation_evidence:
        tools["evidence"].extend(annotation_evidence)
    tools["retrieval_errors"].extend(annotation_errors)
    if "ZFIN nomenclature + curated orthology" not in tools["retrieval_sources"]:
        tools["retrieval_sources"].append("ZFIN nomenclature + curated orthology")
    if annotation_evidence and "QuickGO Danio rerio annotations" not in tools["retrieval_sources"]:
        tools["retrieval_sources"].append("QuickGO Danio rerio annotations")
    return tools


def run_thinking_arm2_repaired(case, k, args):
    started = time.monotonic()

    first_raw, t1 = strict_chat_call(
        bench.base.base_prompt(case["query"]),
        case["query"],
        **bench.four.cfg(args),
    )
    first, first_repair = repair.normalize_plan(
        first_raw,
        max_seeds=bench.base.MAX_SEEDS,
        max_refs=bench.base.MAX_REFS,
    )

    retrieval_started = time.monotonic()
    tools = retrieve_tools_with_repairs(case["query"], first)
    retrieval_seconds = time.monotonic() - retrieval_started

    final_raw, t2 = strict_chat_call(
        bench.base.second_prompt(
            case["query"],
            first,
            tools["resolution"],
            tools["orthology"],
            tools["local_context"],
            tools["evidence"],
        ),
        case["query"],
        **bench.four.cfg(args),
    )
    final, final_repair = repair.normalize_plan(
        final_raw,
        max_seeds=bench.base.MAX_SEEDS,
        max_refs=bench.base.MAX_REFS,
    )

    out = bench.base.score(final, case["expected_examples"], k)
    out.update(
        {
            "first_pass_plan": first_raw,
            "normalized_first_pass_plan": first,
            "plan_before_final_zfin_repair": final_raw,
            "plan": final,
            "zfin_first_pass_repair": first_repair,
            "zfin_final_repair": final_repair,
            "model_trace": {"first_pass": t1, "second_pass": t2},
            "candidate_resolution_checks": tools["resolution"],
            "orthology_checks": tools["orthology"],
            "local_context_records": len(tools["local_context"]),
            "authoritative_evidence_records": len(tools["evidence"]),
            "authoritative_evidence": tools["evidence"],
            "retrieval_sources": tools["retrieval_sources"],
            "retrieval_errors": [
                *first_repair.get("errors", []),
                *tools["retrieval_errors"],
                *final_repair.get("errors", []),
            ],
            "retrieval_seconds": round(retrieval_seconds, 3),
            "total_seconds": round(time.monotonic() - started, 3),
        }
    )
    return out


bench.four.retrieve_tools = retrieve_tools_with_repairs
bench.four.run_thinking_arm2 = run_thinking_arm2_repaired


def preflight(args) -> None:
    cfg = {
        "model": args.model,
        "temperature": args.temperature,
        "num_ctx": args.num_ctx,
        "num_predict": args.num_predict,
    }

    probe = chat_payload("Think briefly, then answer exactly: OK", **cfg)
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

    test_question = "Which proteins mark zebrafish macrophages?"
    strict_chat_call(bench.base.base_prompt(test_question), test_question, **cfg)
    print(
        f"Thinking preflight passed via /api/chat: transport thinking={len(thinking)} chars, "
        f"final={len(final)} chars; benchmark-style JSON parse passed."
    )


bench.parse_args = parse_args
bench.preflight = preflight

if __name__ == "__main__":
    bench.main()
