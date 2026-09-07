#!/usr/bin/env python3
"""Hard-25 benchmark for the dedicated Qwen3 4B Thinking model.
Runs only: (1) thinking base and (2) thinking -> existing Arm-2 tools -> thinking.
Aborts before the benchmark if Ollama does not return a non-empty thinking trace.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app

BASE_PATH = ROOT / "scripts" / "benchmark_qwen3_4b_three_arm_25.py"
FOUR_PATH = ROOT / "scripts" / "benchmark_qwen3_4b_thinking_four_arm_25.py"

base_spec = importlib.util.spec_from_file_location("hard25_base", BASE_PATH)
base = importlib.util.module_from_spec(base_spec)
assert base_spec and base_spec.loader
base_spec.loader.exec_module(base)

four_spec = importlib.util.spec_from_file_location("thinking_four", FOUR_PATH)
four = importlib.util.module_from_spec(four_spec)
assert four_spec and four_spec.loader
four_spec.loader.exec_module(four)

VERSION = "thinking-two-arm-hard25-v1.0"
ARMS = ("thinking_base", "thinking_arm2")
DEFAULT_CASES = ROOT / "hard25_qwen3_4b_three_arm_cases.json"
DEFAULT_DB = ROOT / "data" / "zebrafish_esm.db"
DEFAULT_CP = ROOT / "qwen3_4b_thinking_two_arm_25_checkpoint.json"
DEFAULT_OUT = ROOT / "qwen3_4b_thinking_two_arm_25_results.json"

# Official qwen3:4b-thinking Ollama sampling defaults are temp=.6, top_k=20, top_p=.95.
# 16k context / 6k generation budget gives the thinking model room for reasoning plus JSON output,
# especially on the second pass where the Arm-2 evidence prompt is much larger than the base prompt.
TOP_K = 20
TOP_P = 0.95


def payload(prompt: str, *, model: str, temperature: float, num_ctx: int, num_predict: int) -> dict:
    return {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "think": True,
        "options": {
            "temperature": temperature,
            "top_k": TOP_K,
            "top_p": TOP_P,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
    }


def strict_call(prompt: str, question: str, **cfg):
    attempts = []
    last = None
    for attempt in range(2):
        retry = "\nPrevious final answer was not valid JSON. Return one complete JSON object only." if attempt else ""
        started = time.monotonic()
        try:
            response = app._http_json(
                f"{app.ollama_url()}/api/generate",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload(prompt + retry, **cfg)).encode(),
                timeout=600,
            )
            thinking = str(response.get("thinking") or "").strip()
            final = str(response.get("response") or "").strip()
            if not thinking:
                raise RuntimeError(
                    f"Model {cfg['model']} returned no thinking trace. Aborting: this is not a valid thinking-mode run."
                )
            if not final:
                raise RuntimeError("Thinking consumed the generation budget and no final answer was returned.")
            plan = base.clean(app._parse_json_object(final), question)
            attempts.append({
                "attempt": attempt + 1,
                "thinking": thinking,
                "final_response": final,
                "seconds": round(time.monotonic() - started, 3),
                "prompt_eval_count": response.get("prompt_eval_count"),
                "eval_count": response.get("eval_count"),
            })
            return plan, {"attempts": attempts, "thinking_present": True}
        except Exception as exc:
            last = exc
            attempts.append({"attempt": attempt + 1, "error": str(exc), "seconds": round(time.monotonic() - started, 3)})
            # Missing thinking is not a JSON formatting problem; retrying would waste time.
            if "returned no thinking trace" in str(exc):
                raise
    raise RuntimeError(f"Thinking Ollama failed after one retry. ({last})") from last


# Reuse the already-tested Arm-2 retrieval/validation implementation, but force every model call
# through the strict thinking call above.
four.call_thinking_qwen = strict_call


def preflight(args) -> None:
    probe = {
        "model": args.model,
        "prompt": "Think briefly about whether macrophages are immune cells. Return exactly JSON: {\"ok\": true}",
        "stream": False,
        "format": "json",
        "think": True,
        "options": {
            "temperature": args.temperature,
            "top_k": TOP_K,
            "top_p": TOP_P,
            "num_ctx": args.num_ctx,
            "num_predict": min(args.num_predict, 2048),
        },
    }
    response = app._http_json(
        f"{app.ollama_url()}/api/generate",
        headers={"Content-Type": "application/json"},
        data=json.dumps(probe).encode(),
        timeout=300,
    )
    thinking = str(response.get("thinking") or "").strip()
    final = str(response.get("response") or "").strip()
    if not thinking:
        raise RuntimeError(
            f"PREFLIGHT FAILED: {args.model} returned an empty thinking field. Benchmark was NOT started."
        )
    if not final:
        raise RuntimeError("PREFLIGHT FAILED: model produced thinking but no final response.")
    app._parse_json_object(final)
    print(f"Thinking preflight passed: {args.model} returned {len(thinking)} thinking characters.")


def result_payload(cases, args):
    return {
        "metadata": {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_commit": base.git("rev-parse", "HEAD"),
            "git_branch": base.git("branch", "--show-current"),
            "runner_version": VERSION,
            "ollama_model": args.model,
            "ollama_url": app.ollama_url(),
            "case_count": 25,
            "completed_case_count": len(cases),
            "cases_sha256": base.sha(args.cases),
            "app_sha256": base.sha(ROOT / "app.py"),
            "runner_sha256": base.sha(Path(__file__)),
            "database_sha256": base.sha(args.db),
            "k": args.k,
            "ollama_settings": {
                "think": True,
                "temperature": args.temperature,
                "top_k": TOP_K,
                "top_p": TOP_P,
                "num_ctx": args.num_ctx,
                "num_predict": args.num_predict,
                "format": "json",
            },
            "arms": list(ARMS),
        },
        "summary": {arm: base.summary(cases, arm) for arm in ARMS} if cases else {},
        "pairwise": {"thinking_base_vs_thinking_arm2": base.pair(cases, *ARMS)} if cases else {},
        "cases": cases,
    }


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", default=str(DEFAULT_CASES))
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.add_argument("--model", default="qwen3:4b-thinking")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--checkpoint", default=str(DEFAULT_CP))
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--num-ctx", type=int, default=16384)
    p.add_argument("--num-predict", type=int, default=6000)
    return p.parse_args()


def main():
    args = parse_args()
    data = base.load_cases(args.cases)
    cases = data["cases"]
    if len(cases) != 25:
        raise RuntimeError(f"Expected exactly 25 cases, found {len(cases)}")

    os.environ["AI_PROVIDER"] = "ollama"
    os.environ["OLLAMA_MODEL"] = args.model
    app.load_database(args.db)

    # Critical guard against repeating the prior invalid four-hour run.
    preflight(args)

    cp = Path(args.checkpoint)
    out = Path(args.out)
    completed = []
    if cp.exists():
        saved = json.loads(cp.read_text(encoding="utf-8"))
        meta = saved.get("metadata") or {}
        if meta.get("runner_version") != VERSION:
            raise RuntimeError("Checkpoint runner version does not match.")
        if meta.get("cases_sha256") != base.sha(args.cases):
            raise RuntimeError("Checkpoint cases file does not match.")
        if meta.get("ollama_model") != args.model:
            raise RuntimeError("Checkpoint model does not match.")
        completed = list(saved.get("cases") or [])

    done_ids = {int(x["id"]) for x in completed}
    for case in cases:
        cid = int(case["id"])
        if cid in done_ids:
            continue
        print(f"\n[{len(completed)+1}/25] case {cid}: {case['query']}")
        row = {
            "id": cid,
            "category": case.get("category"),
            "subtopic": case.get("subtopic"),
            "wording": case.get("wording"),
            "query_specificity": case.get("query_specificity"),
            "query": case["query"],
            "expected_examples": case["expected_examples"],
        }

        print("  arm: thinking_base")
        row["thinking_base"] = four.run_thinking_base(case, args.k, args)
        print("  arm: thinking_arm2")
        row["thinking_arm2"] = four.run_thinking_arm2(case, args.k, args)

        completed.append(row)
        write_json(cp, result_payload(completed, args))
        print(
            f"  saved case {cid}: base canonical={row['thinking_base']['canonical_hit_any']} "
            f"tools canonical={row['thinking_arm2']['canonical_hit_any']}"
        )

    final = result_payload(completed, args)
    write_json(out, final)
    write_json(cp, final)

    print("\nFINAL SUMMARY")
    for arm in ARMS:
        s = final["summary"][arm]
        print(
            f"{arm}: canonical={s['canonical_any_rate']*100:.1f}% "
            f"recall={s['mean_canonical_recall']*100:.1f}% H@1={s['hit_at_1_rate']*100:.1f}% "
            f"H@3={s['hit_at_3_rate']*100:.1f}% H@5={s['hit_at_5_rate']*100:.1f}% "
            f"MRR={s['mean_mrr']:.3f} unresolved={s['unresolved_rate']*100:.1f}% "
            f"median={s['median_latency_seconds']:.1f}s"
        )


if __name__ == "__main__":
    main()
