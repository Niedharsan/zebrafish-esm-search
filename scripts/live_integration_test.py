#!/usr/bin/env python3
"""Optional real Gemini/UniProt integration check using local .env and private DB."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app  # noqa: E402


DEFAULT_CASES = [
    {"query": "macrophage proteins", "expected_gene": "mpeg1.1"},
    {"query": "Wnt signaling proteins", "expected_gene": None},
]


def run_case(query: str, expected_gene: str | None, k: int) -> Dict[str, Any]:
    result = app.discovery_api({"q": [query], "k": [str(k)]})
    plan = result.get("plan") or {}
    seeds = result.get("seeds") or []
    names = [str(seed.get("name") or "").lower() for seed in seeds]

    checks = [
        ("request succeeded", bool(result.get("ok"))),
        ("Gemini research was used", bool(plan.get("ai_used"))),
        ("Google Search grounding was used", bool(plan.get("search_grounded"))),
        ("validated zebrafish seeds were produced", bool(seeds)),
        ("old question-type classifier is gone", "question_type" not in plan),
        ("old evidence-priority taxonomy is gone", "evidence_priorities" not in plan),
    ]
    if expected_gene:
        checks.append((f"{expected_gene} reached final seed set", expected_gene.lower() in names))

    return {
        "query": query,
        "result": result,
        "checks": checks,
        "passed": all(ok for _, ok in checks),
    }


def print_report(report: Dict[str, Any]) -> None:
    result, plan = report["result"], report["result"].get("plan") or {}
    print("\n" + "=" * 76)
    print(report["query"])
    print("=" * 76)
    print(f"retrieval_terms: {plan.get('retrieval_terms')}")
    print(f"rationale: {plan.get('rationale')}")
    print(f"evidence_summary: {plan.get('evidence_summary')}")
    if result.get("retrieval_warning"):
        print(f"retrieval_warning: {result['retrieval_warning']}")

    print("\nFinal validated seeds:")
    for rank, seed in enumerate(result.get("seeds") or [], 1):
        print(
            f"  {rank:>2}. {str(seed.get('name') or seed.get('protein_id')):<18} "
            f"source={seed.get('source')} "
            f"resolved_by={seed.get('resolved_by')}"
        )
        if seed.get("ai_reason"):
            print(f"      reason: {seed['ai_reason']}")

    print("\nChecks:")
    for label, ok in report["checks"]:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real local AI-first zebrafish discovery checks.")
    parser.add_argument("--db", default="data/zebrafish_esm.db")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--query", action="append", help="Custom query; repeat for multiple.")
    parser.add_argument("--json-out")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("Zebrafish ESM live integration test")
    print("REAL Gemini + Google Search + UniProt calls; local API key is never printed.")

    if not app.ai_available():
        print("\nFAIL: GEMINI_API_KEY is not configured.")
        return 2

    db = Path(args.db)
    if not db.is_absolute():
        db = ROOT / db
    if not db.exists():
        print(f"\nFAIL: database not found: {db}")
        return 2

    app.load_database(str(db))
    print(f"AI model: {app.gemini_model()}")
    print(f"Proteins loaded: {len(app.PROTEINS):,}")

    cases = (
        [{"query": q, "expected_gene": None} for q in args.query]
        if args.query
        else DEFAULT_CASES
    )
    reports: List[Dict[str, Any]] = []
    for case in cases:
        try:
            report = run_case(case["query"], case["expected_gene"], max(1, min(args.k, 100)))
        except Exception as exc:
            report = {
                "query": case["query"],
                "result": {"ok": False, "message": str(exc), "plan": {}, "seeds": []},
                "checks": [("live request completed", False)],
                "passed": False,
            }
        reports.append(report)
        print_report(report)

    passed = sum(1 for report in reports if report["passed"])
    print("\n" + "=" * 76)
    print(f"SUMMARY: {passed}/{len(reports)} live cases passed")
    print("=" * 76)

    if args.json_out:
        output = Path(args.json_out)
        if not output.is_absolute():
            output = ROOT / output
        output.write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved report: {output}")

    return 0 if passed == len(reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
