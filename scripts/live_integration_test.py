#!/usr/bin/env python3
"""Run optional live biological-discovery checks against local credentials/data.

This script is intentionally not part of CI. It uses the repository's ignored
.env file and private local zebrafish database, then makes real Gemini, UniProt,
Ensembl, and Google Search-grounded requests through the same discovery code as
the dashboard. It never prints the Gemini API key.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app  # noqa: E402


DEFAULT_CASES = [
    {
        "query": "macrophage proteins",
        "expected_question_types": {"cell_type", "tissue"},
        "expected_gene": "mpeg1.1",
        "preferred_evidence": {
            "marker_support",
            "expression_support",
            "expression_annotation",
            "literature_support",
            "functional_evidence",
            "function_annotation",
        },
    },
    {
        "query": "Wnt signaling proteins",
        "expected_question_types": {"pathway", "biological_process"},
        "expected_gene": None,
        "preferred_evidence": {
            "pathway_support",
            "pathway_annotation",
            "go_biological_process",
            "functional_evidence",
            "function_annotation",
            "literature_support",
        },
    },
]

WEAK_ONLY_EVIDENCE = {"name_match", "uniprot_text_search_hit"}


def _seed_name(seed: Dict[str, Any]) -> str:
    return str(seed.get("name") or seed.get("protein_id") or "").strip()


def _seed_evidence(seed: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for value in seed.get("evidence_types") or []:
        item = str(value).strip()
        if item and item not in out:
            out.append(item)
    return out


def _all_seed_evidence(seeds: Iterable[Dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for seed in seeds:
        out.update(_seed_evidence(seed))
    return out


def _print_check(ok: bool, text: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {text}")


def _run_case(case: Dict[str, Any], k: int, include_explanation: bool) -> Dict[str, Any]:
    query = str(case["query"])
    original_explain = app.explain_discovery
    if not include_explanation:
        app.explain_discovery = lambda *args, **kwargs: None  # type: ignore[assignment]
    try:
        result = app.discovery_api({"q": [query], "k": [str(k)]})
    finally:
        app.explain_discovery = original_explain

    plan = result.get("plan") or {}
    seeds = result.get("seeds") or []
    evidence_summary = plan.get("evidence_summary") or {}
    seed_names = {_seed_name(seed).lower() for seed in seeds if _seed_name(seed)}
    all_evidence = _all_seed_evidence(seeds)

    checks: List[Dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    check("discovery request succeeded", bool(result.get("ok")), str(result.get("message") or ""))
    check("AI ranking was used", bool(plan.get("ai_used")))
    check("Google Search grounding was used", bool(plan.get("search_grounded")))
    check("validated local zebrafish seeds were produced", bool(seeds))

    sources = {str(value) for value in evidence_summary.get("sources") or []}
    check("structured UniProt evidence was retrieved", "UniProtKB" in sources)
    check("Ensembl identifier evidence was available", "Ensembl" in sources)

    expected_types = set(case.get("expected_question_types") or [])
    if expected_types:
        actual_type = str(plan.get("question_type") or "")
        check(
            "question type is appropriate",
            actual_type in expected_types,
            f"actual={actual_type}; expected one of {sorted(expected_types)}",
        )

    expected_gene = str(case.get("expected_gene") or "").strip().lower()
    if expected_gene:
        check(
            f"expected biological candidate {expected_gene} reached the final seed set",
            expected_gene in seed_names,
            f"final seeds={sorted(seed_names)}",
        )

    if seeds:
        lexical_only = bool(all_evidence) and all_evidence.issubset(WEAK_ONLY_EVIDENCE)
        check(
            "final seeds are not supported only by lexical/name matches",
            not lexical_only,
            f"seed evidence={sorted(all_evidence)}",
        )

    preferred = set(case.get("preferred_evidence") or [])
    if preferred and seeds:
        observed_preferred = sorted(all_evidence & preferred)
        check(
            "evidence types fit this question class",
            bool(observed_preferred),
            f"matched={observed_preferred}; observed={sorted(all_evidence)}",
        )

    return {
        "query": query,
        "result": result,
        "checks": checks,
        "passed": all(item["ok"] for item in checks),
    }


def _print_case(report: Dict[str, Any]) -> None:
    result = report["result"]
    plan = result.get("plan") or {}
    seeds = result.get("seeds") or []

    print("\n" + "=" * 76)
    print(report["query"])
    print("=" * 76)
    print(f"question_type: {plan.get('question_type')}")
    print(f"retrieval_terms: {plan.get('retrieval_terms')}")
    print(f"evidence_priorities: {plan.get('evidence_priorities')}")
    print(f"evidence_summary: {plan.get('evidence_summary')}")
    if result.get("retrieval_warning"):
        print(f"retrieval_warning: {result['retrieval_warning']}")

    print("\nFinal validated seeds:")
    if not seeds:
        print("  <none>")
    for rank, seed in enumerate(seeds, start=1):
        print(
            f"  {rank:>2}. {_seed_name(seed):<18} "
            f"evidence={_seed_evidence(seed)} "
            f"resolved_by={seed.get('resolved_by')}"
        )

    print("\nChecks:")
    for item in report["checks"]:
        detail = f" — {item['detail']}" if item.get("detail") and not item["ok"] else ""
        _print_check(bool(item["ok"]), f"{item['name']}{detail}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real local Gemini/UniProt/Ensembl integration checks (not CI)."
    )
    parser.add_argument("--db", default="data/zebrafish_esm.db", help="Local private SQLite database path")
    parser.add_argument("--k", type=int, default=10, help="Number of ESM neighbors to request")
    parser.add_argument(
        "--query",
        action="append",
        help="Custom biological query. Repeat for multiple queries. Without this flag, runs macrophage and Wnt checks.",
    )
    parser.add_argument(
        "--include-explanation",
        action="store_true",
        help="Also run the final Gemini explanation call (costs one additional model request per query).",
    )
    parser.add_argument("--json-out", help="Optional path for a sanitized JSON report")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("Zebrafish ESM live integration test")
    print("This makes REAL remote requests and is intentionally excluded from CI.")
    print("The API key is loaded locally and is never printed.")

    if not app.ai_available():
        print("\nFAIL: GEMINI_API_KEY is not configured in the local environment/.env.")
        return 2

    db_path = Path(args.db)
    if not db_path.is_absolute():
        db_path = ROOT / db_path
    if not db_path.exists():
        print(f"\nFAIL: database not found: {db_path}")
        return 2

    app.load_database(str(db_path))
    print(f"AI model: {app.gemini_model()}")
    print(f"Database: {db_path}")
    print(f"Proteins loaded: {len(app.PROTEINS):,}")

    if args.query:
        cases = [
            {
                "query": query,
                "expected_question_types": set(),
                "expected_gene": None,
                "preferred_evidence": set(),
            }
            for query in args.query
        ]
    else:
        cases = DEFAULT_CASES

    reports: List[Dict[str, Any]] = []
    for case in cases:
        try:
            report = _run_case(case, max(1, min(args.k, 100)), args.include_explanation)
        except Exception as exc:
            report = {
                "query": case["query"],
                "result": {"ok": False, "message": str(exc), "plan": {}, "seeds": []},
                "checks": [{"name": "live request completed", "ok": False, "detail": str(exc)}],
                "passed": False,
            }
        reports.append(report)
        _print_case(report)

    passed = sum(1 for report in reports if report["passed"])
    print("\n" + "=" * 76)
    print(f"SUMMARY: {passed}/{len(reports)} live cases passed")
    print("=" * 76)

    if args.json_out:
        output = Path(args.json_out)
        if not output.is_absolute():
            output = ROOT / output
        sanitized = {
            "model": app.gemini_model(),
            "database": str(db_path),
            "protein_count": len(app.PROTEINS),
            "reports": reports,
        }
        output.write_text(json.dumps(sanitized, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved report: {output}")

    return 0 if passed == len(reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
