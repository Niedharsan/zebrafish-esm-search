#!/usr/bin/env python3
"""Benchmark local Ollama interpretation through the full zebrafish discovery pipeline."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app  # noqa: E402


CASES = [
    ("macrophage", "Which proteins mark zebrafish macrophages?", ["mpeg1", "mpeg1.1", "mfap4", "csf1ra"]),
    ("macrophage", "Find proteins involved in macrophage phagocytosis.", ["mpeg1", "mpeg1.1", "marco", "spi1b", "lcp1"]),
    ("macrophage", "What genes identify microglia in the zebrafish brain?", ["apoeb", "p2ry12", "csf1ra", "mpeg1.1"]),
    ("vascular", "Which proteins mark vascular endothelial cells in zebrafish?", ["kdrl", "cdh5", "fli1a", "etv2", "pecam1"]),
    ("vascular", "Find core zebrafish angiogenesis proteins.", ["vegfaa", "kdrl", "kdr", "flt1", "dll4"]),
    ("vascular", "Which genes help form the zebrafish blood-brain barrier?", ["cldn5a", "slc2a1a", "mfsd2aa", "cdh5"]),
    ("neuronal", "Give me general neuronal marker proteins in zebrafish.", ["elavl3", "tubb5", "neurod1", "map2"]),
    ("neuronal", "Which genes identify dopaminergic neurons in zebrafish?", ["th", "slc6a3", "ddc", "nr4a2a"]),
    ("neuronal", "Find proteins associated with zebrafish motor neurons.", ["mnx1", "isl1", "isl2a", "chat", "slc18a3a"]),
    ("immune", "Which proteins respond to bacterial infection in zebrafish?", ["il1b", "tnfa", "mpx", "lyz", "nfkbiaa"]),
    ("immune", "Find zebrafish antiviral interferon-response proteins.", ["stat1a", "stat1b", "mxa", "isg15", "irf7"]),
    ("immune", "Which genes mark adaptive immune lymphocytes in zebrafish?", ["rag1", "rag2", "cd4-1", "cd8a", "ighm"]),
    ("pigmentation", "Find proteins required for zebrafish melanophore pigmentation.", ["mitfa", "tyr", "dct", "tyrp1b"]),
    ("pigmentation", "Which genes are important for zebrafish xanthophores?", ["csf1ra", "pax7a", "gch2"]),
    ("pigmentation", "What proteins control zebrafish pigment stripe formation?", ["kita", "kitlga", "ednrb1a", "gja5b"]),
    ("cardiac", "Which proteins mark zebrafish cardiomyocytes?", ["myl7", "nkx2.5", "tnnt2a", "myh6"]),
    ("cardiac", "Find genes involved in the zebrafish cardiac conduction system.", ["hcn4", "nkx2.5", "isl1", "tbx3"]),
    ("cardiac", "Which proteins contribute to zebrafish heart regeneration?", ["gata4", "hand2", "stat3", "nrg1"]),
    ("crispr", "Suggest candidate CRISPR targets to reduce zebrafish pigmentation.", ["mitfa", "tyr", "slc24a5"]),
    ("crispr", "What genes could I knock out to reduce macrophage development in zebrafish?", ["irf8", "csf1ra", "spi1b"]),
    ("crispr", "Suggest zebrafish CRISPR targets for disrupting blood-vessel development.", ["kdrl", "vegfaa", "etv2", "fli1a"]),
    ("ambiguous", "What proteins make the fish transparent?", ["mitfa", "tyr", "slc24a5", "mpv17"]),
    ("ambiguous", "Which cells-eating-debris proteins matter after a zebrafish brain injury?", ["mpeg1.1", "apoeb", "csf1ra", "p2ry12"]),
    ("ambiguous", "Find genes that make vessels grow around a wound.", ["vegfaa", "kdrl", "flt1", "dll4"]),
]


def run_case(category: str, query: str, expected: List[str], k: int) -> Dict[str, Any]:
    started = time.monotonic()
    plan = app.interpret_biological_query(query)
    candidates = plan.get("zebrafish_candidates") or []
    with ThreadPoolExecutor(max_workers=min(6, max(1, len(candidates)))) as pool:
        resolved_groups = list(pool.map(lambda candidate: app.resolve_targeted_uniprot_candidates([candidate]), candidates))
    direct = app._merge_seeds(*(group[0] for group in resolved_groups))
    trace = [item for group in resolved_groups for item in group[1]]
    errors = [item for group in resolved_groups for item in group[2]]
    references = app.orthology_seeds(plan.get("reference_candidates") or []) if len(direct) < 4 else []
    resolved = app._merge_seeds(direct, references)
    candidate_genes = [str(item.get("gene") or "").lower() for item in plan.get("zebrafish_candidates") or []]
    seed_genes = [str(app.PROTEINS[int(item["index"])].get("name") or "").lower() for item in resolved]
    expected_set = {gene.lower() for gene in expected}
    expected_hits = sorted(expected_set.intersection(candidate_genes + seed_genes))
    return {
        "category": category,
        "query": query,
        "expected_examples": expected,
        "candidate_genes": candidate_genes,
        "candidate_count": len(candidate_genes),
        "validated_seed_genes": seed_genes,
        "validated_seed_count": len(seed_genes),
        "expected_hits": expected_hits,
        "candidate_relevance_pass": bool(expected_hits),
        "pipeline_pass": bool(resolved),
        "ai_used": bool(plan.get("ai_used")),
        "search_grounded": bool(plan.get("search_grounded")),
        "rationale": plan.get("rationale"),
        "retrieval_errors": errors,
        "resolution_trace": trace,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "neighbors": app.discovery_neighbors([int(seed["index"]) for seed in resolved], k) if resolved else [],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a local Ollama model on zebrafish ESM discovery.")
    parser.add_argument("--db", default="data/zebrafish_esm.db")
    parser.add_argument("--model", default="qwen3:4b-instruct")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--json-out", default="local_4b_benchmark.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["AI_PROVIDER"] = "ollama"
    os.environ["OLLAMA_MODEL"] = args.model
    db = Path(args.db)
    if not db.is_absolute():
        db = ROOT / db
    app.load_database(str(db))
    print(f"Local model benchmark: {args.model}")
    print(f"Cases: {len(CASES)}; proteins: {len(app.PROTEINS):,}; external validation: UniProt/Ensembl")

    reports = []
    for number, (category, query, expected) in enumerate(CASES, 1):
        try:
            report = run_case(category, query, expected, max(1, min(args.k, 100)))
        except Exception as exc:
            report = {
                "category": category,
                "query": query,
                "candidate_relevance_pass": False,
                "pipeline_pass": False,
                "error": str(exc),
            }
        reports.append(report)
        print(
            f"[{number:02}/{len(CASES)}] {category:<12} "
            f"relevant={'yes' if report.get('candidate_relevance_pass') else 'no ':<3} "
            f"seeds={report.get('validated_seed_count', 0):>2} "
            f"{report.get('elapsed_seconds', 0):>6}s  {query}"
        )

    relevance_passes = sum(bool(row.get("candidate_relevance_pass")) for row in reports)
    pipeline_passes = sum(bool(row.get("pipeline_pass")) for row in reports)
    total_seconds = round(sum(float(row.get("elapsed_seconds") or 0) for row in reports), 2)
    summary = {
        "model": args.model,
        "case_count": len(reports),
        "candidate_relevance_passes": relevance_passes,
        "candidate_relevance_rate": round(relevance_passes / len(reports), 3),
        "pipeline_passes": pipeline_passes,
        "pipeline_success_rate": round(pipeline_passes / len(reports), 3),
        "total_elapsed_seconds": total_seconds,
        "web_grounding_used": any(bool(row.get("search_grounded")) for row in reports),
    }
    output = Path(args.json_out)
    if not output.is_absolute():
        output = ROOT / output
    output.write_text(json.dumps({"summary": summary, "cases": reports}, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved detailed report: {output}")
    return 0 if pipeline_passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
