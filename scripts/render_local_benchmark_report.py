#!/usr/bin/env python3
"""Render the local-model benchmark JSON as a shareable Markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable


def joined(values: Iterable[Any], fallback: str = "None") -> str:
    cleaned = [f"`{value}`" for value in values if str(value).strip()]
    return ", ".join(cleaned) or fallback


def outcome(case: Dict[str, Any]) -> str:
    if case.get("candidate_relevance_pass") and case.get("pipeline_pass"):
        return "Reference hit + validated seeds"
    if case.get("pipeline_pass"):
        return "No reference hit; validated seeds produced"
    return "No validated seed"


def render(payload: Dict[str, Any]) -> str:
    summary = payload["summary"]
    cases = payload["cases"]
    elapsed = sorted(float(case.get("elapsed_seconds") or 0) for case in cases)
    middle = len(elapsed) // 2
    median = (elapsed[middle - 1] + elapsed[middle]) / 2 if len(elapsed) % 2 == 0 else elapsed[middle]
    categories = sorted({str(case["category"]) for case in cases})

    lines = [
        "# Local Qwen3 4B benchmark: Zebrafish ESM Discovery",
        "",
        "## Executive summary",
        "",
        "This report records one local benchmark run of `qwen3:4b-instruct` (4B, Q4_K_M) through Ollama. "
        "The model interpreted each biological question and proposed candidate genes. The existing pipeline then used "
        "UniProt/Ensembl for deterministic identity resolution and the private local ESM database for similarity search. "
        "Gemini was not called, Google Search grounding was disabled, and embeddings remained local.",
        "",
        f"- Questions: **{summary['case_count']}** across **{len(categories)} categories** ({', '.join(categories)})",
        f"- At least one validated local ESM seed: **{summary['pipeline_passes']}/{summary['case_count']} ({summary['pipeline_success_rate']:.1%})**",
        f"- At least one match to the case's predefined canonical reference examples: **{summary['candidate_relevance_passes']}/{summary['case_count']} ({summary['candidate_relevance_rate']:.1%})**",
        f"- Median end-to-end latency: **{median:.2f} seconds per question**",
        f"- Mean end-to-end latency: **{summary['total_elapsed_seconds'] / summary['case_count']:.2f} seconds per question**",
        f"- Observed latency range: **{elapsed[0]:.2f}–{elapsed[-1]:.2f} seconds**",
        f"- Total runtime: **{summary['total_elapsed_seconds']:.2f} seconds**",
        f"- Web grounding used: **{'yes' if summary['web_grounding_used'] else 'no'}**",
        "",
        "The canonical-reference check is a transparent heuristic, not a biological gold-standard accuracy score. "
        "A reference miss can still contain relevant biology—for example, the general macrophage query returned and validated `c1qa`—" 
        "while a validated identifier is not by itself proof that every proposed protein is relevant to the question.",
        "",
        "## Post-benchmark macrophage fix and retest",
        "",
        "The original broad prompt, “Which proteins mark zebrafish macrophages?”, did not return `mpeg1.1`. Unlike the Gemini path, "
        "which uses zebrafish-specific Google Search grounding, the initial Ollama path relied entirely on the 4B model's internal knowledge. "
        "It consequently favored mammalian-style immune markers such as `ms4a1a`, `tlr4`, and `ccl2`.",
        "",
        "The Ollama path was subsequently given generic lexical context from the local zebrafish database before candidate ranking. "
        "This is not a macrophage-specific rule: the same retrieval step finds database descriptions matching the meaningful words in any "
        "biological question and supplies their exact local gene symbols to the model.",
        "",
        "A real post-fix run of the same prompt returned and validated all of these seeds:",
        "",
        "`csf1a`, `csf1b`, `marco`, `mrc1a`, `mrc1b`, `mif`, `mpeg1.1`, `mpeg1.2`, `slc11a2`, `csf1r`",
        "",
        "The resulting top five ESM neighbors were `kita`, `ly75`, `flt4`, `pdgfra`, and `kitb`. The aggregate figures below remain the "
        "original pre-fix 24-question baseline; a complete post-fix rerun would be required before updating those aggregate measurements.",
        "",
        "## Aggregate case table",
        "",
        "| # | Category | Prompt | Reference outcome | Validated seeds | Seconds |",
        "|---:|---|---|---|---:|---:|",
    ]
    for number, case in enumerate(cases, 1):
        prompt = str(case["query"]).replace("|", "\\|")
        lines.append(
            f"| {number} | {case['category']} | {prompt} | {outcome(case)} | "
            f"{case.get('validated_seed_count', 0)} | {float(case.get('elapsed_seconds') or 0):.2f} |"
        )

    lines.extend(["", "## Detailed results", ""])
    for number, case in enumerate(cases, 1):
        neighbors = case.get("neighbors") or []
        neighbor_text = ", ".join(
            f"`{row.get('name') or row.get('protein_id')}` ({float(row.get('similarity') or 0):.5f}; closest seed `{row.get('closest_seed')}`)"
            for row in neighbors
        ) or "None"
        lines.extend(
            [
                f"### {number}. {case['category']}: {case['query']}",
                "",
                f"- Outcome: **{outcome(case)}**",
                f"- Predefined reference examples: {joined(case.get('expected_examples') or [])}",
                f"- Model-proposed zebrafish genes: {joined(case.get('candidate_genes') or [])}",
                f"- Reference hits: {joined(case.get('expected_hits') or [])}",
                f"- Deterministically validated seed genes: {joined(case.get('validated_seed_genes') or [])}",
                f"- Top ESM neighbors: {neighbor_text}",
                f"- End-to-end latency: {float(case.get('elapsed_seconds') or 0):.2f} seconds",
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation for CV and README use",
            "",
            "The benchmark supports saying that a local 4B model was integrated and evaluated as a bounded biological-query interpreter. "
            "It also supports reporting the exact 24-question seed-resolution and reference-overlap results above. It does not support a claim "
            "that the model achieved 95.8% biological accuracy, matched or outperformed Gemini, or operated fully offline: UniProt and Ensembl "
            "were still used for public identifier resolution.",
            "",
            "A conservative CV bullet:",
            "",
            "> Integrated a local Qwen3 4B model through Ollama into a zebrafish protein-discovery pipeline; benchmarked 24 natural-language questions across eight biological categories, producing deterministically validated ESM search seeds for 23/24 cases with 25.25-second median end-to-end latency.",
            "",
            "A more technical project-description sentence:",
            "",
            "> Added a configurable local Ollama interpretation path while retaining deterministic UniProt/Ensembl identity validation and local ESM similarity search; in one 24-question benchmark, 23 prompts produced validated zebrafish seeds and 17 included at least one predefined canonical reference gene.",
            "",
            "## Handoff prompt for another GPT",
            "",
            "```text",
            "Use this benchmark report and the repository README to propose factual CV and public README updates.",
            "",
            "Requirements:",
            "- State that qwen3:4b-instruct ran locally through Ollama for biological candidate generation.",
            "- State that UniProt/Ensembl validation remained external and deterministic, while ESM embeddings/search stayed local.",
            "- You may report 23/24 prompts producing validated seeds, 17/24 overlapping the predefined reference examples, and 25.25-second median latency.",
            "- Describe the 17/24 measure as reference-example overlap, not accuracy.",
            "- Do not claim comparison, parity, or superiority versus Gemini; no paired Gemini run was performed.",
            "- Mention that exact zebrafish symbol generation was the main observed weakness.",
            "- Keep the CV bullet concise and make the README methodology reproducible.",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
