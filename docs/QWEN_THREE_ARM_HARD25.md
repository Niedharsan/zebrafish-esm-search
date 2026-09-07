# Qwen3 4B three-arm hard-25 benchmark

This experiment compares three frozen architectures on 25 difficult cases selected from the first 50 completed questions of the existing paired benchmark.

## Arms

### A — Base Qwen

`question → Qwen3 4B internal knowledge → deterministic UniProt/Ensembl/local validation → ESM`

No retrieval is available before Qwen generates candidates.

### B — Qwen first, then scientific tools

`question → Qwen3 4B internal reasoning → tools → second Qwen review/rerank → deterministic final validation → ESM`

The first Qwen pass is retrieval-free. Only after that pass does the runner collect local zebrafish lexical metadata, PubMed, Europe PMC, QuickGO, UniProt identifier checks, Ensembl orthology checks, and local database validation. The second Qwen pass receives a compact evidence package and decides what to retain, repair, reject, rerank, or add.

### C — Local zebrafish context first

`question → local zebrafish lexical metadata top 30 → Qwen3 4B → deterministic UniProt/Ensembl/local validation → ESM`

This reproduces the core `3aa9805` approach. PubMed, Europe PMC, and QuickGO are not used before candidate generation.

## Hard-25 selection

Case IDs:

`1, 3, 8, 10, 11, 13, 14, 15, 16, 20, 21, 22, 24, 27, 31, 32, 34, 36, 38, 42, 44, 45, 46, 47, 50`

Selection:
- 18 where base Qwen had canonical overlap and the previous full pre-retrieval augmented arm did not
- 6 where neither arm had canonical overlap
- case 1 macrophage as a known augmentation-rescue control

## Fixed model settings

- model: `qwen3:4b-instruct`
- temperature: `0.1`
- context: `8192`
- max output tokens: `1200`
- thinking: `false`
- output format: `json`
- max zebrafish candidates: `10`
- max reference candidates: `6`

## Primary evaluation

Canonical examples are diagnostic, not exhaustive biological truth.

Primary metrics:
- any canonical hit
- canonical recall
- rank of first canonical hit
- Hit@1
- Hit@3
- Hit@5
- reciprocal rank of the first canonical hit

Secondary metrics:
- validated seed count
- unresolved proposed-gene rate
- pipeline seed success
- latency
- retrieval errors
- ESM neighbours
- provenance and raw model outputs

Pairwise “better” decisions prioritize canonical recall/ranking before identifier quality.

## Run

From the repository root:

```bash
.venv/bin/python scripts/benchmark_qwen3_4b_three_arm_25.py \
  --cases hard25_qwen3_4b_three_arm_cases.json \
  --db data/zebrafish_esm.db \
  --model qwen3:4b-instruct \
  --k 5 \
  --out qwen3_4b_three_arm_25_results.json \
  --checkpoint qwen3_4b_three_arm_25_checkpoint.json
```

The checkpoint is written only after all three arms of a case finish. If interrupted with `Ctrl+C`, the incomplete case is rerun on resume; already completed cases are preserved.

## Important

Do not modify prompts, retrieval code, model settings, dataset, or validation behavior after the benchmark begins. The runner records hashes of the dataset, app, runner, and local database and refuses to resume if the frozen configuration changes.
