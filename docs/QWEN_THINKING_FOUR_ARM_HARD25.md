# Qwen3 4B Thinking — Hard-25 Four-Arm Benchmark

This benchmark uses the same 25 difficult zebrafish questions and the same validation/retrieval/scoring code as the previous hard-25 experiment.

The new runner imports `benchmark_qwen3_4b_three_arm_25.py` and reuses its existing base prompt, Arm-2 second-pass prompt, scientific tools, validation and canonical scoring. This keeps the experiment focused on inference-time reasoning rather than adding a new rule system.

## Four experiments

### 1. `thinking_arm2`

Question → Qwen thinking → existing Arm-2 tools → Qwen thinking again → validation → ESM.

Tools are unchanged:
- local zebrafish metadata
- PubMed
- Europe PMC
- QuickGO
- UniProt
- Ensembl
- local zebrafish DB

### 2. `thinking_base`

Question → Qwen thinking → validation → ESM.

No retrieval before generation.

### 3. `thinking_x3_synthesis`

Question → three independent Qwen thinking runs → one final Qwen thinking synthesis over the three returned candidate sets → validation → ESM.

No scientific tools.

### 4. `thinking_x3_tools_synthesis`

Question → three independent Qwen thinking runs → union of their candidates → existing Arm-2 tools → one final Qwen thinking synthesis over the three results plus tool evidence → validation → ESM.

The three runs are stateless calls to the same Qwen model. This is repeated inference, not a multi-agent system.

## Model settings

All four use the same model-generation settings as the previous hard-25 run, with only `think: true` added:

- `qwen3:4b-instruct`
- `think: true`
- temperature `0.1`
- context `8192`
- prediction budget `1200`
- final response format JSON

## Dataset

Reuses:

`hard25_qwen3_4b_three_arm_cases.json`

IDs:

`1, 3, 8, 10, 11, 13, 14, 15, 16, 20, 21, 22, 24, 27, 31, 32, 34, 36, 38, 42, 44, 45, 46, 47, 50`

## Metrics

Primary:
- any canonical hit
- mean canonical recall
- Hit@1
- Hit@3
- Hit@5
- MRR

Secondary:
- valid seed rate/count
- unresolved identifier rate
- latency
- retrieval errors

Canonical examples remain diagnostic examples, not exhaustive biological truth.

## Checkpointing

Checkpoint occurs only after all four experiments complete for a question.

If interrupted with `Ctrl+C`, rerun the same command. Completed questions are preserved and the incomplete question restarts.

## Run

```bash
cd "/Users/niedharsan/Downloads/Work/AI DATABASE/pg_zfish_project/zebrafish_esm_dashboard" && \
.venv/bin/python scripts/benchmark_qwen3_4b_thinking_four_arm_25.py \
  --cases hard25_qwen3_4b_three_arm_cases.json \
  --db data/zebrafish_esm.db \
  --model qwen3:4b-instruct \
  --k 5 \
  --out qwen3_4b_thinking_four_arm_25_results.json \
  --checkpoint qwen3_4b_thinking_four_arm_25_checkpoint.json
```
