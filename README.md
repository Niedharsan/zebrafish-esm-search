# Zebrafish ESM Search

AI-assisted zebrafish protein discovery combining whole-proteome ESM embeddings, a local zebrafish protein database, cloud or local language-model interpretation, authoritative biological retrieval, and deterministic protein identity validation.

The dashboard supports both direct protein lookup and natural-language biological discovery. Its final search space is always *Danio rerio*.

## What is indexed

The current private search database contains:

- 22,523 zebrafish proteins
- 2,560-dimensional protein embeddings
- SQLite metadata
- NumPy cosine-similarity search

The embedding dataset was generated with the pretrained EvolutionaryScale ESMC 6B model through the Forge API. Forge is used to generate the reusable embedding dataset; normal dashboard searches run against the local database and do not require a new ESMC inference call.

## Search modes

### Protein lookup

Enter a gene symbol, UniProt accession, or protein identifier such as:

```text
mpeg1.1
gata1a
mpx
Q7SXE0
```

The application resolves the protein deterministically against the local database and returns the closest zebrafish proteins by ESM embedding similarity. This mode does not require an LLM.

### Biological discovery

Ask a biological question in natural language, for example:

```text
Which proteins mark zebrafish macrophages?
Which zebrafish proteins are involved in Wnt signaling?
Find proteins involved in autophagy.
Which proteins regulate erythropoiesis in zebrafish?
Find proteins associated with fin regeneration.
Which proteins are related to cilia?
Which proteins are involved in pigmentation?
```

The language-model layer interprets the biological question and proposes candidate proteins. Those candidates must then resolve to real zebrafish proteins before they can seed ESM similarity search.

```text
Biological question
        ↓
LLM biological interpretation
        ↓
scientific retrieval + biological APIs
        ↓
validated zebrafish seed proteins
        ↓
local ESM embedding similarity
        ↓
ranked related proteins
```

## Two AI paths

### Gemini + Google Search grounding

The Gemini path uses zebrafish-specific Google Search grounding to identify biologically relevant candidates, followed by UniProt/Ensembl resolution and exact local validation.

### Local Qwen3 4B + scientific tools

The local path runs `qwen3:4b-instruct` through Ollama. Rather than trusting the small model's internal knowledge alone, the application augments it with:

- lexical context from the local zebrafish protein metadata
- PubMed retrieval
- Europe PMC retrieval
- QuickGO / Gene Ontology retrieval
- UniProt REST validation
- Ensembl orthology/identifier support
- local ESM similarity search

The Qwen model itself remains local. Public biological services receive only question-derived search terms; raw embedding vectors and the local database are not sent to them.

This architecture is designed to test a practical question: **how far can a small local open-weight model be improved for a narrow scientific workflow by giving it domain-specific retrieval, authoritative APIs, deterministic validation, and a specialist protein-embedding search tool?** The model weights are not fine-tuned; the improvement comes from system-level domain augmentation and validation.

## Reliability design

The LLM is not the authority for protein identity.

A proposed candidate can enter the ESM search only if it resolves to an actual zebrafish protein through targeted biological resolution and the local database. This separates responsibilities:

- **LLM:** biological interpretation and candidate ranking
- **PubMed / Europe PMC / QuickGO:** external scientific evidence
- **UniProt / Ensembl / local DB:** identity resolution and validation
- **ESM:** protein-representation similarity and candidate expansion

This is important because a small general-purpose model can understand the biological concept while still producing inaccurate zebrafish nomenclature. The surrounding scientific tools are intended to correct that weakness rather than simply trusting fluent model output.

## Integrations

- EvolutionaryScale Forge API / ESMC 6B
- Gemini API with Google Search grounding
- Ollama + Qwen3 4B local inference
- PubMed E-utilities
- Europe PMC
- QuickGO / Gene Ontology
- UniProt REST API
- Ensembl REST API
- SQLite and NumPy

## Privacy and species boundaries

- Final seed proteins and similarity results are restricted to *Danio rerio*.
- Every AI-assisted candidate must resolve to an exact protein in the local zebrafish database before ESM search.
- The SQLite database and embedding files remain local and are not committed to the repository.
- AI providers and public retrieval services never receive raw embedding vectors or database credentials.
- In Ollama mode, the language model and embeddings remain local.
- Exact protein lookup remains available without an LLM.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy the example configuration:

```bash
cp config.example .env
```

For Gemini:

```text
AI_PROVIDER=gemini
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-flash-lite
```

For local Qwen through Ollama:

```text
AI_PROVIDER=ollama
OLLAMA_MODEL=qwen3:4b-instruct
OLLAMA_URL=http://127.0.0.1:11434
```

Optional `NCBI_EMAIL` and `NCBI_API_KEY` settings are supported for PubMed E-utilities.

## Run locally

```bash
python app.py --db data/zebrafish_esm.db
```

On macOS:

```bash
./start_dashboard.command
```

Then open `http://127.0.0.1:5000`.

## Build the private database

```bash
python build_database.py \
  --embeddings /path/to/embeddings.npy \
  --metadata /path/to/metadata.csv \
  --id-column protein_id \
  --name-column gene_name \
  --description-column description \
  --out-db data/zebrafish_esm.db
```

## Evaluation

### Initial 24-question local-model benchmark

In one pre-augmentation local run of `qwen3:4b-instruct`, 23 of 24 questions produced at least one deterministically validated zebrafish ESM seed, and 17 of 24 included at least one predefined canonical reference example. Reference overlap is a diagnostic measure rather than a biological accuracy score.

The broad macrophage-marker case exposed a useful failure mode: the small model understood macrophage biology but missed the canonical zebrafish `mpeg1.1` symbol. Adding local zebrafish metadata context and authoritative biological retrieval subsequently allowed the same local model to identify and validate `mpeg1.1` without a macrophage-specific hard-coded rule.

See [`docs/local-qwen3-4b-benchmark.md`](docs/local-qwen3-4b-benchmark.md) for the original benchmark details and post-benchmark retest.

### Paired 100-question benchmark

A paired benchmark is being run across 100 specific biological questions. Each question is evaluated twice with the same `qwen3:4b-instruct` model:

1. **Qwen alone** — question + structured seed-selection prompt; no retrieval context before generation.
2. **Augmented Qwen** — local zebrafish metadata + PubMed + Europe PMC + QuickGO before Qwen candidate ranking, followed by deterministic validation and ESM search.

The benchmark records raw model output, proposed genes, unresolved identifiers, validated seeds, reference overlap, ESM neighbours, retrieval provenance, latency, and paired rescue/improvement metrics. The test is designed to quantify whether scientific retrieval and deterministic tools improve a small local model for zebrafish protein discovery without fine-tuning the model weights.

No final 100-query result is claimed here until the paired run is complete.

## Tests

Run the local regression suite:

```bash
python -m unittest discover -s tests -v
python -m py_compile app.py build_database.py scripts/live_integration_test.py
```

Live integration tests are intentionally excluded from public CI when they require local data, local Ollama, or external biological services.

## Limitations

- ESM similarity is a discovery signal, not proof of shared function, pathway membership, interaction, or homology.
- A validated identifier does not by itself prove biological relevance.
- LLM ranking can still be incomplete or wrong even when grounded.
- Retrieval quality depends on the query and source coverage.
- The current vector search uses a simple local NumPy implementation rather than a dedicated ANN index.
- Mean-pooled ESM embeddings are a practical baseline and could be compared with alternative representations.

## Project framing

This project is primarily an example of scientific AI system design rather than model training: combining a protein foundation model, local vector search, local/open-weight and cloud LLMs, authoritative biological APIs, deterministic validation, benchmarking, and iterative scientific failure analysis into one usable workflow.
