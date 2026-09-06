# Zebrafish ESM Search

AI-assisted zebrafish protein discovery combining ESM protein embeddings, a local zebrafish protein database, Gemini biological-language interpretation, public bioinformatics APIs, and deterministic protein identity validation.

The dashboard supports both direct protein lookup and natural-language biological discovery. Its final search space is always *Danio rerio*.

## Search modes

### Protein lookup

Enter a gene symbol, UniProt accession, or protein identifier such as:

```text
mpeg1.1
gata1a
mpx
Q7SXE0
```

The application resolves the protein deterministically against the local database and returns the closest zebrafish proteins by ESM embedding similarity. This mode does not require AI.

### Biological discovery

Ask a biological question in natural language, for example:

```text
Which proteins are associated with zebrafish macrophages?
Which zebrafish proteins are involved in Wnt signaling?
Find proteins involved in autophagy.
Which proteins are expressed in the zebrafish heart?
Find proteins associated with fin regeneration.
Which proteins are related to cilia?
Which proteins are involved in pigmentation?
Which proteins bind actin?
```

The AI layer interprets the question, uses public biological resources to identify relevant zebrafish proteins, validates their identities against the local protein database, and expands the validated set through ESM similarity.

```text
Biological question
        ↓
AI-assisted zebrafish discovery
        ↓
public biological databases/APIs
        ↓
validated zebrafish proteins
        ↓
ESM embedding similarity
        ↓
ranked related proteins
```

## Integrations

- Gemini API with Google Search grounding
- Optional local Ollama interpretation
- UniProt REST API
- Ensembl REST API
- ESM protein embeddings
- SQLite and NumPy

## Privacy and species boundaries

- Final seed proteins and similarity results are restricted to *Danio rerio*.
- Every AI-assisted candidate must resolve to an exact protein in the local zebrafish database before ESM search.
- The SQLite database and embedding files remain local and are not committed to the repository.
- Gemini never receives raw embedding vectors or database credentials.
- The API key remains server-side.
- Exact protein lookup remains available without Gemini.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy the example configuration and add a Gemini API key:

```bash
cp config.example .env
```

```text
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-flash-lite
```

To use a local Ollama model instead of Gemini, configure:

```text
AI_PROVIDER=ollama
OLLAMA_MODEL=qwen3:4b-instruct
OLLAMA_URL=http://127.0.0.1:11434
```

The local model proposes candidate genes from its internal knowledge and is explicitly reported as not web-grounded. UniProt/Ensembl identity resolution and the local ESM similarity search remain unchanged.

## Run locally

```bash
python app.py --db data/zebrafish_esm.db
```

On macOS, you can also run:

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

## Tests

Run the local regression suite:

```bash
python -m unittest discover -s tests -v
python -m py_compile app.py build_database.py scripts/live_integration_test.py
```

An optional live integration test uses the local API key and private database to exercise Gemini, Google Search, UniProt, local validation, and ESM:

```bash
.venv/bin/python scripts/live_integration_test.py
```

The live test covers macrophage discovery and a non-macrophage Wnt signaling query. It is intentionally excluded from public CI because it requires local credentials and data.

Run the 24-question local-model benchmark with:

```bash
.venv/bin/python scripts/benchmark_local_model.py --json-out local_4b_benchmark.json
```
