# Zebrafish ESM Search

A dual-mode zebrafish protein discovery dashboard that combines a private ESM embedding database with a grounded AI query layer.

> **AI researches and interprets biological language. Deterministic systems control species identity, orthology mapping, protein identity, embedding similarity, and ranking.**

The real embedding database is never committed to this repository and embedding vectors are never sent to the browser or to Gemini.

## Two search modes

### 1. Protein lookup — deterministic

Examples:

```text
gata1a
mpeg1.1
mpx
ENSDARP...
```

Flow:

```text
local identity resolution
→ private ESM embedding database
→ cosine similarity
→ ranked zebrafish proteins
```

No AI is required.

### 2. Biological question — zebrafish-first discovery

Example:

```text
macrophage proteins
```

Flow:

```text
natural-language question
→ Gemini + Google Search researches the question with Danio rerio as the fixed target species
→ direct zebrafish candidate genes are proposed from zebrafish-specific evidence
→ every proposed zebrafish gene must resolve exactly in the local zebrafish database
→ deeper Danio rerio UniProt retrieval adds additional validated seeds
→ if zebrafish evidence is sparse, strong human/mouse reference genes may be used
→ mammalian reference genes are mapped to Danio rerio orthologues with Ensembl REST
→ only mapped, locally validated zebrafish proteins can enter the final seed set
→ private ESM similarity expands from those validated zebrafish seeds
→ optional Gemini explanation of the ranked results
```

### Species policy

- Final search space and final ESM seeds are always **Danio rerio**.
- Zebrafish-specific evidence is searched and prioritized first.
- For cell-type queries, established zebrafish markers or enriched/specific genes are preferred over generic pathway members.
- Human/mouse evidence is allowed when zebrafish evidence is sparse, but mammalian genes cannot enter the ESM search directly.
- Mammalian genes must first be mapped to zebrafish orthologues through Ensembl and then resolve exactly into the local zebrafish database.
- Seed provenance is preserved so direct zebrafish evidence can be distinguished from mammalian-evidence/orthology inference.

## Grounding and retrieval

Biological discovery now uses three complementary routes:

1. **Gemini Google Search, zebrafish-first** — identifies highly relevant zebrafish candidates and biological concepts.
2. **UniProt Danio rerio retrieval** — searches a deeper result pool per concept and validates candidates against the local database.
3. **Ensembl orthology fallback** — maps human/mouse reference genes to zebrafish only when direct zebrafish evidence is insufficient.

AI-nominated genes are never accepted solely because the model named them. A direct zebrafish candidate must exist as an exact local zebrafish identity; cross-species candidates require deterministic orthology mapping first.

## Privacy / data-egress boundary

- real `.db`, `.sqlite`, `.npy`, `.npz`, and `.pt` artifacts stay private;
- the browser receives ranked protein metadata, never raw embeddings;
- `GEMINI_API_KEY` remains server-side;
- Gemini receives biological questions and compact protein metadata, never embedding vectors or database credentials;
- UniProt and Ensembl receive only ordinary public biological identifiers/search terms.

## Architecture

```text
Browser UI
   │
   ├── Protein lookup
   │      └── local resolver → NumPy/SQLite ESM similarity
   │
   └── Biological question
          ├── Gemini + Google Search (Danio rerio first)
          ├── direct zebrafish candidate validation
          ├── UniProt Danio rerio retrieval
          └── optional human/mouse reference evidence
                    ↓
               Ensembl orthology
                    ↓
        exact local zebrafish seed validation
                    ↓
             private ESM similarity
                    ↓
          optional Gemini explanation
```

Current stack:

- Python standard-library HTTP server
- SQLite
- NumPy
- Gemini REST API + Google Search grounding
- UniProt REST API
- Ensembl REST API
- vanilla HTML/CSS/JavaScript
- GitHub Actions + `unittest`

No Flask, FastAPI, Streamlit, or Gemini SDK is required for the local version.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If your source ESM embeddings are `.pt` files, install PyTorch separately:

```bash
pip install torch
```

## Configure AI

Copy the tracked configuration template into the ignored local `.env` file:

```bash
cp config.example .env
```

Then set:

```text
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-flash-lite
```

Exact protein lookup works without Gemini. Biological-question mode has deterministic fallbacks when AI or remote retrieval is unavailable.

## Build the private database

### From `.npy` embeddings + metadata CSV

```bash
python build_database.py \
  --embeddings /path/to/embeddings.npy \
  --metadata /path/to/metadata.csv \
  --id-column protein_id \
  --name-column gene_name \
  --description-column description \
  --out-db data/zebrafish_esm.db
```

### From `.npz`

```bash
python build_database.py \
  --embeddings /path/to/embeddings.npz \
  --metadata /path/to/metadata.csv \
  --out-db data/zebrafish_esm.db
```

### From an ESM `.pt` directory

```bash
python build_database.py \
  --pt-dir /path/to/esm_pt_files \
  --metadata /path/to/metadata.csv \
  --out-db data/zebrafish_esm.db
```

## Run locally

```bash
python app.py --db data/zebrafish_esm.db
```

Open:

```text
http://127.0.0.1:5000
```

The local server intentionally refuses non-loopback binding.

## Demo database

```bash
python scripts/create_demo_db.py
python app.py --db data/demo_zebrafish_esm.db
```

The demo database is synthetic and is not the private production embedding set.

## API routes

```text
GET /api/health
GET /api/status
GET /api/search?q=mpeg1.1&k=20
GET /api/discover?q=macrophage%20proteins&k=20
GET /api/suggest?q=mpeg
```

## Tests

```bash
python -m unittest discover -s tests -v
python -m py_compile app.py build_database.py
```

Tests cover deterministic protein lookup, no-key fallback, Google-Search-backed species-scoped planning, exact local validation of AI zebrafish candidates, Ensembl orthology mapping, and validated-seed ESM ranking.
