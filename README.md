# Zebrafish ESM Search

A dual-mode zebrafish protein discovery dashboard that combines a private ESM embedding database with an optional, grounded AI query layer.

The important architectural rule is simple:

> **AI interprets biological language and explains results. Deterministic systems control protein identity, external retrieval, embedding similarity, and ranking.**

The real embedding database is never committed to this repository and embedding vectors are never sent to the browser or to the AI provider.

## Two search modes

### 1. Protein lookup — deterministic

Input examples:

```text
gata1a
mpx
ENSDARP...
```

Flow:

```text
exact/fuzzy local identity resolution
→ private ESM embedding database
→ cosine similarity
→ ranked zebrafish proteins
```

No AI is required for this path.

### 2. Biological question — AI-assisted, evidence-grounded

Input example:

```text
proteins involved in erythropoiesis
```

Flow:

```text
natural-language question
→ Gemini converts the question into biological retrieval concepts
→ UniProt retrieves Danio rerio proteins for those concepts
→ retrieved proteins must resolve exactly into the local ESM database
→ ESM ranks proteins by similarity to the validated seed set
→ optional Gemini explanation of the already-ranked results
```

Gemini is deliberately **not asked to invent seed genes**. If Gemini is unavailable, the system falls back to deterministic retrieval using the original question. If UniProt is unavailable, a local annotation fallback can still generate seeds.

## Privacy / data-egress boundary

The application is designed so that:

- real `.db`, `.sqlite`, `.npy`, `.npz`, and `.pt` embedding artifacts stay private;
- the browser receives ranked protein metadata, never raw embeddings;
- `GEMINI_API_KEY` remains server-side;
- Gemini receives the biological question and compact result/seed metadata only;
- Gemini never receives embedding vectors or database credentials.

## Architecture

```text
Browser UI
   │
   ├── Protein lookup
   │      └── local resolver → NumPy/SQLite ESM similarity
   │
   └── Biological question
          └── Gemini query planner
                 ↓
             UniProt REST
                 ↓
        exact local seed validation
                 ↓
         private ESM similarity
                 ↓
       optional Gemini explanation
```

Current stack:

- Python standard-library HTTP server
- SQLite
- NumPy
- optional Gemini REST API
- UniProt REST API
- vanilla HTML/CSS/JavaScript
- GitHub Actions + `unittest`

No Flask, FastAPI, Streamlit, or Gemini SDK is required for the current local version.

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

## Configure optional AI

Copy the tracked configuration template into the ignored local `.env` file:

```bash
cp config.example .env
```

Then edit `.env` and set:

```text
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-flash-lite
```

The app also accepts these as normal environment variables. Exact protein lookup works without a Gemini key.

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

The local server intentionally refuses non-loopback binding. Public deployment should put the application behind a proper hosted service rather than exposing the development server directly.

## Demo database

```bash
python scripts/create_demo_db.py
python app.py --db data/demo_zebrafish_esm.db
```

Then try a deterministic protein lookup such as:

```text
mpx
```

The demo database is synthetic and is not the private production embedding set.

## API routes

```text
GET /api/health
GET /api/status
GET /api/search?q=gata1a&k=20
GET /api/discover?q=proteins%20involved%20in%20erythropoiesis&k=20
GET /api/suggest?q=gat
```

## Tests

```bash
python -m unittest discover -s tests -v
python -m py_compile app.py build_database.py
```

The tests explicitly check that deterministic protein lookup remains functional without an AI key and that biological discovery ranks from validated seed indices rather than AI-generated gene guesses.

## Next deployment step

For a public portfolio demo, the intended topology is:

```text
public GitHub source
→ hosted Python service
→ private server-side ESM database / object storage
→ browser receives ranked results only
```

The production deployment should add request limits, caching for UniProt/AI calls, deployment-specific secret management, and a persistent private storage layer for the real embeddings.
