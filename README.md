# Zebrafish ESM Search

A local zebrafish protein-discovery dashboard that combines a private ESM embedding database with a small AI research layer.

> **Gemini does the biological reasoning. Deterministic code only controls protein identity, zebrafish validation, orthology fallback, and ESM similarity.**

The real embedding database is not committed to this repository and raw embeddings are never sent to Gemini or the browser.

## Two modes

### 1. Protein lookup

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
→ private ESM database
→ cosine similarity
→ ranked zebrafish proteins
```

This path is deterministic and does not require AI.

### 2. Biological question

Examples:

```text
macrophage proteins
Wnt signaling proteins
proteins involved in autophagy
proteins expressed in the heart
```

Flow:

```text
user question
→ simple Danio rerio UniProt retrieval
→ Gemini + Google Search researches the biology
→ Gemini ranks the strongest zebrafish candidates
→ deterministic local zebrafish validation
→ optional human/mouse → zebrafish Ensembl orthology fallback
→ private ESM similarity expansion
```

There is deliberately **no hand-built question-type classifier, biological evidence taxonomy, or weighted relevance engine**.

UniProt search order is only retrieval context. Gemini is explicitly told that a hit may rank highly just because the query word occurs in the protein name, and that it must use independent zebrafish evidence to decide biological relevance.

For example, a UniProt search for `macrophage` may surface `mpeg1.1` because its protein name contains “Macrophage-expressed gene 1”. That lexical match is useful for retrieval, but Gemini still has to check the zebrafish biology before ranking it.

## Species policy

- Final ESM seeds are always **Danio rerio** proteins.
- Gemini searches zebrafish evidence first.
- Human/mouse evidence is only a fallback when zebrafish evidence is sparse.
- Mammalian genes cannot enter the ESM search directly; they must first map to zebrafish orthologues through Ensembl.
- Every final seed must resolve into the local zebrafish protein database.

## Privacy

- `.db`, `.sqlite`, `.npy`, `.npz`, and `.pt` files stay local/private.
- `GEMINI_API_KEY` stays server-side.
- Gemini receives the biological question and compact public UniProt metadata, not embedding vectors or database credentials.
- The browser receives protein metadata and similarity results, not raw embeddings.

## Stack

- Python standard-library HTTP server
- SQLite
- NumPy
- Gemini REST API + Google Search grounding
- UniProt REST API
- Ensembl REST API
- vanilla HTML/CSS/JavaScript
- GitHub Actions + `unittest`

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configure AI

```bash
cp config.example .env
```

Set:

```text
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-flash-lite
```

Exact protein lookup works without Gemini.

## Run locally

```bash
python app.py --db data/zebrafish_esm.db
```

or on macOS:

```bash
./start_dashboard.command
```

## Build the private database

Example:

```bash
python build_database.py \
  --embeddings /path/to/embeddings.npy \
  --metadata /path/to/metadata.csv \
  --id-column protein_id \
  --name-column gene_name \
  --description-column description \
  --out-db data/zebrafish_esm.db
```

## Live integration test

This uses your local `.env` and private database and makes real Gemini + Google Search + UniProt calls:

```bash
.venv/bin/python scripts/live_integration_test.py
```

The default checks include:

```text
macrophage proteins
Wnt signaling proteins
```

For the macrophage case, the script explicitly checks that `mpeg1.1` reaches the final validated seed set.

The live test is intentionally not run in public CI because it needs your private local API key and database.

## Normal tests

```bash
python -m unittest discover -s tests -v
python -m py_compile app.py build_database.py
```

The regression suite checks deterministic protein lookup, Gemini grounding payloads, simple zebrafish UniProt retrieval, AI-first ranking, UniProt synonym/accession resolution, mammalian orthology fallback, and ESM ranking.
