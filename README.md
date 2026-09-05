# Zebrafish ESM Search

A dual-mode zebrafish protein discovery dashboard that combines a private ESM embedding database with an evidence-aware AI retrieval layer.

> **AI decides what evidence matters for the biological question. Deterministic systems control species identity, identifier resolution, orthology mapping, local protein identity, and ESM similarity.**

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

### 2. Biological question — evidence-aware zebrafish discovery

Examples:

```text
macrophage proteins
Wnt signaling proteins
proteins involved in autophagy
proteins expressed in the heart
proteins associated with fin regeneration
proteins that bind actin
```

Flow:

```text
natural-language question
→ Gemini classifies the biological question type and proposes retrieval concepts only
→ UniProtKB is queried directly for Danio rerio records and annotations
→ each UniProt hit is decomposed into evidence types instead of treating search position as biological rank
→ Ensembl resolves zebrafish symbols/identifiers and aliases
→ Gemini + Google Search gathers independent zebrafish evidence from ZFIN, literature, expression, pathway, phenotype, and other relevant sources
→ Gemini ranks candidates using evidence appropriate to the question type
→ every candidate must still resolve deterministically into the local zebrafish protein database
→ if zebrafish evidence is sparse, human/mouse reference genes may be mapped to zebrafish orthologues through Ensembl
→ private ESM similarity expands from the validated zebrafish seeds
→ optional Gemini explanation of the ranked results
```

## Evidence-aware ranking

A database search hit is not automatically treated as biological evidence.

For example, a UniProt search for `macrophage` can rank `mpeg1.1` highly because the protein name contains **Macrophage-expressed gene 1**. The application records that as:

```text
name_match
```

not as evidence of high expression or macrophage specificity.

Structured UniProt evidence can instead be labelled as:

```text
name_match
function_annotation
pathway_annotation
expression_annotation
phenotype_annotation
go_biological_process
go_molecular_function
go_cellular_component
```

The AI then interprets those evidence classes together with independent grounded zebrafish evidence.

The importance of each evidence type depends on the question:

- **cell type / tissue** — marker, expression, enrichment, and direct biological evidence matter most;
- **pathway / biological process** — curated pathway, GO biological process, functional and experimental evidence matter most;
- **molecular function** — GO molecular function and direct functional annotation matter most;
- **phenotype** — genetic, phenotype, ZFIN, and experimental evidence matter most;
- **protein family / other questions** — the planner chooses an appropriate evidence policy for the request.

A lexical protein-name match is always treated as weak supporting evidence unless another source independently supports the biological interpretation.

## Identifier resolution

Candidate ranking and protein identity are separate steps.

A model-generated name is never accepted simply because Gemini produced it. Candidate resolution can use:

```text
exact local zebrafish gene symbol
exact local UniProt accession
UniProt evidence-linked gene/accession
Ensembl zebrafish symbol or synonym resolution
```

Only after one of those routes resolves to an exact protein in the local 22,523-protein database can the candidate become an ESM seed.

This also handles cases where biological sources use a broader/older symbol while the local database contains the current zebrafish symbol.

## Species policy

- Final search space and final ESM seeds are always **Danio rerio**.
- Zebrafish-specific evidence is searched and prioritized first.
- Human/mouse evidence is allowed only when zebrafish evidence is sparse or useful as conserved reference biology.
- Mammalian genes cannot enter the ESM search directly.
- Mammalian genes must first be mapped to zebrafish orthologues through Ensembl and then resolve into the local zebrafish database.
- Seed provenance and evidence classes are retained.

## Grounding and retrieval

Biological discovery uses complementary evidence routes:

1. **UniProtKB structured zebrafish retrieval** — gene/protein names plus function, pathway, expression, phenotype, and GO annotations.
2. **Ensembl structured identifier resolution** — canonical zebrafish symbols, Ensembl IDs, synonyms/aliases, and cross-species orthology.
3. **Gemini Google Search grounding** — independent zebrafish evidence from ZFIN, primary literature, expression/single-cell resources, pathway resources, phenotype studies, and other authoritative sources.
4. **Local deterministic validation** — every final candidate must map to a protein in the private zebrafish ESM database.

UniProt search order is explicitly preserved only as retrieval metadata. It is never treated as the final biological ranking.

## Privacy / data-egress boundary

- real `.db`, `.sqlite`, `.npy`, `.npz`, and `.pt` artifacts stay private;
- the browser receives ranked protein metadata, never raw embeddings;
- `GEMINI_API_KEY` remains server-side;
- Gemini receives biological questions and compact biological evidence, never embedding vectors or database credentials;
- UniProt and Ensembl receive only ordinary public biological identifiers/search terms.

## Architecture

```text
Browser UI
   │
   ├── Protein lookup
   │      └── local resolver → NumPy/SQLite ESM similarity
   │
   └── Biological question
          │
          ├── Gemini retrieval planner
          │      └── question type + search concepts + evidence priorities
          │
          ├── UniProtKB Danio rerio structured retrieval
          │      └── evidence-type extraction
          │
          ├── Ensembl zebrafish identifier resolution
          │
          ├── Gemini + Google Search
          │      └── independent zebrafish evidence / ZFIN / literature
          │
          └── evidence-aware Gemini candidate ranking
                    ↓
          deterministic local zebrafish resolution
                    ↓
           optional mammalian orthology fallback
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

Tests cover deterministic protein lookup, no-key fallback, Google Search payload compatibility, evidence-type extraction, separation of lexical name matches from biological evidence, GO/process evidence, Ensembl alias resolution, UniProt accession resolution, question-type-aware retrieval planning, cross-species orthology, prevention of raw UniProt filler seeds, and validated-seed ESM ranking.
