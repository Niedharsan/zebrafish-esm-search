# Zebrafish ESM Search

AI-assisted zebrafish protein discovery using Gemini, biological APIs, and whole-proteome ESM protein embeddings.

The zebrafish reference proteome used in this project was embedded with EvolutionaryScale's ESMC 6B protein language model through the Forge API. Those embeddings create a local vector search space where proteins can be compared by learned sequence representation, enabling semantic protein similarity search across the zebrafish proteome.

The current local dataset indexes 22,523 zebrafish proteins. Gemini adds a natural-language biological search layer, while UniProt and Ensembl REST APIs resolve biological findings to real *Danio rerio* protein identities before the ESM search is run. Raw embedding vectors remain local and are never sent to Gemini.

## Search modes

### Protein similarity

Enter a gene symbol, UniProt accession, or protein identifier such as:

```text
mpeg1.1
gata1a
mpx
Q7SXE0
```

The application resolves the protein against the local database and returns the most similar zebrafish proteins in ESM embedding space. This mode does not require Gemini.

### AI biological search

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

Gemini interprets the biological question using Google Search grounding. UniProt and Ensembl APIs are then used to resolve relevant findings to zebrafish proteins, which are validated against the local database before whole-proteome ESM similarity search.

```text
Biological question
        ↓
Gemini + Google Search
        ↓
UniProt + Ensembl APIs
        ↓
validated zebrafish proteins
        ↓
whole-proteome ESM embedding search
        ↓
related zebrafish proteins
```

## ESM embedding provenance

The embedding dataset was generated through EvolutionaryScale's Forge API using the ESMC 6B model. Protein sequences were sent to the model to obtain ESM embeddings, which were stored locally and later assembled into the SQLite/NumPy search database used by this application.

The Forge API is used for embedding generation, not for every dashboard query. Once the embedding database exists, protein similarity search runs locally.

## Integrations

- EvolutionaryScale Forge API / ESMC 6B — protein embedding generation
- Gemini API with Google Search grounding — natural-language biological search
- UniProt REST API — zebrafish protein and identifier resolution
- Ensembl REST API — orthology and identifier support
- ESM protein embeddings — whole-proteome similarity search
- SQLite and NumPy — local vector storage and ranking

## Data and scope

- The final search space is restricted to *Danio rerio*.
- The current project database contains 22,523 zebrafish proteins with 2,560-dimensional ESM vectors.
- AI-assisted candidates must resolve to a real protein in the local zebrafish database before they are used for ESM search.
- The SQLite database and embedding files remain local and are not committed to the repository.
- Gemini never receives raw embedding vectors or database credentials.
- API keys remain server-side.
- Direct protein similarity search remains available without Gemini.

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

## Run locally

Direct Python launch:

```bash
python app.py --db data/zebrafish_esm.db
```

Then open `http://127.0.0.1:5000`.

On macOS, you can instead run:

```bash
./start_dashboard.command
```

The macOS launcher uses `http://127.0.0.1:8000` and falls back to port `3000` if needed.

## Build the local search database

`build_database.py` assembles saved ESM embedding chunks and metadata into the local SQLite database:

```bash
python build_database.py \
  --pt-dir /path/to/embedding_chunks \
  --metadata /path/to/metadata.csv \
  --out-db data/zebrafish_esm.db
```

The original ESM embedding generation was performed separately through the EvolutionaryScale Forge API.

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
