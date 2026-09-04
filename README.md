# Zebrafish ESM Similarity Dashboard

A small local black-dashboard app for searching a zebrafish ESM embedding database. It uses Python’s built-in local web server, so there is no Streamlit or Flask dependency.

It is designed for this workflow:

1. Keep your original ESM files in their current folder.
2. Use `build_database.py` to create a separate SQLite `.db` file.
3. Run `app.py` to open a local browser dashboard.
4. Search a protein/gene name and get the most similar zebrafish proteins by cosine similarity.

The dashboard runs locally only, at `http://127.0.0.1:5000`.

---

## Folder layout

```text
zebrafish_esm_dashboard/
├── app.py
├── build_database.py
├── requirements.txt
├── README.md
├── data/
│   └── zebrafish_esm.db      # created by you
├── templates/
│   └── index.html
├── static/
│   ├── styles.css
│   └── app.js
└── scripts/
    └── create_demo_db.py
```

Your original embedding files stay outside this folder unless you choose otherwise.

---

## Install

From Terminal on Mac:

```bash
cd /path/to/zebrafish_esm_dashboard
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This installs NumPy and pandas for database building/search. The dashboard server itself uses Python’s standard library.

If your embeddings are `.pt` files made by the ESM extraction script, install PyTorch too:

```bash
pip install torch
```

---

## Option A — Build DB from one `.npy` matrix + metadata CSV

Use this if you have one embeddings matrix where rows match your metadata rows.

```bash
python build_database.py \
  --embeddings /path/to/embeddings.npy \
  --metadata /path/to/metadata.csv \
  --id-column protein_id \
  --name-column gene_name \
  --description-column description \
  --out-db data/zebrafish_esm.db
```

If your metadata column names are different, change `protein_id`, `gene_name`, and `description`.

---

## Option B — Build DB from `.npz`

Use this if you have a compressed NumPy file containing embeddings and IDs.

```bash
python build_database.py \
  --embeddings /path/to/embeddings.npz \
  --metadata /path/to/metadata.csv \
  --out-db data/zebrafish_esm.db
```

The script looks for common keys like `embeddings`, `vectors`, `X`, `ids`, `protein_ids`, and `labels`.

---

## Option C — Build DB from an ESM `.pt` directory

Use this if your ESM output folder contains one `.pt` file per protein.

```bash
python build_database.py \
  --pt-dir /path/to/esm_pt_files \
  --metadata /path/to/metadata.csv \
  --out-db data/zebrafish_esm.db
```

The script extracts `mean_representations` when available, choosing the highest ESM layer automatically.

---

## Run the dashboard

```bash
python app.py --db data/zebrafish_esm.db
```

Open:

```text
http://127.0.0.1:5000
```

Search for a gene/protein name, for example:

```text
mpx
```

The app resolves the closest matching protein in your database and returns the most similar proteins by cosine similarity.

---

## Test with a fake demo database

This is only to confirm the dashboard works before connecting your real zebrafish data.

```bash
python scripts/create_demo_db.py
python app.py --db data/demo_zebrafish_esm.db
```

Then open `http://127.0.0.1:5000` and search for:

```text
mpx
```

---

## Notes

- Cosine similarity uses normalized ESM vectors.
- For 26k zebrafish proteins, the app loads vectors into memory once at startup. This should be fine on a normal laptop.
- SQLite stores the metadata and float32 normalized vector blobs.
- Original ESM files are not modified.
