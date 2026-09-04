import argparse
import json
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def extract_gene_symbol(description):
    if not isinstance(description, str):
        return ""
    match = re.search(r"\bGN=([A-Za-z0-9_.:-]+)", description)
    return match.group(1) if match else ""


def load_chunk_tensor(path):
    obj = torch.load(path, map_location="cpu")

    if not torch.is_tensor(obj):
        raise ValueError(f"{path.name} is not a plain torch tensor. Got: {type(obj)}")

    tensor = obj.detach().float().cpu()

    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)

    if tensor.ndim > 2:
        tensor = tensor.reshape(tensor.shape[0], -1)

    return tensor.numpy().astype("float32")


def normalize_matrix(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return matrix / norms


def build_database(pt_dir, metadata_csv, out_db):
    pt_dir = Path(pt_dir)
    metadata_csv = Path(metadata_csv)
    out_db = Path(out_db)
    out_db.parent.mkdir(parents=True, exist_ok=True)

    print("Loading metadata...")
    meta = pd.read_csv(metadata_csv)

    if {"chunk_id", "index_in_chunk"}.issubset(meta.columns):
        meta = meta.sort_values(["chunk_id", "index_in_chunk"]).reset_index(drop=True)
    elif "global_index" in meta.columns:
        meta = meta.sort_values("global_index").reset_index(drop=True)

    print(f"Metadata proteins: {len(meta)}")
    print(f"Metadata columns: {list(meta.columns)}")

    chunk_files = sorted(pt_dir.glob("chunk_*.pt"))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk_*.pt files found in {pt_dir}")

    print(f"Loading {len(chunk_files)} chunk files...")

    chunks = []
    for i, chunk_file in enumerate(chunk_files, start=1):
        arr = load_chunk_tensor(chunk_file)
        chunks.append(arr)
        print(f"[{i}/{len(chunk_files)}] {chunk_file.name}: {arr.shape}")

    matrix = np.vstack(chunks).astype("float32")
    print(f"Raw embedding rows from chunks: {matrix.shape[0]}")
    print(f"Embedding dimensions: {matrix.shape[1]}")

    # The chunk files are already loaded in chunk order.
    # Metadata is sorted by chunk_id + index_in_chunk above, so rows should align directly.
    matrix = matrix[:len(meta)]

    matrix = normalize_matrix(matrix)

    print(f"Final proteins to store: {len(meta)}")

    if out_db.exists():
        out_db.unlink()

    conn = sqlite3.connect(out_db)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE proteins (
            row_id INTEGER PRIMARY KEY,
            protein_id TEXT,
            gene_name TEXT,
            description TEXT,
            search_text TEXT,
            metadata_json TEXT,
            embedding BLOB
        )
    """)

    cur.execute("""
        CREATE VIRTUAL TABLE protein_search USING fts5(
            protein_id,
            gene_name,
            description,
            search_text,
            content='proteins',
            content_rowid='row_id'
        )
    """)

    print("Writing SQLite database...")

    for i, row in meta.iterrows():
        protein_id = str(row.get("record_id", f"protein_{i}"))
        description = str(row.get("description", ""))
        gene_name = extract_gene_symbol(description)

        metadata_dict = {
            str(k): None if pd.isna(v) else str(v)
            for k, v in row.to_dict().items()
        }
        metadata_dict["gene_name_extracted"] = gene_name

        search_text = " ".join([
            protein_id,
            gene_name,
            description,
            json.dumps(metadata_dict, ensure_ascii=False)
        ])

        cur.execute(
            """
            INSERT INTO proteins
            (row_id, protein_id, gene_name, description, search_text, metadata_json, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                i + 1,
                protein_id,
                gene_name,
                description,
                search_text,
                json.dumps(metadata_dict, ensure_ascii=False),
                matrix[i].astype("float32").tobytes(),
            )
        )

        cur.execute(
            """
            INSERT INTO protein_search
            (rowid, protein_id, gene_name, description, search_text)
            VALUES (?, ?, ?, ?, ?)
            """,
            (i + 1, protein_id, gene_name, description, search_text)
        )

    cur.execute("CREATE INDEX idx_protein_id ON proteins(protein_id)")
    cur.execute("CREATE INDEX idx_gene_name ON proteins(gene_name)")

    conn.commit()
    conn.close()

    print("")
    print(f"Done: {out_db}")
    print(f"Proteins stored: {len(meta)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt-dir", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--out-db", required=True)
    args = parser.parse_args()

    build_database(args.pt_dir, args.metadata, args.out_db)


if __name__ == "__main__":
    main()
