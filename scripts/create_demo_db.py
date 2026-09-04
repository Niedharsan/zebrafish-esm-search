#!/usr/bin/env python3
"""Create a tiny fake DB so you can test the dashboard UI."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from build_database import create_database, make_records_from_metadata  # noqa: E402

rng = np.random.default_rng(7)
ids = [
    "ENSDARP00000000001", "ENSDARP00000000002", "ENSDARP00000000003",
    "ENSDARP00000000004", "ENSDARP00000000005", "ENSDARP00000000006",
]
names = ["mpx", "ela2", "lyz", "tp53", "mki67", "sox10"]
desc = [
    "myeloperoxidase-like neutrophil granule protein",
    "elastase 2 neutrophil granule serine protease",
    "lysozyme immune defense protein",
    "tumor protein p53 transcription factor",
    "marker of proliferation ki-67",
    "SRY-box transcription factor 10 neural crest marker",
]
metadata = pd.DataFrame({"protein_id": ids, "gene_name": names, "description": desc})

# Make immune proteins slightly clustered, unrelated proteins elsewhere.
base_immune = rng.normal(size=128).astype(np.float32)
base_tf = rng.normal(size=128).astype(np.float32)
vectors = []
for i in range(len(ids)):
    base = base_immune if i < 3 else base_tf
    vectors.append(base + rng.normal(scale=0.25, size=128).astype(np.float32))
matrix = np.vstack(vectors)
records = make_records_from_metadata(
    metadata,
    ids=ids,
    id_column="protein_id",
    name_column="gene_name",
    description_column="description",
    sequence_column=None,
)
create_database(str(ROOT / "data" / "demo_zebrafish_esm.db"), matrix, ids, records)
