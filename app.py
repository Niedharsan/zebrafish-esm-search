#!/usr/bin/env python3
"""Local black dashboard for zebrafish ESM protein similarity search.

No Streamlit or Flask required. This uses Python's standard-library HTTP server.
"""

from __future__ import annotations

import argparse
import difflib
import html
import json
import mimetypes
import sqlite3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np


ROOT = Path(__file__).resolve().parent
DB_PATH = "data/zebrafish_esm.db"
HOST = "127.0.0.1"
PORT = 5000
PROTEINS: List[Dict[str, Any]] = []
ID_TO_INDEX: Dict[str, int] = {}
VECTORS: Optional[np.ndarray] = None
SEARCH_TEXTS: List[str] = []


def connect_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def load_database(db_path: str) -> None:
    global DB_PATH, PROTEINS, ID_TO_INDEX, VECTORS, SEARCH_TEXTS
    DB_PATH = db_path

    if not Path(DB_PATH).exists():
        raise FileNotFoundError(
            f"Database not found: {DB_PATH}. Build it first with build_database.py."
        )

    conn = connect_db()
    rows = conn.execute(
        """
        SELECT row_id, protein_id, gene_name, description, metadata_json, embedding
        FROM proteins
        ORDER BY row_id
        """
    ).fetchall()
    conn.close()

    if not rows:
        raise ValueError("Database contains no proteins.")

    proteins = []
    vectors = []

    for row in rows:
        blob = row["embedding"]
        vec = np.frombuffer(blob, dtype=np.float32).copy()

        proteins.append(
            {
                "protein_id": row["protein_id"] or "",
                "name": row["gene_name"] or "",
                "description": row["description"] or "",
                "sequence": "",
                "extra_json": row["metadata_json"] or "{}",
            }
        )
        vectors.append(vec)

    PROTEINS = proteins
    VECTORS = np.vstack(vectors).astype(np.float32, copy=False)
    ID_TO_INDEX = {p["protein_id"]: i for i, p in enumerate(PROTEINS)}
    SEARCH_TEXTS = [
        " ".join([p["protein_id"], p["name"], p["description"]]).lower()
        for p in PROTEINS
    ]

    print(f"Loaded {len(PROTEINS):,} proteins from {DB_PATH}. Vectors: {VECTORS.shape}")


def protein_public(p: Dict[str, Any]) -> Dict[str, Any]:
    sequence = p.get("sequence") or ""
    return {
        "protein_id": p.get("protein_id", ""),
        "name": p.get("name", ""),
        "description": p.get("description", ""),
        "sequence_length": len(sequence) if sequence else None,
    }


def resolve_query(query: str) -> Optional[Tuple[int, str, float]]:
    """Resolve free text to the best protein row index.

    Returns: (index, method, match_score)
    """
    q = query.strip().lower()
    if not q:
        return None

    # Exact ID or exact name.
    for i, p in enumerate(PROTEINS):
        if p["protein_id"].lower() == q:
            return i, "exact protein ID", 1.0
        if p["name"] and p["name"].lower() == q:
            return i, "exact name", 1.0

    # Contains match over ID/name/description.
    contains = []
    for i, text in enumerate(SEARCH_TEXTS):
        if q in text:
            p = PROTEINS[i]
            id_or_name = q in p["protein_id"].lower() or (p["name"] and q in p["name"].lower())
            boost = 0.45 if id_or_name else 0.0
            score = min(0.96, len(q) / max(len(text), 1) + boost)
            contains.append((score, i))
    if contains:
        contains.sort(reverse=True)
        return contains[0][1], "contains match", float(contains[0][0])

    # Fuzzy match against protein IDs and names using only the Python standard library.
    choices: List[str] = []
    choice_to_index: Dict[str, int] = {}
    for i, p in enumerate(PROTEINS):
        for field in [p["protein_id"], p["name"]]:
            if field:
                key = field.lower()
                choices.append(key)
                choice_to_index[key] = i
    matches = difflib.get_close_matches(q, choices, n=1, cutoff=0.55)
    if matches:
        m = matches[0]
        ratio = difflib.SequenceMatcher(a=q, b=m).ratio()
        return choice_to_index[m], "fuzzy match", float(ratio)

    return None


def nearest_neighbors(index: int, k: int) -> List[Dict[str, Any]]:
    if VECTORS is None:
        raise RuntimeError("Vectors are not loaded.")
    query_vec = VECTORS[index]
    sims = VECTORS @ query_vec
    sims[index] = -np.inf
    k = max(1, min(k, len(PROTEINS) - 1))
    candidate_idx = np.argpartition(-sims, kth=k - 1)[:k]
    ordered_idx = candidate_idx[np.argsort(-sims[candidate_idx])]
    out = []
    for rank, idx in enumerate(ordered_idx, start=1):
        p = protein_public(PROTEINS[int(idx)])
        p.update({"rank": rank, "similarity": round(float(sims[int(idx)]), 5)})
        out.append(p)
    return out


def search_api(params: Dict[str, List[str]]) -> Dict[str, Any]:
    q = (params.get("q") or [""])[0].strip()
    try:
        k = int((params.get("k") or ["20"])[0])
    except ValueError:
        k = 20
    k = max(1, min(k, 100))

    resolved = resolve_query(q)
    if resolved is None:
        return {
            "ok": False,
            "message": "No matching protein was found. Try a gene symbol, protein ID, or part of a description.",
            "query": q,
            "results": [],
        }

    idx, method, match_score = resolved
    return {
        "ok": True,
        "query": q,
        "match_method": method,
        "match_score": round(match_score, 4),
        "matched_protein": protein_public(PROTEINS[idx]),
        "results": nearest_neighbors(idx, k),
    }


def suggest_api(params: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    q = (params.get("q") or [""])[0].strip().lower()
    if not q:
        return []
    suggestions = []
    for p in PROTEINS:
        text = " ".join([p["protein_id"], p["name"], p["description"]]).lower()
        if q in text:
            suggestions.append(protein_public(p))
        if len(suggestions) >= 8:
            break
    return suggestions


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "ZebrafishESMDashboard/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        # Keep the terminal clean but still show important requests if needed.
        return

    def send_bytes(self, payload: bytes, content_type: str = "application/octet-stream", status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def send_json(self, payload: Any, status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_bytes(data, "application/json; charset=utf-8", status=status)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        params = parse_qs(parsed.query)

        try:
            if path == "/":
                self.serve_index()
            elif path == "/api/search":
                self.send_json(search_api(params))
            elif path == "/api/suggest":
                self.send_json(suggest_api(params))
            elif path.startswith("/static/"):
                self.serve_static(path)
            else:
                self.send_json({"ok": False, "message": "Not found"}, status=404)
        except Exception as exc:  # fail visibly in browser JSON
            self.send_json({"ok": False, "message": str(exc)}, status=500)

    def serve_index(self) -> None:
        template_path = ROOT / "templates" / "index.html"
        text = template_path.read_text(encoding="utf-8")
        text = text.replace("{{ '{:,}'.format(protein_count) }}", f"{len(PROTEINS):,}")
        text = text.replace("{{ db_path }}", html.escape(DB_PATH))
        text = text.replace("{{ url_for('static', filename='styles.css') }}", "/static/styles.css")
        text = text.replace("{{ url_for('static', filename='app.js') }}", "/static/app.js")
        self.send_bytes(text.encode("utf-8"), "text/html; charset=utf-8")

    def serve_static(self, path: str) -> None:
        rel = path.removeprefix("/static/")
        # prevent path traversal
        target = (ROOT / "static" / rel).resolve()
        if ROOT.resolve() not in target.parents:
            self.send_json({"ok": False, "message": "Invalid path"}, status=400)
            return
        if not target.exists() or not target.is_file():
            self.send_json({"ok": False, "message": "Static file not found"}, status=404)
            return
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        self.send_bytes(target.read_bytes(), content_type)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local zebrafish ESM dashboard.")
    parser.add_argument("--db", default="data/zebrafish_esm.db", help="Path to SQLite DB")
    parser.add_argument("--host", default=HOST, help="Host interface")
    parser.add_argument("--port", type=int, default=PORT, help="Port")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.host != HOST:
        raise ValueError("For local-only use, run this dashboard with --host 127.0.0.1.")
    load_database(args.db)
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Open http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
