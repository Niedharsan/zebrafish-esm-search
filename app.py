#!/usr/bin/env python3
"""Zebrafish ESM dashboard with deterministic and AI-assisted discovery modes.

The private embedding database stays server-side. Exact protein lookup and ESM
similarity are deterministic. Optional Gemini integration is used only to turn
natural-language biological questions into retrieval concepts and to explain
already-ranked results. Seed proteins come from UniProt and must resolve back
into the local zebrafish database before they can influence ranking.
"""

from __future__ import annotations

import argparse
import difflib
import html
import json
import mimetypes
import os
import re
import sqlite3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, unquote, urlencode, urlparse
from urllib.request import Request, urlopen

import numpy as np


ROOT = Path(__file__).resolve().parent
DB_PATH = "data/zebrafish_esm.db"
HOST = "127.0.0.1"
PORT = 5000
DANIO_RERIO_TAXON_ID = 7955
UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
GEMINI_GENERATE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"
HTTP_TIMEOUT_SECONDS = 12

PROTEINS: List[Dict[str, Any]] = []
ID_TO_INDEX: Dict[str, int] = {}
NAME_TO_INDEX: Dict[str, int] = {}
VECTORS: Optional[np.ndarray] = None
SEARCH_TEXTS: List[str] = []


def load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE pairs without adding a dependency."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_env_file(ROOT / ".env")


def gemini_api_key() -> str:
    return os.environ.get("GEMINI_API_KEY", "").strip()


def gemini_model() -> str:
    return os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL


def ai_available() -> bool:
    return bool(gemini_api_key())


def connect_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def load_database(db_path: str) -> None:
    global DB_PATH, PROTEINS, ID_TO_INDEX, NAME_TO_INDEX, VECTORS, SEARCH_TEXTS
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

    proteins: List[Dict[str, Any]] = []
    vectors: List[np.ndarray] = []

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
    ID_TO_INDEX = {
        p["protein_id"].strip().lower(): i for i, p in enumerate(PROTEINS) if p["protein_id"].strip()
    }
    NAME_TO_INDEX = {
        p["name"].strip().lower(): i for i, p in enumerate(PROTEINS) if p["name"].strip()
    }
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


def resolve_exact_identifier(value: str) -> Optional[int]:
    key = value.strip().lower()
    if not key:
        return None
    if key in ID_TO_INDEX:
        return ID_TO_INDEX[key]
    if key in NAME_TO_INDEX:
        return NAME_TO_INDEX[key]
    return None


def resolve_query(query: str) -> Optional[Tuple[int, str, float]]:
    """Resolve protein-oriented free text to the best local protein row index."""
    q = query.strip().lower()
    if not q:
        return None

    exact = resolve_exact_identifier(q)
    if exact is not None:
        p = PROTEINS[exact]
        method = "exact protein ID" if p["protein_id"].lower() == q else "exact gene name"
        return exact, method, 1.0

    contains: List[Tuple[float, int]] = []
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
        match = matches[0]
        ratio = difflib.SequenceMatcher(a=q, b=match).ratio()
        return choice_to_index[match], "fuzzy match", float(ratio)

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
    out: List[Dict[str, Any]] = []
    for rank, idx in enumerate(ordered_idx, start=1):
        p = protein_public(PROTEINS[int(idx)])
        p.update({"rank": rank, "similarity": round(float(sims[int(idx)]), 5)})
        out.append(p)
    return out


def search_api(params: Dict[str, List[str]]) -> Dict[str, Any]:
    q = (params.get("q") or [""])[0].strip()
    k = parse_k(params)

    resolved = resolve_query(q)
    if resolved is None:
        return {
            "ok": False,
            "mode": "protein",
            "message": "No matching protein was found. Try a gene symbol, protein ID, or part of a description.",
            "query": q,
            "results": [],
        }

    idx, method, match_score = resolved
    return {
        "ok": True,
        "mode": "protein",
        "query": q,
        "match_method": method,
        "match_score": round(match_score, 4),
        "matched_protein": protein_public(PROTEINS[idx]),
        "results": nearest_neighbors(idx, k),
    }


def parse_k(params: Dict[str, List[str]]) -> int:
    try:
        k = int((params.get("k") or ["20"])[0])
    except ValueError:
        k = 20
    return max(1, min(k, 100))


def _http_json(url: str, *, headers: Optional[Dict[str, str]] = None, data: Optional[bytes] = None) -> Any:
    request_headers = {
        "Accept": "application/json",
        "User-Agent": "zebrafish-esm-search/2.0",
        **(headers or {}),
    }
    req = Request(url, data=data, headers=request_headers, method="POST" if data is not None else "GET")
    try:
        with urlopen(req, timeout=HTTP_TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"Remote service returned HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Remote service unavailable: {exc.reason}") from exc


def _gemini_text(prompt: str) -> str:
    key = gemini_api_key()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    url = GEMINI_GENERATE_URL.format(model=gemini_model())
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.15,
            "responseMimeType": "application/json",
        },
    }
    response = _http_json(
        url,
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
        data=json.dumps(payload).encode("utf-8"),
    )
    candidates = response.get("candidates") or []
    if not candidates:
        raise RuntimeError("Gemini returned no candidate response.")
    parts = (((candidates[0] or {}).get("content") or {}).get("parts") or [])
    text = "".join(str(part.get("text", "")) for part in parts if isinstance(part, dict)).strip()
    if not text:
        raise RuntimeError("Gemini returned an empty response.")
    return text


def _parse_json_object(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise RuntimeError("AI response was not valid JSON.") from exc
    if not isinstance(data, dict):
        raise RuntimeError("AI response must be a JSON object.")
    return data


def interpret_biological_query(question: str) -> Dict[str, Any]:
    """Use AI only to create retrieval concepts, never authoritative gene seeds."""
    question = question.strip()
    fallback = {
        "normalized_question": question,
        "retrieval_terms": [question],
        "rationale": "Direct biological keyword retrieval (AI interpreter unavailable).",
        "ai_used": False,
    }
    if not ai_available():
        return fallback

    prompt = f"""You are a query-planning component for a zebrafish protein discovery system.
The user question is: {question!r}

Convert the question into concise biological concepts that can be used as text queries against UniProt for Danio rerio.
Do NOT propose, guess, or list gene symbols, protein IDs, UniProt accessions, or candidate proteins. Seed proteins are retrieved separately from UniProt.
Return only JSON with this exact shape:
{{
  "normalized_question": "short normalized question",
  "retrieval_terms": ["3 to 5 short biological concepts"],
  "rationale": "one sentence explaining the search interpretation"
}}
Prefer pathway, process, cell-type, phenotype, molecular-function, or biological-role language that is likely to occur in curated protein annotations.
"""
    try:
        data = _parse_json_object(_gemini_text(prompt))
        terms = []
        for value in data.get("retrieval_terms") or []:
            term = str(value).strip()
            if term and term.lower() not in {x.lower() for x in terms}:
                terms.append(term[:100])
        if not terms:
            return fallback
        return {
            "normalized_question": str(data.get("normalized_question") or question).strip()[:240],
            "retrieval_terms": terms[:5],
            "rationale": str(data.get("rationale") or "AI-generated biological retrieval plan.").strip()[:500],
            "ai_used": True,
        }
    except Exception as exc:
        fallback["rationale"] = f"AI interpreter unavailable; used direct retrieval instead. ({exc})"
        return fallback


def _primary_gene_name(uniprot_record: Dict[str, Any]) -> str:
    genes = uniprot_record.get("genes") or []
    for gene in genes:
        gene_name = (gene or {}).get("geneName") or {}
        value = str(gene_name.get("value") or "").strip()
        if value:
            return value
    return ""


def _protein_description(uniprot_record: Dict[str, Any]) -> str:
    description = uniprot_record.get("proteinDescription") or {}
    for key in ("recommendedName", "submissionNames"):
        value = description.get(key)
        if isinstance(value, dict):
            full = value.get("fullName") or {}
            if isinstance(full, dict) and full.get("value"):
                return str(full["value"])
        if isinstance(value, list):
            for item in value:
                full = (item or {}).get("fullName") or {}
                if isinstance(full, dict) and full.get("value"):
                    return str(full["value"])
    return ""


def fetch_uniprot_seeds(terms: Iterable[str], *, per_term: int = 6) -> List[Dict[str, Any]]:
    """Retrieve Danio rerio proteins from UniProt, then require exact local resolution."""
    seeds: List[Dict[str, Any]] = []
    seen_local: set[int] = set()

    for term in terms:
        query = f"(organism_id:{DANIO_RERIO_TAXON_ID}) AND ({term})"
        params = urlencode(
            {
                "query": query,
                "format": "json",
                "fields": "accession,gene_primary,protein_name",
                "size": str(per_term),
            }
        )
        payload = _http_json(f"{UNIPROT_SEARCH_URL}?{params}")
        for record in payload.get("results") or []:
            accession = str(record.get("primaryAccession") or "").strip()
            gene_name = _primary_gene_name(record)
            protein_name = _protein_description(record)

            local_index: Optional[int] = None
            resolved_by = ""
            for label, candidate in (("gene name", gene_name), ("UniProt accession", accession)):
                if not candidate:
                    continue
                local_index = resolve_exact_identifier(candidate)
                if local_index is not None:
                    resolved_by = label
                    break

            if local_index is None or local_index in seen_local:
                continue
            seen_local.add(local_index)
            seeds.append(
                {
                    "index": local_index,
                    "source": "UniProt",
                    "retrieval_term": term,
                    "uniprot_accession": accession,
                    "uniprot_gene": gene_name,
                    "uniprot_protein_name": protein_name,
                    "resolved_by": resolved_by,
                }
            )
            if len(seeds) >= 10:
                return seeds
    return seeds


def local_annotation_seeds(terms: Iterable[str], *, limit: int = 6) -> List[Dict[str, Any]]:
    """Deterministic fallback when UniProt is unavailable or yields no local matches."""
    scored: List[Tuple[float, int, str]] = []
    for term in terms:
        tokens = [t for t in re.findall(r"[a-z0-9]+", term.lower()) if len(t) >= 4]
        if not tokens:
            continue
        for idx, text in enumerate(SEARCH_TEXTS):
            hits = sum(1 for token in tokens if token in text)
            if hits:
                score = hits / len(tokens)
                scored.append((score, idx, term))
    scored.sort(key=lambda item: item[0], reverse=True)

    out: List[Dict[str, Any]] = []
    seen: set[int] = set()
    for score, idx, term in scored:
        if idx in seen:
            continue
        seen.add(idx)
        out.append(
            {
                "index": idx,
                "source": "local annotation fallback",
                "retrieval_term": term,
                "resolved_by": f"annotation token overlap ({score:.2f})",
            }
        )
        if len(out) >= limit:
            break
    return out


def discovery_neighbors(seed_indices: List[int], k: int) -> List[Dict[str, Any]]:
    """Rank proteins by best ESM similarity to any validated seed protein."""
    if VECTORS is None:
        raise RuntimeError("Vectors are not loaded.")
    if not seed_indices:
        return []

    unique_seed_indices = list(dict.fromkeys(seed_indices))
    seed_matrix = VECTORS[unique_seed_indices]
    similarities = VECTORS @ seed_matrix.T
    best_seed_column = np.argmax(similarities, axis=1)
    best_scores = np.max(similarities, axis=1)
    for idx in unique_seed_indices:
        best_scores[idx] = -np.inf

    available = len(PROTEINS) - len(unique_seed_indices)
    if available <= 0:
        return []
    k = max(1, min(k, available))
    candidate_idx = np.argpartition(-best_scores, kth=k - 1)[:k]
    ordered_idx = candidate_idx[np.argsort(-best_scores[candidate_idx])]

    out: List[Dict[str, Any]] = []
    for rank, idx_value in enumerate(ordered_idx, start=1):
        idx = int(idx_value)
        seed_idx = unique_seed_indices[int(best_seed_column[idx])]
        p = protein_public(PROTEINS[idx])
        seed = protein_public(PROTEINS[seed_idx])
        p.update(
            {
                "rank": rank,
                "similarity": round(float(best_scores[idx]), 5),
                "closest_seed": seed.get("name") or seed.get("protein_id"),
                "closest_seed_id": seed.get("protein_id"),
            }
        )
        out.append(p)
    return out


def explain_discovery(question: str, plan: Dict[str, Any], seeds: List[Dict[str, Any]], results: List[Dict[str, Any]]) -> Optional[str]:
    if not ai_available() or not results:
        return None

    compact_seeds = []
    for seed in seeds[:8]:
        p = protein_public(PROTEINS[int(seed["index"])])
        compact_seeds.append({"gene": p["name"], "protein_id": p["protein_id"], "description": p["description"]})
    compact_results = [
        {
            "gene": r.get("name"),
            "protein_id": r.get("protein_id"),
            "description": r.get("description"),
            "similarity": r.get("similarity"),
            "closest_seed": r.get("closest_seed"),
        }
        for r in results[:8]
    ]

    prompt = f"""You are explaining results from a zebrafish ESM protein-similarity search.
Question: {question}
Retrieval plan: {json.dumps(plan, ensure_ascii=False)}
Validated seed proteins: {json.dumps(compact_seeds, ensure_ascii=False)}
Top ranked candidates: {json.dumps(compact_results, ensure_ascii=False)}

Return only JSON: {{"summary": "2-4 concise sentences"}}.
Explain what the ranking suggests, but do not claim functional proof from ESM similarity alone. Do not invent annotations not present above. Clearly distinguish sequence/representation similarity from evidence of biological function.
"""
    try:
        data = _parse_json_object(_gemini_text(prompt))
        summary = str(data.get("summary") or "").strip()
        return summary[:1200] or None
    except Exception:
        return None


def discovery_api(params: Dict[str, List[str]]) -> Dict[str, Any]:
    question = (params.get("q") or [""])[0].strip()
    k = parse_k(params)
    if not question:
        return {"ok": False, "mode": "discovery", "message": "Enter a biological question.", "results": []}

    plan = interpret_biological_query(question)
    retrieval_error: Optional[str] = None
    try:
        seeds = fetch_uniprot_seeds(plan["retrieval_terms"])
    except Exception as exc:
        retrieval_error = str(exc)
        seeds = []

    if not seeds:
        seeds = local_annotation_seeds(plan["retrieval_terms"])

    if not seeds:
        message = "No validated zebrafish seed proteins could be resolved into the local ESM database."
        if retrieval_error:
            message += f" UniProt error: {retrieval_error}"
        return {
            "ok": False,
            "mode": "discovery",
            "query": question,
            "plan": plan,
            "message": message,
            "results": [],
        }

    results = discovery_neighbors([int(seed["index"]) for seed in seeds], k)
    public_seeds = []
    for seed in seeds:
        p = protein_public(PROTEINS[int(seed["index"])])
        public_seeds.append(
            {
                **p,
                "source": seed.get("source"),
                "retrieval_term": seed.get("retrieval_term"),
                "resolved_by": seed.get("resolved_by"),
                "uniprot_accession": seed.get("uniprot_accession"),
            }
        )

    return {
        "ok": True,
        "mode": "discovery",
        "query": question,
        "plan": plan,
        "seed_source": "UniProt validated against local DB" if any(seed.get("source") == "UniProt" for seed in seeds) else "local annotation fallback",
        "retrieval_warning": retrieval_error,
        "seeds": public_seeds,
        "results": results,
        "ai_explanation": explain_discovery(question, plan, seeds, results),
        "privacy": "Embeddings remain server-side. Gemini receives only the question and compact protein metadata, never embedding vectors or database credentials.",
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


def status_api() -> Dict[str, Any]:
    return {
        "ok": True,
        "protein_count": len(PROTEINS),
        "ai_available": ai_available(),
        "ai_model": gemini_model() if ai_available() else None,
        "embedding_egress": False,
    }


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "ZebrafishESMDashboard/2.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def send_bytes(self, payload: bytes, content_type: str = "application/octet-stream", status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
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
            elif path == "/api/health" or path == "/api/status":
                self.send_json(status_api())
            elif path == "/api/search":
                self.send_json(search_api(params))
            elif path == "/api/discover":
                self.send_json(discovery_api(params))
            elif path == "/api/suggest":
                self.send_json(suggest_api(params))
            elif path.startswith("/static/"):
                self.serve_static(path)
            else:
                self.send_json({"ok": False, "message": "Not found"}, status=404)
        except Exception as exc:
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
    parser = argparse.ArgumentParser(description="Run zebrafish ESM dashboard.")
    parser.add_argument("--db", default="data/zebrafish_esm.db", help="Path to SQLite DB")
    parser.add_argument("--host", default=HOST, help="Host interface")
    parser.add_argument("--port", type=int, default=PORT, help="Port")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.host != HOST:
        raise ValueError("Local mode is intentionally bound to 127.0.0.1. Add a deployment layer before exposing it publicly.")
    load_database(args.db)
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Open http://{args.host}:{args.port}")
    print(f"AI interpreter: {'enabled (' + gemini_model() + ')' if ai_available() else 'disabled; deterministic modes still work'}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
