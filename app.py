#!/usr/bin/env python3
"""Local zebrafish ESM search: deterministic protein lookup + simple AI-first discovery."""

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
from urllib.parse import parse_qs, quote, unquote, urlencode, urlparse
from urllib.request import Request, urlopen

import numpy as np

ROOT = Path(__file__).resolve().parent
DB_PATH = "data/zebrafish_esm.db"
HOST, PORT = "127.0.0.1", 5000
UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
ENSEMBL_REST_URL = "https://rest.ensembl.org"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"
HTTP_TIMEOUT_SECONDS = 15
MAX_SEEDS = 10

PROTEINS: List[Dict[str, Any]] = []
ID_TO_INDEX: Dict[str, int] = {}
NAME_TO_INDEX: Dict[str, int] = {}
VECTORS: Optional[np.ndarray] = None
SEARCH_TEXTS: List[str] = []


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() and key.strip() not in os.environ:
            os.environ[key.strip()] = value.strip().strip('"').strip("'")


load_env_file(ROOT / ".env")


def gemini_api_key() -> str:
    return os.environ.get("GEMINI_API_KEY", "").strip()


def gemini_model() -> str:
    return os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL


def ai_available() -> bool:
    return bool(gemini_api_key())


def load_database(db_path: str) -> None:
    global DB_PATH, PROTEINS, ID_TO_INDEX, NAME_TO_INDEX, VECTORS, SEARCH_TEXTS
    DB_PATH = db_path
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT row_id, protein_id, gene_name, description, metadata_json, embedding FROM proteins ORDER BY row_id"
    ).fetchall()
    conn.close()
    if not rows:
        raise ValueError("Database contains no proteins.")

    PROTEINS, vectors = [], []
    for row in rows:
        PROTEINS.append(
            {
                "protein_id": row["protein_id"] or "",
                "name": row["gene_name"] or "",
                "description": row["description"] or "",
                "sequence": "",
                "extra_json": row["metadata_json"] or "{}",
            }
        )
        vectors.append(np.frombuffer(row["embedding"], dtype=np.float32).copy())

    VECTORS = np.vstack(vectors).astype(np.float32, copy=False)
    ID_TO_INDEX = {p["protein_id"].lower(): i for i, p in enumerate(PROTEINS) if p["protein_id"]}
    NAME_TO_INDEX = {p["name"].lower(): i for i, p in enumerate(PROTEINS) if p["name"]}
    SEARCH_TEXTS = [" ".join([p["protein_id"], p["name"], p["description"]]).lower() for p in PROTEINS]
    print(f"Loaded {len(PROTEINS):,} proteins from {db_path}. Vectors: {VECTORS.shape}")


def protein_public(p: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "protein_id": p.get("protein_id", ""),
        "name": p.get("name", ""),
        "description": p.get("description", ""),
        "sequence_length": None,
    }


def resolve_exact_identifier(value: str) -> Optional[int]:
    key = value.strip().lower()
    if not key:
        return None
    return ID_TO_INDEX.get(key, NAME_TO_INDEX.get(key))


def resolve_uniprot_accession(value: str) -> Optional[int]:
    accession = value.strip().lower()
    if not accession:
        return None
    exact = resolve_exact_identifier(accession)
    if exact is not None:
        return exact
    for i, p in enumerate(PROTEINS):
        if accession in [x.strip().lower() for x in str(p.get("protein_id") or "").split("|")]:
            return i
    return None


def resolve_query(query: str) -> Optional[Tuple[int, str, float]]:
    q = query.strip().lower()
    if not q:
        return None
    exact = resolve_exact_identifier(q)
    if exact is not None:
        p = PROTEINS[exact]
        return exact, "exact protein ID" if p["protein_id"].lower() == q else "exact gene name", 1.0

    contains = []
    for i, text in enumerate(SEARCH_TEXTS):
        if q in text:
            p = PROTEINS[i]
            boost = 0.45 if q in p["protein_id"].lower() or q in p["name"].lower() else 0.0
            contains.append((min(0.96, len(q) / max(len(text), 1) + boost), i))
    if contains:
        contains.sort(reverse=True)
        return contains[0][1], "contains match", float(contains[0][0])

    choices, mapping = [], {}
    for i, p in enumerate(PROTEINS):
        for field in (p["protein_id"], p["name"]):
            if field:
                key = field.lower()
                choices.append(key)
                mapping[key] = i
    matches = difflib.get_close_matches(q, choices, n=1, cutoff=0.55)
    if not matches:
        return None
    match = matches[0]
    return mapping[match], "fuzzy match", float(difflib.SequenceMatcher(a=q, b=match).ratio())


def nearest_neighbors(index: int, k: int) -> List[Dict[str, Any]]:
    if VECTORS is None:
        raise RuntimeError("Vectors are not loaded.")
    sims = VECTORS @ VECTORS[index]
    sims[index] = -np.inf
    k = max(1, min(k, len(PROTEINS) - 1))
    idxs = np.argpartition(-sims, kth=k - 1)[:k]
    idxs = idxs[np.argsort(-sims[idxs])]
    out = []
    for rank, idx in enumerate(idxs, start=1):
        p = protein_public(PROTEINS[int(idx)])
        p.update({"rank": rank, "similarity": round(float(sims[int(idx)]), 5)})
        out.append(p)
    return out


def parse_k(params: Dict[str, List[str]]) -> int:
    try:
        return max(1, min(int((params.get("k") or ["20"])[0]), 100))
    except ValueError:
        return 20


def search_api(params: Dict[str, List[str]]) -> Dict[str, Any]:
    q = (params.get("q") or [""])[0].strip()
    resolved = resolve_query(q)
    if resolved is None:
        return {"ok": False, "mode": "protein", "query": q, "message": "No matching protein was found.", "results": []}
    idx, method, score = resolved
    return {
        "ok": True,
        "mode": "protein",
        "query": q,
        "match_method": method,
        "match_score": round(score, 4),
        "matched_protein": protein_public(PROTEINS[idx]),
        "results": nearest_neighbors(idx, parse_k(params)),
    }


def _http_json(url: str, *, headers: Optional[Dict[str, str]] = None, data: Optional[bytes] = None) -> Any:
    req = Request(
        url,
        data=data,
        headers={"Accept": "application/json", "User-Agent": "zebrafish-esm-search/3.0", **(headers or {})},
        method="POST" if data is not None else "GET",
    )
    try:
        with urlopen(req, timeout=HTTP_TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"Remote service returned HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Remote service unavailable: {exc.reason}") from exc


def _parse_json_object(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise RuntimeError("AI response was not valid JSON.")
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise RuntimeError("AI response must be a JSON object.")
    return value


def _gemini_text(prompt: str, *, use_google_search: bool = False) -> str:
    key = gemini_api_key()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")
    config: Dict[str, Any] = {"temperature": 0.1}
    if not use_google_search:
        config["responseMimeType"] = "application/json"
    payload: Dict[str, Any] = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": config}
    if use_google_search:
        payload["tools"] = [{"google_search": {}}]
    response = _http_json(
        GEMINI_URL.format(model=gemini_model()),
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
        data=json.dumps(payload).encode(),
    )
    candidates = response.get("candidates") or []
    if not candidates:
        raise RuntimeError("Gemini returned no candidate response.")
    candidate = candidates[0] or {}
    text = "".join(
        str(part.get("text", "")) for part in ((candidate.get("content") or {}).get("parts") or []) if isinstance(part, dict)
    ).strip()
    if not text:
        raise RuntimeError(f"Gemini returned no text. finishReason={candidate.get('finishReason') or 'unknown'}")
    return text


def _search_term(question: str) -> str:
    text = re.sub(r"\b(danio\s+rerio|zebrafish|proteins?|genes?)\b", " ", question, flags=re.I)
    return re.sub(r"\s+", " ", text).strip(" ,;:-") or question.strip()


def _uniprot_gene(record: Dict[str, Any]) -> str:
    for gene in record.get("genes") or []:
        value = str(((gene or {}).get("geneName") or {}).get("value") or "").strip()
        if value:
            return value
    return ""


def _uniprot_synonyms(record: Dict[str, Any]) -> List[str]:
    out = []
    for gene in record.get("genes") or []:
        for syn in (gene or {}).get("synonyms") or []:
            value = str((syn or {}).get("value") or "").strip()
            if value and value not in out:
                out.append(value)
    return out[:8]


def _uniprot_name(record: Dict[str, Any]) -> str:
    description = record.get("proteinDescription") or {}
    for key in ("recommendedName", "submissionNames"):
        value = description.get(key)
        values = [value] if isinstance(value, dict) else value if isinstance(value, list) else []
        for item in values:
            full = (item or {}).get("fullName") or {}
            if isinstance(full, dict) and full.get("value"):
                return str(full["value"])
    return ""


def fetch_uniprot_candidates(question: str, size: int = 25) -> List[Dict[str, Any]]:
    term = _search_term(question)
    params = urlencode(
        {
            "query": f"(organism_id:7955) AND ({term})",
            "format": "json",
            "fields": "accession,gene_primary,gene_synonym,protein_name",
            "size": str(size),
        }
    )
    payload = _http_json(f"{UNIPROT_SEARCH_URL}?{params}")
    return [
        {
            "search_rank": rank,
            "gene": _uniprot_gene(record),
            "gene_synonyms": _uniprot_synonyms(record),
            "uniprot_accession": str(record.get("primaryAccession") or "").strip(),
            "protein_name": _uniprot_name(record),
        }
        for rank, record in enumerate(payload.get("results") or [], start=1)
    ]


def _clean_candidates(values: Any, allowed_species: set[str], limit: int) -> List[Dict[str, str]]:
    out, seen = [], set()
    for raw in values or []:
        if not isinstance(raw, dict):
            continue
        gene = str(raw.get("gene") or "").strip()
        species = str(raw.get("species") or "zebrafish").strip().lower()
        key = (species, gene.lower())
        if not gene or species not in allowed_species or key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "gene": gene[:80],
                "species": species,
                "uniprot_accession": str(raw.get("uniprot_accession") or "").strip()[:40],
                "reason": str(raw.get("reason") or "").strip()[:700],
            }
        )
        if len(out) >= limit:
            break
    return out


def interpret_biological_query(question: str) -> Dict[str, Any]:
    question = question.strip()
    term = _search_term(question)
    errors: List[str] = []
    try:
        uniprot = fetch_uniprot_candidates(question)
    except Exception as exc:
        uniprot = []
        errors.append(f"UniProt: {exc}")

    fallback = {
        "normalized_question": f"Danio rerio: {question}",
        "retrieval_terms": [term],
        "zebrafish_candidates": [],
        "reference_candidates": [],
        "rationale": "AI unavailable; using direct zebrafish UniProt fallback.",
        "ai_used": False,
        "search_grounded": False,
        "evidence_summary": {"sources": ["UniProtKB"], "uniprot_records": len(uniprot)},
        "_uniprot_candidates": uniprot,
        "_retrieval_errors": errors,
    }
    if not ai_available():
        return fallback

    research_prompt = f"""Act as a zebrafish biologist selecting seed proteins for an ESM search.

USER QUESTION:
{question}

A Danio rerio UniProt search for {term!r} returned:
{json.dumps(uniprot[:25], ensure_ascii=False)}

Use Google Search and your own biological reasoning. Search zebrafish-specific sources first, especially ZFIN, UniProt, Ensembl, Gene Ontology, expression/single-cell resources, pathway resources, and primary zebrafish papers where relevant.

Do the biology yourself:
- UniProt search position is NOT biological rank; a name match can rank high for lexical reasons.
- You may keep, reject, or reorder UniProt hits, and add zebrafish genes missing from this list.
- Do not classify the question into a fixed category and do not use a hand-built scoring scheme.
- Use the evidence that actually answers this question.
- Keep broad questions broad; do not silently narrow a pathway/process/cell-type question to one subtype or mechanism.
- Prefer direct Danio rerio evidence. Use human/mouse only as a clearly separated fallback when zebrafish evidence is sparse.
- Use exact current zebrafish gene symbols where possible.

Return a concise plain-text note headed TOP ZEBRAFISH CANDIDATES with the strongest candidates first and a short biological reason for each.
"""
    try:
        note = _gemini_text(research_prompt, use_google_search=True)
        structured = _parse_json_object(
            _gemini_text(
                f"""Convert this zebrafish research note into JSON.

Original question: {question!r}

NOTE:
---
{note[:14000]}
---

Return only:
{{
  "normalized_question": "...",
  "zebrafish_candidates": [
    {{"gene":"exact zebrafish symbol","species":"zebrafish","uniprot_accession":"","reason":"short reason"}}
  ],
  "reference_candidates": [
    {{"gene":"human or mouse symbol","species":"human or mouse","reason":"why fallback evidence matters"}}
  ],
  "rationale": "..."
}}

Preserve the note's biological ranking. Do not use UniProt search position as the ranking. Return at most {MAX_SEEDS} zebrafish candidates.
"""
            )
        )
    except Exception as exc:
        fallback["rationale"] = f"AI research unavailable; using direct zebrafish UniProt fallback. ({exc})"
        return fallback

    return {
        "normalized_question": str(structured.get("normalized_question") or f"Danio rerio: {question}")[:240],
        "retrieval_terms": [term],
        "zebrafish_candidates": _clean_candidates(
            structured.get("zebrafish_candidates"), {"zebrafish", "danio rerio"}, MAX_SEEDS
        ),
        "reference_candidates": _clean_candidates(structured.get("reference_candidates"), {"human", "mouse"}, 6),
        "rationale": str(structured.get("rationale") or "Gemini-ranked zebrafish research.")[:800],
        "ai_used": True,
        "search_grounded": True,
        "evidence_summary": {
            "sources": ["UniProtKB", "Gemini Google Search"],
            "uniprot_records": len(uniprot),
            "ranking_policy": "Gemini decides biological relevance; UniProt order is retrieval only.",
        },
        "_uniprot_candidates": uniprot,
        "_retrieval_errors": errors,
    }


def _resolve_candidate(candidate: Dict[str, str], uniprot: List[Dict[str, Any]]) -> Tuple[Optional[int], str]:
    gene = str(candidate.get("gene") or "").strip()
    accession = str(candidate.get("uniprot_accession") or "").strip()

    idx = resolve_exact_identifier(gene)
    if idx is not None:
        return idx, "exact local zebrafish gene"
    if accession:
        idx = resolve_uniprot_accession(accession)
        if idx is not None:
            return idx, "exact local UniProt accession"

    for record in uniprot:
        aliases = [str(record.get("gene") or ""), *[str(x) for x in record.get("gene_synonyms") or []]]
        if gene and gene.lower() in {x.lower() for x in aliases if x}:
            idx = resolve_uniprot_accession(str(record.get("uniprot_accession") or ""))
            if idx is not None:
                return idx, f"UniProt name/synonym resolution: {gene} → {record.get('gene')}"
    return None, ""


def validate_ai_zebrafish_candidates(
    candidates: Iterable[Dict[str, str]], uniprot: Iterable[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    records = list(uniprot)
    out, seen = [], set()
    for candidate in candidates:
        idx, resolved_by = _resolve_candidate(candidate, records)
        if idx is None or idx in seen:
            continue
        seen.add(idx)
        out.append(
            {
                "index": idx,
                "source": "Gemini zebrafish research",
                "retrieval_term": candidate.get("gene"),
                "resolved_by": resolved_by,
                "uniprot_accession": candidate.get("uniprot_accession"),
                "ai_reason": candidate.get("reason"),
            }
        )
    return out


def _ensembl_zebrafish_ortholog_symbols(source_species: str, gene: str) -> List[str]:
    species = {"human": "homo_sapiens", "mouse": "mus_musculus"}.get(source_species.lower())
    if not species:
        return []
    params = urlencode({"target_species": "danio_rerio", "type": "orthologues", "format": "condensed", "sequence": "none"})
    payload = _http_json(
        f"{ENSEMBL_REST_URL}/homology/symbol/{quote(species)}/{quote(gene)}?{params}",
        headers={"Content-Type": "application/json"},
    )
    out = []
    for datum in payload.get("data") or []:
        for homology in (datum or {}).get("homologies") or []:
            target = (homology or {}).get("target") or {}
            symbol = str(target.get("display_id") or target.get("display_name") or "").strip() if isinstance(target, dict) else ""
            if symbol and symbol not in out:
                out.append(symbol)
    return out


def orthology_seeds(reference_candidates: Iterable[Dict[str, str]]) -> List[Dict[str, Any]]:
    out, seen = [], set()
    for candidate in reference_candidates:
        species, gene = str(candidate.get("species") or "").lower(), str(candidate.get("gene") or "").strip()
        if species not in {"human", "mouse"} or not gene:
            continue
        try:
            symbols = _ensembl_zebrafish_ortholog_symbols(species, gene)
        except Exception:
            continue
        for symbol in symbols:
            idx = resolve_exact_identifier(symbol)
            if idx is None or idx in seen:
                continue
            seen.add(idx)
            out.append(
                {
                    "index": idx,
                    "source": f"Ensembl orthology ({species} → zebrafish)",
                    "retrieval_term": gene,
                    "resolved_by": f"{gene} → {symbol}; exact local zebrafish gene",
                    "reference_species": species,
                    "reference_gene": gene,
                    "ai_reason": candidate.get("reason"),
                }
            )
    return out


def uniprot_fallback_seeds(uniprot: Iterable[Dict[str, Any]], limit: int = 6) -> List[Dict[str, Any]]:
    out, seen = [], set()
    for record in uniprot:
        idx = resolve_exact_identifier(str(record.get("gene") or ""))
        if idx is None:
            idx = resolve_uniprot_accession(str(record.get("uniprot_accession") or ""))
        if idx is None or idx in seen:
            continue
        seen.add(idx)
        out.append(
            {
                "index": idx,
                "source": "UniProt fallback",
                "retrieval_term": record.get("gene") or record.get("protein_name"),
                "resolved_by": "deterministic local UniProt match",
                "uniprot_accession": record.get("uniprot_accession"),
                "ai_reason": "Fallback because AI research returned no resolvable seeds.",
            }
        )
        if len(out) >= limit:
            break
    return out


def _merge_seeds(*groups: Iterable[Dict[str, Any]], limit: int = MAX_SEEDS) -> List[Dict[str, Any]]:
    out, seen = [], set()
    for group in groups:
        for seed in group:
            idx = int(seed["index"])
            if idx in seen:
                continue
            seen.add(idx)
            out.append(seed)
            if len(out) >= limit:
                return out
    return out


def discovery_neighbors(seed_indices: List[int], k: int) -> List[Dict[str, Any]]:
    if VECTORS is None:
        raise RuntimeError("Vectors are not loaded.")
    unique = list(dict.fromkeys(seed_indices))
    sims = VECTORS @ VECTORS[unique].T
    best_col, best = np.argmax(sims, axis=1), np.max(sims, axis=1)
    for idx in unique:
        best[idx] = -np.inf
    k = max(1, min(k, len(PROTEINS) - len(unique)))
    idxs = np.argpartition(-best, kth=k - 1)[:k]
    idxs = idxs[np.argsort(-best[idxs])]
    out = []
    for rank, idx_value in enumerate(idxs, 1):
        idx = int(idx_value)
        seed = PROTEINS[unique[int(best_col[idx])]]
        p = protein_public(PROTEINS[idx])
        p.update(
            {
                "rank": rank,
                "similarity": round(float(best[idx]), 5),
                "closest_seed": seed.get("name") or seed.get("protein_id"),
                "closest_seed_id": seed.get("protein_id"),
            }
        )
        out.append(p)
    return out


def _public_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in plan.items() if not k.startswith("_")}


def discovery_api(params: Dict[str, List[str]]) -> Dict[str, Any]:
    question = (params.get("q") or [""])[0].strip()
    if not question:
        return {"ok": False, "mode": "discovery", "message": "Enter a biological question.", "results": []}

    plan = interpret_biological_query(question)
    uniprot = list(plan.get("_uniprot_candidates") or [])
    direct = validate_ai_zebrafish_candidates(plan.get("zebrafish_candidates") or [], uniprot)
    refs = orthology_seeds(plan.get("reference_candidates") or []) if len(direct) < 4 else []
    seeds = _merge_seeds(direct, refs)
    if not seeds:
        seeds = uniprot_fallback_seeds(uniprot)

    warning = "; ".join(plan.get("_retrieval_errors") or []) or None
    if not seeds:
        return {
            "ok": False,
            "mode": "discovery",
            "query": question,
            "plan": _public_plan(plan),
            "message": "No zebrafish seed proteins could be resolved into the local ESM database.",
            "results": [],
        }

    results = discovery_neighbors([int(seed["index"]) for seed in seeds], parse_k(params))
    public_seeds = []
    for seed in seeds:
        public_seeds.append(
            {
                **protein_public(PROTEINS[int(seed["index"])]),
                "source": seed.get("source"),
                "retrieval_term": seed.get("retrieval_term"),
                "resolved_by": seed.get("resolved_by"),
                "uniprot_accession": seed.get("uniprot_accession"),
                "reference_species": seed.get("reference_species"),
                "reference_gene": seed.get("reference_gene"),
                "ai_reason": seed.get("ai_reason"),
            }
        )

    source_names = list(dict.fromkeys(str(seed.get("source")) for seed in seeds if seed.get("source")))
    return {
        "ok": True,
        "mode": "discovery",
        "query": question,
        "plan": _public_plan(plan),
        "seed_source": "AI-ranked zebrafish: " + "; ".join(source_names),
        "retrieval_warning": warning,
        "seeds": public_seeds,
        "results": results,
        "ai_explanation": None,
        "privacy": "Embeddings remain local. Gemini sees the question and compact public UniProt metadata, never vectors or database credentials.",
    }


def suggest_api(params: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    q = (params.get("q") or [""])[0].strip().lower()
    if not q:
        return []
    out = []
    for p in PROTEINS:
        if q in " ".join([p["protein_id"], p["name"], p["description"]]).lower():
            out.append(protein_public(p))
        if len(out) >= 8:
            break
    return out


def status_api() -> Dict[str, Any]:
    return {
        "ok": True,
        "protein_count": len(PROTEINS),
        "ai_available": ai_available(),
        "ai_model": gemini_model() if ai_available() else None,
        "species_scope": "Danio rerio",
        "google_search_grounding": ai_available(),
        "structured_retrieval": ["UniProtKB", "Ensembl orthology fallback"],
        "embedding_egress": False,
    }


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "ZebrafishESMDashboard/3.0"

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
        self.send_bytes(json.dumps(payload).encode(), "application/json; charset=utf-8", status)

    def do_GET(self) -> None:  # noqa: N802
        parsed, path = urlparse(self.path), unquote(urlparse(self.path).path)
        params = parse_qs(parsed.query)
        try:
            if path == "/":
                self.serve_index()
            elif path in {"/api/health", "/api/status"}:
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
                self.send_json({"ok": False, "message": "Not found"}, 404)
        except Exception as exc:
            self.send_json({"ok": False, "message": str(exc)}, 500)

    def serve_index(self) -> None:
        text = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
        text = text.replace("{{ '{:,}'.format(protein_count) }}", f"{len(PROTEINS):,}")
        text = text.replace("{{ db_path }}", html.escape(DB_PATH))
        text = text.replace("{{ url_for('static', filename='styles.css') }}", "/static/styles.css")
        text = text.replace("{{ url_for('static', filename='app.js') }}", "/static/app.js")
        self.send_bytes(text.encode(), "text/html; charset=utf-8")

    def serve_static(self, path: str) -> None:
        target = (ROOT / "static" / path.removeprefix("/static/")).resolve()
        if ROOT.resolve() not in target.parents or not target.is_file():
            self.send_json({"ok": False, "message": "Static file not found"}, 404)
            return
        self.send_bytes(target.read_bytes(), mimetypes.guess_type(str(target))[0] or "application/octet-stream")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run zebrafish ESM dashboard.")
    parser.add_argument("--db", default="data/zebrafish_esm.db")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.host != HOST:
        raise ValueError("Local mode is intentionally bound to 127.0.0.1.")
    load_database(args.db)
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Open http://{args.host}:{args.port}")
    print(f"AI interpreter: {'enabled (' + gemini_model() + ')' if ai_available() else 'disabled; deterministic modes still work'}")
    print("Biological discovery: Gemini research → deterministic zebrafish validation → ESM similarity")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
