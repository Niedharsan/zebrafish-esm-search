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
DEFAULT_OLLAMA_MODEL = "qwen3:4b-instruct"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
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


def ai_provider() -> str:
    return os.environ.get("AI_PROVIDER", "gemini").strip().lower() or "gemini"


def ollama_model() -> str:
    return os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL).strip() or DEFAULT_OLLAMA_MODEL


def ollama_url() -> str:
    return os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL).strip().rstrip("/") or DEFAULT_OLLAMA_URL


def ai_model() -> str:
    return ollama_model() if ai_provider() == "ollama" else gemini_model()


def ai_label() -> str:
    return "Local Ollama" if ai_provider() == "ollama" else "Gemini"


def ai_available() -> bool:
    return bool(ollama_model()) if ai_provider() == "ollama" else bool(gemini_api_key())


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


def _http_json(
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    data: Optional[bytes] = None,
    timeout: int = HTTP_TIMEOUT_SECONDS,
) -> Any:
    req = Request(
        url,
        data=data,
        headers={"Accept": "application/json", "User-Agent": "zebrafish-esm-search/3.0", **(headers or {})},
        method="POST" if data is not None else "GET",
    )
    try:
        with urlopen(req, timeout=timeout) as response:
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


def _gemini_response(prompt: str, *, use_google_search: bool = False) -> Tuple[str, Dict[str, Any]]:
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
    grounding = candidate.get("groundingMetadata") or {}
    return text, grounding if isinstance(grounding, dict) else {}


def _gemini_text(prompt: str, *, use_google_search: bool = False) -> str:
    return _gemini_response(prompt, use_google_search=use_google_search)[0]


def _ollama_text(prompt: str) -> str:
    payload = {
        "model": ollama_model(),
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "think": False,
        "options": {"temperature": 0.1, "num_ctx": 8192, "num_predict": 1200},
    }
    response = _http_json(
        f"{ollama_url()}/api/generate",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload).encode(),
        timeout=120,
    )
    text = str(response.get("response") or "").strip()
    if not text:
        raise RuntimeError("Ollama returned no text.")
    return text


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


def fetch_uniprot_candidates(term: str, size: int = 10) -> List[Dict[str, Any]]:
    term = term.strip()
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


def _local_question_context(question: str, limit: int = 30) -> List[Dict[str, str]]:
    stopwords = {
        "a", "an", "and", "are", "associated", "danio", "find", "for", "in", "involved",
        "mark", "of", "or", "protein", "proteins", "rerio", "the", "what", "which", "with",
        "zebrafish",
    }
    terms = []
    for token in re.findall(r"[a-z0-9.]+", question.lower()):
        term = token[:-1] if token.endswith("s") and len(token) > 4 else token
        if len(term) >= 3 and term not in stopwords and term not in terms:
            terms.append(term)
    if not terms:
        return []

    scored = []
    for protein in PROTEINS:
        gene = str(protein.get("name") or "").strip()
        description = str(protein.get("description") or "").strip()
        protein_id = str(protein.get("protein_id") or "").strip()
        gene_lower, description_lower = gene.lower(), description.lower()
        score = 0
        for term in terms:
            if gene_lower == term:
                score += 12
            if re.search(rf"\b{re.escape(term)}\b", description_lower):
                score += 4
        if not score:
            continue
        if gene and not gene_lower.startswith(("loc", "si:", "zgc:")):
            score += 1
        evidence = re.search(r"\bPE=(\d)\b", description)
        if evidence:
            score += max(0, 3 - int(evidence.group(1)))
        scored.append((score, gene_lower, gene, protein_id, description))

    scored.sort(key=lambda row: (-row[0], row[1]))
    return [
        {"gene": gene, "protein_id": protein_id, "description": description[:220]}
        for _, _, gene, protein_id, description in scored[:limit]
    ]


def interpret_biological_query(question: str) -> Dict[str, Any]:
    question = question.strip()
    fallback = {
        "normalized_question": f"Danio rerio: {question}",
        "retrieval_terms": [],
        "zebrafish_candidates": [],
        "reference_candidates": [],
        "rationale": "AI biological discovery is unavailable.",
        "ai_used": False,
        "search_grounded": False,
        "evidence_summary": {"sources": []},
        "_research_note": "",
        "_grounding_metadata": {},
        "_retrieval_errors": [],
    }
    if not ai_available():
        return fallback

    if ai_provider() == "ollama":
        local_context = _local_question_context(question)
        prompt = f"""Act as a zebrafish biologist selecting seed proteins for an ESM similarity search.

USER QUESTION:
{question}

Identify the most biologically relevant Danio rerio genes or proteins using your internal knowledge. You have no web access, so do not claim that you searched sources or verified current literature.

LOCAL ZEBRAFISH DATABASE CONTEXT (lexical retrieval, not biological ranking):
{json.dumps(local_context, ensure_ascii=False)}

Rules:
- Do not classify the question into a fixed category or use a hand-built scoring scheme.
- Keep broad questions broad; do not silently narrow a pathway, process, or cell-type question.
- Prefer canonical, commonly used, directly relevant zebrafish genes over lexical overlap.
- Use the local database context to recognize exact zebrafish symbols, but keep only entries that are biologically relevant to the question.
- Use exact zebrafish gene symbols when known, including zebrafish paralog suffixes such as a/b or .1/.2.
- Put uncertain human or mouse candidates in reference_candidates so Ensembl can resolve orthologs.
- Return at most {MAX_SEEDS} zebrafish candidates and at most 6 reference candidates.

Return only this JSON object:
{{
  "normalized_question": "...",
  "zebrafish_candidates": [
    {{"gene":"exact zebrafish symbol","species":"zebrafish","uniprot_accession":"","reason":"short reason"}}
  ],
  "reference_candidates": [
    {{"gene":"human or mouse symbol","species":"human or mouse","reason":"why fallback evidence matters"}}
  ],
  "rationale": "brief summary, including uncertainty where appropriate"
}}
"""
        try:
            note = _ollama_text(prompt)
            structured = _parse_json_object(note)
        except Exception as exc:
            fallback["rationale"] = f"AI biological discovery is unavailable. ({exc})"
            return fallback

        zebrafish = _clean_candidates(structured.get("zebrafish_candidates"), {"zebrafish", "danio rerio"}, MAX_SEEDS)
        return {
            "normalized_question": str(structured.get("normalized_question") or f"Danio rerio: {question}")[:240],
            "retrieval_terms": [candidate["gene"] for candidate in zebrafish],
            "zebrafish_candidates": zebrafish,
            "reference_candidates": _clean_candidates(structured.get("reference_candidates"), {"human", "mouse"}, 6),
            "rationale": str(structured.get("rationale") or "Local-model zebrafish candidate selection.")[:800],
            "ai_used": True,
            "search_grounded": False,
            "evidence_summary": {
                "sources": ["Local Ollama model knowledge (not web-grounded)", "Local zebrafish database lexical context"],
                "search_queries": 0,
                "local_context_records": len(local_context),
                "ranking_policy": "The local model proposes biological candidates; databases resolve identifiers.",
            },
            "_research_note": note,
            "_grounding_metadata": {},
            "_retrieval_errors": [],
        }

    research_prompt = f"""Use Google Search before answering. Act as a zebrafish biologist selecting seed proteins for an ESM search.

USER QUESTION:
{question}

Independently identify the most biologically relevant Danio rerio genes or proteins for the question. Search zebrafish-specific sources first, especially ZFIN, UniProt, Ensembl, Gene Ontology, expression/single-cell resources, pathway resources, and primary zebrafish papers where relevant.

Do the biology yourself:
- Do not classify the question into a fixed category and do not use a hand-built scoring scheme.
- Use the evidence that actually answers this question.
- Keep broad questions broad; do not silently narrow a pathway/process/cell-type question to one subtype or mechanism.
- Prefer direct Danio rerio evidence. Use human/mouse only as a clearly separated fallback when zebrafish evidence is sparse.
- Prefer canonical, commonly used, directly evidenced zebrafish genes over lexical overlap in protein names.
- Give a concise gene symbol or protein identifier that can be searched in a biological database.

Return a concise plain-text note headed TOP ZEBRAFISH CANDIDATES with the strongest candidates first and a short biological reason for each, followed by a SOURCES section.
"""
    try:
        note, grounding = _gemini_response(research_prompt, use_google_search=True)
        grounded = bool(grounding.get("webSearchQueries") or grounding.get("groundingChunks"))
        if not grounded:
            raise RuntimeError("Gemini returned research without Google Search grounding.")
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
        fallback["rationale"] = f"AI biological discovery is unavailable. ({exc})"
        return fallback

    zebrafish = _clean_candidates(structured.get("zebrafish_candidates"), {"zebrafish", "danio rerio"}, MAX_SEEDS)
    return {
        "normalized_question": str(structured.get("normalized_question") or f"Danio rerio: {question}")[:240],
        "retrieval_terms": [candidate["gene"] for candidate in zebrafish],
        "zebrafish_candidates": zebrafish,
        "reference_candidates": _clean_candidates(structured.get("reference_candidates"), {"human", "mouse"}, 6),
        "rationale": str(structured.get("rationale") or "Gemini-ranked zebrafish research.")[:800],
        "ai_used": True,
        "search_grounded": grounded,
        "evidence_summary": {
            "sources": ["Gemini Google Search"],
            "search_queries": len(grounding.get("webSearchQueries") or []),
            "ranking_policy": "Gemini decides biological relevance; databases resolve identifiers.",
        },
        "_research_note": note,
        "_grounding_metadata": grounding,
        "_retrieval_errors": [],
    }


def _uniprot_match(record: Dict[str, Any], term: str) -> Tuple[bool, str]:
    key = term.strip().lower()
    if not key:
        return False, "empty AI term"
    gene = str(record.get("gene") or "").strip()
    accession = str(record.get("uniprot_accession") or "").strip()
    synonyms = [str(value).strip() for value in record.get("gene_synonyms") or []]
    if key == gene.lower():
        return True, "exact UniProt gene"
    if key in {value.lower() for value in synonyms if value}:
        return True, f"exact UniProt gene synonym for {gene}"
    if key == accession.lower():
        return True, "exact UniProt accession"
    return False, "no exact gene, synonym, or accession match"


def resolve_targeted_uniprot_candidates(
    candidates: Iterable[Dict[str, str]], size: int = 10
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    seeds, traces, errors, seen = [], [], [], set()
    for candidate in candidates:
        term = str(candidate.get("gene") or "").strip()
        if not term:
            continue
        try:
            records = fetch_uniprot_candidates(term, size=size)
        except Exception as exc:
            records = []
            errors.append(f"UniProt ({term}): {exc}")

        trace = {"term": term, "records": [], "local_resolution": None}
        for record in records:
            accepted, reason = _uniprot_match(record, term)
            item = {**record, "accepted": accepted, "decision": reason}
            idx = resolve_uniprot_accession(str(record.get("uniprot_accession") or "")) if accepted else None
            if idx is None and accepted:
                idx = resolve_exact_identifier(str(record.get("gene") or ""))
            item["local_match"] = protein_public(PROTEINS[idx]) if idx is not None else None
            trace["records"].append(item)
            if accepted and idx is not None and idx not in seen and trace["local_resolution"] is None:
                seen.add(idx)
                trace["local_resolution"] = {
                    "protein": protein_public(PROTEINS[idx]),
                    "resolved_by": reason,
                }
                seeds.append(
                    {
                        "index": idx,
                        "source": f"{ai_label()} interpretation → targeted UniProt",
                        "retrieval_term": term,
                        "resolved_by": reason,
                        "uniprot_accession": record.get("uniprot_accession"),
                        "ai_reason": candidate.get("reason"),
                    }
                )

        if trace["local_resolution"] is None:
            idx = resolve_exact_identifier(term)
            if idx is not None and idx not in seen:
                seen.add(idx)
                trace["local_resolution"] = {
                    "protein": protein_public(PROTEINS[idx]),
                    "resolved_by": "exact local zebrafish gene",
                }
                seeds.append(
                    {
                        "index": idx,
                        "source": f"{ai_label()} interpretation → exact local validation",
                        "retrieval_term": term,
                        "resolved_by": "exact local zebrafish gene",
                        "uniprot_accession": "",
                        "ai_reason": candidate.get("reason"),
                    }
                )
        traces.append(trace)
    return seeds, traces, errors


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
    direct, resolution_trace, resolution_errors = resolve_targeted_uniprot_candidates(
        plan.get("zebrafish_candidates") or []
    )
    plan["_targeted_uniprot_resolution"] = resolution_trace
    plan["_retrieval_errors"] = [*(plan.get("_retrieval_errors") or []), *resolution_errors]
    refs = orthology_seeds(plan.get("reference_candidates") or []) if len(direct) < 4 else []
    seeds = _merge_seeds(direct, refs)

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
        "privacy": (
            "Embeddings and the biological question remain local. UniProt receives targeted public search terms; "
            "no service receives vectors or database credentials."
            if ai_provider() == "ollama"
            else "Embeddings remain local. Gemini sees the biological question; UniProt receives targeted public search terms. Neither receives vectors or database credentials."
        ),
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
        "ai_provider": ai_provider() if ai_available() else None,
        "ai_model": ai_model() if ai_available() else None,
        "species_scope": "Danio rerio",
        "google_search_grounding": ai_available() and ai_provider() == "gemini",
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
    print(f"AI interpreter: {'enabled (' + ai_provider() + ': ' + ai_model() + ')' if ai_available() else 'disabled; deterministic modes still work'}")
    print(f"Biological discovery: {ai_label()} interpretation → deterministic zebrafish validation → ESM similarity")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
