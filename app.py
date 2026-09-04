#!/usr/bin/env python3
"""Zebrafish ESM dashboard with deterministic and AI-assisted discovery modes.

Exact protein lookup and ESM similarity remain deterministic. Biological-query
mode is zebrafish-first: Gemini may use Google Search to identify relevant
Danio rerio biology, but every final ESM seed must resolve to a protein in the
local zebrafish database. When zebrafish evidence is sparse, human/mouse genes
may be used only as reference evidence and are deterministically mapped to
zebrafish orthologues with Ensembl before they can become seeds.
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
from urllib.parse import parse_qs, quote, unquote, urlencode, urlparse
from urllib.request import Request, urlopen

import numpy as np


ROOT = Path(__file__).resolve().parent
DB_PATH = "data/zebrafish_esm.db"
HOST = "127.0.0.1"
PORT = 5000
DANIO_RERIO_TAXON_ID = 7955
UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
ENSEMBL_REST_URL = "https://rest.ensembl.org"
GEMINI_GENERATE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"
HTTP_TIMEOUT_SECONDS = 12
UNIPROT_RESULTS_PER_TERM = 20
MAX_DISCOVERY_SEEDS = 12
MIN_ZEBRAFISH_SEEDS_BEFORE_ORTHOLOGY = 8

PROTEINS: List[Dict[str, Any]] = []
ID_TO_INDEX: Dict[str, int] = {}
NAME_TO_INDEX: Dict[str, int] = {}
VECTORS: Optional[np.ndarray] = None
SEARCH_TEXTS: List[str] = []


def load_env_file(path: Path) -> None:
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
        raise FileNotFoundError(f"Database not found: {DB_PATH}. Build it first with build_database.py.")

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
        vectors.append(np.frombuffer(row["embedding"], dtype=np.float32).copy())
        proteins.append(
            {
                "protein_id": row["protein_id"] or "",
                "name": row["gene_name"] or "",
                "description": row["description"] or "",
                "sequence": "",
                "extra_json": row["metadata_json"] or "{}",
            }
        )

    PROTEINS = proteins
    VECTORS = np.vstack(vectors).astype(np.float32, copy=False)
    ID_TO_INDEX = {p["protein_id"].strip().lower(): i for i, p in enumerate(PROTEINS) if p["protein_id"].strip()}
    NAME_TO_INDEX = {p["name"].strip().lower(): i for i, p in enumerate(PROTEINS) if p["name"].strip()}
    SEARCH_TEXTS = [" ".join([p["protein_id"], p["name"], p["description"]]).lower() for p in PROTEINS]
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
            contains.append((min(0.96, len(q) / max(len(text), 1) + boost), i))
    if contains:
        contains.sort(reverse=True)
        return contains[0][1], "contains match", float(contains[0][0])

    choices: List[str] = []
    choice_to_index: Dict[str, int] = {}
    for i, p in enumerate(PROTEINS):
        for field in (p["protein_id"], p["name"]):
            if field:
                key = field.lower()
                choices.append(key)
                choice_to_index[key] = i
    matches = difflib.get_close_matches(q, choices, n=1, cutoff=0.55)
    if not matches:
        return None
    match = matches[0]
    return choice_to_index[match], "fuzzy match", float(difflib.SequenceMatcher(a=q, b=match).ratio())


def nearest_neighbors(index: int, k: int) -> List[Dict[str, Any]]:
    if VECTORS is None:
        raise RuntimeError("Vectors are not loaded.")
    sims = VECTORS @ VECTORS[index]
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


def parse_k(params: Dict[str, List[str]]) -> int:
    try:
        k = int((params.get("k") or ["20"])[0])
    except ValueError:
        k = 20
    return max(1, min(k, 100))


def search_api(params: Dict[str, List[str]]) -> Dict[str, Any]:
    q = (params.get("q") or [""])[0].strip()
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
        "results": nearest_neighbors(idx, parse_k(params)),
    }


def _http_json(url: str, *, headers: Optional[Dict[str, str]] = None, data: Optional[bytes] = None) -> Any:
    request_headers = {
        "Accept": "application/json",
        "User-Agent": "zebrafish-esm-search/2.1",
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


def _parse_json_object(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise RuntimeError("AI response was not valid JSON.")
        try:
            data = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise RuntimeError("AI response was not valid JSON.") from exc
    if not isinstance(data, dict):
        raise RuntimeError("AI response must be a JSON object.")
    return data


def _gemini_text(prompt: str, *, use_google_search: bool = False) -> str:
    key = gemini_api_key()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    generation_config: Dict[str, Any] = {"temperature": 0.1}
    # Gemini 2.5 Search grounding rejects responseMimeType=application/json.
    # For grounded calls the prompt requests JSON and the parser validates it.
    if not use_google_search:
        generation_config["responseMimeType"] = "application/json"

    payload: Dict[str, Any] = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": generation_config,
    }
    if use_google_search:
        payload["tools"] = [{"google_search": {}}]

    response = _http_json(
        GEMINI_GENERATE_URL.format(model=gemini_model()),
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


def _clean_candidate_list(values: Any, *, allowed_species: Optional[set[str]] = None, limit: int = 8) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    seen: set[Tuple[str, str]] = set()
    for raw in values or []:
        if not isinstance(raw, dict):
            continue
        gene = str(raw.get("gene") or "").strip()
        species = str(raw.get("species") or "zebrafish").strip().lower()
        reason = str(raw.get("reason") or "").strip()[:400]
        if allowed_species is not None and species not in allowed_species:
            continue
        key = (species, gene.lower())
        if not gene or key in seen:
            continue
        seen.add(key)
        out.append({"gene": gene[:80], "species": species, "reason": reason})
        if len(out) >= limit:
            break
    return out


def interpret_biological_query(question: str) -> Dict[str, Any]:
    """Create a zebrafish-first, search-grounded biological retrieval plan."""
    question = question.strip()
    fallback = {
        "normalized_question": question,
        "retrieval_terms": [question],
        "zebrafish_candidates": [],
        "reference_candidates": [],
        "rationale": "Direct zebrafish biological keyword retrieval (AI interpreter unavailable).",
        "ai_used": False,
        "search_grounded": False,
    }
    if not ai_available():
        return fallback

    prompt = f"""You are the biological-search planner for a DANIO RERIO (zebrafish) protein-discovery system.
User question: {question!r}

Species policy:
1. The final biological target is ALWAYS zebrafish / Danio rerio.
2. Search zebrafish-specific evidence first. Prefer zebrafish cell markers, expression data, genetic studies, pathway annotations, ZFIN/UniProt/Ensembl resources, and zebrafish papers.
3. For every web search you perform for primary evidence, include zebrafish or Danio rerio context. Do not let generic human search results define the zebrafish answer.
4. If zebrafish evidence for a protein/pathway is sparse, you MAY use strong human or mouse evidence to identify plausible conserved reference genes. Report those separately. Do not claim they are zebrafish genes and do not invent the zebrafish ortholog; the application will map them deterministically with Ensembl.
5. Prefer highly specific/directly relevant genes or proteins over broad generic pathway members. For a cell-type question, prioritize established zebrafish markers or highly enriched/specific genes when available.

Use Google Search to research the question before answering. Return ONLY JSON with this shape:
{{
  "normalized_question": "zebrafish-specific normalized question",
  "retrieval_terms": ["3 to 5 concise zebrafish biological search concepts"],
  "zebrafish_candidates": [
    {{"gene": "Danio rerio gene symbol", "species": "zebrafish", "reason": "brief zebrafish-specific evidence/relevance"}}
  ],
  "reference_candidates": [
    {{"gene": "human or mouse gene symbol", "species": "human or mouse", "reason": "why mammalian evidence is useful because zebrafish evidence is sparse"}}
  ],
  "rationale": "one or two sentences"
}}

Give up to 8 strong zebrafish candidates. Only use reference_candidates when useful; zebrafish evidence takes priority.
"""
    try:
        data = _parse_json_object(_gemini_text(prompt, use_google_search=True))
        terms: List[str] = []
        for value in data.get("retrieval_terms") or []:
            term = str(value).strip()
            if term and term.lower() not in {x.lower() for x in terms}:
                terms.append(term[:120])
        if not terms:
            terms = [question]
        return {
            "normalized_question": str(data.get("normalized_question") or question).strip()[:240],
            "retrieval_terms": terms[:5],
            "zebrafish_candidates": _clean_candidate_list(data.get("zebrafish_candidates"), allowed_species={"zebrafish", "danio rerio"}, limit=8),
            "reference_candidates": _clean_candidate_list(data.get("reference_candidates"), allowed_species={"human", "mouse"}, limit=6),
            "rationale": str(data.get("rationale") or "Zebrafish-first grounded search plan.").strip()[:600],
            "ai_used": True,
            "search_grounded": True,
        }
    except Exception as exc:
        fallback["rationale"] = f"AI search unavailable; used zebrafish-only direct retrieval instead. ({exc})"
        return fallback


def _primary_gene_name(uniprot_record: Dict[str, Any]) -> str:
    for gene in uniprot_record.get("genes") or []:
        value = str(((gene or {}).get("geneName") or {}).get("value") or "").strip()
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


def validate_ai_zebrafish_candidates(candidates: Iterable[Dict[str, str]]) -> List[Dict[str, Any]]:
    """AI may nominate genes, but only exact local zebrafish identities become seeds."""
    seeds: List[Dict[str, Any]] = []
    seen: set[int] = set()
    for candidate in candidates:
        gene = str(candidate.get("gene") or "").strip()
        idx = resolve_exact_identifier(gene)
        if idx is None or idx in seen:
            continue
        seen.add(idx)
        seeds.append(
            {
                "index": idx,
                "source": "Gemini Google Search (zebrafish)",
                "retrieval_term": gene,
                "resolved_by": "exact local zebrafish gene",
                "evidence_class": "zebrafish-supported",
                "ai_reason": candidate.get("reason") or "",
            }
        )
    return seeds


def fetch_uniprot_seeds(
    terms: Iterable[str],
    *,
    per_term: int = UNIPROT_RESULTS_PER_TERM,
    max_seeds: int = MAX_DISCOVERY_SEEDS,
) -> List[Dict[str, Any]]:
    """Retrieve deeper Danio rerio UniProt results and validate them locally."""
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
                if candidate:
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
                    "source": "UniProt zebrafish search",
                    "retrieval_term": term,
                    "uniprot_accession": accession,
                    "uniprot_gene": gene_name,
                    "uniprot_protein_name": protein_name,
                    "resolved_by": resolved_by,
                    "evidence_class": "zebrafish-supported",
                }
            )
            if len(seeds) >= max_seeds:
                return seeds
    return seeds


def _ensembl_zebrafish_ortholog_symbols(source_species: str, gene: str) -> List[str]:
    species_map = {"human": "homo_sapiens", "mouse": "mus_musculus"}
    ensembl_species = species_map.get(source_species.lower())
    if not ensembl_species:
        return []

    params = urlencode({"target_species": "danio_rerio", "type": "orthologues", "format": "condensed", "sequence": "none"})
    payload = _http_json(
        f"{ENSEMBL_REST_URL}/homology/symbol/{quote(ensembl_species)}/{quote(gene)}?{params}",
        headers={"Content-Type": "application/json"},
    )
    symbols: List[str] = []
    seen: set[str] = set()
    for datum in payload.get("data") or []:
        for homology in (datum or {}).get("homologies") or []:
            target = (homology or {}).get("target") or {}
            direct_symbol = ""
            target_id = ""
            if isinstance(target, dict):
                direct_symbol = str(target.get("display_id") or target.get("display_name") or "").strip()
                target_id = str(target.get("id") or "").strip()
            elif isinstance(target, str):
                target_id = target.strip()

            if direct_symbol and direct_symbol.lower() not in seen:
                seen.add(direct_symbol.lower())
                symbols.append(direct_symbol)
                continue
            if not target_id:
                continue
            try:
                lookup = _http_json(
                    f"{ENSEMBL_REST_URL}/lookup/id/{quote(target_id)}?content-type=application/json",
                    headers={"Content-Type": "application/json"},
                )
                symbol = str(lookup.get("display_name") or "").strip()
            except Exception:
                symbol = ""
            if symbol and symbol.lower() not in seen:
                seen.add(symbol.lower())
                symbols.append(symbol)
    return symbols


def orthology_seeds(reference_candidates: Iterable[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Map mammalian reference evidence to exact local Danio rerio proteins."""
    seeds: List[Dict[str, Any]] = []
    seen: set[int] = set()
    for candidate in reference_candidates:
        source_species = str(candidate.get("species") or "").lower()
        source_gene = str(candidate.get("gene") or "").strip()
        if source_species not in {"human", "mouse"} or not source_gene:
            continue
        try:
            zebrafish_symbols = _ensembl_zebrafish_ortholog_symbols(source_species, source_gene)
        except Exception:
            continue
        for symbol in zebrafish_symbols:
            idx = resolve_exact_identifier(symbol)
            if idx is None or idx in seen:
                continue
            seen.add(idx)
            seeds.append(
                {
                    "index": idx,
                    "source": f"Ensembl orthology ({source_species} → zebrafish)",
                    "retrieval_term": source_gene,
                    "resolved_by": f"{source_gene} → {symbol}; exact local zebrafish gene",
                    "evidence_class": "mammalian evidence + zebrafish orthology",
                    "reference_species": source_species,
                    "reference_gene": source_gene,
                    "ai_reason": candidate.get("reason") or "",
                }
            )
    return seeds


def local_annotation_seeds(terms: Iterable[str], *, limit: int = 6) -> List[Dict[str, Any]]:
    scored: List[Tuple[float, int, str]] = []
    for term in terms:
        tokens = [t for t in re.findall(r"[a-z0-9]+", term.lower()) if len(t) >= 4]
        if not tokens:
            continue
        for idx, text in enumerate(SEARCH_TEXTS):
            hits = sum(1 for token in tokens if token in text)
            if hits:
                scored.append((hits / len(tokens), idx, term))
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
                "source": "local zebrafish annotation fallback",
                "retrieval_term": term,
                "resolved_by": f"annotation token overlap ({score:.2f})",
                "evidence_class": "zebrafish local annotation",
            }
        )
        if len(out) >= limit:
            break
    return out


def _merge_seeds(*groups: Iterable[Dict[str, Any]], limit: int = MAX_DISCOVERY_SEEDS) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[int] = set()
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
    if not seed_indices:
        return []
    unique_seed_indices = list(dict.fromkeys(seed_indices))
    similarities = VECTORS @ VECTORS[unique_seed_indices].T
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
    for seed in seeds[:10]:
        p = protein_public(PROTEINS[int(seed["index"])])
        compact_seeds.append(
            {
                "gene": p["name"],
                "protein_id": p["protein_id"],
                "description": p["description"],
                "source": seed.get("source"),
                "evidence_class": seed.get("evidence_class"),
            }
        )
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
    prompt = f"""Explain a DANIO RERIO zebrafish ESM protein-similarity result.
Question: {question}
Plan: {json.dumps(plan, ensure_ascii=False)}
Validated zebrafish seeds: {json.dumps(compact_seeds, ensure_ascii=False)}
Top ESM candidates: {json.dumps(compact_results, ensure_ascii=False)}
Return only JSON: {{"summary": "2-4 concise sentences"}}.
Keep the interpretation zebrafish-specific. Distinguish direct zebrafish evidence from mammalian evidence transferred through orthology. Do not claim functional proof from ESM similarity alone and do not invent annotations.
"""
    try:
        data = _parse_json_object(_gemini_text(prompt))
        summary = str(data.get("summary") or "").strip()
        return summary[:1200] or None
    except Exception:
        return None


def discovery_api(params: Dict[str, List[str]]) -> Dict[str, Any]:
    question = (params.get("q") or [""])[0].strip()
    if not question:
        return {"ok": False, "mode": "discovery", "message": "Enter a biological question.", "results": []}

    plan = interpret_biological_query(question)
    direct_ai_seeds = validate_ai_zebrafish_candidates(plan.get("zebrafish_candidates") or [])

    retrieval_error: Optional[str] = None
    try:
        uniprot_seeds = fetch_uniprot_seeds(plan["retrieval_terms"])
    except Exception as exc:
        retrieval_error = str(exc)
        uniprot_seeds = []

    zebrafish_seeds = _merge_seeds(direct_ai_seeds, uniprot_seeds)
    mapped_reference_seeds: List[Dict[str, Any]] = []
    if len(zebrafish_seeds) < MIN_ZEBRAFISH_SEEDS_BEFORE_ORTHOLOGY and plan.get("reference_candidates"):
        mapped_reference_seeds = orthology_seeds(plan["reference_candidates"])

    seeds = _merge_seeds(zebrafish_seeds, mapped_reference_seeds)
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

    results = discovery_neighbors([int(seed["index"]) for seed in seeds], parse_k(params))
    public_seeds: List[Dict[str, Any]] = []
    for seed in seeds:
        p = protein_public(PROTEINS[int(seed["index"])])
        public_seeds.append(
            {
                **p,
                "source": seed.get("source"),
                "retrieval_term": seed.get("retrieval_term"),
                "resolved_by": seed.get("resolved_by"),
                "uniprot_accession": seed.get("uniprot_accession"),
                "evidence_class": seed.get("evidence_class"),
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
        "plan": plan,
        "seed_source": "Zebrafish-first: " + "; ".join(source_names),
        "retrieval_warning": retrieval_error,
        "seeds": public_seeds,
        "results": results,
        "ai_explanation": explain_discovery(question, plan, seeds, results),
        "privacy": "Embeddings remain server-side. Gemini receives the biological question and compact protein metadata, never embedding vectors or database credentials.",
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
        "species_scope": "Danio rerio",
        "google_search_grounding": ai_available(),
        "embedding_egress": False,
    }


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "ZebrafishESMDashboard/2.1"

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
        self.send_bytes(json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8", status=status)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
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
                self.send_json({"ok": False, "message": "Not found"}, status=404)
        except Exception as exc:
            self.send_json({"ok": False, "message": str(exc)}, status=500)

    def serve_index(self) -> None:
        text = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
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
        self.send_bytes(target.read_bytes(), mimetypes.guess_type(str(target))[0] or "application/octet-stream")


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
    print("Biological discovery species scope: Danio rerio (zebrafish-first; mammalian orthology fallback only when needed)")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
