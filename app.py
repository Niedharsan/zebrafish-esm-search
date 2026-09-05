#!/usr/bin/env python3
"""Zebrafish ESM dashboard with deterministic and evidence-grounded discovery modes.

Exact protein lookup and ESM similarity remain deterministic. Biological-query
mode is zebrafish-first: Gemini plans retrieval, structured UniProt and Ensembl
evidence is collected, Google Search grounding adds independent biological
context, and a final evidence-ranking call selects candidates. Every final ESM
seed must resolve deterministically to a protein in the local zebrafish database.
Human/mouse genes may be used only as reference evidence and are mapped to
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
UNIPROT_RESULTS_PER_TERM = 10
MAX_UNIPROT_EVIDENCE_RECORDS = 30
MAX_DISCOVERY_SEEDS = 12
MIN_ZEBRAFISH_SEEDS_BEFORE_ORTHOLOGY = 8

QUESTION_TYPES = {
    "cell_type",
    "tissue",
    "biological_process",
    "pathway",
    "phenotype",
    "molecular_function",
    "protein_family",
    "other",
}

DEFAULT_EVIDENCE_PRIORITIES = [
    "direct zebrafish experimental or curated biological evidence",
    "relevant zebrafish functional/GO/pathway/expression evidence",
    "independent zebrafish literature or ZFIN evidence",
    "lexical protein-name matches only as weak supporting evidence",
]

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


def resolve_uniprot_accession(value: str) -> Optional[int]:
    accession = value.strip().lower()
    if not accession:
        return None
    exact = resolve_exact_identifier(accession)
    if exact is not None:
        return exact
    for i, protein in enumerate(PROTEINS):
        parts = [part.strip().lower() for part in str(protein.get("protein_id") or "").split("|")]
        if accession in parts:
            return i
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
        "User-Agent": "zebrafish-esm-search/2.3",
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

    candidate = candidates[0] or {}
    parts = ((candidate.get("content") or {}).get("parts") or [])
    text = "".join(str(part.get("text", "")) for part in parts if isinstance(part, dict)).strip()
    if not text:
        metadata = candidate.get("groundingMetadata") or {}
        queries = metadata.get("webSearchQueries") or []
        finish_reason = candidate.get("finishReason") or "unknown"
        suffix = f" finishReason={finish_reason}"
        if queries:
            suffix += f"; search_queries={queries[:3]}"
        raise RuntimeError(f"Gemini returned no text.{suffix}")
    return text


def _clean_candidate_list(values: Any, *, allowed_species: Optional[set[str]] = None, limit: int = 12) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str]] = set()
    for raw in values or []:
        if not isinstance(raw, dict):
            continue
        gene = str(raw.get("gene") or "").strip()
        species = str(raw.get("species") or "zebrafish").strip().lower()
        reason = str(raw.get("reason") or "").strip()[:600]
        accession = str(raw.get("uniprot_accession") or "").strip()[:40]
        evidence_types = []
        for item in raw.get("evidence_types") or []:
            value = str(item).strip()
            if value and value not in evidence_types:
                evidence_types.append(value[:80])
        if allowed_species is not None and species not in allowed_species:
            continue
        key = (species, gene.lower())
        if not gene or key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "gene": gene[:80],
                "species": species,
                "reason": reason,
                "uniprot_accession": accession,
                "evidence_types": evidence_types[:8],
            }
        )
        if len(out) >= limit:
            break
    return out


def _all_strings(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        out: List[str] = []
        for item in value.values():
            out.extend(_all_strings(item))
        return out
    if isinstance(value, list):
        out = []
        for item in value:
            out.extend(_all_strings(item))
        return out
    return []


def _primary_gene_name(uniprot_record: Dict[str, Any]) -> str:
    for gene in uniprot_record.get("genes") or []:
        value = str(((gene or {}).get("geneName") or {}).get("value") or "").strip()
        if value:
            return value
    return ""


def _gene_synonyms(uniprot_record: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for gene in uniprot_record.get("genes") or []:
        for synonym in (gene or {}).get("synonyms") or []:
            value = str((synonym or {}).get("value") or "").strip()
            if value and value not in out:
                out.append(value)
    return out


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


def _comment_text(uniprot_record: Dict[str, Any], comment_types: set[str]) -> str:
    pieces: List[str] = []
    for comment in uniprot_record.get("comments") or []:
        if str((comment or {}).get("commentType") or "").upper() not in comment_types:
            continue
        pieces.extend(_all_strings(comment))
    return " ".join(dict.fromkeys(piece.strip() for piece in pieces if piece.strip()))


def _go_text(uniprot_record: Dict[str, Any], aspect: str) -> str:
    prefix = {"biological_process": "P:", "molecular_function": "F:", "cellular_component": "C:"}[aspect]
    terms: List[str] = []
    for xref in uniprot_record.get("uniProtKBCrossReferences") or []:
        if str((xref or {}).get("database") or "") != "GO":
            continue
        for prop in (xref or {}).get("properties") or []:
            value = str((prop or {}).get("value") or "").strip()
            if value.startswith(prefix):
                terms.append(value)
    return " | ".join(dict.fromkeys(terms))


def _term_tokens(term: str) -> List[str]:
    stop = {
        "protein", "proteins", "gene", "genes", "zebrafish", "danio", "rerio",
        "marker", "markers", "related", "associated", "involved", "role", "roles",
    }
    tokens = [token for token in re.findall(r"[a-z0-9]+", term.lower()) if len(token) >= 3 and token not in stop]
    return list(dict.fromkeys(tokens))


def _text_matches(tokens: Iterable[str], text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in tokens)


def uniprot_record_to_evidence(record: Dict[str, Any], term: str, search_rank: int) -> Dict[str, Any]:
    accession = str(record.get("primaryAccession") or "").strip()
    gene = _primary_gene_name(record)
    synonyms = _gene_synonyms(record)
    protein_name = _protein_description(record)
    function_text = _comment_text(record, {"FUNCTION"})
    pathway_text = _comment_text(record, {"PATHWAY"})
    expression_text = _comment_text(record, {"TISSUE SPECIFICITY", "DEVELOPMENTAL STAGE", "INDUCTION"})
    phenotype_text = _comment_text(record, {"DISRUPTION PHENOTYPE"})
    go_bp = _go_text(record, "biological_process")
    go_mf = _go_text(record, "molecular_function")
    go_cc = _go_text(record, "cellular_component")
    tokens = _term_tokens(term)

    evidence_types: List[str] = []
    matched_fields: List[str] = []
    name_text = " ".join([gene, *synonyms, protein_name])
    checks = [
        ("name_match", "name", name_text),
        ("function_annotation", "function", function_text),
        ("pathway_annotation", "pathway", pathway_text),
        ("expression_annotation", "expression", expression_text),
        ("phenotype_annotation", "phenotype", phenotype_text),
        ("go_biological_process", "go_biological_process", go_bp),
        ("go_molecular_function", "go_molecular_function", go_mf),
        ("go_cellular_component", "go_cellular_component", go_cc),
    ]
    for evidence_type, field, text in checks:
        if tokens and text and _text_matches(tokens, text):
            evidence_types.append(evidence_type)
            matched_fields.append(field)
    if not evidence_types:
        evidence_types.append("uniprot_text_search_hit")

    return {
        "source": "UniProtKB",
        "search_term": term,
        "search_rank": int(search_rank),
        "gene": gene,
        "gene_synonyms": synonyms[:8],
        "uniprot_accession": accession,
        "protein_name": protein_name,
        "evidence_types": evidence_types,
        "matched_fields": matched_fields,
        "annotations": {
            "function": function_text[:700],
            "pathway": pathway_text[:700],
            "expression": expression_text[:700],
            "phenotype": phenotype_text[:700],
            "go_biological_process": go_bp[:700],
            "go_molecular_function": go_mf[:700],
            "go_cellular_component": go_cc[:700],
        },
        "retrieval_note": "UniProt search rank is retrieval order, not a biological relevance score.",
    }


def fetch_uniprot_evidence(
    terms: Iterable[str],
    *,
    per_term: int = UNIPROT_RESULTS_PER_TERM,
    max_records: int = MAX_UNIPROT_EVIDENCE_RECORDS,
) -> List[Dict[str, Any]]:
    fields = ",".join(
        [
            "accession", "gene_primary", "gene_synonym", "protein_name",
            "cc_function", "cc_pathway", "cc_tissue_specificity",
            "cc_developmental_stage", "cc_disruption_phenotype",
            "go_p", "go_f", "go_c",
        ]
    )
    by_accession: Dict[str, Dict[str, Any]] = {}
    for term in terms:
        term = str(term).strip()
        if not term:
            continue
        query = f"(organism_id:{DANIO_RERIO_TAXON_ID}) AND ({term})"
        params = urlencode(
            {
                "query": query,
                "format": "json",
                "fields": fields,
                "size": str(per_term),
            }
        )
        payload = _http_json(f"{UNIPROT_SEARCH_URL}?{params}")
        for rank, record in enumerate(payload.get("results") or [], start=1):
            evidence = uniprot_record_to_evidence(record, term, rank)
            key = evidence.get("uniprot_accession") or f"{evidence.get('gene')}::{term}::{rank}"
            if key not in by_accession:
                by_accession[key] = evidence
            else:
                current = by_accession[key]
                current["search_rank"] = min(int(current["search_rank"]), rank)
                current_terms = current.setdefault("search_terms", [current.get("search_term")])
                if term not in current_terms:
                    current_terms.append(term)
                for evidence_type in evidence.get("evidence_types") or []:
                    if evidence_type not in current["evidence_types"]:
                        current["evidence_types"].append(evidence_type)
                for field in evidence.get("matched_fields") or []:
                    if field not in current["matched_fields"]:
                        current["matched_fields"].append(field)
            if len(by_accession) >= max_records:
                break
        if len(by_accession) >= max_records:
            break
    return list(by_accession.values())[:max_records]


def fetch_ensembl_evidence(symbols: Iterable[str]) -> List[Dict[str, Any]]:
    requested = []
    for symbol in symbols:
        value = str(symbol).strip()
        if value and value.lower() not in {x.lower() for x in requested}:
            requested.append(value)
        if len(requested) >= 40:
            break
    if not requested:
        return []

    payload = _http_json(
        f"{ENSEMBL_REST_URL}/lookup/symbol/danio_rerio",
        headers={"Content-Type": "application/json"},
        data=json.dumps({"symbols": requested}).encode("utf-8"),
    )
    out: List[Dict[str, Any]] = []
    resolved_requested: set[str] = set()
    if isinstance(payload, dict):
        for requested_symbol, record in payload.items():
            if not isinstance(record, dict):
                continue
            canonical = str(record.get("display_name") or requested_symbol).strip()
            out.append(
                {
                    "source": "Ensembl",
                    "requested_symbol": requested_symbol,
                    "canonical_symbol": canonical,
                    "ensembl_id": str(record.get("id") or "").strip(),
                    "description": str(record.get("description") or "").strip()[:700],
                    "biotype": str(record.get("biotype") or "").strip(),
                    "evidence_types": ["identifier_resolution"],
                }
            )
            resolved_requested.add(requested_symbol.lower())

    for symbol in requested:
        if symbol.lower() in resolved_requested:
            continue
        try:
            params = urlencode({"object_type": "gene"})
            xrefs = _http_json(
                f"{ENSEMBL_REST_URL}/xrefs/symbol/danio_rerio/{quote(symbol)}?{params}",
                headers={"Content-Type": "application/json"},
            )
        except Exception:
            continue
        for xref in (xrefs or [])[:3]:
            ensembl_id = str((xref or {}).get("id") or "").strip()
            if not ensembl_id:
                continue
            try:
                lookup = _http_json(
                    f"{ENSEMBL_REST_URL}/lookup/id/{quote(ensembl_id)}?content-type=application/json",
                    headers={"Content-Type": "application/json"},
                )
            except Exception:
                lookup = {}
            canonical = str((lookup or {}).get("display_name") or (xref or {}).get("display_id") or symbol).strip()
            out.append(
                {
                    "source": "Ensembl",
                    "requested_symbol": symbol,
                    "canonical_symbol": canonical,
                    "ensembl_id": ensembl_id,
                    "description": str((lookup or {}).get("description") or "").strip()[:700],
                    "biotype": str((lookup or {}).get("biotype") or "").strip(),
                    "evidence_types": ["identifier_resolution", "symbol_or_synonym_resolution"],
                }
            )
            break
    return out


def _retrieval_plan(question: str) -> Dict[str, Any]:
    fallback = {
        "normalized_question": question,
        "question_type": "other",
        "retrieval_terms": [question],
        "evidence_priorities": DEFAULT_EVIDENCE_PRIORITIES,
    }
    prompt = f"""Classify this biological request for a DANIO RERIO evidence-retrieval system.
Question: {question!r}

Do not answer the biological question and do not nominate genes. Return only JSON:
{{
  "normalized_question": "zebrafish-specific restatement",
  "question_type": "one of cell_type, tissue, biological_process, pathway, phenotype, molecular_function, protein_family, other",
  "retrieval_terms": ["2 to 5 concise concepts suitable for zebrafish UniProt/database search"],
  "evidence_priorities": ["3 to 5 evidence types that should matter most for this question type"]
}}

Use evidence appropriate to the actual question. For example, cell/tissue questions should emphasize marker/expression evidence; pathway/process questions should emphasize curated pathway, GO process and functional evidence; molecular-function questions should emphasize GO molecular function and functional annotation; phenotype questions should emphasize genetic/phenotype evidence. A lexical protein-name match is weak evidence and must never be treated as proof of expression, pathway membership, phenotype, or function.
"""
    try:
        data = _parse_json_object(_gemini_text(prompt))
    except Exception:
        return fallback

    question_type = str(data.get("question_type") or "other").strip().lower()
    if question_type not in QUESTION_TYPES:
        question_type = "other"
    terms: List[str] = []
    for raw in data.get("retrieval_terms") or []:
        value = str(raw).strip()
        if value and value.lower() not in {x.lower() for x in terms}:
            terms.append(value[:120])
    if not terms:
        terms = [question]
    priorities = []
    for raw in data.get("evidence_priorities") or []:
        value = str(raw).strip()
        if value and value not in priorities:
            priorities.append(value[:180])
    return {
        "normalized_question": str(data.get("normalized_question") or question).strip()[:240],
        "question_type": question_type,
        "retrieval_terms": terms[:5],
        "evidence_priorities": priorities[:5] or DEFAULT_EVIDENCE_PRIORITIES,
    }


def _compact_uniprot_evidence(records: List[Dict[str, Any]], limit: int = 24) -> List[Dict[str, Any]]:
    return [
        {
            "gene": record.get("gene"),
            "uniprot_accession": record.get("uniprot_accession"),
            "protein_name": record.get("protein_name"),
            "search_term": record.get("search_term"),
            "search_rank": record.get("search_rank"),
            "evidence_types": record.get("evidence_types"),
            "matched_fields": record.get("matched_fields"),
            "annotations": record.get("annotations"),
        }
        for record in records[:limit]
    ]


def _ground_evidence(
    question: str,
    retrieval_plan: Dict[str, Any],
    uniprot_evidence: List[Dict[str, Any]],
    ensembl_evidence: List[Dict[str, Any]],
) -> str:
    compact_uniprot = _compact_uniprot_evidence(uniprot_evidence, 20)
    compact_ensembl = ensembl_evidence[:20]
    prompt = f"""Research this DANIO RERIO biological question using Google Search: {question!r}

Question type: {retrieval_plan.get('question_type')}
Evidence priorities: {json.dumps(retrieval_plan.get('evidence_priorities'), ensure_ascii=False)}

Structured UniProtKB retrieval results:
{json.dumps(compact_uniprot, ensure_ascii=False)}

Structured Ensembl identifier evidence:
{json.dumps(compact_ensembl, ensure_ascii=False)}

Use independent zebrafish evidence appropriate to the question type: ZFIN, zebrafish papers, expression/single-cell resources, curated pathway or GO resources, phenotype/genetic studies, and other relevant primary or authoritative sources.

Important reasoning rule: UniProt search order is not biological ranking. A result can rank highly simply because the query word occurs in its protein name. Treat `name_match` as lexical evidence only. It is NOT evidence that a protein is highly expressed, cell-type specific, a pathway member, functionally important, or phenotype-causing unless independent annotations or evidence support that conclusion.

For pathways/processes/functions, look for direct curated/functional/GO or experimental support. For cell types/tissues, look for marker/expression/enrichment support. For phenotypes, look for genetic/phenotype evidence. You may identify additional zebrafish genes not present in the structured retrieval if independent evidence supports them. If zebrafish evidence is sparse, clearly separate human/mouse reference evidence.

Return a concise plain-text evidence note, not JSON. Include exact zebrafish gene symbols when supported and explain the evidence type.
"""
    return _gemini_text(prompt, use_google_search=True)


def _rank_evidence(
    question: str,
    retrieval_plan: Dict[str, Any],
    uniprot_evidence: List[Dict[str, Any]],
    ensembl_evidence: List[Dict[str, Any]],
    grounded_note: str,
) -> Dict[str, Any]:
    prompt = f"""Rank candidate proteins for this DANIO RERIO biological question using the evidence provided.
Question: {question!r}
Question type: {retrieval_plan.get('question_type')}
Evidence priorities: {json.dumps(retrieval_plan.get('evidence_priorities'), ensure_ascii=False)}
Retrieval terms: {json.dumps(retrieval_plan.get('retrieval_terms'), ensure_ascii=False)}

UniProtKB structured evidence:
{json.dumps(_compact_uniprot_evidence(uniprot_evidence, 24), ensure_ascii=False)}

Ensembl identifier evidence:
{json.dumps(ensembl_evidence[:24], ensure_ascii=False)}

Independent grounded evidence note:
---
{grounded_note[:12000]}
---

Reason over evidence types, not database result order. A `name_match` alone is weak lexical evidence. Never convert a name match into an expression, marker, pathway, phenotype, or functional claim. Weight evidence according to the question type and evidence priorities above. Prefer direct zebrafish evidence over indirect association. Human/mouse genes belong only in reference_candidates and will be mapped deterministically to zebrafish orthologues later.

Return only JSON:
{{
  "zebrafish_candidates": [
    {{
      "gene": "exact zebrafish gene symbol",
      "species": "zebrafish",
      "uniprot_accession": "accession if supported by the supplied evidence, otherwise empty string",
      "evidence_types": ["specific evidence labels such as marker_support, expression_support, go_biological_process, pathway_support, functional_evidence, phenotype_support, literature_support, name_match"],
      "reason": "brief explanation of why this candidate fits this specific question"
    }}
  ],
  "reference_candidates": [
    {{"gene": "human or mouse gene", "species": "human or mouse", "evidence_types": ["reference evidence type"], "reason": "why reference evidence is needed"}}
  ],
  "rationale": "one or two sentences explaining the ranking logic"
}}

Return up to 12 zebrafish candidates in strongest-evidence-first order. Fewer is better than filling the list with weak lexical or indirect matches.
"""
    return _parse_json_object(_gemini_text(prompt))


def interpret_biological_query(question: str) -> Dict[str, Any]:
    """Retrieve structured evidence first, then let AI reason over evidence types."""
    question = question.strip()
    fallback = {
        "normalized_question": question,
        "question_type": "other",
        "retrieval_terms": [question],
        "evidence_priorities": DEFAULT_EVIDENCE_PRIORITIES,
        "zebrafish_candidates": [],
        "reference_candidates": [],
        "rationale": "Direct zebrafish biological keyword retrieval (AI interpreter unavailable).",
        "ai_used": False,
        "search_grounded": False,
        "evidence_summary": {"sources": [], "uniprot_records": 0, "ensembl_records": 0},
        "_uniprot_evidence": [],
        "_ensembl_evidence": [],
        "_retrieval_errors": [],
    }
    if not ai_available():
        return fallback

    retrieval_plan = _retrieval_plan(question)
    errors: List[str] = []
    try:
        uniprot_evidence = fetch_uniprot_evidence(retrieval_plan["retrieval_terms"])
    except Exception as exc:
        uniprot_evidence = []
        errors.append(f"UniProt: {exc}")

    try:
        initial_symbols = [record.get("gene") for record in uniprot_evidence if record.get("gene")]
        ensembl_evidence = fetch_ensembl_evidence(initial_symbols)
    except Exception as exc:
        ensembl_evidence = []
        errors.append(f"Ensembl: {exc}")

    try:
        grounded_note = _ground_evidence(question, retrieval_plan, uniprot_evidence, ensembl_evidence)
        ranked = _rank_evidence(question, retrieval_plan, uniprot_evidence, ensembl_evidence, grounded_note)
    except Exception as exc:
        fallback.update(retrieval_plan)
        fallback["rationale"] = f"AI evidence ranking unavailable; structured zebrafish evidence retained for deterministic fallback. ({exc})"
        fallback["_uniprot_evidence"] = uniprot_evidence
        fallback["_ensembl_evidence"] = ensembl_evidence
        fallback["_retrieval_errors"] = errors
        fallback["evidence_summary"] = {
            "sources": [source for source, present in (("UniProtKB", bool(uniprot_evidence)), ("Ensembl", bool(ensembl_evidence))) if present],
            "uniprot_records": len(uniprot_evidence),
            "ensembl_records": len(ensembl_evidence),
        }
        return fallback

    zebrafish_candidates = _clean_candidate_list(
        ranked.get("zebrafish_candidates"), allowed_species={"zebrafish", "danio rerio"}, limit=MAX_DISCOVERY_SEEDS
    )
    reference_candidates = _clean_candidate_list(
        ranked.get("reference_candidates"), allowed_species={"human", "mouse"}, limit=6
    )

    try:
        candidate_symbols = [candidate["gene"] for candidate in zebrafish_candidates]
        candidate_ensembl = fetch_ensembl_evidence(candidate_symbols)
        existing_keys = {(item.get("requested_symbol"), item.get("canonical_symbol"), item.get("ensembl_id")) for item in ensembl_evidence}
        for item in candidate_ensembl:
            key = (item.get("requested_symbol"), item.get("canonical_symbol"), item.get("ensembl_id"))
            if key not in existing_keys:
                ensembl_evidence.append(item)
                existing_keys.add(key)
    except Exception as exc:
        errors.append(f"Ensembl candidate resolution: {exc}")

    sources = ["Gemini Google Search"]
    if uniprot_evidence:
        sources.insert(0, "UniProtKB")
    if ensembl_evidence:
        sources.insert(1 if uniprot_evidence else 0, "Ensembl")

    return {
        **retrieval_plan,
        "zebrafish_candidates": zebrafish_candidates,
        "reference_candidates": reference_candidates,
        "rationale": str(ranked.get("rationale") or "Evidence-ranked zebrafish discovery plan.").strip()[:800],
        "ai_used": True,
        "search_grounded": True,
        "evidence_summary": {
            "sources": sources,
            "uniprot_records": len(uniprot_evidence),
            "ensembl_records": len(ensembl_evidence),
            "ranking_policy": "Evidence-type aware; database search order is not biological rank.",
        },
        "_uniprot_evidence": uniprot_evidence,
        "_ensembl_evidence": ensembl_evidence,
        "_retrieval_errors": errors,
    }


def _ensembl_canonical_candidates(gene: str, ensembl_evidence: Iterable[Dict[str, Any]]) -> List[str]:
    key = gene.strip().lower()
    out: List[str] = []
    for item in ensembl_evidence:
        requested = str(item.get("requested_symbol") or "").strip().lower()
        canonical = str(item.get("canonical_symbol") or "").strip()
        if requested == key and canonical and canonical.lower() not in {x.lower() for x in out}:
            out.append(canonical)
        if canonical.lower() == key and canonical and canonical.lower() not in {x.lower() for x in out}:
            out.append(canonical)
    return out


def validate_ai_zebrafish_candidates(
    candidates: Iterable[Dict[str, Any]],
    uniprot_evidence: Optional[Iterable[Dict[str, Any]]] = None,
    ensembl_evidence: Optional[Iterable[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """AI ranks evidence; only deterministic local identities become ESM seeds."""
    uniprot_records = list(uniprot_evidence or [])
    ensembl_records = list(ensembl_evidence or [])
    by_gene = {str(r.get("gene") or "").lower(): r for r in uniprot_records if r.get("gene")}
    by_accession = {str(r.get("uniprot_accession") or "").lower(): r for r in uniprot_records if r.get("uniprot_accession")}

    seeds: List[Dict[str, Any]] = []
    seen: set[int] = set()
    for candidate in candidates:
        gene = str(candidate.get("gene") or "").strip()
        accession = str(candidate.get("uniprot_accession") or "").strip()
        idx: Optional[int] = None
        resolved_by = ""
        evidence_record: Optional[Dict[str, Any]] = None

        if gene:
            idx = resolve_exact_identifier(gene)
            if idx is not None:
                resolved_by = "exact local zebrafish gene"
                evidence_record = by_gene.get(gene.lower())

        if idx is None and accession:
            idx = resolve_uniprot_accession(accession)
            if idx is not None:
                resolved_by = "exact local UniProt accession"
                evidence_record = by_accession.get(accession.lower())

        if idx is None and gene.lower() in by_gene:
            record = by_gene[gene.lower()]
            for label, identifier in (
                ("UniProt evidence gene", str(record.get("gene") or "")),
                ("UniProt evidence accession", str(record.get("uniprot_accession") or "")),
            ):
                idx = resolve_exact_identifier(identifier) if "gene" in label else resolve_uniprot_accession(identifier)
                if idx is not None:
                    resolved_by = label
                    evidence_record = record
                    break

        if idx is None and gene:
            for canonical in _ensembl_canonical_candidates(gene, ensembl_records):
                idx = resolve_exact_identifier(canonical)
                if idx is not None:
                    resolved_by = f"Ensembl symbol resolution: {gene} → {canonical}"
                    break

        if idx is None or idx in seen:
            continue
        seen.add(idx)
        evidence_types = list(candidate.get("evidence_types") or [])
        if not evidence_types and evidence_record:
            evidence_types = list(evidence_record.get("evidence_types") or [])
        evidence_class = " + ".join(str(item) for item in evidence_types[:5]) or "zebrafish-supported"
        seeds.append(
            {
                "index": idx,
                "source": "Evidence-ranked zebrafish candidate",
                "retrieval_term": gene or accession,
                "resolved_by": resolved_by,
                "uniprot_accession": accession or (evidence_record or {}).get("uniprot_accession"),
                "evidence_class": evidence_class,
                "evidence_types": evidence_types,
                "ai_reason": candidate.get("reason") or "",
            }
        )
    return seeds


def evidence_fallback_seeds(uniprot_evidence: Iterable[Dict[str, Any]], *, limit: int = 6) -> List[Dict[str, Any]]:
    weights = {
        "expression_annotation": 5,
        "phenotype_annotation": 5,
        "pathway_annotation": 4,
        "function_annotation": 4,
        "go_biological_process": 4,
        "go_molecular_function": 4,
        "go_cellular_component": 2,
        "name_match": 1,
        "uniprot_text_search_hit": 0,
    }
    scored: List[Tuple[int, int, Dict[str, Any]]] = []
    for record in uniprot_evidence:
        score = sum(weights.get(str(item), 0) for item in record.get("evidence_types") or [])
        rank = int(record.get("search_rank") or 999)
        scored.append((score, -rank, record))
    scored.sort(reverse=True, key=lambda item: (item[0], item[1]))

    out: List[Dict[str, Any]] = []
    seen: set[int] = set()
    for score, _, record in scored:
        gene = str(record.get("gene") or "").strip()
        accession = str(record.get("uniprot_accession") or "").strip()
        idx = resolve_exact_identifier(gene)
        if idx is None:
            idx = resolve_uniprot_accession(accession)
        if idx is None or idx in seen:
            continue
        seen.add(idx)
        out.append(
            {
                "index": idx,
                "source": "UniProt evidence fallback",
                "retrieval_term": record.get("search_term"),
                "uniprot_accession": accession,
                "resolved_by": "deterministic local evidence fallback",
                "evidence_class": " + ".join(record.get("evidence_types") or []) or "uniprot retrieval",
                "evidence_types": record.get("evidence_types") or [],
                "ai_reason": f"Fallback evidence score {score}; UniProt rank not treated as biological rank.",
            }
        )
        if len(out) >= limit:
            break
    return out


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


def orthology_seeds(reference_candidates: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
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


def _public_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in plan.items() if not key.startswith("_")}


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
                "evidence_types": seed.get("evidence_types"),
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
    public_plan = _public_plan(plan)
    prompt = f"""Explain a DANIO RERIO zebrafish ESM protein-similarity result.
Question: {question}
Plan: {json.dumps(public_plan, ensure_ascii=False)}
Validated zebrafish seeds: {json.dumps(compact_seeds, ensure_ascii=False)}
Top ESM candidates: {json.dumps(compact_results, ensure_ascii=False)}
Return only JSON: {{"summary": "2-4 concise sentences"}}.
Keep the interpretation zebrafish-specific. Distinguish evidence types. Do not infer expression, pathway membership, phenotype, or function from a lexical name match. Do not claim functional proof from ESM similarity alone and do not invent annotations.
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
    uniprot_evidence = list(plan.get("_uniprot_evidence") or [])
    ensembl_evidence = list(plan.get("_ensembl_evidence") or [])
    direct_ai_seeds = validate_ai_zebrafish_candidates(
        plan.get("zebrafish_candidates") or [], uniprot_evidence, ensembl_evidence
    )

    mapped_reference_seeds: List[Dict[str, Any]] = []
    if len(direct_ai_seeds) < MIN_ZEBRAFISH_SEEDS_BEFORE_ORTHOLOGY and plan.get("reference_candidates"):
        mapped_reference_seeds = orthology_seeds(plan["reference_candidates"])

    seeds = _merge_seeds(direct_ai_seeds, mapped_reference_seeds)
    if not seeds and uniprot_evidence:
        seeds = evidence_fallback_seeds(uniprot_evidence)
    if not seeds:
        seeds = local_annotation_seeds(plan["retrieval_terms"])

    retrieval_errors = list(plan.get("_retrieval_errors") or [])
    retrieval_warning = "; ".join(retrieval_errors) if retrieval_errors else None
    if not seeds:
        message = "No validated zebrafish seed proteins could be resolved into the local ESM database."
        if retrieval_warning:
            message += f" Retrieval error: {retrieval_warning}"
        return {
            "ok": False,
            "mode": "discovery",
            "query": question,
            "plan": _public_plan(plan),
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
                "evidence_types": seed.get("evidence_types"),
                "reference_species": seed.get("reference_species"),
                "reference_gene": seed.get("reference_gene"),
                "ai_reason": seed.get("ai_reason"),
            }
        )

    source_names = list(dict.fromkeys(str(seed.get("source")) for seed in seeds if seed.get("source")))
    public_plan = _public_plan(plan)
    return {
        "ok": True,
        "mode": "discovery",
        "query": question,
        "plan": public_plan,
        "seed_source": "Evidence-ranked zebrafish: " + "; ".join(source_names),
        "retrieval_warning": retrieval_warning,
        "seeds": public_seeds,
        "results": results,
        "ai_explanation": explain_discovery(question, plan, seeds, results),
        "privacy": "Embeddings remain server-side. Gemini receives the biological question and compact biological evidence, never embedding vectors or database credentials.",
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
        "structured_evidence_sources": ["UniProtKB", "Ensembl"],
        "embedding_egress": False,
    }


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "ZebrafishESMDashboard/2.3"

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
    print("Biological discovery: evidence-ranked Danio rerio retrieval (UniProtKB + Ensembl + Google Search; mammalian orthology fallback only when needed)")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
