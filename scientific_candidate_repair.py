"""Deterministic zebrafish nomenclature repair and QuickGO annotation expansion.

Experimental tool layer for the hard-25 benchmark:
- ZFIN Previous Names -> current zebrafish symbols
- ZFIN curated human/mouse orthology -> zebrafish symbols
- ZFIN UniProt associations -> current zebrafish symbols
- QuickGO annotation search -> actual Danio rerio gene products for retrieved GO terms

The layer is fail-open: network/cache failures leave the original model plan unchanged.
"""
from __future__ import annotations

import copy
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = ROOT / "data" / "zfin_cache"
CACHE_MAX_AGE_SECONDS = 24 * 60 * 60

ZFIN_FILES = {
    "aliases": ("https://zfin.org/downloads/aliases.txt", "aliases.txt"),
    "human_orthos": ("https://zfin.org/downloads/human_orthos.txt", "human_orthos.txt"),
    "mouse_orthos": ("https://zfin.org/downloads/mouse_orthos.txt", "mouse_orthos.txt"),
    "uniprot": ("https://zfin.org/downloads/uniprot.txt", "uniprot.txt"),
}
QUICKGO_ANNOTATION_URL = "https://www.ebi.ac.uk/QuickGO/services/annotation/search"


def _cache_dir() -> Path:
    value = os.environ.get("ZFIN_CACHE_DIR", "").strip()
    return Path(value) if value else DEFAULT_CACHE_DIR


def _download_text(url: str, timeout: int = 60) -> str:
    req = Request(
        url,
        headers={
            "Accept": "text/plain,*/*",
            "User-Agent": "zebrafish-esm-search/3.0",
        },
    )
    with urlopen(req, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def _load_cached_text(key: str) -> str:
    url, filename = ZFIN_FILES[key]
    cache_dir = _cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / filename
    fresh = path.exists() and (time.time() - path.stat().st_mtime) < CACHE_MAX_AGE_SECONDS
    if fresh:
        return path.read_text(encoding="utf-8", errors="replace")

    try:
        text = _download_text(url)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
        return text
    except Exception:
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace")
        raise


def _iter_tsv(text: str) -> Iterable[List[str]]:
    for raw in text.splitlines():
        line = raw.strip("\n\r")
        if not line or line.startswith("#"):
            continue
        cols = line.split("\t")
        if cols:
            yield cols


class ZfinLookup:
    def __init__(self) -> None:
        self.loaded = False
        self.aliases: Dict[str, List[str]] = {}
        self.human: Dict[str, List[str]] = {}
        self.mouse: Dict[str, List[str]] = {}
        self.uniprot: Dict[str, str] = {}

    @staticmethod
    def _append(mapping: Dict[str, List[str]], key: str, value: str) -> None:
        k, v = key.strip().lower(), value.strip()
        if not k or not v:
            return
        bucket = mapping.setdefault(k, [])
        if v not in bucket:
            bucket.append(v)

    def load(self) -> None:
        if self.loaded:
            return

        aliases: Dict[str, List[str]] = {}
        for cols in _iter_tsv(_load_cached_text("aliases")):
            if len(cols) < 4:
                continue
            zfin_id, _current_name, current_symbol, previous_name = cols[:4]
            if not zfin_id.startswith("ZDB-GENE-") or not current_symbol:
                continue
            self._append(aliases, current_symbol, current_symbol)
            self._append(aliases, previous_name, current_symbol)

        human: Dict[str, List[str]] = {}
        for cols in _iter_tsv(_load_cached_text("human_orthos")):
            if len(cols) < 4:
                continue
            zfin_id, zfin_symbol, _zfin_name, human_symbol = cols[:4]
            if zfin_id.startswith("ZDB-GENE-"):
                self._append(human, human_symbol, zfin_symbol)

        mouse: Dict[str, List[str]] = {}
        for cols in _iter_tsv(_load_cached_text("mouse_orthos")):
            if len(cols) < 4:
                continue
            zfin_id, zfin_symbol, _zfin_name, mouse_symbol = cols[:4]
            if zfin_id.startswith("ZDB-GENE-"):
                self._append(mouse, mouse_symbol, zfin_symbol)

        uniprot: Dict[str, str] = {}
        for cols in _iter_tsv(_load_cached_text("uniprot")):
            if len(cols) < 4:
                continue
            zfin_id, _so_id, symbol, accession = cols[:4]
            if zfin_id.startswith("ZDB-GENE-") and symbol and accession:
                uniprot.setdefault(accession.strip().lower(), symbol.strip())

        self.aliases = aliases
        self.human = human
        self.mouse = mouse
        self.uniprot = uniprot
        self.loaded = True

    def normalize(self, term: str) -> List[Tuple[str, str]]:
        """Return current zebrafish symbols for an alias/mammalian symbol."""
        self.load()
        key = term.strip().lower()
        if not key:
            return []

        if key in self.aliases:
            return [(symbol, "ZFIN previous/current-name mapping") for symbol in self.aliases[key]]
        if key in self.human:
            return [(symbol, "ZFIN curated human orthology") for symbol in self.human[key]]
        if key in self.mouse:
            return [(symbol, "ZFIN curated mouse orthology") for symbol in self.mouse[key]]
        return []

    def orthologs(self, gene: str, species: str) -> List[str]:
        self.load()
        key = gene.strip().lower()
        species = species.strip().lower()
        mapping = self.human if species == "human" else self.mouse if species == "mouse" else {}
        return list(mapping.get(key, []))

    def symbol_for_uniprot(self, gene_product_id: str) -> str:
        self.load()
        key = gene_product_id.strip().split(":")[-1].lower()
        return self.uniprot.get(key, "")


_LOOKUP = ZfinLookup()


def normalize_plan(
    plan: Dict[str, Any],
    *,
    max_seeds: int = 10,
    max_refs: int = 6,
    lookup: ZfinLookup | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Normalize model candidates and add deterministic ZFIN-curated orthologs."""
    lookup = lookup or _LOOKUP
    out = copy.deepcopy(plan)
    zf_out: List[Dict[str, str]] = []
    seen = set()
    trace: List[Dict[str, Any]] = []
    errors: List[str] = []

    def add_candidate(gene: str, reason: str, original: str, source: str) -> None:
        key = gene.strip().lower()
        if not key or key in seen or len(zf_out) >= max_seeds:
            return
        seen.add(key)
        zf_out.append(
            {
                "gene": gene.strip(),
                "species": "zebrafish",
                "uniprot_accession": "",
                "reason": reason[:240],
            }
        )
        trace.append(
            {
                "input": original,
                "normalized": gene.strip(),
                "source": source,
            }
        )

    try:
        for candidate in plan.get("zebrafish_candidates") or []:
            original = str(candidate.get("gene") or "").strip()
            if not original:
                continue
            mappings = lookup.normalize(original)
            if mappings:
                for symbol, source in mappings:
                    reason = str(candidate.get("reason") or "")
                    if symbol.lower() != original.lower():
                        reason = f"{reason} [normalized by {source}: {original} -> {symbol}]".strip()
                    add_candidate(symbol, reason, original, source)
            else:
                add_candidate(
                    original,
                    str(candidate.get("reason") or ""),
                    original,
                    "unchanged; no ZFIN repair",
                )

        refs = list(plan.get("reference_candidates") or [])[:max_refs]
        for candidate in refs:
            gene = str(candidate.get("gene") or "").strip()
            species = str(candidate.get("species") or "").strip().lower()
            if not gene or species not in {"human", "mouse"}:
                continue
            for symbol in lookup.orthologs(gene, species):
                add_candidate(
                    symbol,
                    f"ZFIN curated ortholog of {species} {gene}",
                    gene,
                    f"ZFIN curated {species} orthology",
                )
    except Exception as exc:
        errors.append(f"ZFIN normalization: {exc}")
        return copy.deepcopy(plan), {"repairs": [], "errors": errors}

    out["zebrafish_candidates"] = zf_out
    out["reference_candidates"] = list(plan.get("reference_candidates") or [])[:max_refs]
    return out, {"repairs": trace, "errors": errors}


def _annotation_symbol(item: Dict[str, Any], lookup: ZfinLookup) -> str:
    symbol = str(item.get("symbol") or "").strip()
    if symbol:
        mappings = lookup.normalize(symbol)
        return mappings[0][0] if mappings else symbol
    return lookup.symbol_for_uniprot(str(item.get("geneProductId") or ""))


def quickgo_annotation_evidence(
    term_evidence: Iterable[Dict[str, Any]],
    http_json: Callable[..., Dict[str, Any]],
    *,
    lookup: ZfinLookup | None = None,
    max_terms: int = 5,
    max_symbols_per_term: int = 25,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Retrieve Danio rerio gene products annotated to GO terms already found by QuickGO."""
    lookup = lookup or _LOOKUP
    terms: List[Tuple[str, str]] = []
    seen_terms = set()
    for item in term_evidence:
        if str(item.get("source") or "") != "QuickGO":
            continue
        go_id = str(item.get("go_id") or "").strip()
        if not go_id.startswith("GO:") or go_id in seen_terms:
            continue
        seen_terms.add(go_id)
        terms.append((go_id, str(item.get("name") or "")))
        if len(terms) >= max_terms:
            break

    out: List[Dict[str, Any]] = []
    errors: List[str] = []
    for go_id, go_name in terms:
        params = urlencode(
            {
                "goId": go_id,
                "taxonId": "7955",
                "goUsage": "descendants",
                "goUsageRelationships": "is_a,part_of",
                "includeFields": "goName,name,synonyms",
                "limit": "100",
                "page": "1",
            }
        )
        try:
            payload = http_json(
                f"{QUICKGO_ANNOTATION_URL}?{params}",
                headers={"Accept": "application/json"},
            )
            symbols: List[str] = []
            seen = set()
            for item in payload.get("results") or []:
                if not isinstance(item, dict):
                    continue
                symbol = _annotation_symbol(item, lookup)
                key = symbol.lower()
                if symbol and key not in seen:
                    seen.add(key)
                    symbols.append(symbol)
                if len(symbols) >= max_symbols_per_term:
                    break
            if symbols:
                out.append(
                    {
                        "source": "QuickGO",
                        "go_id": go_id,
                        "name": f"{go_name or go_id} — Danio rerio annotated gene products",
                        "definition": "Annotated zebrafish gene products: " + ", ".join(symbols),
                        "annotation_candidates": symbols,
                    }
                )
        except Exception as exc:
            errors.append(f"QuickGO annotations ({go_id}): {exc}")
    return out, errors
