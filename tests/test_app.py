import json
import os
import unittest
from unittest.mock import patch

import numpy as np

import app


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.old_key = os.environ.pop("GEMINI_API_KEY", None)
        app.PROTEINS = [
            {"protein_id": "P1", "name": "gata1a", "description": "erythroid transcription factor", "sequence": "", "extra_json": "{}"},
            {"protein_id": "tr|Q7SXE0|Q7SXE0_DANRE", "name": "mpeg1.1", "description": "Macrophage-expressed gene 1 protein", "sequence": "", "extra_json": "{}"},
            {"protein_id": "P3", "name": "wnt8a", "description": "Wnt family member 8a", "sequence": "", "extra_json": "{}"},
            {"protein_id": "P4", "name": "gene4", "description": "related protein", "sequence": "", "extra_json": "{}"},
        ]
        app.ID_TO_INDEX = {p["protein_id"].lower(): i for i, p in enumerate(app.PROTEINS)}
        app.NAME_TO_INDEX = {p["name"].lower(): i for i, p in enumerate(app.PROTEINS)}
        app.SEARCH_TEXTS = [" ".join([p["protein_id"], p["name"], p["description"]]).lower() for p in app.PROTEINS]
        app.VECTORS = np.array([[1, 0], [0, 1], [0.2, 0.98], [0.9, 0.1]], dtype=np.float32)
        app.VECTORS /= np.linalg.norm(app.VECTORS, axis=1, keepdims=True)

    def tearDown(self):
        if self.old_key is None:
            os.environ.pop("GEMINI_API_KEY", None)
        else:
            os.environ["GEMINI_API_KEY"] = self.old_key

    def test_exact_lookup_stays_deterministic(self):
        result = app.search_api({"q": ["mpeg1.1"], "k": ["2"]})
        self.assertTrue(result["ok"])
        self.assertEqual(result["matched_protein"]["name"], "mpeg1.1")
        self.assertEqual(result["match_method"], "exact gene name")

    def test_no_key_fallback_is_simple(self):
        plan = app.interpret_biological_query("macrophage proteins")
        self.assertFalse(plan["ai_used"])
        self.assertFalse(plan["search_grounded"])
        self.assertEqual(plan["retrieval_terms"], [])
        self.assertNotIn("question_type", plan)

    @patch("app._http_json")
    def test_grounded_gemini_payload_and_metadata(self, mock_http):
        os.environ["GEMINI_API_KEY"] = "test-key"
        mock_http.return_value = {
            "candidates": [{
                "content": {"parts": [{"text": "research"}]},
                "groundingMetadata": {"webSearchQueries": ["zebrafish macrophage markers"]},
            }]
        }
        text, grounding = app._gemini_response("x", use_google_search=True)
        payload = json.loads(mock_http.call_args.kwargs["data"].decode())
        self.assertEqual(text, "research")
        self.assertEqual(grounding["webSearchQueries"], ["zebrafish macrophage markers"])
        self.assertEqual(payload["tools"], [{"google_search": {}}])
        self.assertNotIn("responseMimeType", payload["generationConfig"])

    @patch("app._http_json")
    def test_targeted_uniprot_search_is_zebrafish(self, mock_http):
        mock_http.return_value = {
            "results": [{
                "primaryAccession": "Q7SXE0",
                "genes": [{"geneName": {"value": "mpeg1.1"}, "synonyms": [{"value": "mpeg1"}]}],
                "proteinDescription": {"recommendedName": {"fullName": {"value": "Macrophage-expressed gene 1 protein"}}},
            }]
        }
        rows = app.fetch_uniprot_candidates("mpeg1")
        self.assertEqual(rows[0]["gene"], "mpeg1.1")
        called_url = mock_http.call_args.args[0]
        self.assertIn("organism_id%3A7955", called_url)
        self.assertIn("mpeg1", called_url)
        self.assertNotIn("macrophage", called_url)

    @patch("app._gemini_text")
    @patch("app._gemini_response")
    @patch("app.fetch_uniprot_candidates")
    def test_ai_research_precedes_uniprot_resolution(self, mock_uniprot, mock_response, mock_text):
        os.environ["GEMINI_API_KEY"] = "test-key"
        mock_response.return_value = (
            "TOP ZEBRAFISH CANDIDATES\n1. mpeg1 — canonical marker\nSOURCES\nZFIN",
            {"webSearchQueries": ["zebrafish macrophage markers"]},
        )
        mock_text.return_value = json.dumps({
            "normalized_question": "zebrafish macrophage proteins",
            "zebrafish_candidates": [{"gene": "mpeg1", "species": "zebrafish", "reason": "direct zebrafish evidence"}],
            "reference_candidates": [],
            "rationale": "Grounded zebrafish research.",
        })
        plan = app.interpret_biological_query("macrophage proteins")
        self.assertEqual(plan["retrieval_terms"], ["mpeg1"])
        self.assertTrue(plan["search_grounded"])
        mock_uniprot.assert_not_called()
        research_prompt = mock_response.call_args.args[0]
        self.assertNotIn("UniProt search for", research_prompt)
        self.assertIn("Independently identify", research_prompt)

    @patch("app._gemini_response")
    def test_ungrounded_research_is_not_reported_as_grounded(self, mock_response):
        os.environ["GEMINI_API_KEY"] = "test-key"
        mock_response.return_value = ("ungrounded note", {})
        plan = app.interpret_biological_query("macrophage proteins")
        self.assertFalse(plan["ai_used"])
        self.assertFalse(plan["search_grounded"])
        self.assertIn("without Google Search grounding", plan["rationale"])

    def test_uniprot_accession_inside_pipe_id_resolves(self):
        self.assertEqual(app.resolve_uniprot_accession("Q7SXE0"), 1)

    @patch("app.fetch_uniprot_candidates")
    def test_targeted_synonym_resolves_and_unrelated_hits_are_rejected(self, mock_fetch):
        mock_fetch.return_value = [
            {"search_rank": 1, "gene": "mpeg1.1", "gene_synonyms": ["mpeg1"], "uniprot_accession": "Q7SXE0", "protein_name": "Macrophage-expressed gene 1 protein"},
            {"search_rank": 2, "gene": "gata1a", "gene_synonyms": [], "uniprot_accession": "P1", "protein_name": "GATA-binding factor 1"},
        ]
        candidates = [{"gene": "mpeg1", "species": "zebrafish", "reason": "canonical marker"}]
        seeds, traces, errors = app.resolve_targeted_uniprot_candidates(candidates)
        self.assertEqual(errors, [])
        self.assertEqual([app.PROTEINS[seed["index"]]["name"] for seed in seeds], ["mpeg1.1"])
        self.assertIn("synonym", seeds[0]["resolved_by"])
        self.assertTrue(traces[0]["records"][0]["accepted"])
        self.assertFalse(traces[0]["records"][1]["accepted"])
        self.assertIsNone(traces[0]["records"][1]["local_match"])

    @patch("app.fetch_uniprot_candidates")
    def test_no_arbitrary_uniprot_hit_becomes_a_seed(self, mock_fetch):
        mock_fetch.return_value = [
            {"search_rank": 1, "gene": "gata1a", "gene_synonyms": [], "uniprot_accession": "P1", "protein_name": "GATA-binding factor 1"}
        ]
        candidates = [{"gene": "unknown-marker", "species": "zebrafish", "reason": "research result"}]
        seeds, traces, _ = app.resolve_targeted_uniprot_candidates(candidates)
        self.assertEqual(seeds, [])
        self.assertFalse(traces[0]["records"][0]["accepted"])

    @patch("app._ensembl_zebrafish_ortholog_symbols", return_value=["wnt8a"])
    def test_mammalian_reference_uses_ensembl_before_local_validation(self, _):
        seeds = app.orthology_seeds([{"gene": "WNT8A", "species": "human", "reason": "fallback"}])
        self.assertEqual(len(seeds), 1)
        self.assertEqual(app.PROTEINS[seeds[0]["index"]]["name"], "wnt8a")

    def test_no_hand_built_question_type_or_evidence_taxonomy(self):
        self.assertFalse(hasattr(app, "QUESTION_TYPES"))
        self.assertFalse(hasattr(app, "DEFAULT_EVIDENCE_PRIORITIES"))
        self.assertFalse(hasattr(app, "uniprot_record_to_evidence"))

    def test_discovery_neighbors_use_best_seed_similarity(self):
        results = app.discovery_neighbors([1], 2)
        self.assertEqual(results[0]["name"], "wnt8a")
        self.assertEqual(results[0]["closest_seed"], "mpeg1.1")


if __name__ == "__main__":
    unittest.main()
