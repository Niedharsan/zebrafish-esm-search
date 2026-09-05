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

    def test_search_term_only_removes_generic_words(self):
        self.assertEqual(app._search_term("macrophage proteins"), "macrophage")
        self.assertEqual(app._search_term("Wnt signaling proteins in zebrafish"), "Wnt signaling in")

    @patch("app.fetch_uniprot_candidates", return_value=[])
    def test_no_key_fallback_is_simple(self, _):
        plan = app.interpret_biological_query("macrophage proteins")
        self.assertFalse(plan["ai_used"])
        self.assertEqual(plan["retrieval_terms"], ["macrophage"])
        self.assertNotIn("question_type", plan)

    @patch("app._http_json")
    def test_grounded_gemini_omits_json_mime(self, mock_http):
        os.environ["GEMINI_API_KEY"] = "test-key"
        mock_http.return_value = {"candidates": [{"content": {"parts": [{"text": "research"}]}}]}
        app._gemini_text("x", use_google_search=True)
        payload = json.loads(mock_http.call_args.kwargs["data"].decode())
        self.assertEqual(payload["tools"], [{"google_search": {}}])
        self.assertNotIn("responseMimeType", payload["generationConfig"])

    @patch("app._http_json")
    def test_uniprot_search_is_zebrafish_and_simple(self, mock_http):
        mock_http.return_value = {
            "results": [{
                "primaryAccession": "Q7SXE0",
                "genes": [{"geneName": {"value": "mpeg1.1"}, "synonyms": [{"value": "mpeg1"}]}],
                "proteinDescription": {"recommendedName": {"fullName": {"value": "Macrophage-expressed gene 1 protein"}}},
            }]
        }
        rows = app.fetch_uniprot_candidates("macrophage proteins")
        self.assertEqual(rows[0]["gene"], "mpeg1.1")
        called_url = mock_http.call_args.args[0]
        self.assertIn("organism_id%3A7955", called_url)
        self.assertIn("macrophage", called_url)
        self.assertNotIn("cc_function", called_url)

    @patch("app._gemini_text")
    @patch("app.fetch_uniprot_candidates")
    def test_ai_does_biological_ranking_without_question_classifier(self, mock_uniprot, mock_gemini):
        os.environ["GEMINI_API_KEY"] = "test-key"
        mock_uniprot.return_value = [
            {"search_rank": 1, "gene": "mpeg1.1", "gene_synonyms": ["mpeg1"], "uniprot_accession": "Q7SXE0", "protein_name": "Macrophage-expressed gene 1 protein"},
            {"search_rank": 2, "gene": "gata1a", "gene_synonyms": [], "uniprot_accession": "P1", "protein_name": "Gata1a"},
        ]
        mock_gemini.side_effect = [
            "TOP ZEBRAFISH CANDIDATES\n1. mpeg1.1 — canonical zebrafish macrophage-associated marker with independent zebrafish support.",
            json.dumps({
                "normalized_question": "zebrafish macrophage proteins",
                "zebrafish_candidates": [{"gene": "mpeg1.1", "species": "zebrafish", "uniprot_accession": "Q7SXE0", "reason": "direct zebrafish evidence"}],
                "reference_candidates": [],
                "rationale": "AI compared UniProt retrieval with independent zebrafish evidence.",
            }),
        ]
        plan = app.interpret_biological_query("macrophage proteins")
        self.assertEqual(plan["zebrafish_candidates"][0]["gene"], "mpeg1.1")
        self.assertEqual(mock_gemini.call_count, 2)
        research_prompt = mock_gemini.call_args_list[0].args[0]
        self.assertNotIn("question_type", research_prompt)
        self.assertIn("Do the biology yourself", research_prompt)
        self.assertIn("UniProt search position is NOT biological rank", research_prompt)

    def test_uniprot_accession_inside_pipe_id_resolves(self):
        self.assertEqual(app.resolve_uniprot_accession("Q7SXE0"), 1)

    def test_ai_candidate_can_resolve_via_uniprot_synonym(self):
        seeds = app.validate_ai_zebrafish_candidates(
            [{"gene": "mpeg1", "species": "zebrafish", "uniprot_accession": "", "reason": "marker"}],
            [{"gene": "mpeg1.1", "gene_synonyms": ["mpeg1"], "uniprot_accession": "Q7SXE0", "protein_name": "Macrophage-expressed gene 1 protein"}],
        )
        self.assertEqual(len(seeds), 1)
        self.assertEqual(seeds[0]["index"], 1)
        self.assertIn("UniProt name/synonym", seeds[0]["resolved_by"])

    @patch("app.interpret_biological_query")
    def test_discovery_uses_ai_ranked_seeds_without_uniprot_fillers(self, mock_plan):
        mock_plan.return_value = {
            "normalized_question": "zebrafish macrophage proteins",
            "retrieval_terms": ["macrophage"],
            "zebrafish_candidates": [{"gene": "mpeg1.1", "species": "zebrafish", "uniprot_accession": "Q7SXE0", "reason": "marker"}],
            "reference_candidates": [],
            "rationale": "AI research",
            "ai_used": True,
            "search_grounded": True,
            "evidence_summary": {},
            "_uniprot_candidates": [
                {"gene": "mpeg1.1", "gene_synonyms": ["mpeg1"], "uniprot_accession": "Q7SXE0", "protein_name": "Macrophage-expressed gene 1 protein"},
                {"gene": "gata1a", "gene_synonyms": [], "uniprot_accession": "P1", "protein_name": "Gata1a"},
            ],
            "_retrieval_errors": [],
        }
        result = app.discovery_api({"q": ["macrophage proteins"], "k": ["2"]})
        self.assertTrue(result["ok"])
        self.assertEqual([s["name"] for s in result["seeds"]], ["mpeg1.1"])

    @patch("app._gemini_text")
    @patch("app.fetch_uniprot_candidates", return_value=[])
    def test_broad_pathway_prompt_explicitly_preserves_scope(self, _, mock_gemini):
        os.environ["GEMINI_API_KEY"] = "test-key"
        mock_gemini.side_effect = [
            "TOP ZEBRAFISH CANDIDATES\n1. wnt8a — pathway ligand\n2. fzd7a — receptor\n3. ctnnb1 — intracellular transducer",
            json.dumps({
                "normalized_question": "zebrafish Wnt signaling proteins",
                "zebrafish_candidates": [{"gene": "wnt8a", "species": "zebrafish", "reason": "pathway member"}],
                "reference_candidates": [],
                "rationale": "broad pathway scope",
            }),
        ]
        app.interpret_biological_query("Wnt signaling proteins")
        prompt = mock_gemini.call_args_list[0].args[0]
        self.assertIn("Keep broad questions broad", prompt)
        self.assertIn("do not silently narrow", prompt)

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
