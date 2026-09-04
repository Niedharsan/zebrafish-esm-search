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
            {"protein_id": "P2", "name": "gene2", "description": "related erythroid protein", "sequence": "", "extra_json": "{}"},
            {"protein_id": "P3", "name": "mpeg1.1", "description": "macrophage-expressed gene 1 protein", "sequence": "", "extra_json": "{}"},
            {"protein_id": "P4", "name": "gene4", "description": "another erythroid protein", "sequence": "", "extra_json": "{}"},
        ]
        app.ID_TO_INDEX = {p["protein_id"].lower(): i for i, p in enumerate(app.PROTEINS)}
        app.NAME_TO_INDEX = {p["name"].lower(): i for i, p in enumerate(app.PROTEINS)}
        app.SEARCH_TEXTS = [" ".join([p["protein_id"], p["name"], p["description"]]).lower() for p in app.PROTEINS]
        app.VECTORS = np.array(
            [
                [1.0, 0.0],
                [0.95, 0.05],
                [0.0, 1.0],
                [0.8, 0.2],
            ],
            dtype=np.float32,
        )
        app.VECTORS /= np.linalg.norm(app.VECTORS, axis=1, keepdims=True)

    def tearDown(self):
        if self.old_key is not None:
            os.environ["GEMINI_API_KEY"] = self.old_key
        else:
            os.environ.pop("GEMINI_API_KEY", None)

    def test_exact_protein_lookup_remains_deterministic(self):
        result = app.search_api({"q": ["gata1a"], "k": ["2"]})
        self.assertTrue(result["ok"])
        self.assertEqual(result["mode"], "protein")
        self.assertEqual(result["matched_protein"]["protein_id"], "P1")
        self.assertEqual(result["results"][0]["protein_id"], "P2")

    def test_ai_interpreter_has_safe_no_key_fallback(self):
        plan = app.interpret_biological_query("proteins involved in erythropoiesis")
        self.assertFalse(plan["ai_used"])
        self.assertFalse(plan["search_grounded"])
        self.assertEqual(plan["retrieval_terms"], ["proteins involved in erythropoiesis"])
        self.assertEqual(plan["zebrafish_candidates"], [])

    @patch("app._http_json")
    def test_google_search_grounding_omits_json_response_mime_type(self, mock_http):
        os.environ["GEMINI_API_KEY"] = "test-key"
        mock_http.return_value = {
            "candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}]
        }

        self.assertEqual(app._gemini_text("test", use_google_search=True), '{"ok": true}')
        payload = json.loads(mock_http.call_args.kwargs["data"].decode("utf-8"))
        self.assertEqual(payload["tools"], [{"google_search": {}}])
        self.assertNotIn("responseMimeType", payload["generationConfig"])

    @patch("app._http_json")
    def test_non_grounded_gemini_call_keeps_json_response_mode(self, mock_http):
        os.environ["GEMINI_API_KEY"] = "test-key"
        mock_http.return_value = {
            "candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}]
        }

        app._gemini_text("test")
        payload = json.loads(mock_http.call_args.kwargs["data"].decode("utf-8"))
        self.assertEqual(payload["generationConfig"]["responseMimeType"], "application/json")
        self.assertNotIn("tools", payload)

    def test_json_parser_can_extract_grounded_json_from_wrapping_text(self):
        parsed = app._parse_json_object('Grounded result:\n{"gene": "mpeg1.1"}\n')
        self.assertEqual(parsed["gene"], "mpeg1.1")

    @patch("app._gemini_text")
    def test_ai_planner_is_search_grounded_and_species_scoped(self, mock_gemini):
        os.environ["GEMINI_API_KEY"] = "test-key"
        mock_gemini.return_value = '''{
          "normalized_question": "zebrafish macrophage proteins",
          "retrieval_terms": ["macrophage", "phagocytosis"],
          "zebrafish_candidates": [{"gene": "mpeg1.1", "species": "zebrafish", "reason": "zebrafish macrophage marker"}],
          "reference_candidates": [{"gene": "CD68", "species": "human", "reason": "mammalian macrophage reference"}],
          "rationale": "zebrafish first"
        }'''
        plan = app.interpret_biological_query("macrophage proteins")
        self.assertTrue(plan["ai_used"])
        self.assertTrue(plan["search_grounded"])
        self.assertEqual(plan["zebrafish_candidates"][0]["gene"], "mpeg1.1")
        self.assertEqual(plan["reference_candidates"][0]["species"], "human")
        _, kwargs = mock_gemini.call_args
        self.assertTrue(kwargs["use_google_search"])
        prompt = mock_gemini.call_args.args[0]
        self.assertIn("DANIO RERIO", prompt)
        self.assertIn("Search zebrafish-specific evidence first", prompt)

    def test_ai_zebrafish_candidate_must_resolve_exactly_locally(self):
        seeds = app.validate_ai_zebrafish_candidates(
            [
                {"gene": "mpeg1.1", "species": "zebrafish", "reason": "marker"},
                {"gene": "inventedGene", "species": "zebrafish", "reason": "hallucination"},
            ]
        )
        self.assertEqual(len(seeds), 1)
        self.assertEqual(seeds[0]["index"], 2)
        self.assertEqual(seeds[0]["evidence_class"], "zebrafish-supported")

    def test_discovery_ranks_by_best_validated_seed_similarity(self):
        results = app.discovery_neighbors([0], 2)
        self.assertEqual([r["protein_id"] for r in results], ["P2", "P4"])
        self.assertEqual(results[0]["closest_seed"], "gata1a")

    @patch("app._http_json")
    def test_human_reference_is_mapped_to_zebrafish_with_ensembl(self, mock_http):
        mock_http.side_effect = [
            {"data": [{"homologies": [{"target": {"id": "ENSDARG000TEST"}}]}]},
            {"display_name": "mpeg1.1"},
        ]
        seeds = app.orthology_seeds(
            [{"gene": "CD68", "species": "human", "reason": "mammalian macrophage evidence"}]
        )
        self.assertEqual(len(seeds), 1)
        self.assertEqual(seeds[0]["index"], 2)
        self.assertEqual(seeds[0]["reference_gene"], "CD68")
        self.assertIn("orthology", seeds[0]["evidence_class"])

    @patch("app.fetch_uniprot_seeds")
    @patch("app.interpret_biological_query")
    def test_discovery_prioritizes_validated_zebrafish_ai_seed(self, mock_plan, mock_uniprot):
        mock_plan.return_value = {
            "normalized_question": "zebrafish macrophage proteins",
            "retrieval_terms": ["macrophage"],
            "zebrafish_candidates": [{"gene": "mpeg1.1", "species": "zebrafish", "reason": "marker"}],
            "reference_candidates": [],
            "rationale": "zebrafish first",
            "ai_used": True,
            "search_grounded": True,
        }
        mock_uniprot.return_value = [
            {
                "index": 0,
                "source": "UniProt zebrafish search",
                "retrieval_term": "macrophage",
                "uniprot_accession": "QTEST",
                "resolved_by": "gene name",
                "evidence_class": "zebrafish-supported",
            }
        ]
        with patch("app.explain_discovery", return_value=None):
            result = app.discovery_api({"q": ["macrophage proteins"], "k": ["2"]})
        self.assertTrue(result["ok"])
        self.assertEqual(result["seeds"][0]["name"], "mpeg1.1")
        self.assertEqual(result["seeds"][0]["source"], "Gemini Google Search (zebrafish)")


if __name__ == "__main__":
    unittest.main()
