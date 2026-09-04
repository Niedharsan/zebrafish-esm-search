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
            {"protein_id": "P3", "name": "gene3", "description": "unrelated neural protein", "sequence": "", "extra_json": "{}"},
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
        self.assertEqual(plan["retrieval_terms"], ["proteins involved in erythropoiesis"])

    def test_discovery_ranks_by_best_validated_seed_similarity(self):
        results = app.discovery_neighbors([0], 2)
        self.assertEqual([r["protein_id"] for r in results], ["P2", "P4"])
        self.assertEqual(results[0]["closest_seed"], "gata1a")

    @patch("app.fetch_uniprot_seeds")
    @patch("app.interpret_biological_query")
    def test_discovery_uses_grounded_seed_not_ai_gene_guess(self, mock_plan, mock_uniprot):
        mock_plan.return_value = {
            "normalized_question": "erythropoiesis proteins",
            "retrieval_terms": ["erythropoiesis"],
            "rationale": "process search",
            "ai_used": True,
        }
        mock_uniprot.return_value = [
            {
                "index": 0,
                "source": "UniProt",
                "retrieval_term": "erythropoiesis",
                "uniprot_accession": "QTEST",
                "resolved_by": "gene name",
            }
        ]
        with patch("app.explain_discovery", return_value=None):
            result = app.discovery_api({"q": ["proteins involved in erythropoiesis"], "k": ["2"]})
        self.assertTrue(result["ok"])
        self.assertEqual(result["seeds"][0]["name"], "gata1a")
        self.assertEqual(result["results"][0]["protein_id"], "P2")


if __name__ == "__main__":
    unittest.main()
