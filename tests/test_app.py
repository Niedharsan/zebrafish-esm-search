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
            {"protein_id": "tr|Q7SXE0|Q7SXE0_DANRE", "name": "mpeg1.1", "description": "macrophage-expressed gene 1 protein", "sequence": "", "extra_json": "{}"},
            {"protein_id": "P4", "name": "gene4", "description": "another erythroid protein", "sequence": "", "extra_json": "{}"},
        ]
        app.ID_TO_INDEX = {p["protein_id"].lower(): i for i, p in enumerate(app.PROTEINS)}
        app.NAME_TO_INDEX = {p["name"].lower(): i for i, p in enumerate(app.PROTEINS)}
        app.SEARCH_TEXTS = [" ".join([p["protein_id"], p["name"], p["description"]]).lower() for p in app.PROTEINS]
        app.VECTORS = np.array(
            [[1.0, 0.0], [0.95, 0.05], [0.0, 1.0], [0.8, 0.2]], dtype=np.float32
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
        mock_http.return_value = {"candidates": [{"content": {"parts": [{"text": "grounded research"}]}}]}
        self.assertEqual(app._gemini_text("test", use_google_search=True), "grounded research")
        payload = json.loads(mock_http.call_args.kwargs["data"].decode("utf-8"))
        self.assertEqual(payload["tools"], [{"google_search": {}}])
        self.assertNotIn("responseMimeType", payload["generationConfig"])

    @patch("app._http_json")
    def test_non_grounded_gemini_call_keeps_json_response_mode(self, mock_http):
        os.environ["GEMINI_API_KEY"] = "test-key"
        mock_http.return_value = {"candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}]}
        app._gemini_text("test")
        payload = json.loads(mock_http.call_args.kwargs["data"].decode("utf-8"))
        self.assertEqual(payload["generationConfig"]["responseMimeType"], "application/json")
        self.assertNotIn("tools", payload)

    @patch("app._http_json")
    def test_empty_grounded_response_reports_finish_reason(self, mock_http):
        os.environ["GEMINI_API_KEY"] = "test-key"
        mock_http.return_value = {
            "candidates": [{
                "content": {"parts": []},
                "finishReason": "STOP",
                "groundingMetadata": {"webSearchQueries": ["zebrafish macrophage marker"]},
            }]
        }
        with self.assertRaisesRegex(RuntimeError, "finishReason=STOP"):
            app._gemini_text("test", use_google_search=True)

    def test_name_match_is_not_expression_evidence(self):
        record = {
            "primaryAccession": "Q7SXE0",
            "genes": [{"geneName": {"value": "mpeg1.1"}}],
            "proteinDescription": {"recommendedName": {"fullName": {"value": "Macrophage-expressed gene 1 protein"}}},
        }
        evidence = app.uniprot_record_to_evidence(record, "macrophage", 1)
        self.assertIn("name_match", evidence["evidence_types"])
        self.assertNotIn("expression_annotation", evidence["evidence_types"])
        self.assertEqual(evidence["search_rank"], 1)
        self.assertIn("not a biological relevance score", evidence["retrieval_note"])

    def test_go_process_evidence_is_distinct_from_name_match(self):
        record = {
            "primaryAccession": "PTEST",
            "genes": [{"geneName": {"value": "atg1"}}],
            "proteinDescription": {"recommendedName": {"fullName": {"value": "Protein kinase"}}},
            "uniProtKBCrossReferences": [
                {"database": "GO", "properties": [{"key": "GoTerm", "value": "P:autophagy"}]}
            ],
        }
        evidence = app.uniprot_record_to_evidence(record, "autophagy", 1)
        self.assertIn("go_biological_process", evidence["evidence_types"])
        self.assertNotIn("name_match", evidence["evidence_types"])

    def test_accession_resolves_pipe_delimited_local_id(self):
        self.assertEqual(app.resolve_uniprot_accession("Q7SXE0"), 2)

    @patch("app._http_json")
    def test_ensembl_alias_can_resolve_to_canonical_local_gene(self, mock_http):
        mock_http.side_effect = [
            {},
            [{"id": "ENSDARG0001"}],
            {"id": "ENSDARG0001", "display_name": "mpeg1.1", "description": "macrophage gene"},
        ]
        ensembl = app.fetch_ensembl_evidence(["mpeg1"])
        self.assertEqual(ensembl[0]["canonical_symbol"], "mpeg1.1")
        seeds = app.validate_ai_zebrafish_candidates(
            [{"gene": "mpeg1", "species": "zebrafish", "reason": "marker", "evidence_types": ["marker_support"]}],
            [],
            ensembl,
        )
        self.assertEqual(len(seeds), 1)
        self.assertEqual(seeds[0]["index"], 2)
        self.assertIn("mpeg1 → mpeg1.1", seeds[0]["resolved_by"])

    @patch("app.fetch_ensembl_evidence")
    @patch("app.fetch_uniprot_evidence")
    @patch("app._gemini_text")
    def test_ai_receives_structured_databases_before_candidate_ranking(self, mock_gemini, mock_uniprot, mock_ensembl):
        os.environ["GEMINI_API_KEY"] = "test-key"
        mock_gemini.side_effect = [
            json.dumps({
                "normalized_question": "zebrafish macrophage proteins",
                "question_type": "cell_type",
                "retrieval_terms": ["macrophage"],
                "evidence_priorities": ["marker and expression evidence"],
            }),
            "Independent zebrafish evidence supports mpeg1.1 as a macrophage marker.",
            json.dumps({
                "zebrafish_candidates": [{
                    "gene": "mpeg1.1",
                    "species": "zebrafish",
                    "uniprot_accession": "Q7SXE0",
                    "evidence_types": ["name_match", "marker_support", "literature_support"],
                    "reason": "independent marker evidence",
                }],
                "reference_candidates": [],
                "rationale": "direct zebrafish evidence",
            }),
        ]
        mock_uniprot.return_value = [{
            "source": "UniProtKB",
            "search_term": "macrophage",
            "search_rank": 1,
            "gene": "mpeg1.1",
            "uniprot_accession": "Q7SXE0",
            "protein_name": "Macrophage-expressed gene 1 protein",
            "evidence_types": ["name_match"],
            "matched_fields": ["name"],
            "annotations": {},
        }]
        mock_ensembl.return_value = [{
            "source": "Ensembl",
            "requested_symbol": "mpeg1.1",
            "canonical_symbol": "mpeg1.1",
            "ensembl_id": "ENSDARG1",
            "description": "",
            "biotype": "protein_coding",
            "evidence_types": ["identifier_resolution"],
        }]

        plan = app.interpret_biological_query("macrophage proteins")
        self.assertTrue(plan["ai_used"])
        self.assertTrue(plan["search_grounded"])
        self.assertEqual(plan["question_type"], "cell_type")
        self.assertEqual(plan["zebrafish_candidates"][0]["gene"], "mpeg1.1")
        self.assertEqual(mock_gemini.call_count, 3)
        self.assertFalse(mock_gemini.call_args_list[0].kwargs.get("use_google_search", False))
        self.assertTrue(mock_gemini.call_args_list[1].kwargs["use_google_search"])
        self.assertFalse(mock_gemini.call_args_list[2].kwargs.get("use_google_search", False))
        rank_prompt = mock_gemini.call_args_list[2].args[0]
        self.assertIn("UniProtKB structured evidence", rank_prompt)
        self.assertIn("name_match", rank_prompt)
        self.assertIn("not database result order", rank_prompt.lower())

    @patch("app._gemini_text")
    def test_retrieval_planner_is_universal_across_question_types(self, mock_gemini):
        os.environ["GEMINI_API_KEY"] = "test-key"
        mock_gemini.return_value = json.dumps({
            "normalized_question": "zebrafish Wnt signaling proteins",
            "question_type": "pathway",
            "retrieval_terms": ["Wnt signaling", "canonical Wnt pathway"],
            "evidence_priorities": ["curated pathway membership", "GO biological process", "functional evidence"],
        })
        plan = app._retrieval_plan("Wnt signaling proteins")
        self.assertEqual(plan["question_type"], "pathway")
        self.assertIn("Wnt signaling", plan["retrieval_terms"])
        self.assertIn("lexical protein-name match", mock_gemini.call_args.args[0])

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

    @patch("app.interpret_biological_query")
    def test_discovery_does_not_append_raw_uniprot_fillers_when_ai_seed_resolves(self, mock_plan):
        mock_plan.return_value = {
            "normalized_question": "zebrafish macrophage proteins",
            "question_type": "cell_type",
            "retrieval_terms": ["macrophage"],
            "evidence_priorities": ["marker evidence"],
            "zebrafish_candidates": [{
                "gene": "mpeg1.1",
                "species": "zebrafish",
                "reason": "marker",
                "evidence_types": ["marker_support"],
                "uniprot_accession": "Q7SXE0",
            }],
            "reference_candidates": [],
            "rationale": "ranked evidence",
            "ai_used": True,
            "search_grounded": True,
            "evidence_summary": {},
            "_uniprot_evidence": [
                {"gene": "mpeg1.1", "uniprot_accession": "Q7SXE0", "search_rank": 1, "search_term": "macrophage", "evidence_types": ["name_match"]},
                {"gene": "gata1a", "uniprot_accession": "P1", "search_rank": 2, "search_term": "macrophage", "evidence_types": ["uniprot_text_search_hit"]},
            ],
            "_ensembl_evidence": [],
            "_retrieval_errors": [],
        }
        with patch("app.explain_discovery", return_value=None):
            result = app.discovery_api({"q": ["macrophage proteins"], "k": ["2"]})
        self.assertTrue(result["ok"])
        self.assertEqual([seed["name"] for seed in result["seeds"]], ["mpeg1.1"])
        self.assertEqual(result["seeds"][0]["source"], "Evidence-ranked zebrafish candidate")

    def test_discovery_ranks_by_best_validated_seed_similarity(self):
        results = app.discovery_neighbors([0], 2)
        self.assertEqual([r["protein_id"] for r in results], ["P2", "P4"])
        self.assertEqual(results[0]["closest_seed"], "gata1a")


if __name__ == "__main__":
    unittest.main()
