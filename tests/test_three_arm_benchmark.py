from __future__ import annotations
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/"scripts"/"benchmark_qwen3_4b_three_arm_25.py"
spec=importlib.util.spec_from_file_location("bench",SCRIPT)
bench=importlib.util.module_from_spec(spec); assert spec and spec.loader; spec.loader.exec_module(bench)

class ThreeArmTests(unittest.TestCase):
    def setUp(self):
        self.case={"id":1,"category":"cell_type_lineage","subtopic":"macrophage","wording":"direct","query_specificity":"specific","query":"Which proteins mark zebrafish macrophages?","expected_examples":["mpeg1.1","mfap4","csf1ra","marco"]}
        self.empty={"canonical_hit_any":False,"canonical_hits":[],"canonical_recall":0.0,"first_canonical_rank":None,"hit_at_1":False,"hit_at_3":False,"hit_at_5":False,"mrr_first_canonical":0.0,"validated_seed_count":0,"pipeline_has_seed":False,"unresolved_gene_count":0,"proposed_gene_count":0,"unresolved_rate":0.0}

    def test_hard25_ids(self):
        data=bench.load_cases(ROOT/"hard25_qwen3_4b_three_arm_cases.json")
        self.assertEqual([c["id"] for c in data["cases"]],[1,3,8,10,11,13,14,15,16,20,21,22,24,27,31,32,34,36,38,42,44,45,46,47,50])

    def test_base_prompt_is_retrieval_free(self):
        p=bench.base_prompt(self.case["query"])
        self.assertIn("Use only your internal knowledge",p)
        self.assertIn("no local database context",p)
        self.assertNotIn("AUTHORITATIVE RETRIEVAL",p)

    def test_local_prompt_has_metadata_not_literature(self):
        p=bench.local_prompt(self.case["query"],[{"gene":"mpeg1.1","protein_id":"x","description":"Macrophage-expressed gene 1"}])
        self.assertIn("mpeg1.1",p)
        self.assertNotIn("PubMed",p)
        self.assertNotIn("Europe PMC",p)
        self.assertNotIn("QuickGO",p)

    def test_tools_arm_reasons_first_then_retrieves(self):
        prompts=[]
        first={"normalized_question":self.case["query"],"zebrafish_candidates":[{"gene":"mpeg1","species":"zebrafish","uniprot_accession":"","reason":"marker"}],"reference_candidates":[],"rationale":"first"}
        second={"normalized_question":self.case["query"],"zebrafish_candidates":[{"gene":"mpeg1.1","species":"zebrafish","uniprot_accession":"","reason":"repaired"}],"reference_candidates":[],"rationale":"second"}
        def fake_call(prompt,q):
            prompts.append(prompt); return (first if len(prompts)==1 else second),["{}"]
        with patch.object(bench,"call_qwen",side_effect=fake_call), \
             patch.object(bench,"score",return_value=dict(self.empty)), \
             patch.object(bench.app,"_local_question_context",return_value=[{"gene":"mpeg1.1","protein_id":"x","description":"Macrophage"}]), \
             patch.object(bench.app,"fetch_authoritative_evidence",return_value=([{"source":"PubMed","title":"zebrafish macrophage"}],[])), \
             patch.object(bench.app,"resolve_targeted_uniprot_candidates",return_value=([],[{"term":"mpeg1","local_resolution":None}],[])), \
             patch.object(bench.app,"orthology_seeds",return_value=[]), \
             patch.object(bench.app,"_gemini_response",side_effect=AssertionError("Gemini forbidden")), \
             patch.object(bench.app,"_gemini_text",side_effect=AssertionError("Gemini forbidden")):
            bench.run_tools(self.case,5)
        self.assertEqual(len(prompts),2)
        self.assertIn("Use only your internal knowledge",prompts[0])
        self.assertNotIn("AUTHORITATIVE RETRIEVAL",prompts[0])
        self.assertIn("AUTHORITATIVE RETRIEVAL",prompts[1])
        self.assertIn("CANDIDATE IDENTIFIER CHECKS",prompts[1])

    def test_local_arm_never_calls_literature_or_gemini(self):
        plan={"normalized_question":self.case["query"],"zebrafish_candidates":[],"reference_candidates":[],"rationale":""}
        with patch.object(bench.app,"_local_question_context",return_value=[]), \
             patch.object(bench,"call_qwen",return_value=(plan,["{}"])), \
             patch.object(bench,"score",return_value=dict(self.empty)), \
             patch.object(bench.app,"fetch_authoritative_evidence",side_effect=AssertionError("literature forbidden")), \
             patch.object(bench.app,"_gemini_response",side_effect=AssertionError("Gemini forbidden")), \
             patch.object(bench.app,"_gemini_text",side_effect=AssertionError("Gemini forbidden")):
            bench.run_local(self.case,5)

    def test_base_arm_never_calls_retrieval_or_gemini(self):
        plan={"normalized_question":self.case["query"],"zebrafish_candidates":[],"reference_candidates":[],"rationale":""}
        with patch.object(bench,"call_qwen",return_value=(plan,["{}"])), \
             patch.object(bench,"score",return_value=dict(self.empty)), \
             patch.object(bench.app,"_local_question_context",side_effect=AssertionError("local retrieval forbidden")), \
             patch.object(bench.app,"fetch_authoritative_evidence",side_effect=AssertionError("literature forbidden")), \
             patch.object(bench.app,"_gemini_response",side_effect=AssertionError("Gemini forbidden")), \
             patch.object(bench.app,"_gemini_text",side_effect=AssertionError("Gemini forbidden")):
            bench.run_base(self.case,5)

    def test_pairwise_prefers_canonical_over_identifier_cleanliness(self):
        canonical={"canonical_recall":.25,"mrr_first_canonical":.2,"hit_at_1":False,"hit_at_3":False,"hit_at_5":True,"unresolved_rate":.9,"validated_seed_count":1}
        clean={"canonical_recall":0,"mrr_first_canonical":0,"hit_at_1":False,"hit_at_3":False,"hit_at_5":False,"unresolved_rate":0,"validated_seed_count":10}
        self.assertGreater(bench.key(canonical),bench.key(clean))

if __name__=="__main__": unittest.main()
