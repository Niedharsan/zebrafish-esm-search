from __future__ import annotations
import argparse, importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/"scripts"/"benchmark_qwen3_4b_thinking_four_arm_25.py"
spec=importlib.util.spec_from_file_location("thinking_bench",SCRIPT)
bench=importlib.util.module_from_spec(spec); assert spec and spec.loader; spec.loader.exec_module(bench)

class ThinkingFourArmTests(unittest.TestCase):
    def setUp(self):
        self.case={"id":1,"category":"cell_type_lineage","subtopic":"macrophage","wording":"direct",
                   "query_specificity":"specific","query":"Which proteins mark zebrafish macrophages?",
                   "expected_examples":["mpeg1.1","mfap4","csf1ra","marco"]}
        self.args=argparse.Namespace(model="qwen3:4b-instruct",temperature=.6,top_p=.95,top_k=20,num_ctx=8192,num_predict=2400)
        self.plan={"normalized_question":self.case["query"],
                   "zebrafish_candidates":[{"gene":"mpeg1.1","species":"zebrafish","uniprot_accession":"","reason":"marker"}],
                   "reference_candidates":[],"rationale":"test"}
        self.score={"canonical_hit_any":False,"canonical_hits":[],"canonical_recall":0.0,"first_canonical_rank":None,
                    "hit_at_1":False,"hit_at_3":False,"hit_at_5":False,"mrr_first_canonical":0.0,
                    "validated_seed_count":0,"pipeline_has_seed":False,"unresolved_gene_count":0,
                    "proposed_gene_count":0,"unresolved_rate":0.0}
        self.tools={"local_context":[],"evidence":[],"resolution":[],"orthology":[],"retrieval_errors":[],
                    "retrieval_sources":["Local zebrafish database lexical context","PubMed","Europe PMC","QuickGO",
                                         "UniProt","Ensembl","Local zebrafish DB"]}

    def test_same_hard25(self):
        data=bench.base.load_cases(ROOT/"hard25_qwen3_4b_three_arm_cases.json")
        self.assertEqual([c["id"] for c in data["cases"]],
                         [1,3,8,10,11,13,14,15,16,20,21,22,24,27,31,32,34,36,38,42,44,45,46,47,50])

    def test_exact_four_arms(self):
        self.assertEqual(bench.ARMS,("thinking_arm2","thinking_base","thinking_x3_synthesis","thinking_x3_tools_synthesis"))

    def test_payload_turns_thinking_on(self):
        p=bench.thinking_payload("x",model="qwen3:4b-instruct",temperature=.6,top_p=.95,top_k=20,num_ctx=8192,num_predict=2400)
        self.assertIs(p["think"],True); self.assertEqual(p["format"],"json")
        self.assertEqual(p["options"],{"temperature":.6,"top_p":.95,"top_k":20,"num_ctx":8192,"num_predict":2400})

    def test_thinking_base_one_call_no_tools(self):
        with patch.object(bench,"call_thinking_qwen",return_value=(self.plan,{"thinking_present":True})) as qwen, \
             patch.object(bench.base,"score",return_value=dict(self.score)), \
             patch.object(bench,"retrieve_tools",side_effect=AssertionError("tools forbidden")):
            out=bench.run_thinking_base(self.case,5,self.args)
        self.assertEqual(qwen.call_count,1); self.assertEqual(out["retrieval_sources"],[])

    def test_thinking_arm2_is_think_tools_think(self):
        events=[]
        def q(*a,**k): events.append("qwen"); return self.plan,{"thinking_present":True}
        def t(*a,**k): events.append("tools"); return dict(self.tools)
        with patch.object(bench,"call_thinking_qwen",side_effect=q), \
             patch.object(bench,"retrieve_tools",side_effect=t), \
             patch.object(bench.base,"score",return_value=dict(self.score)):
            out=bench.run_thinking_arm2(self.case,5,self.args)
        self.assertEqual(events,["qwen","tools","qwen"]); self.assertIn("PubMed",out["retrieval_sources"])

    def test_three_means_three_independent_qwen_calls(self):
        with patch.object(bench,"call_thinking_qwen",return_value=(self.plan,{"thinking_present":True})) as qwen:
            plans,traces=bench.three(self.case,self.args)
        self.assertEqual(qwen.call_count,3); self.assertEqual(len(plans),3); self.assertEqual(len(traces),3)

    def test_x3_synthesis_has_no_tools(self):
        plans=[dict(self.plan) for _ in range(3)]
        with patch.object(bench,"three",return_value=(plans,[{}]*3)), \
             patch.object(bench,"call_thinking_qwen",return_value=(self.plan,{"thinking_present":True})) as final, \
             patch.object(bench.base,"score",return_value=dict(self.score)), \
             patch.object(bench,"retrieve_tools",side_effect=AssertionError("tools forbidden")):
            out=bench.run_thinking_x3_synthesis(self.case,5,self.args)
        self.assertEqual(final.call_count,1); self.assertEqual(len(out["independent_plans"]),3)

    def test_x3_tools_unions_first_then_tools_then_final(self):
        p2={**self.plan,"zebrafish_candidates":[{"gene":"marco","species":"zebrafish","reason":"marker"}]}
        captured={}
        def tools(question,union): captured["union"]=union; return dict(self.tools)
        with patch.object(bench,"three",return_value=([self.plan,p2,self.plan],[{}]*3)), \
             patch.object(bench,"retrieve_tools",side_effect=tools) as toolcall, \
             patch.object(bench,"call_thinking_qwen",return_value=(self.plan,{"thinking_present":True})) as final, \
             patch.object(bench.base,"score",return_value=dict(self.score)):
            out=bench.run_thinking_x3_tools_synthesis(self.case,5,self.args)
        self.assertEqual(toolcall.call_count,1); self.assertEqual(final.call_count,1)
        self.assertEqual([x["gene"] for x in captured["union"]["zebrafish_candidates"]],["mpeg1.1","marco"])
        self.assertIn("Ensembl",out["retrieval_sources"])

    def test_reuses_existing_base_and_arm2_prompts(self):
        base_prompt=bench.base.base_prompt(self.case["query"])
        self.assertIn("Use only your internal knowledge",base_prompt)
        self.assertNotIn("AUTHORITATIVE RETRIEVAL",base_prompt)
        second=bench.base.second_prompt(self.case["query"],self.plan,[],[],[],[])
        self.assertIn("SECOND-PASS",second)
        self.assertIn("AUTHORITATIVE RETRIEVAL",second)

if __name__=="__main__": unittest.main()
