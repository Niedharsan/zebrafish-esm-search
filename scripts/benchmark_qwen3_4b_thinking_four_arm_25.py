#!/usr/bin/env python3
"""Hard-25 Qwen3 thinking benchmark: 4 simple inference strategies. Reuses the previous Arm-2 prompts/tools/scoring."""
from __future__ import annotations
import argparse, importlib.util, json, os, sys, time
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import app

BASE=ROOT/"scripts"/"benchmark_qwen3_4b_three_arm_25.py"
spec=importlib.util.spec_from_file_location("hard25_base",BASE)
base=importlib.util.module_from_spec(spec); assert spec and spec.loader; spec.loader.exec_module(base)

VERSION="thinking-four-arm-hard25-v1.0"
ARMS=("thinking_arm2","thinking_base","thinking_x3_synthesis","thinking_x3_tools_synthesis")
DEFAULT_CASES=ROOT/"hard25_qwen3_4b_three_arm_cases.json"
DEFAULT_DB=ROOT/"data"/"zebrafish_esm.db"
DEFAULT_CP=ROOT/"qwen3_4b_thinking_four_arm_25_checkpoint.json"
DEFAULT_OUT=ROOT/"qwen3_4b_thinking_four_arm_25_results.json"

def thinking_payload(prompt,*,model,temperature,num_ctx,num_predict):
    return {"model":model,"prompt":prompt,"stream":False,"format":"json","think":True,
            "options":{"temperature":temperature,"num_ctx":num_ctx,"num_predict":num_predict}}

def call_thinking_qwen(prompt,question,**cfg):
    attempts=[]; last=None
    for attempt in range(2):
        retry="\nPrevious final answer was not valid JSON. Return one complete JSON object only." if attempt else ""
        started=time.monotonic()
        try:
            response=app._http_json(f"{app.ollama_url()}/api/generate",
                headers={"Content-Type":"application/json"},
                data=json.dumps(thinking_payload(prompt+retry,**cfg)).encode(),timeout=300)
            final=str(response.get("response") or "").strip()
            thinking=str(response.get("thinking") or "").strip()
            plan=base.clean(app._parse_json_object(final),question)
            attempts.append({"attempt":attempt+1,"thinking":thinking,"final_response":final,
                             "seconds":round(time.monotonic()-started,3),
                             "prompt_eval_count":response.get("prompt_eval_count"),"eval_count":response.get("eval_count")})
            return plan,{"attempts":attempts,"thinking_present":bool(thinking)}
        except Exception as exc:
            last=exc; attempts.append({"attempt":attempt+1,"error":str(exc),"seconds":round(time.monotonic()-started,3)})
    raise RuntimeError(f"Thinking Ollama failed to return valid JSON after one retry. ({last})") from last

def cfg(args):
    return {"model":args.model,"temperature":args.temperature,"num_ctx":args.num_ctx,"num_predict":args.num_predict}

def compact_plan(plan):
    return {"normalized_question":str(plan.get("normalized_question") or "")[:240],
            "zebrafish_candidates":[{"gene":str(x.get("gene") or "")[:80],"species":str(x.get("species") or "")[:30],
                                     "reason":str(x.get("reason") or "")[:160]}
                                    for x in (plan.get("zebrafish_candidates") or [])[:base.MAX_SEEDS]],
            "reference_candidates":[{"gene":str(x.get("gene") or "")[:80],"species":str(x.get("species") or "")[:30],
                                     "reason":str(x.get("reason") or "")[:120]}
                                    for x in (plan.get("reference_candidates") or [])[:base.MAX_REFS]],
            "rationale":str(plan.get("rationale") or "")[:320]}

def candidate_union(plans):
    zf=[]; refs=[]; sz=set(); sr=set()
    for plan in plans:
        for x in plan.get("zebrafish_candidates") or []:
            g=str(x.get("gene") or "").strip(); k=g.lower()
            if g and k not in sz: sz.add(k); zf.append(x)
        for x in plan.get("reference_candidates") or []:
            g=str(x.get("gene") or "").strip(); s=str(x.get("species") or "").strip().lower(); k=(s,g.lower())
            if g and s in {"human","mouse"} and k not in sr: sr.add(k); refs.append(x)
    return {"zebrafish_candidates":zf,"reference_candidates":refs}

def retrieve_tools(question,plan):
    local=app._local_question_context(question,limit=30)
    evidence,eerrs=app.fetch_authoritative_evidence(question)
    _,traces,rerrs=app.resolve_targeted_uniprot_candidates(plan.get("zebrafish_candidates") or [])
    orth=app.orthology_seeds(plan.get("reference_candidates") or [])
    return {"local_context":local,"evidence":base.compact_evidence(evidence),
            "resolution":base.compact_trace(traces),"orthology":base.compact_orthology(orth),
            "retrieval_errors":[*eerrs,*rerrs],
            "retrieval_sources":["Local zebrafish database lexical context","PubMed","Europe PMC","QuickGO",
                                 "UniProt","Ensembl","Local zebrafish DB"]}

def synthesis_prompt(question,plans):
    return f"""Act as a zebrafish biologist. Three independent thinking runs answered the same question.

QUESTION:
{question}

THREE RUNS:
{json.dumps([compact_plan(p) for p in plans],ensure_ascii=False)}

Think through the three results and produce one final ranked answer. Combine strong candidates when appropriate, reject weak ones, and prefer canonical, directly relevant Danio rerio genes with exact zebrafish nomenclature.

Return at most {base.MAX_SEEDS} zebrafish candidates and {base.MAX_REFS} human/mouse reference candidates.
Return only:
{{"normalized_question":"...","zebrafish_candidates":[{{"gene":"exact zebrafish symbol","species":"zebrafish","uniprot_accession":"","reason":"short reason"}}],"reference_candidates":[{{"gene":"human or mouse symbol","species":"human or mouse","reason":"short reason"}}],"rationale":"brief synthesis"}}
"""

def compact_tools(tools):
    local=[{"gene":str(x.get("gene") or "")[:80],"protein_id":str(x.get("protein_id") or "")[:80],
            "description":str(x.get("description") or "")[:120]} for x in tools["local_context"][:20]]
    ev=[]
    for x in tools["evidence"]:
        src=str(x.get("source") or "")
        if src in {"PubMed","Europe PMC"}:
            ev.append({"source":src,"title":str(x.get("title") or "")[:180],"abstract":str(x.get("abstract") or "")[:180],
                       "pmid":x.get("pmid"),"year":x.get("year")})
        elif src=="QuickGO":
            ev.append({"source":src,"go_id":x.get("go_id"),"name":str(x.get("name") or "")[:140],
                       "definition":str(x.get("definition") or "")[:160]})
    return {"resolution":tools["resolution"],"orthology":tools["orthology"],"local_context":local,"evidence":ev}

def tools_synthesis_prompt(question,plans,tools):
    t=compact_tools(tools)
    return f"""Act as a zebrafish biologist. Three independent thinking runs answered the same question, then the existing Arm-2 scientific tools returned evidence.

QUESTION:
{question}

THREE RUNS:
{json.dumps([compact_plan(p) for p in plans],ensure_ascii=False)}

IDENTIFIER CHECKS:
{json.dumps(t["resolution"],ensure_ascii=False)}
ORTHOLOGY CHECKS:
{json.dumps(t["orthology"],ensure_ascii=False)}
LOCAL ZEBRAFISH CONTEXT:
{json.dumps(t["local_context"],ensure_ascii=False)}
PUBMED / EUROPE PMC / QUICKGO EVIDENCE:
{json.dumps(t["evidence"],ensure_ascii=False)}

Think through the three results and tool evidence and produce one final ranked answer. Correct nomenclature mistakes, reject weak candidates, preserve strong biology, and add strongly supported missing zebrafish candidates when the supplied evidence identifies them.

Return at most {base.MAX_SEEDS} zebrafish candidates and {base.MAX_REFS} human/mouse reference candidates.
Return only:
{{"normalized_question":"...","zebrafish_candidates":[{{"gene":"exact zebrafish symbol","species":"zebrafish","uniprot_accession":"","reason":"short reason"}}],"reference_candidates":[{{"gene":"human or mouse symbol","species":"human or mouse","reason":"short reason"}}],"rationale":"brief synthesis"}}
"""

def three(case,args):
    plans=[]; traces=[]
    for _ in range(3):
        p,t=call_thinking_qwen(base.base_prompt(case["query"]),case["query"],**cfg(args)); plans.append(p); traces.append(t)
    return plans,traces

def run_thinking_base(case,k,args):
    started=time.monotonic()
    plan,trace=call_thinking_qwen(base.base_prompt(case["query"]),case["query"],**cfg(args))
    out=base.score(plan,case["expected_examples"],k)
    out.update({"plan":plan,"model_trace":trace,"retrieval_sources":[],"retrieval_errors":[],
                "total_seconds":round(time.monotonic()-started,3)})
    return out

def run_thinking_arm2(case,k,args):
    started=time.monotonic()
    first,t1=call_thinking_qwen(base.base_prompt(case["query"]),case["query"],**cfg(args))
    rt=time.monotonic(); tools=retrieve_tools(case["query"],first); rsec=time.monotonic()-rt
    final,t2=call_thinking_qwen(base.second_prompt(case["query"],first,tools["resolution"],tools["orthology"],
                                                   tools["local_context"],tools["evidence"]),case["query"],**cfg(args))
    out=base.score(final,case["expected_examples"],k)
    out.update({"first_pass_plan":first,"plan":final,"model_trace":{"first_pass":t1,"second_pass":t2},
                "candidate_resolution_checks":tools["resolution"],"orthology_checks":tools["orthology"],
                "local_context_records":len(tools["local_context"]),"authoritative_evidence_records":len(tools["evidence"]),
                "authoritative_evidence":tools["evidence"],"retrieval_sources":tools["retrieval_sources"],
                "retrieval_errors":tools["retrieval_errors"],"retrieval_seconds":round(rsec,3),
                "total_seconds":round(time.monotonic()-started,3)})
    return out

def run_thinking_x3_synthesis(case,k,args):
    started=time.monotonic(); plans,traces=three(case,args)
    final,tf=call_thinking_qwen(synthesis_prompt(case["query"],plans),case["query"],**cfg(args))
    out=base.score(final,case["expected_examples"],k)
    out.update({"independent_plans":plans,"plan":final,"model_trace":{"independent_runs":traces,"synthesis":tf},
                "retrieval_sources":[],"retrieval_errors":[],"total_seconds":round(time.monotonic()-started,3)})
    return out

def run_thinking_x3_tools_synthesis(case,k,args):
    started=time.monotonic(); plans,traces=three(case,args); union=candidate_union(plans)
    rt=time.monotonic(); tools=retrieve_tools(case["query"],union); rsec=time.monotonic()-rt
    final,tf=call_thinking_qwen(tools_synthesis_prompt(case["query"],plans,tools),case["query"],**cfg(args))
    out=base.score(final,case["expected_examples"],k)
    out.update({"independent_plans":plans,"candidate_union":compact_plan(union),"plan":final,
                "model_trace":{"independent_runs":traces,"synthesis":tf},
                "candidate_resolution_checks":tools["resolution"],"orthology_checks":tools["orthology"],
                "local_context_records":len(tools["local_context"]),"authoritative_evidence_records":len(tools["evidence"]),
                "authoritative_evidence":tools["evidence"],"retrieval_sources":tools["retrieval_sources"],
                "retrieval_errors":tools["retrieval_errors"],"retrieval_seconds":round(rsec,3),
                "total_seconds":round(time.monotonic()-started,3)})
    return out

RUNNERS={"thinking_arm2":run_thinking_arm2,"thinking_base":run_thinking_base,
         "thinking_x3_synthesis":run_thinking_x3_synthesis,"thinking_x3_tools_synthesis":run_thinking_x3_tools_synthesis}

def output(cases,args):
    summaries={a:base.summary(cases,a) for a in ARMS} if cases else {}
    pairs={}
    if cases:
        for i,a in enumerate(ARMS):
            for b in ARMS[i+1:]: pairs[f"{a}_vs_{b}"]=base.pair(cases,a,b)
    return {"metadata":{"timestamp_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
                        "git_commit":base.git("rev-parse","HEAD"),"git_branch":base.git("branch","--show-current"),
                        "runner_version":VERSION,"ollama_model":args.model,"ollama_url":app.ollama_url(),
                        "case_count":25,"completed_case_count":len(cases),"cases_sha256":base.sha(args.cases),
                        "app_sha256":base.sha(ROOT/"app.py"),"runner_sha256":base.sha(Path(__file__)),
                        "database_sha256":base.sha(args.db),"k":args.k,
                        "ollama_settings":{"think":True,"temperature":args.temperature,"num_ctx":args.num_ctx,
                                           "num_predict":args.num_predict,"format":"json"},"arms":list(ARMS)},
            "summary":summaries,"pairwise":pairs,"cases":cases}

def print_summary(cases):
    print("\nFINAL SUMMARY")
    for arm in ARMS:
        s=base.summary(cases,arm)
        print(f"{arm}: canonical={s['canonical_any_rate']*100:.1f}% recall={s['mean_canonical_recall']*100:.1f}% "
              f"H@1={s['hit_at_1_rate']*100:.1f}% H@3={s['hit_at_3_rate']*100:.1f}% H@5={s['hit_at_5_rate']*100:.1f}% "
              f"MRR={s['mean_mrr']:.3f} unresolved={s['unresolved_rate']*100:.1f}% median={s['median_latency_seconds']:.1f}s")

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--cases",default=str(DEFAULT_CASES)); p.add_argument("--db",default=str(DEFAULT_DB))
    p.add_argument("--model",default="qwen3:4b-instruct"); p.add_argument("--k",type=int,default=5)
    p.add_argument("--out",default=str(DEFAULT_OUT)); p.add_argument("--checkpoint",default=str(DEFAULT_CP))
    p.add_argument("--temperature",type=float,default=.1); p.add_argument("--num-ctx",type=int,default=8192)
    p.add_argument("--num-predict",type=int,default=1200)
    return p.parse_args()

def main():
    args=parse_args(); data=base.load_cases(args.cases)
    os.environ["AI_PROVIDER"]="ollama"; os.environ["OLLAMA_MODEL"]=args.model
    app.load_database(args.db)
    cp=Path(args.checkpoint); completed=[]
    if cp.exists():
        saved=json.loads(cp.read_text(encoding="utf-8")); meta=saved.get("metadata") or {}
        if meta.get("runner_version")!=VERSION: raise RuntimeError("Checkpoint runner version does not match.")
        if meta.get("cases_sha256")!=base.sha(args.cases): raise RuntimeError("Checkpoint cases file does not match.")
        completed=saved.get("cases") or []; print(f"Resuming from {len(completed)}/25 complete cases.")
    done={x["id"] for x in completed}
    for pos,case in enumerate(data["cases"],1):
        if case["id"] in done: continue
        print(f"\n[{pos}/25] case {case['id']}: {case['query']}")
        row={k:case[k] for k in ("id","category","subtopic","wording","query_specificity","query","expected_examples")}
        for arm in ARMS:
            print(f"  -> {arm}"); started=time.monotonic()
            try: row[arm]=RUNNERS[arm](case,args.k,args)
            except KeyboardInterrupt:
                print("\nInterrupted. Incomplete case was not checkpointed."); raise
            except Exception as exc:
                print(f"     ERROR: {exc}")
                row[arm]={"error":str(exc),"canonical_hit_any":False,"canonical_hits":[],"canonical_recall":0.0,
                          "first_canonical_rank":None,"hit_at_1":False,"hit_at_3":False,"hit_at_5":False,
                          "mrr_first_canonical":0.0,"pipeline_has_seed":False,"validated_seed_count":0,
                          "unresolved_gene_count":0,"proposed_gene_count":0,"unresolved_rate":0.0,
                          "total_seconds":round(time.monotonic()-started,3)}
        completed.append(row); base.write_json(cp,output(completed,args)); print(f"  checkpointed {len(completed)}/25")
    base.write_json(args.out,output(completed,args)); print_summary(completed); print(f"\nResults written to: {args.out}")
    return 0

if __name__=="__main__": raise SystemExit(main())
