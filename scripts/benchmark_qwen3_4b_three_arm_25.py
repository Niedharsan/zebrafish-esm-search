#!/usr/bin/env python3
"""Hard-25 three-arm benchmark for local Qwen3 4B. Never calls Gemini."""
from __future__ import annotations
import argparse, hashlib, json, math, os, platform, statistics, subprocess, sys, time
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import app

VERSION="three-arm-hard25-v1.0"
ARMS=("base_qwen","qwen_then_tools","local_context_qwen")
MAX_SEEDS=10
MAX_REFS=6
DEFAULT_CASES=ROOT/"hard25_qwen3_4b_three_arm_cases.json"
DEFAULT_DB=ROOT/"data/zebrafish_esm.db"
DEFAULT_CP=ROOT/"qwen3_4b_three_arm_25_checkpoint.json"
DEFAULT_OUT=ROOT/"qwen3_4b_three_arm_25_results.json"

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def git(*args):
    try: return subprocess.check_output(["git",*args],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
    except Exception: return ""

def write_json(path,obj):
    path=Path(path); tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+"\n",encoding="utf-8"); tmp.replace(path)

def load_cases(path):
    d=json.loads(Path(path).read_text(encoding="utf-8")); cases=d.get("cases") or []
    if len(cases)!=25 or len({c.get("id") for c in cases})!=25: raise ValueError("Need exactly 25 unique hard cases.")
    return d

def base_prompt(q):
    return f"""Act as a zebrafish biologist selecting seed proteins for an ESM similarity search.

USER QUESTION:
{q}

Use only your internal knowledge. You have no web search, no local database context, no PubMed/Europe PMC/QuickGO evidence, and no validation feedback before answering.

Rules:
- Keep the scope unchanged.
- Prefer canonical, commonly used, directly relevant Danio rerio genes.
- Use exact zebrafish symbols when known, including paralog suffixes such as a/b or .1/.2.
- If unsure of the exact zebrafish symbol, put a known human/mouse gene in reference_candidates instead of inventing one.
- Return at most {MAX_SEEDS} zebrafish candidates and {MAX_REFS} reference candidates.
- Do not claim external search or verification.

Return only:
{{"normalized_question":"...","zebrafish_candidates":[{{"gene":"exact zebrafish symbol","species":"zebrafish","uniprot_accession":"","reason":"short reason"}}],"reference_candidates":[{{"gene":"human or mouse symbol","species":"human or mouse","reason":"short reason"}}],"rationale":"brief summary"}}
"""

def local_prompt(q,ctx):
    return f"""Act as a zebrafish biologist selecting seed proteins for an ESM similarity search.

USER QUESTION:
{q}

Use your internal biology knowledge plus the local zebrafish vocabulary below. You have no web/literature retrieval.

LOCAL ZEBRAFISH DATABASE CONTEXT (lexical retrieval, NOT biological ranking):
{json.dumps(ctx,ensure_ascii=False)}

Rules:
- Keep the scope unchanged.
- Prefer canonical, directly relevant zebrafish genes over lexical overlap.
- Use this context to recognize exact zebrafish symbols, not as a relevance ranking.
- Use exact zebrafish paralog suffixes when known.
- Put uncertain human/mouse candidates in reference_candidates.
- Return at most {MAX_SEEDS} zebrafish candidates and {MAX_REFS} reference candidates.

Return only:
{{"normalized_question":"...","zebrafish_candidates":[{{"gene":"exact zebrafish symbol","species":"zebrafish","uniprot_accession":"","reason":"short reason"}}],"reference_candidates":[{{"gene":"human or mouse symbol","species":"human or mouse","reason":"short reason"}}],"rationale":"brief summary"}}
"""

def compact_trace(traces):
    out=[]
    for t in traces:
        r=t.get("local_resolution") or {}; p=r.get("protein") or {}
        out.append({"proposed_term":t.get("term"),"resolved":bool(r),"resolved_gene":p.get("name") or "","protein_id":p.get("protein_id") or "","resolved_by":r.get("resolved_by") or ""})
    return out

def compact_orthology(seeds):
    out=[]
    for s in seeds:
        p=app.PROTEINS[int(s["index"])]
        out.append({"reference_species":s.get("reference_species") or "","reference_gene":s.get("reference_gene") or s.get("retrieval_term") or "","zebrafish_gene":p.get("name") or "","protein_id":p.get("protein_id") or "","resolved_by":s.get("resolved_by") or ""})
    return out

def compact_evidence(items):
    out=[]
    for x in items:
        src=str(x.get("source") or "")
        if src in ("PubMed","Europe PMC"):
            out.append({"source":src,"title":str(x.get("title") or "")[:260],"abstract":str(x.get("abstract") or "")[:420],"pmid":x.get("pmid"),"year":x.get("year")})
        elif src=="QuickGO":
            out.append({"source":src,"go_id":x.get("go_id"),"name":str(x.get("name") or "")[:180],"definition":str(x.get("definition") or "")[:300]})
    return out

def second_prompt(q,first,resolution,orthology,ctx,evidence):
    return f"""Act as a zebrafish biologist performing a SECOND-PASS review of your own initial answer.

ORIGINAL QUESTION:
{q}

FIRST-PASS ANSWER (made before retrieval):
{json.dumps(first,ensure_ascii=False)}

CANDIDATE IDENTIFIER CHECKS (UniProt + local DB):
{json.dumps(resolution,ensure_ascii=False)}

REFERENCE ORTHOLOGY CHECKS (Ensembl -> zebrafish):
{json.dumps(orthology,ensure_ascii=False)}

LOCAL ZEBRAFISH VOCABULARY (lexical only, NOT ranking):
{json.dumps(ctx,ensure_ascii=False)}

AUTHORITATIVE RETRIEVAL (PubMed + Europe PMC + QuickGO):
{json.dumps(evidence,ensure_ascii=False)}

Use the tools to verify, repair, reject, or rerank the first-pass candidates. You may add a missing candidate only when strongly supported and an exact zebrafish symbol is supplied by the tool output.

Rules:
- Canonical biological relevance is more important than lexical similarity.
- Do not promote a protein merely because its identifier is valid.
- Preserve strong first-pass biology unless evidence/nomenclature justifies a change.
- Repair mammalian-style/outdated names when exact zebrafish symbols are supplied.
- Prefer direct zebrafish evidence.
- Return at most {MAX_SEEDS} zebrafish candidates and {MAX_REFS} reference candidates.

Return only:
{{"normalized_question":"...","zebrafish_candidates":[{{"gene":"exact zebrafish symbol","species":"zebrafish","uniprot_accession":"","reason":"short reason"}}],"reference_candidates":[{{"gene":"human or mouse symbol","species":"human or mouse","reason":"short reason"}}],"rationale":"brief second-pass summary"}}
"""

def clean(structured,q):
    return {
        "normalized_question":str(structured.get("normalized_question") or f"Danio rerio: {q}")[:240],
        "zebrafish_candidates":app._clean_candidates(structured.get("zebrafish_candidates"),{"zebrafish","danio rerio"},MAX_SEEDS),
        "reference_candidates":app._clean_candidates(structured.get("reference_candidates"),{"human","mouse"},MAX_REFS),
        "rationale":str(structured.get("rationale") or "")[:1200],
    }

def call_qwen(prompt,q):
    raws=[]; original=app._ollama_text
    def capture(p):
        text=original(p); raws.append(text); return text
    app._ollama_text=capture
    try: structured,_=app._ollama_json(prompt)
    finally: app._ollama_text=original
    return clean(structured,q),raws

def seed_public(s):
    p=app.PROTEINS[int(s["index"])]
    return {"gene":p.get("name") or "","protein_id":p.get("protein_id") or "","source":s.get("source") or "","retrieval_term":s.get("retrieval_term") or "","resolved_by":s.get("resolved_by") or "","uniprot_accession":s.get("uniprot_accession") or "","ai_reason":s.get("ai_reason") or ""}

def score(plan,expected,k):
    t=time.monotonic()
    direct,traces,errors=app.resolve_targeted_uniprot_candidates(plan.get("zebrafish_candidates") or [])
    refs=app.orthology_seeds(plan.get("reference_candidates") or []) if len(direct)<4 else []
    seeds=app._merge_seeds(direct,refs,limit=MAX_SEEDS)
    validation=time.monotonic()-t
    t=time.monotonic(); neighbors=app.discovery_neighbors([int(s["index"]) for s in seeds],k) if seeds else []; esm=time.monotonic()-t
    valid=[seed_public(s) for s in seeds]; genes=[x["gene"] for x in valid if x["gene"]]; lower=[g.lower() for g in genes]
    hits=[]; ranks=[]
    for e in expected:
        if e.lower() in lower: hits.append(e); ranks.append(lower.index(e.lower())+1)
    first=min(ranks) if ranks else None
    byterm={str(x.get("term") or "").lower():x for x in traces}; unresolved=[]
    for c in plan.get("zebrafish_candidates") or []:
        term=str(c.get("gene") or ""); tr=byterm.get(term.lower())
        if not tr or not tr.get("local_resolution"): unresolved.append(term)
    proposed=len(plan.get("zebrafish_candidates") or [])
    return {
        "validated_seed_genes":genes,"validated_seeds":valid,"validated_seed_count":len(valid),"pipeline_has_seed":bool(valid),
        "unresolved_proposed_genes":unresolved,"unresolved_gene_count":len(unresolved),"proposed_gene_count":proposed,
        "unresolved_rate":len(unresolved)/proposed if proposed else 0.0,"resolution_trace":traces,"validation_errors":errors,
        "esm_neighbors":neighbors,"canonical_hits":hits,"canonical_hit_any":bool(hits),
        "canonical_recall":len(hits)/len(expected) if expected else 0.0,"first_canonical_rank":first,
        "hit_at_1":bool(first and first<=1),"hit_at_3":bool(first and first<=3),"hit_at_5":bool(first and first<=5),
        "mrr_first_canonical":1/first if first else 0.0,"validation_seconds":round(validation,3),"esm_seconds":round(esm,3)
    }

def run_base(c,k):
    t=time.monotonic(); plan,raw=call_qwen(base_prompt(c["query"]),c["query"]); r=score(plan,c["expected_examples"],k)
    r.update({"plan":plan,"raw_model_outputs":raw,"retrieval_sources":[],"retrieval_errors":[],"total_seconds":round(time.monotonic()-t,3)})
    return r

def run_local(c,k):
    t=time.monotonic(); ctx=app._local_question_context(c["query"],limit=30); plan,raw=call_qwen(local_prompt(c["query"],ctx),c["query"]); r=score(plan,c["expected_examples"],k)
    r.update({"plan":plan,"raw_model_outputs":raw,"retrieval_sources":["Local zebrafish database lexical context"],"local_context_records":len(ctx),"local_context":ctx,"retrieval_errors":[],"total_seconds":round(time.monotonic()-t,3)})
    return r

def run_tools(c,k):
    t0=time.monotonic()
    first,raw1=call_qwen(base_prompt(c["query"]),c["query"]); t1=time.monotonic()
    search_q=first.get("normalized_question") or c["query"]
    ctx=app._local_question_context(search_q,limit=30)
    evidence,eerrs=app.fetch_authoritative_evidence(search_q)
    _,traces,rerrs=app.resolve_targeted_uniprot_candidates(first.get("zebrafish_candidates") or [])
    orth=app.orthology_seeds(first.get("reference_candidates") or [])
    resolution=compact_trace(traces); orthology=compact_orthology(orth); ev=compact_evidence(evidence); t2=time.monotonic()
    final,raw2=call_qwen(second_prompt(c["query"],first,resolution,orthology,ctx,ev),c["query"]); t3=time.monotonic()
    r=score(final,c["expected_examples"],k)
    r.update({
        "first_pass_plan":first,"plan":final,"raw_model_outputs":{"first_pass":raw1,"second_pass":raw2},
        "retrieval_sources":["Local zebrafish database lexical context","PubMed","Europe PMC","QuickGO","UniProt","Ensembl","Local zebrafish DB"],
        "local_context_records":len(ctx),"authoritative_evidence_records":len(evidence),"candidate_resolution_checks":resolution,
        "orthology_checks":orthology,"authoritative_evidence":ev,"retrieval_errors":[*eerrs,*rerrs],
        "first_pass_seconds":round(t1-t0,3),"retrieval_seconds":round(t2-t1,3),"second_pass_seconds":round(t3-t2,3),
        "total_seconds":round(time.monotonic()-t0,3)
    })
    return r

def summary(cases,arm):
    rows=[c[arm] for c in cases]; n=len(rows); lat=sorted(float(r.get("total_seconds") or 0) for r in rows)
    unresolved=sum(int(r.get("unresolved_gene_count") or 0) for r in rows); proposed=sum(int(r.get("proposed_gene_count") or 0) for r in rows)
    return {
        "n":n,"canonical_any_rate":sum(bool(r.get("canonical_hit_any")) for r in rows)/n,
        "mean_canonical_recall":statistics.mean(float(r.get("canonical_recall") or 0) for r in rows),
        "hit_at_1_rate":sum(bool(r.get("hit_at_1")) for r in rows)/n,"hit_at_3_rate":sum(bool(r.get("hit_at_3")) for r in rows)/n,
        "hit_at_5_rate":sum(bool(r.get("hit_at_5")) for r in rows)/n,"mean_mrr":statistics.mean(float(r.get("mrr_first_canonical") or 0) for r in rows),
        "pipeline_seed_rate":sum(bool(r.get("pipeline_has_seed")) for r in rows)/n,"mean_validated_seeds":statistics.mean(int(r.get("validated_seed_count") or 0) for r in rows),
        "unresolved_rate":unresolved/proposed if proposed else 0.0,"median_latency_seconds":statistics.median(lat),
        "p90_latency_seconds":lat[max(0,math.ceil(.9*n)-1)]
    }

def key(r):
    return (float(r.get("canonical_recall") or 0),float(r.get("mrr_first_canonical") or 0),float(bool(r.get("hit_at_5"))),float(bool(r.get("hit_at_3"))),float(bool(r.get("hit_at_1"))),-float(r.get("unresolved_rate") or 0),float(r.get("validated_seed_count") or 0))

def pair(cases,a,b):
    out={"left_better":0,"right_better":0,"tie":0}
    for c in cases:
        ka,kb=key(c[a]),key(c[b]); out["left_better" if ka>kb else "right_better" if kb>ka else "tie"]+=1
    return out

def print_summary(cases):
    s={a:summary(cases,a) for a in ARMS}
    print("\nFINAL THREE-ARM SUMMARY")
    print(f"{'metric':26}{'BASE':>14}{'QWEN→TOOLS':>16}{'LOCAL-CONTEXT':>18}")
    fields=[("canonical any","canonical_any_rate","pct"),("mean canonical recall","mean_canonical_recall","pct"),("Hit@1","hit_at_1_rate","pct"),("Hit@3","hit_at_3_rate","pct"),("Hit@5","hit_at_5_rate","pct"),("MRR","mean_mrr","num"),("queries >=1 seed","pipeline_seed_rate","pct"),("mean validated seeds","mean_validated_seeds","num"),("unresolved rate","unresolved_rate","pct"),("median latency","median_latency_seconds","sec"),("P90 latency","p90_latency_seconds","sec")]
    for label,f,typ in fields:
        vals=[]
        for a in ARMS:
            v=s[a][f]; vals.append(f"{100*v:.1f}%" if typ=="pct" else f"{v:.2f}s" if typ=="sec" else f"{v:.3f}")
        print(f"{label:26}{vals[0]:>14}{vals[1]:>16}{vals[2]:>18}")
    print("\nPAIRWISE: canonical recall/ranking first; identifier quality second")
    for a,b in [("qwen_then_tools","base_qwen"),("local_context_qwen","base_qwen"),("qwen_then_tools","local_context_qwen")]:
        x=pair(cases,a,b); print(f"{a} vs {b}: {x}")

def metadata(args,cases,runner):
    return {"timestamp_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"git_commit":git("rev-parse","HEAD"),"git_branch":git("branch","--show-current"),
        "ollama_model":args.model,"ollama_url":args.ollama_url,"python_version":sys.version,"platform":platform.platform(),"database_filename":Path(args.db).name,
        "protein_count":len(app.PROTEINS),"case_count":25,"runner_version":VERSION,"cases_sha256":sha(cases),"app_sha256":sha(ROOT/"app.py"),
        "runner_sha256":sha(runner),"database_sha256":sha(args.db),"k":args.k,"ollama_settings":{"temperature":0.1,"num_ctx":8192,"num_predict":1200,"think":False,"format":"json"}}

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--cases",default=str(DEFAULT_CASES)); p.add_argument("--db",default=str(DEFAULT_DB)); p.add_argument("--model",default="qwen3:4b-instruct")
    p.add_argument("--ollama-url",default="http://127.0.0.1:11434"); p.add_argument("--k",type=int,default=5); p.add_argument("--checkpoint",default=str(DEFAULT_CP)); p.add_argument("--out",default=str(DEFAULT_OUT)); return p.parse_args()

def main():
    args=parse_args(); cases_path=Path(args.cases).resolve(); runner=Path(__file__).resolve(); cp=Path(args.checkpoint).resolve(); out=Path(args.out).resolve(); cases=load_cases(cases_path)["cases"]
    os.environ["AI_PROVIDER"]="ollama"; os.environ["OLLAMA_MODEL"]=args.model; os.environ["OLLAMA_URL"]=args.ollama_url
    app.load_database(args.db); meta=metadata(args,cases_path,runner); completed=[]
    if cp.exists():
        old=json.loads(cp.read_text(encoding="utf-8")); changed=[k for k in ("cases_sha256","app_sha256","runner_sha256","database_sha256","ollama_model","k") if (old.get("metadata") or {}).get(k)!=meta.get(k)]
        if changed: raise RuntimeError("Checkpoint mismatch: "+", ".join(changed))
        completed=list(old.get("cases") or []); print(f"Resuming with {len(completed)} completed three-arm cases.")
    done={int(c["id"]) for c in completed}
    for pos,c in enumerate(cases,1):
        if int(c["id"]) in done: continue
        print(f"\n[{pos:02d}/25] {c['category']} / {c['subtopic']} / {c['wording']}\n{c['query']}")
        row={k:c[k] for k in ("id","category","subtopic","wording","query_specificity","query","expected_examples")}
        for arm in ARMS:
            try: r=run_base(c,args.k) if arm=="base_qwen" else run_tools(c,args.k) if arm=="qwen_then_tools" else run_local(c,args.k)
            except KeyboardInterrupt: raise
            except Exception as e: r={"error":f"{type(e).__name__}: {e}","canonical_hit_any":False,"canonical_hits":[],"canonical_recall":0.0,"first_canonical_rank":None,"hit_at_1":False,"hit_at_3":False,"hit_at_5":False,"mrr_first_canonical":0.0,"validated_seed_count":0,"pipeline_has_seed":False,"unresolved_gene_count":0,"proposed_gene_count":0,"unresolved_rate":0.0,"total_seconds":0.0}
            row[arm]=r; print(f"  {arm:18} canonical={r.get('canonical_recall',0):.2f} first={str(r.get('first_canonical_rank')):>4} seeds={r.get('validated_seed_count',0):>2} unresolved={r.get('unresolved_gene_count',0):>2} {r.get('total_seconds',0):>7.2f}s"+(f" ERROR={r['error']}" if r.get("error") else ""))
        completed.append(row); done.add(int(c["id"]))
        write_json(cp,{"metadata":meta,"protocol":{"A":"base Qwen","B":"Qwen first then tools then Qwen","C":"local context then Qwen","primary_metric":"canonical recall/ranking; examples are diagnostic, not exhaustive truth"},"cases":completed})
    final={"metadata":meta,"summary":{a:summary(completed,a) for a in ARMS},"pairwise":{"tools_vs_base":pair(completed,"qwen_then_tools","base_qwen"),"local_vs_base":pair(completed,"local_context_qwen","base_qwen"),"tools_vs_local":pair(completed,"qwen_then_tools","local_context_qwen")},"cases":completed}
    write_json(out,final); print_summary(completed); print(f"\nCheckpoint: {cp}\nFinal results: {out}")
    return 0

if __name__=="__main__": raise SystemExit(main())
