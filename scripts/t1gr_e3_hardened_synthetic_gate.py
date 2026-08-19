#!/usr/bin/env python3
"""End-to-end synthetic gate for the hardened E3 closure + split tooling.

Creates an isolated temporary repo; NEVER touches formal data or formal E3 reports.
Exercises 2000 synthetic IDs, accepted upstream-shape provenance, closure, split, and
idempotent reruns. No network access.
"""
from __future__ import annotations
import hashlib, json, shutil, subprocess, sys, tempfile, time, zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def sha256_json(x):
    return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def run_cmd(cmd,cwd,timeout):
    r=subprocess.run(cmd,cwd=cwd,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout)
    if r.returncode:
        raise RuntimeError(f"SYNTHETIC_SUBPROCESS_FAIL:{r.returncode}:{r.stderr.strip()[:120]}")
    return json.loads(r.stdout)

def main():
    base=Path(tempfile.mkdtemp(prefix="t1gr_e3_hardened_gate_"))
    try:
        repo=base/"repo";private=base/"private";private.mkdir()
        for d in ("src","scripts","config"):
            shutil.copytree(ROOT/d,repo/d)
        (repo/"reports/step4_t1gr").mkdir(parents=True)
        pol=json.loads((repo/"config/t1gr_e3_closure_split_policy.json").read_text(encoding="utf-8"))
        ids=[f"s{i:04d}" for i in range(2000)]
        sizes=[27,24,20,11,10,6,5,5,3,3,3]+[2]*14
        comps=[];pos=0
        for z in sizes: comps.append(ids[pos:pos+z]);pos+=z
        comps += [[x] for x in ids[pos:]]
        strong=[];pairs=set()
        for c in comps[:25]:
            for a,b in zip(c,c[1:]):
                pair=tuple(sorted((a,b)));pairs.add(pair);strong.append({"a":a,"b":b,"class":"STRONG_NEAR_DUPLICATE"})
        for c in comps[:25]:
            if len(strong)>=808: break
            for i,a in enumerate(c):
                if len(strong)>=808: break
                for b in c[i+1:]:
                    pair=tuple(sorted((a,b)))
                    if pair not in pairs:
                        pairs.add(pair);strong.append({"a":a,"b":b,"class":"STRONG_NEAR_DUPLICATE"})
                        if len(strong)>=808: break
        if len(strong)!=808: raise RuntimeError("SYNTHETIC_STRONG_EDGE_BUILD_FAIL")
        single=[c[0] for c in comps[25:]]
        review=[{"a":single[2*i],"b":single[2*i+1],"class":"REVIEW_NEAR_DUPLICATE"} for i in range(180)]
        force=ids[:18];hist_contract="a"*64;zip_commit="synthetic_zip_commitment"
        visual={"schema":"t1gr-visual-leakage-private-v1","policy_sha256":pol["source_visual_leakage"]["source_policy_sha256"],
                "historical_contract_sha256":hist_contract,"historical_hash_entries_verified":34,
                "formal_zip_member_metadata_commitment":zip_commit,"internal_components_all":comps,
                "internal_strong_edges":strong,"internal_review_edges":review,
                "force_train_ids_due_historical_contamination":force}
        visual_p=private/"visual.json";visual_p.write_text(json.dumps(visual),encoding="utf-8")
        upstream={"schema":"t1gr-visual-leakage-public-v1","formal_n":2000,"historical_n":17,
                  "policy_sha256":pol["source_visual_leakage"]["source_policy_sha256"],
                  "historical_contract_sha256":hist_contract,"historical_hash_entries_verified":34,
                  "formal_zip_member_metadata_commitment":zip_commit,"historical_contaminated_formal_count":18,
                  "force_train_ids_commitment":sha256_json(sorted(force)),"internal_strong_edge_count":808,
                  "internal_review_edge_count":180,"internal_component_count":1880,
                  "internal_nontrivial_component_count":25,"internal_max_component_size":27,"internal_graph_gate":"PASS"}
        (repo/"reports/step4_t1gr/visual_leakage_public.json").write_text(json.dumps(upstream),encoding="utf-8")
        formal=private/"formal.zip"
        with zipfile.ZipFile(formal,"w",zipfile.ZIP_DEFLATED) as z:
            for i,s in enumerate(ids):
                ext=".jpg" if i<149 else ".png"
                z.writestr(f"visible/{s}{ext}",b"")
                c1=i%12;c2=(i*5+3)%12
                z.writestr(f"labels/{s}.txt",f"{c1} 0.5 0.5 0.2 0.2\n{c2} 0.4 0.4 0.1 0.1\n")
        comp_p=private/"components.json";split_p=private/"split.json"
        ccmd=[sys.executable,str(repo/"scripts/t1gr_finalize_leakage_components.py"),"--visual-private",str(visual_p),"--private-out",str(comp_p),"--timeout-seconds","30"]
        t=time.monotonic();c1=run_cmd(ccmd,repo,45);closure_seconds=time.monotonic()-t
        c2=run_cmd(ccmd,repo,45)
        scmd=[sys.executable,str(repo/"scripts/t1gr_propose_component_split.py"),"--formal-zip",str(formal),"--components-private",str(comp_p),"--private-out",str(split_p),"--timeout-seconds","60"]
        t=time.monotonic();s1=run_cmd(scmd,repo,90);split_seconds=time.monotonic()-t
        t=time.monotonic();s2=run_cmd(scmd,repo,90);split_rerun_seconds=time.monotonic()-t
        pub=json.loads((repo/"reports/step4_t1gr/split_candidate_public.json").read_text())
        checks={
          "closure_first_pass":c1["status"]=="PASS" and not c1["idempotent_reuse"],
          "closure_rerun_idempotent":c2["status"]=="PASS" and c2["idempotent_reuse"],
          "split_first_pass":s1["status"]=="PASS" and not s1["idempotent_reuse"],
          "split_rerun_idempotent":s2["status"]=="PASS" and s2["idempotent_reuse"],
          "split_hard_gate":pub["hard_gate_passed"] is True,
          "union_equals_2000":pub["union_equals_2000"] is True,
          "sample_overlap_empty":pub["sample_overlap_empty"] is True,
          "components_not_split":pub["components_not_split"] is True,
          "historical_quarantine_respected":pub["historical_component_quarantine_respected"] is True,
          "public_has_no_private_ids":pub["private_ids_in_public_report"] is False,
          "candidate_does_not_authorize_e4":pub["e4_seal_authorized"] is False,
          "candidate_does_not_authorize_step1":pub["step1_authorized"] is False,
        }
        report={"schema":"t1gr-e3-hardened-synthetic-gate-v1.2","all_passed":all(checks.values()),"checks":checks,
                "timing_seconds":{"closure":round(closure_seconds,3),"split":round(split_seconds,3),"split_idempotent_rerun":round(split_rerun_seconds,3)},
                "synthetic_split_counts":pub["counts"]}
        print(json.dumps(report,ensure_ascii=False,indent=2))
        if not report["all_passed"]: raise SystemExit(3)
    finally:
        shutil.rmtree(base,ignore_errors=True)
if __name__=="__main__":main()
