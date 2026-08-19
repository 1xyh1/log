#!/usr/bin/env python3
"""Hardened E3 conservative leakage closure.

Scientific rule (unchanged):
  final graph = strong edges U all review edges
  if a final component intersects any historical contaminated ID, the entire
  final component is FORCE_TRAIN and therefore ineligible for FINAL HOLDOUT.

Engineering rule: fail closed on malformed/null/duplicate input, concurrent runs,
output conflicts, permission/path errors, timeouts, or public sensitive-data leakage.
"""
from __future__ import annotations

import argparse, collections, json, math, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from multimodal.t1gr_secure_io import (  # noqa:E402
    Deadline, GateError, assert_public_safe, atomic_json_write, check_existing_output, ensure_private_input, ensure_private_output, ensure_public_output, ensure_repo_input, fail, file_lock, read_json_bounded, require_dict, require_keys,
    require_list, require_unchanged, safe_error_message, sha256_file, sha256_json, stat_token, validate_identifier,
)

SCRIPT_VERSION = "t1gr-e3-finalize-components-hardened-v1.2"
FROZEN_POLICY_SHA256="9f60361ae393983b1f972b428005f354429f2b9abd71c2ee50459706b7ebe9ae"
ALLOWED_EDGE_CLASSES = {"EXACT_BOTH_MODALITIES", "STRONG_NEAR_DUPLICATE", "REVIEW_NEAR_DUPLICATE"}

class DSU:
    def __init__(self, ids):
        self.p={x:x for x in ids}; self.s={x:1 for x in ids}
    def find(self,x):
        while self.p[x]!=x:
            self.p[x]=self.p[self.p[x]]; x=self.p[x]
        return x
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a==b:return
        if self.s[a]<self.s[b]:a,b=b,a
        self.p[b]=a;self.s[a]+=self.s[b]
    def comps(self):
        d=collections.defaultdict(list)
        for x in self.p:d[self.find(x)].append(x)
        return [sorted(v) for v in d.values()]


def validate_policy(p: dict) -> None:
    require_keys(p, ("schema","source_visual_leakage","review_edge_adjudication","component_gate","security"), "POLICY_MISSING_FIELDS")
    if p["schema"] != "t1gr-e3-closure-split-policy-v1.2": fail("BAD_POLICY_SCHEMA")
    if p["review_edge_adjudication"] != "MERGE_ALL_CONSERVATIVELY": fail("REVIEW_POLICY_DRIFT")
    src=require_dict(p["source_visual_leakage"],"BAD_SOURCE_POLICY")
    require_keys(src,("schema","source_policy_sha256","public_report","expected_formal_n","expected_historical_n","expected_historical_hash_entries_verified","expected_historical_seed_ids","expected_strong_edges","expected_review_edges","expected_strong_only_component_count","expected_strong_only_nontrivial_components","expected_strong_only_max_component"),"SOURCE_POLICY_MISSING")
    sec=require_dict(p["security"],"BAD_SECURITY_POLICY")
    require_keys(sec,("private_parent_must_preexist","public_output_prefix","lock_wait_seconds","lock_stale_seconds","closure_timeout_seconds","max_private_json_bytes","max_policy_json_bytes"),"SECURITY_POLICY_MISSING")


def validate_visual_private(v: dict, p: dict, deadline: Deadline) -> tuple[list[str], list[dict], list[dict], set[str]]:
    require_keys(v,("schema","policy_sha256","internal_components_all","internal_strong_edges","internal_review_edges","force_train_ids_due_historical_contamination"),"VISUAL_PRIVATE_MISSING_FIELDS")
    src=p["source_visual_leakage"]
    if v["schema"] != src["schema"]: fail("BAD_VISUAL_PRIVATE_SCHEMA")
    if v["policy_sha256"] != src["source_policy_sha256"]: fail("UPSTREAM_POLICY_SHA_DRIFT")
    raw_components=require_list(v["internal_components_all"],"BAD_COMPONENTS_TYPE")
    ids=[]; seen=set()
    for comp in raw_components:
        deadline.check()
        comp=require_list(comp,"BAD_COMPONENT_TYPE")
        if not comp: fail("EMPTY_COMPONENT")
        local=set()
        for x in comp:
            sx=validate_identifier(x)
            if sx in local: fail("DUPLICATE_ID_WITHIN_COMPONENT")
            if sx in seen: fail("DUPLICATE_ID_ACROSS_COMPONENTS")
            local.add(sx);seen.add(sx);ids.append(sx)
    expected=int(src["expected_formal_n"])
    if len(ids)!=expected: fail("FORMAL_ID_COUNT_DRIFT",f"observed={len(ids)}")

    def edges(key, expected_count, allowed_classes):
        raw=require_list(v[key],f"BAD_{key.upper()}_TYPE")
        if len(raw)!=int(expected_count): fail("UPSTREAM_EDGE_COUNT_DRIFT",f"kind={key};observed={len(raw)}")
        out=[];pairs=set()
        for e in raw:
            deadline.check()
            e=require_dict(e,"BAD_EDGE_OBJECT")
            require_keys(e,("a","b","class"),"EDGE_MISSING_FIELDS")
            a=validate_identifier(e["a"]);b=validate_identifier(e["b"]);cl=e["class"]
            if a==b: fail("SELF_EDGE")
            if a not in seen or b not in seen: fail("EDGE_ENDPOINT_OUTSIDE_UNIVERSE")
            if cl not in allowed_classes: fail("EDGE_CLASS_WRONG_FOR_LIST")
            pair=tuple(sorted((a,b)))
            if pair in pairs: fail("DUPLICATE_EDGE")
            pairs.add(pair);out.append({"a":a,"b":b,"class":cl})
        return out,pairs
    strong,strong_pairs=edges("internal_strong_edges",src["expected_strong_edges"],{"EXACT_BOTH_MODALITIES","STRONG_NEAR_DUPLICATE"})
    review,review_pairs=edges("internal_review_edges",src["expected_review_edges"],{"REVIEW_NEAR_DUPLICATE"})
    if strong_pairs & review_pairs: fail("EDGE_PRESENT_IN_STRONG_AND_REVIEW")
    force_raw=require_list(v["force_train_ids_due_historical_contamination"],"BAD_FORCE_TRAIN_TYPE")
    force=[];fs=set()
    for x in force_raw:
        sx=validate_identifier(x)
        if sx in fs: fail("DUPLICATE_FORCE_TRAIN_ID")
        if sx not in seen: fail("FORCE_TRAIN_ID_OUTSIDE_UNIVERSE")
        fs.add(sx);force.append(sx)
    if len(force)!=int(src["expected_historical_seed_ids"]): fail("HISTORICAL_SEED_COUNT_DRIFT",f"observed={len(force)}")
    # Rebuild the accepted strong-only graph and verify the private component partition itself.
    sd=DSU(sorted(ids))
    for e in strong: sd.union(e["a"],e["b"])
    rebuilt=sorted((tuple(c) for c in sd.comps()), key=lambda c:(-len(c),c))
    supplied=sorted((tuple(sorted(c)) for c in raw_components), key=lambda c:(-len(c),c))
    if rebuilt!=supplied: fail("STRONG_ONLY_COMPONENT_PROVENANCE_MISMATCH")
    if len(rebuilt)!=int(src["expected_strong_only_component_count"]): fail("STRONG_ONLY_COMPONENT_COUNT_DRIFT")
    if sum(len(c)>1 for c in rebuilt)!=int(src["expected_strong_only_nontrivial_components"]): fail("STRONG_ONLY_NONTRIVIAL_COUNT_DRIFT")
    if max(map(len,rebuilt),default=0)!=int(src["expected_strong_only_max_component"]): fail("STRONG_ONLY_MAX_COMPONENT_DRIFT")
    return sorted(ids),strong,review,fs


def validate_upstream_public(pub: dict, v: dict, p: dict, force: set[str]) -> None:
    src=p["source_visual_leakage"]
    require_keys(pub,("schema","formal_n","historical_n","policy_sha256","historical_contract_sha256","historical_hash_entries_verified","formal_zip_member_metadata_commitment","historical_contaminated_formal_count","force_train_ids_commitment","internal_strong_edge_count","internal_review_edge_count","internal_component_count","internal_nontrivial_component_count","internal_max_component_size","internal_graph_gate"),"UPSTREAM_PUBLIC_MISSING_FIELDS")
    if pub["schema"]!="t1gr-visual-leakage-public-v1": fail("BAD_UPSTREAM_PUBLIC_SCHEMA")
    checks=[
      (pub["formal_n"],src["expected_formal_n"],"UPSTREAM_FORMAL_N_DRIFT"),
      (pub["historical_n"],src["expected_historical_n"],"UPSTREAM_HISTORICAL_N_DRIFT"),
      (pub["historical_hash_entries_verified"],src["expected_historical_hash_entries_verified"],"UPSTREAM_HIST_HASH_VERIFY_DRIFT"),
      (pub["historical_contaminated_formal_count"],src["expected_historical_seed_ids"],"UPSTREAM_HIST_CONTAMINATION_DRIFT"),
      (pub["internal_strong_edge_count"],src["expected_strong_edges"],"UPSTREAM_STRONG_EDGE_DRIFT"),
      (pub["internal_review_edge_count"],src["expected_review_edges"],"UPSTREAM_REVIEW_EDGE_DRIFT"),
      (pub["internal_component_count"],src["expected_strong_only_component_count"],"UPSTREAM_COMPONENT_COUNT_DRIFT"),
      (pub["internal_nontrivial_component_count"],src["expected_strong_only_nontrivial_components"],"UPSTREAM_NONTRIVIAL_DRIFT"),
      (pub["internal_max_component_size"],src["expected_strong_only_max_component"],"UPSTREAM_MAX_COMPONENT_DRIFT"),
    ]
    for observed,expected,code in checks:
        if int(observed)!=int(expected): fail(code)
    if pub["policy_sha256"]!=src["source_policy_sha256"]: fail("UPSTREAM_PUBLIC_POLICY_SHA_DRIFT")
    if pub["internal_graph_gate"]!="PASS": fail("UPSTREAM_GRAPH_NOT_PASS")
    if sha256_json(sorted(force))!=pub["force_train_ids_commitment"]: fail("UPSTREAM_FORCE_TRAIN_COMMITMENT_MISMATCH")
    require_keys(v,("historical_contract_sha256","historical_hash_entries_verified","formal_zip_member_metadata_commitment"),"VISUAL_PRIVATE_PROVENANCE_MISSING")
    if v["historical_contract_sha256"]!=pub["historical_contract_sha256"]: fail("HISTORICAL_CONTRACT_SHA_MISMATCH")
    if int(v["historical_hash_entries_verified"])!=int(pub["historical_hash_entries_verified"]): fail("HISTORICAL_VERIFY_COUNT_MISMATCH")
    if v["formal_zip_member_metadata_commitment"]!=pub["formal_zip_member_metadata_commitment"]: fail("FORMAL_ZIP_COMMITMENT_MISMATCH")

def run(args) -> dict:
    repo=ROOT.resolve(strict=True)
    pol_path=ensure_repo_input(repo,args.policy,"config")
    # Load a minimal safe security default before policy validation.
    policy_token=stat_token(pol_path)
    p=read_json_bounded(pol_path, 1<<20)
    validate_policy(p); sec=p["security"]
    deadline=Deadline(float(args.timeout_seconds or sec["closure_timeout_seconds"]))
    visual=ensure_private_input(Path(args.visual_private),repo)
    private_out=ensure_private_output(Path(args.private_out),repo,bool(sec["private_parent_must_preexist"]))
    public_out=ensure_public_output(repo,args.public_out,sec["public_output_prefix"])
    lock=public_out.with_suffix(public_out.suffix+".lock")
    with file_lock(lock,float(sec["lock_wait_seconds"]),float(sec["lock_stale_seconds"])):
        deadline.check()
        visual_token=stat_token(visual)
        v=read_json_bounded(visual,int(sec["max_private_json_bytes"]),"t1gr-visual-leakage-private-v1")
        ids,strong,review,force=validate_visual_private(v,p,deadline)
        source_public_path=ensure_repo_input(repo,p["source_visual_leakage"]["public_report"],"reports/step4_t1gr")
        source_public_token=stat_token(source_public_path)
        source_public=read_json_bounded(source_public_path,1<<20,"t1gr-visual-leakage-public-v1")
        validate_upstream_public(source_public,v,p,force)
        visual_sha=sha256_file(visual,deadline); policy_sha=sha256_file(pol_path,deadline)
        require_unchanged(visual,visual_token,"VISUAL_PRIVATE_CHANGED_DURING_RUN")
        require_unchanged(source_public_path,source_public_token,"UPSTREAM_PUBLIC_CHANGED_DURING_RUN")
        require_unchanged(pol_path,policy_token,"POLICY_CHANGED_DURING_RUN")
        if policy_sha!=FROZEN_POLICY_SHA256: fail("FROZEN_POLICY_SHA_DRIFT")
        request_fp=sha256_json({"script":SCRIPT_VERSION,"visual_private_sha256":visual_sha,"policy_sha256":policy_sha})
        existing_private=check_existing_output(private_out,request_fp)
        existing_public=check_existing_output(public_out,request_fp)
        if existing_private is not None and existing_public is not None:
            pub_obj,pub_sha=existing_public;_,priv_sha=existing_private
            if not bool(pub_obj.get("component_gate_passed")): fail("COMPONENT_GATE_HOLD")
            return {"status":"PASS","request_fingerprint":request_fp,"private_output_sha256":priv_sha,"public_output_sha256":pub_sha,"idempotent_reuse":True}
        d=DSU(ids)
        for e in strong+review:
            deadline.check(); d.union(e["a"],e["b"])
        comps=sorted(d.comps(),key=lambda x:(-len(x),x))
        if sum(map(len,comps))!=len(ids) or len({x for c in comps for x in c})!=len(ids): fail("COMPONENT_COVERAGE_FAIL")
        force_comps=[]; force_ids=set()
        for c in comps:
            deadline.check()
            if force.intersection(c): force_comps.append(c);force_ids.update(c)
        maxc=max(map(len,comps),default=0)
        lim=max(int(p["component_gate"]["max_component_abs"]),math.ceil(len(ids)*float(p["component_gate"]["max_component_fraction"])))
        gate=maxc<=lim
        hist=collections.Counter(map(len,comps))
        private={
          "schema":"t1gr-e3-final-components-private-v1.2","script_version":SCRIPT_VERSION,
          "source_visual_private_sha256":visual_sha,"policy_sha256":policy_sha,
          "review_edge_adjudication":"MERGE_ALL_CONSERVATIVELY","all_components":comps,
          "force_train_seed_ids":sorted(force),"force_train_components":force_comps,
          "force_train_ids_after_component_propagation":sorted(force_ids),"component_gate_passed":gate,
          "max_component_size":maxc,"component_count":len(comps),"strong_edge_count":len(strong),"review_edge_count":len(review)
        }
        public={
          "schema":"t1gr-e3-final-components-public-v1.2","script_version":SCRIPT_VERSION,
          "source_visual_private_sha256":visual_sha,"policy_sha256":policy_sha,
          "review_edge_adjudication":"MERGE_ALL_CONSERVATIVELY","formal_n":len(ids),
          "source_strong_edge_count":len(strong),"source_review_edge_count":len(review),
          "component_count":len(comps),"nontrivial_component_count":sum(len(c)>1 for c in comps),
          "component_size_distribution":{str(k):int(vv) for k,vv in sorted(hist.items())},
          "max_component_size":maxc,"component_review_limit":lim,"component_gate_passed":gate,
          "historical_seed_contaminated_id_count":len(force),"force_train_component_count":len(force_comps),
          "force_train_id_count_after_component_propagation":len(force_ids),
          "force_train_ids_commitment":sha256_json(sorted(force_ids)),"all_components_commitment":sha256_json(comps),
          "private_ids_in_public_report":False,"e3_component_status":"PASS" if gate else "HOLD_OVERSIZED_COMPONENT",
          "split_status":"HOLD_UNTIL_COMPONENT_GATE_PASS_AND_SPLIT_PROPOSAL"
        }
        assert_public_safe(public)
        psha,preuse=atomic_json_write(private_out,private,private=True,request_fingerprint=request_fp)
        try:
            usha,ureuse=atomic_json_write(public_out,public,private=False,request_fingerprint=request_fp)
        except Exception:
            # Private is deterministic and idempotent; rerun safely resumes public write.
            raise
        result={"status":"PASS" if gate else "HOLD","request_fingerprint":request_fp,"private_output_sha256":psha,"public_output_sha256":usha,"idempotent_reuse":bool(preuse and ureuse)}
        if not gate: fail("COMPONENT_GATE_HOLD")
        return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--visual-private",required=True);ap.add_argument("--policy",default="config/t1gr_e3_closure_split_policy.json")
    ap.add_argument("--private-out",required=True);ap.add_argument("--public-out",default="reports/step4_t1gr/leakage_components_final_public.json")
    ap.add_argument("--timeout-seconds",type=float,default=None)
    args=ap.parse_args()
    try:
        r=run(args);print(json.dumps(r,ensure_ascii=False,indent=2))
    except Exception as e:
        print(json.dumps({"status":"FAIL","error":safe_error_message(e)},ensure_ascii=False),file=sys.stderr);raise SystemExit(2)
if __name__=="__main__":main()
