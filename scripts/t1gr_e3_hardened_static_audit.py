#!/usr/bin/env python3
"""Static implementation audit for T1-GR E3 hardened closure/split tooling."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def load(name,path):
    s=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='reports/step4_t1gr/e3_hardened_static_audit.json');a=ap.parse_args()
    f=load('f',Path('scripts/t1gr_finalize_leakage_components.py'))
    p=load('p',Path('scripts/t1gr_propose_component_split.py'))
    policy_path=ROOT/'config/t1gr_e3_closure_split_policy.json';pol=json.loads(policy_path.read_text(encoding='utf-8'))
    policy_sha=hashlib.sha256(policy_path.read_bytes()).hexdigest()
    fs=(ROOT/'scripts/t1gr_finalize_leakage_components.py').read_text(encoding='utf-8')
    ps=(ROOT/'scripts/t1gr_propose_component_split.py').read_text(encoding='utf-8')
    ss=(ROOT/'src/multimodal/t1gr_secure_io.py').read_text(encoding='utf-8')
    checks={
      'policy_sha_pinned_finalize':f.FROZEN_POLICY_SHA256==policy_sha,
      'policy_sha_pinned_split':p.FROZEN_POLICY_SHA256==policy_sha,
      'accepted_formal_n_2000':pol['source_visual_leakage']['expected_formal_n']==2000,
      'accepted_historical_seed_18':pol['source_visual_leakage']['expected_historical_seed_ids']==18,
      'accepted_strong_edges_808':pol['source_visual_leakage']['expected_strong_edges']==808,
      'accepted_review_edges_180':pol['source_visual_leakage']['expected_review_edges']==180,
      'accepted_strong_components_1880':pol['source_visual_leakage']['expected_strong_only_component_count']==1880,
      'accepted_strong_nontrivial_25':pol['source_visual_leakage']['expected_strong_only_nontrivial_components']==25,
      'accepted_strong_max_27':pol['source_visual_leakage']['expected_strong_only_max_component']==27,
      'review_merge_all_frozen':pol['review_edge_adjudication']=='MERGE_ALL_CONSERVATIVELY',
      'source_public_report_pinned':pol['source_visual_leakage']['public_report']=='reports/step4_t1gr/visual_leakage_public.json',
      'finalize_uses_private_input_guard':'ensure_private_input' in fs,
      'split_uses_private_input_guard':'ensure_private_input' in ps,
      'finalize_uses_lock':'file_lock(' in fs,
      'split_uses_lock':'file_lock(' in ps,
      'finalize_uses_deadline':'Deadline(' in fs,
      'split_uses_deadline':'Deadline(' in ps,
      'finalize_public_safety_scan':'assert_public_safe(public)' in fs,
      'split_public_safety_scan':'assert_public_safe(public)' in ps,
      'idempotent_integrity_function':'check_existing_output' in ss and 'payload_sha256' in ss,
      'atomic_replace':'os.replace(' in ss,
      'fsync':'os.fsync(' in ss,
      'private_chmod':'_chmod_private' in ss,
      'unsafe_zip_name_guard':'validate_zip_name' in ps,
      'encrypted_zip_guard':'ZIP_ENCRYPTED_MEMBER_FORBIDDEN' in ps,
      'no_arbitrary_data_yaml':'--data' not in ps and 'args.data' not in ps,
      'no_direct_write_in_formal_scripts':'.write_text(' not in fs and '.write_text(' not in ps,
      'no_traceback_publication':'traceback' not in fs.lower() and 'traceback' not in ps.lower(),
      'e4_not_authorized_literal':'"e4_seal_authorized":False' in ps,
      'step1_not_authorized_literal':'"step1_authorized":False' in ps,
    }
    report={'schema':'t1gr-e3-hardened-static-audit-v1.2','passed':sum(checks.values()),'total':len(checks),'all_passed':all(checks.values()),'checks':checks,'policy_sha256':policy_sha}
    out=ROOT/a.out;out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if not report['all_passed']:raise SystemExit(2)
if __name__=='__main__':main()
