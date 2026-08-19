#!/usr/bin/env python3
"""Package-only static audit for T1-GR E2-E5 v2 P0/P1 fixes."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def check(name, cond, checks, detail=None):
    row={"passed":bool(cond)}
    if detail is not None: row["detail"]=detail
    checks[name]=row

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',default='reports/step4_t1gr/e2_e5_static_audit.json'); a=ap.parse_args()
    files={p.name:p.read_text(encoding='utf-8') for p in (ROOT/'scripts').glob('*.py')}
    core=(ROOT/'src/multimodal/t1gr_e2e5.py').read_text(encoding='utf-8')
    layout=json.loads((ROOT/'config/t1gr_layout_spec.template.json').read_text(encoding='utf-8'))
    ts=json.loads((ROOT/'config/step1_training_spec.template.json').read_text(encoding='utf-8'))
    c={}
    check('S01_private_contract_outside_repo','PRIVATE_CONTRACT_MUST_BE_OUTSIDE_REPO' in files['t1gr_build_contract.py'],c)
    check('S02_private_split_proposal_outside_repo','PRIVATE_SPLIT_PROPOSAL_MUST_BE_OUTSIDE_REPO' in files['t1gr_propose_split.py'],c)
    check('S03_public_contract_no_ids','contains_sample_ids' in files['t1gr_build_contract.py'],c)
    check('S04_public_freeze_no_ids','PUBLIC_FREEZE_EXPOSES_SAMPLE_IDS' in files['t1gr_freeze_split.py'],c)
    check('S05_no_frozen_before_training_literal','"frozen_before_training": True' not in files['t1gr_freeze_split.py'],c)
    check('S06_runner_proves_time_order','freeze_precedes_training_derived' in files['t1gr_run_step1_baseline.py'],c)
    check('S07_runner_no_data_cli','add_argument("--data"' not in files['t1gr_run_step1_baseline.py'],c)
    check('S08_eval_no_data_cli','add_argument("--data"' not in files['t1gr_eval_step1_baseline.py'],c)
    check('S09_runner_view_manifest','--view-manifest' in files['t1gr_run_step1_baseline.py'],c)
    check('S10_eval_view_manifest','--view-manifest' in files['t1gr_eval_step1_baseline.py'],c)
    check('S11_view_manifest_every_mapping','"mappings": mappings' in files['t1gr_build_step1_rgb_view.py'],c)
    check('S12_view_copy_only','FORMAL_STEP1_VIEW_MUST_COPY' in files['t1gr_build_step1_rgb_view.py'],c)
    check('S13_runtime_checkpoint_hash','BASE_CHECKPOINT_SHA_DRIFT' in files['t1gr_run_step1_baseline.py'],c)
    check('S14_ultralytics_pin','ULTRALYTICS_VERSION_DRIFT' in files['t1gr_run_step1_baseline.py'],c)
    check('S15_effective_args_pre','EFFECTIVE_ARGS_PREFLIGHT_MISMATCH' in files['t1gr_run_step1_baseline.py'],c)
    check('S16_effective_args_post','EFFECTIVE_ARGS_POSTRUN_MISMATCH' in files['t1gr_run_step1_baseline.py'],c)
    check('S17_physical_nc_pre','PRETRAIN_PHYSICAL_HEAD_NC_FAIL' in files['t1gr_run_step1_baseline.py'],c)
    check('S18_physical_nc_eval','STEP1_PHYSICAL_HEAD_NC_FAIL' in files['t1gr_eval_step1_baseline.py'],c)
    check('S19_full_hash_unconditional','--full-hash' not in files['t1gr_build_contract.py'] and 'Formal contract ALWAYS hashes every paired file' in files['t1gr_build_contract.py'],c)
    check('S20_format_hard_gate','"format_valid": not format_failures' in files['t1gr_build_contract.py'],c)
    check('S21_spatial_hard_gate','"cross_modal_hw_valid": not spatial_failures' in files['t1gr_build_contract.py'],c)
    check('S22_expected_count_gate','expected_sample_count_match' in files['t1gr_build_contract.py'],c)
    check('S23_class_names_count_gate','class_names_length_mismatch' in files['t1gr_build_contract.py'],c)
    check('S24_label_exact_fields','field_count_not_exact' in core,c)
    check('S25_label_edge_geometry','bbox_edges_outside_image' in core,c)
    check('S26_group_rule_executed','group_rule_validation' in files['t1gr_build_contract.py'] and 'group_map(common' in files['t1gr_build_contract.py'],c)
    check('S27_group_class_stratified','group_stratified_split' in files['t1gr_propose_split.py'],c)
    check('S28_class_image_support','image_counts' in core and 'class_coverage_audit' in files['t1gr_propose_split.py'],c)
    check('S29_class_box_support','box_counts' in core and 'class_coverage_audit' in files['t1gr_propose_split.py'],c)
    check('S30_nonempty_split_gate','three_splits_nonempty' in files['t1gr_propose_split.py'],c)
    check('S31_duplicate_all_modalities','duplicate_groups_by_kind' in files['t1gr_build_contract.py'] and 'triplet' in files['t1gr_build_contract.py'],c)
    check('S32_duplicate_split_gate','cross_split_exact_duplicate_leakage_empty' in files['t1gr_propose_split.py'],c)
    check('S33_synthetic_gate','t1gr_synthetic_integration_gate.py' in files,c)
    check('S34_layout_format_expectations',all(layout['format_expectations'][m] for m in ('rgb','ir','depth')),c)
    check('S35_training_spec_optimizer',all(k in ts['train_args'] for k in ('optimizer','lr0','nbs','warmup_epochs','mosaic','close_mosaic')),c)
    check('S36_training_spec_eval',all(k in ts['eval_args'] for k in ('iou','max_det','conf')),c)
    check('S37_no_t1gr_arm_runner',not any(('g0' in n.lower() or 'g1_p' in n.lower() or 'g2_s' in n.lower()) for n in files),c)
    failed=[k for k,v in c.items() if not v['passed']]
    rep={'schema':'t1gr-e2-e5-static-audit-v2','passed_count':sum(v['passed'] for v in c.values()),'total_count':len(c),'failed':failed,'all_passed':not failed,'checks':c}
    out=ROOT/a.out; out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists(): raise RuntimeError(f'REFUSE_OVERWRITE:{out}')
    out.write_text(json.dumps(rep,indent=2,ensure_ascii=False),encoding='utf-8'); print(json.dumps({k:rep[k] for k in ('passed_count','total_count','failed','all_passed')},indent=2))
    raise SystemExit(0 if not failed else 2)
if __name__=='__main__': main()
