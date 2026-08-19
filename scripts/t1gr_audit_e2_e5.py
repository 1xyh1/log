#!/usr/bin/env python3
"""Formal E2-E5 entry auditor v2. Compliance facts are derived from hashes/IDs/timestamps."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from multimodal.t1gr_e2e5 import (  # noqa: E402
    SCHEMA_CONTRACT_PUBLIC, SCHEMA_SPLIT_FREEZE_PUBLIC, SCHEMA_STEP1_RECIPE,
    SCHEMA_VIEW_MANIFEST, parse_utc, sha256_file,
)


def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8-sig"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--public-contract", required=True)
    ap.add_argument("--split-freeze-public", required=True)
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--view-manifest", required=True)
    ap.add_argument("--run-manifest", default=None)
    ap.add_argument("--baseline-report", default=None)
    ap.add_argument("--out", default="reports/step4_t1gr/e2_e5_entry_audit.json")
    args = ap.parse_args()

    cp, fp, rp, vp = map(Path, (args.public_contract, args.split_freeze_public, args.recipe, args.view_manifest))
    c, f, r, v = map(load, (cp, fp, rp, vp))
    checks = {
        "E2_contract_schema": c.get("schema") == SCHEMA_CONTRACT_PUBLIC,
        "E2_contract_gate": c.get("contract_gate_passed") is True,
        "E2_full_hash_mandatory": c.get("full_hash_mode") is True and c.get("gates", {}).get("full_hash_complete") is True,
        "E2_format_gate": c.get("format_gate_passed") is True,
        "E2_pairing_gate": c.get("gates", {}).get("pairing_complete") is True,
        "E2_label_gate": c.get("gates", {}).get("labels_valid") is True,
        "E2_expected_count_gate": c.get("gates", {}).get("expected_sample_count_match") is True,
        "E2_class_config_gate": c.get("gates", {}).get("class_config_valid") is True,
        "E3_group_rule_executed": c.get("group_rule_validation", {}).get("passed") is True,
        "E4_freeze_schema": f.get("schema") == SCHEMA_SPLIT_FREEZE_PUBLIC,
        "E4_proposal_gate": f.get("proposal_gate_passed") is True,
        "E4_no_sample_ids_public": f.get("contains_any_sample_ids") is False and not any(k in f for k in ("train_ids", "dev_ids", "final_holdout_ids", "paired_ids")),
        "E4_holdout_not_exposed": f.get("final_holdout_ids_exposed") is False,
        "E4_three_nonempty": all(int(f.get("sample_counts", {}).get(s, 0)) > 0 for s in ("train", "dev", "final_holdout")),
        "E5_recipe_schema": r.get("schema") == SCHEMA_STEP1_RECIPE,
        "E5_recipe_public_contract_pin": r.get("public_contract_sha256") == sha256_file(cp),
        "E5_recipe_split_freeze_pin": r.get("split_freeze_public_sha256") == sha256_file(fp),
        "E5_recipe_holdout_forbidden": r.get("final_holdout_access") == "FORBIDDEN_UNTIL_T1GR_FINAL_ADJUDICATION",
        "E5_view_schema": v.get("schema") == SCHEMA_VIEW_MANIFEST,
        "E5_view_recipe_pin": v.get("recipe_sha256") == sha256_file(rp),
        "E5_view_train_commitment": v.get("train_ids_sha256") == r.get("split_ids_sha256", {}).get("train"),
        "E5_view_dev_commitment": v.get("dev_ids_sha256") == r.get("split_ids_sha256", {}).get("dev"),
        "E5_view_holdout_commitment": v.get("final_holdout_ids_sha256") == r.get("split_ids_sha256", {}).get("final_holdout"),
        "E5_view_holdout_actual_exclusion": v.get("final_holdout_excluded_by_actual_id_set") is True and int(v.get("final_holdout_intersection_count", -1)) == 0,
        "E5_view_dataset_yaml_pin": Path(v.get("dataset_yaml", "")).is_file() and sha256_file(Path(v["dataset_yaml"])) == v.get("dataset_yaml_sha256"),
    }

    run = baseline = None
    if args.run_manifest and args.baseline_report:
        mp, bp = Path(args.run_manifest), Path(args.baseline_report)
        run, baseline = load(mp), load(bp)
        checks.update({
            "E5_run_schema": run.get("schema") == "t1gr-step1-run-manifest-v2",
            "E5_run_recipe_pin": run.get("recipe_sha256") == sha256_file(rp),
            "E5_run_view_pin": run.get("view_manifest_sha256") == sha256_file(vp),
            "E5_run_train_ids_pin": run.get("actual_train_ids_sha256") == r.get("split_ids_sha256", {}).get("train"),
            "E5_run_dev_ids_pin": run.get("actual_dev_ids_sha256") == r.get("split_ids_sha256", {}).get("dev"),
            "E5_run_holdout_derived_excluded": run.get("final_holdout_access_derived") == "EXCLUDED_FROM_PINNED_VIEW" and int(run.get("view_holdout_intersection_count", -1)) == 0,
            "E5_runtime_checkpoint_pin": run.get("base_checkpoint_sha256_runtime") == r.get("base_checkpoint_sha256"),
            "E5_runtime_ultralytics_pin": run.get("ultralytics_version_runtime") == r.get("ultralytics_version"),
            "E5_effective_args_frozen": run.get("effective_args_frozen_keys_match") is True and int(run.get("effective_args_frozen_key_count", 0)) > 0,
            "E5_freeze_precedes_training_derived": run.get("freeze_precedes_training_derived") is True and parse_utc(f["freeze_timestamp_utc"]) < parse_utc(run["training_started_at_utc"]),
            "E5_baseline_report_schema": baseline.get("schema") == "t1gr-step1-baseline-report-v2",
            "E5_baseline_report_status": baseline.get("status") == "STEP1_BASELINE_EXECUTED",
            "E5_baseline_run_pin": baseline.get("run_manifest_sha256") == sha256_file(mp),
            "E5_baseline_view_pin": baseline.get("view_manifest_sha256") == sha256_file(vp),
            "E5_baseline_dev_ids_pin": baseline.get("actual_eval_ids_sha256") == r.get("split_ids_sha256", {}).get("dev"),
            "E5_baseline_holdout_derived_excluded": baseline.get("final_holdout_access_derived") == "EXCLUDED_FROM_PINNED_VIEW",
            "E5_physical_head_nc": baseline.get("physical_head_nc") == baseline.get("expected_nc") == r.get("num_classes"),
            "E5_head_mode_pin": baseline.get("head_end2end") == r.get("train_args", {}).get("end2end"),
        })
    else:
        checks["E5_baseline_complete"] = False

    gates = {
        "E2": all(v for k, v in checks.items() if k.startswith("E2_")),
        "E3": all(v for k, v in checks.items() if k.startswith("E3_")),
        "E4": all(v for k, v in checks.items() if k.startswith("E4_")),
        "E5": all(v for k, v in checks.items() if k.startswith("E5_")),
    }
    report = {
        "schema": "t1gr-e2-e5-entry-audit-v2",
        "checks": checks,
        "gates": gates,
        "all_passed": all(gates.values()),
        "t1gr_training_authorized": all(gates.values()),
        "final_holdout_go_authority": "STILL_SEALED; Step1 only used TRAIN/DEV",
        "next_action": "generate final T1-GR DESIGN_FREEZE and G0/G1/G2 multi-seed runner" if all(gates.values()) else "complete/fix pending E2-E5 evidence gates",
    }
    out = ROOT / args.out
    if out.exists():
        raise RuntimeError(f"REFUSE_OVERWRITE:{out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["all_passed"] else 2)


if __name__ == "__main__":
    main()
