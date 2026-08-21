#!/usr/bin/env python3
"""Fail-closed server preflight for the T1-U6 four-arm experiment."""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal.t1gr_u6_core import (  # noqa: E402
    ARMS,
    SCHEMA_PREFLIGHT,
    SCHEMA_SPEC,
    SCHEMA_VIEW_PUBLIC,
    ZERO_IR,
    implementation_source_hashes,
    validate_spec,
)
from multimodal.t1gr_u6_dataset import T1GRU6Dataset  # noqa: E402
from multimodal.t1gr_u6_model import (  # noqa: E402
    assert_same_seed_arm_identity,
    build_t1gr_u6_model,
    full_detector_zero_aux_equivalence,
)
from multimodal.t1gr_u6_runtime import (  # noqa: E402
    ensure_u6_dataset_yaml,
    server_environment_preflight,
    verify_u6_view,
)
from multimodal.t1gr_e5_core import (  # noqa: E402
    FROZEN_E5_SECURITY_POLICY_SHA256,
    SCHEMA_RECIPE,
    environment_probe,
    payload_ok as e5_payload_ok,
)
from multimodal.t1gr_g_core import SEEDS  # noqa: E402
from multimodal.t1gr_g_runtime import implementation_source_hashes as primary_source_hashes  # noqa: E402
from multimodal.t1gr_secure_io import (  # noqa: E402
    Deadline,
    assert_public_safe,
    atomic_json_write,
    check_existing_output,
    ensure_private_input,
    ensure_public_output,
    ensure_repo_input,
    fail,
    file_lock,
    read_json_bounded,
    safe_error_message,
    sha256_file,
    sha256_json,
)

SCRIPT_VERSION = "t1gr-u6-server-preflight-v1"


def _probe_dataset(cfg, data: dict, view: dict, *, arm: str) -> T1GRU6Dataset:
    return T1GRU6Dataset(
        img_path=str(data["train"]),
        imgsz=cfg.imgsz,
        batch_size=cfg.batch,
        augment=True,
        hyp=cfg,
        rect=False,
        cache=None,
        single_cls=False,
        stride=32,
        pad=0.0,
        prefix="T1-U6 preflight: ",
        task="detect",
        classes=cfg.classes,
        data=data,
        fraction=1.0,
        ir_by_sid=view["ir_maps"]["train"],
        depth_by_sid=view["depth_maps"]["train"],
        depth_kind_by_sid=view["depth_kind_maps"]["train"],
        arm=arm,
        seed=int(SEEDS[0]),
        split="train",
        ir_condition="ARM_NATIVE",
        depth_condition="NATIVE",
    )


def _dataset_contract(recipe: dict, view: dict) -> dict:
    try:
        import torch
        from ultralytics.cfg import DEFAULT_CFG, get_cfg
        from ultralytics.data.utils import check_det_dataset
    except Exception:
        fail("T1GR_U6_PREFLIGHT_DATASET_IMPORT_FAIL")
    overrides = dict(recipe["train_args"])
    overrides.update(recipe["eval_args"])
    overrides.update({"data": str(view["dataset_yaml"]), "model": recipe["model_yaml"], "task": "detect", "mode": "train"})
    cfg = get_cfg(DEFAULT_CFG, overrides=overrides)
    data = check_det_dataset(str(view["dataset_yaml"]))
    data["channels"] = 6
    datasets = {arm: _probe_dataset(cfg, data, view, arm=arm) for arm in ARMS}
    ids = datasets["G0-N"].ids
    if any(dataset.ids != ids for dataset in datasets.values()):
        fail("T1GR_U6_PREFLIGHT_ARM_ID_ORDER_DRIFT")
    by_kind = view["depth_kind_maps"]["train"]
    metric_candidates = [sid for sid in ids if by_kind[sid] == "METRIC_UINT16_PNG"]
    metric_sid = next(
        (
            sid
            for sid in metric_candidates
            if datasets["G3-D"].raw_probe(ids.index(sid))["record"]["emitted_valid_pixels"] > 0
        ),
        None,
    )
    jpg_sid = next((sid for sid in ids if by_kind[sid] == "UNKNOWN_SCALE_JPG_QUARANTINED"), None)
    if metric_sid is None or jpg_sid is None:
        fail("T1GR_U6_PREFLIGHT_DEPTH_DOMAIN_MISSING")

    raw_domains = {}
    for domain, sid in (("metric", metric_sid), ("jpg", jpg_sid)):
        index = ids.index(sid)
        probes = {arm: datasets[arm].raw_probe(index) for arm in ARMS}
        visible_equal = len({row["visible_sha256"] for row in probes.values()}) == 1
        g1_g3_ir_equal = probes["G1-P"]["ir_sha256"] == probes["G3-D"]["ir_sha256"]
        controls_depth_zero = all(
            probes[arm]["depth_nonzero"] == 0
            and probes[arm]["mask_nonzero"] == 0
            and probes[arm]["record"]["depth_source_decoded"] is False
            for arm in ("G0-N", "G1-P", "G2-S")
        )
        safe_records = {}
        for arm, probe in probes.items():
            record = dict(probe["record"])
            record.pop("sample_id", None)
            safe_records[arm] = record
        raw_domains[domain] = {
            "records": safe_records,
            "visible_equal_all_arms": visible_equal,
            "g1_g3_ir_equal": g1_g3_ir_equal,
            "g0_ir_zero": probes["G0-N"]["ir_nonzero"] == 0 and probes["G0-N"]["donor_id"] == ZERO_IR,
            "g2_donor_is_wrong": probes["G2-S"]["donor_id"] != sid,
            "control_depth_zero_without_decode": controls_depth_zero,
            "g3_depth_nonzero": probes["G3-D"]["depth_nonzero"] > 0,
            "g3_mask_nonzero": probes["G3-D"]["mask_nonzero"] > 0,
        }
    if raw_domains["jpg"]["records"]["G3-D"]["depth_source_decoded"] is not False:
        fail("T1GR_U6_PREFLIGHT_JPG_DECODE_POLICY_FAIL")
    if raw_domains["jpg"]["g3_depth_nonzero"] or raw_domains["jpg"]["g3_mask_nonzero"]:
        fail("T1GR_U6_PREFLIGHT_JPG_QUARANTINE_FAIL")

    transformed = None
    ordered = [metric_sid] + [sid for sid in metric_candidates if sid != metric_sid]
    for sid in ordered[:32]:
        index = ids.index(sid)
        images = {arm: datasets[arm][index]["img"] for arm in ARMS}
        mask = images["G3-D"][5]
        candidate = {
            "rgb_bitwise_equal_all_arms": all(torch.equal(images["G0-N"][:3], images[arm][:3]) for arm in ARMS[1:]),
            "g1_g3_ir_bitwise_equal": torch.equal(images["G1-P"][3], images["G3-D"][3]),
            "six_channels_all_arms": all(int(image.shape[0]) == 6 for image in images.values()),
            "g0_aux_zero": int(torch.count_nonzero(images["G0-N"][3:]).item()) == 0,
            "g1_g2_depth_mask_zero": all(int(torch.count_nonzero(images[arm][4:]).item()) == 0 for arm in ("G1-P", "G2-S")),
            "g3_mask_binary": bool(torch.all((mask == 0) | (mask == 255)).item()),
            "g3_mask_zero_implies_depth_zero": int(torch.count_nonzero(images["G3-D"][4][mask == 0]).item()) == 0,
            "g3_depth_nonzero": int(torch.count_nonzero(images["G3-D"][4]).item()) > 0,
        }
        if all(candidate.values()):
            transformed = candidate
            break
    if transformed is None:
        fail("T1GR_U6_PREFLIGHT_TRANSFORM_CONTRACT_FAIL")

    g2 = datasets["G2-S"]
    mapping = {sid: g2.source_sid(sid) for sid in ids}
    donors = list(mapping.values())
    wrong_schedule = {
        "recipient_count": len(mapping),
        "zero_self_match": all(recipient != donor for recipient, donor in mapping.items()),
        "bijection": len(set(donors)) == len(ids) and set(donors) == set(ids),
        "mapping_commitment": sha256_json(mapping),
    }
    wrong_schedule["passed"] = bool(wrong_schedule["zero_self_match"] and wrong_schedule["bijection"])
    required_raw = (
        raw_domains["metric"]["visible_equal_all_arms"],
        raw_domains["metric"]["g1_g3_ir_equal"],
        raw_domains["metric"]["g0_ir_zero"],
        raw_domains["metric"]["g2_donor_is_wrong"],
        raw_domains["metric"]["control_depth_zero_without_decode"],
        raw_domains["metric"]["g3_depth_nonzero"],
        raw_domains["metric"]["g3_mask_nonzero"],
        raw_domains["jpg"]["visible_equal_all_arms"],
        raw_domains["jpg"]["g1_g3_ir_equal"],
        raw_domains["jpg"]["g0_ir_zero"],
        raw_domains["jpg"]["g2_donor_is_wrong"],
        raw_domains["jpg"]["control_depth_zero_without_decode"],
    )
    if not all(required_raw) or not wrong_schedule["passed"]:
        fail("T1GR_U6_PREFLIGHT_RAW_ARM_CONTRACT_FAIL")
    return {
        "raw_domain_probes": raw_domains,
        "transformed_four_arm_probe": transformed,
        "g2_epoch0_wrong_schedule": wrong_schedule,
        "control_depth_decode_skipped": True,
    }


def run(args) -> dict:
    repo = ROOT.resolve(strict=True)
    security_path = ensure_repo_input(repo, "config/t1gr_e5_security_policy.json", "config")
    if sha256_file(security_path) != FROZEN_E5_SECURITY_POLICY_SHA256:
        fail("E5_SECURITY_POLICY_SHA_DRIFT")
    security = read_json_bounded(security_path, 1 << 20, "t1gr-e5-security-policy-v2")
    spec_path = ensure_repo_input(repo, "config/t1gr_u6_design.frozen.json", "config")
    recipe_path = ensure_repo_input(repo, "reports/step4_t1gr/e5_v2_step1_recipe_public.json", "reports/step4_t1gr")
    primary_audit_path = ensure_repo_input(repo, "reports/step4_t1gr/t1gr_g_smoke_audit_public.json", "reports/step4_t1gr")
    view_public_path = ensure_repo_input(repo, "reports/step4_t1gr/t1gr_u6_view_public.json", "reports/step4_t1gr")
    view_manifest_path = ensure_private_input(Path(args.u6_view_manifest), repo)
    checkpoint = ensure_private_input(Path(args.base_checkpoint), repo)
    output = ensure_public_output(repo, "reports/step4_t1gr/t1gr_u6_server_preflight_public.json", security["public_output_prefix"])
    deadline = Deadline(float(args.timeout_seconds))
    with file_lock(output.with_suffix(output.suffix + ".lock"), 5.0, 900.0):
        spec = read_json_bounded(spec_path, int(security["max_public_json_bytes"]), SCHEMA_SPEC)
        recipe = read_json_bounded(recipe_path, int(security["max_public_json_bytes"]), SCHEMA_RECIPE)
        primary_audit = read_json_bounded(primary_audit_path, int(security["max_public_json_bytes"]))
        view_public = read_json_bounded(view_public_path, int(security["max_public_json_bytes"]), SCHEMA_VIEW_PUBLIC)
        validate_spec(spec)
        if not all(e5_payload_ok(value) for value in (recipe, primary_audit, view_public)):
            fail("T1GR_U6_PREFLIGHT_PUBLIC_INTEGRITY_FAIL")
        if primary_audit.get("smoke_gate_passed") is not True or primary_audit.get("final_holdout_open_authorized") is not False:
            fail("T1GR_U6_PRIMARY_G_AUDIT_NOT_PASS")
        view_sha = sha256_file(view_manifest_path, deadline)
        if view_sha != view_public.get("u6_view_manifest_private_sha256"):
            fail("T1GR_U6_VIEW_SHA_DRIFT")
        view = verify_u6_view(view_manifest_path, recipe, deadline=deadline)
        dataset_yaml = ensure_u6_dataset_yaml(view_manifest_path, recipe)
        checkpoint_sha = sha256_file(checkpoint, deadline)
        if checkpoint_sha != recipe.get("base_checkpoint_sha256"):
            fail("T1GR_U6_CHECKPOINT_SHA_DRIFT")
        environment = environment_probe()
        environment_check = server_environment_preflight(environment, recipe["environment"])
        source_hashes = implementation_source_hashes(repo)
        primary_sources = primary_source_hashes(repo)
        if primary_sources != primary_audit.get("implementation_source_hashes"):
            fail("T1GR_U6_PRIMARY_G_SOURCE_DRIFT")
        identity_rows, seed_checks, equivalence_checks = [], [], []
        for seed in SEEDS:
            group = []
            for arm in ARMS:
                model, identity = build_t1gr_u6_model(checkpoint, recipe, arm=arm, seed=int(seed))
                group.append(identity)
                identity_rows.append(identity)
                del model
                gc.collect()
            seed_checks.append(assert_same_seed_arm_identity(group))
            equivalence = full_detector_zero_aux_equivalence(checkpoint, recipe, seed=int(seed))
            if not equivalence["passed"]:
                fail("T1GR_U6_FULL_DETECTOR_EQUIVALENCE_FAIL")
            equivalence_checks.append(equivalence)
        dataset_contract = _dataset_contract(recipe, view)
        dev_depth_kinds = {
            kind: sum(value == kind for value in view["depth_kind_maps"]["dev"].values())
            for kind in ("METRIC_UINT16_PNG", "UNKNOWN_SCALE_JPG_QUARANTINED")
        }
        checks = {
            "spec_frozen": True,
            "legacy_primary_g_sources_unchanged": True,
            "server_software_matches_e5": environment_check["software_match"],
            "one_visible_gpu": environment.get("cuda_device_count") == 1,
            "view_train_count": view["train_count"] == 1504,
            "view_dev_count": view["dev_count"] == 198,
            "twelve_seed_arm_models_checked": len(identity_rows) == 12,
            "same_seed_four_arm_initial_identity": all(row["all_identity_fields_equal"] for row in seed_checks),
            "rgb_zero_aux_full_detector_equivalence": all(row["passed"] for row in equivalence_checks),
            "four_arm_dataset_contract": all(dataset_contract["transformed_four_arm_probe"].values()),
            "g2_wrong_schedule": dataset_contract["g2_epoch0_wrong_schedule"]["passed"],
            "unknown_jpg_quarantined": dataset_contract["raw_domain_probes"]["jpg"]["records"]["G3-D"]["emitted_valid_pixels"] == 0,
            "both_depth_domains_present_in_dev": all(value > 0 for value in dev_depth_kinds.values()),
            "final_holdout_absent": view["u6_manifest"].get("final_holdout_ids_present") is False,
        }
        passed = all(checks.values())
        request = sha256_json({
            "script": SCRIPT_VERSION,
            "spec": sha256_file(spec_path, deadline),
            "recipe": sha256_file(recipe_path, deadline),
            "primary_audit": sha256_file(primary_audit_path, deadline),
            "view_public": sha256_file(view_public_path, deadline),
            "u6_view": view_sha,
            "dataset_yaml": sha256_file(dataset_yaml, deadline),
            "checkpoint": checkpoint_sha,
            "environment": environment,
            "sources": source_hashes,
        })
        existing = check_existing_output(output, request)
        if existing is not None:
            if existing[0].get("preflight_gate_passed") is not True:
                fail("T1GR_U6_EXISTING_PREFLIGHT_NOT_PASS")
            return {"status": "PASS", "idempotent_reuse": True, "public_output_sha256": existing[1]}
        report = {
            "schema": SCHEMA_PREFLIGHT,
            "script_version": SCRIPT_VERSION,
            "spec_file_sha256": sha256_file(spec_path, deadline),
            "recipe_public_sha256": sha256_file(recipe_path, deadline),
            "primary_g_smoke_audit_sha256": sha256_file(primary_audit_path, deadline),
            "u6_view_public_sha256": sha256_file(view_public_path, deadline),
            "u6_view_manifest_private_sha256": view_sha,
            "u6_dataset_yaml_private_sha256": sha256_file(dataset_yaml, deadline),
            "base_checkpoint_sha256": checkpoint_sha,
            "server_environment": environment,
            "environment_adjudication": environment_check,
            "legacy_primary_suite_source_hashes": primary_sources,
            "implementation_source_hashes": source_hashes,
            "model_identity_rows": identity_rows,
            "same_seed_checks": seed_checks,
            "rgb_zero_aux_equivalence_checks": equivalence_checks,
            "dataset_contract": dataset_contract,
            "dev_depth_kind_counts": dev_depth_kinds,
            "checks": checks,
            "preflight_gate_passed": passed,
            "smoke_training_authorized": passed,
            "parallel_seed_lanes_authorized": passed,
            "formal_training_authorized": False,
            "legacy_primary_g_suite_mutation_authorized": False,
            "final_holdout_open_authorized": False,
            "production_go": False,
            "next_action": "run twelve one-epoch U6 smokes across G0-G1-G2-G3, then run the U6 smoke audit",
        }
        assert_public_safe(report)
        digest, _ = atomic_json_write(output, report, private=False, request_fingerprint=request)
        if not passed:
            fail("T1GR_U6_PREFLIGHT_FAIL")
        return {
            "status": "PASS",
            "idempotent_reuse": False,
            "public_output_sha256": digest,
            "smoke_training_authorized": True,
            "parallel_seed_lanes_authorized": True,
            "final_holdout_open_authorized": False,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--u6-view-manifest", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args), ensure_ascii=False, indent=2))
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error": safe_error_message(exc)}), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
