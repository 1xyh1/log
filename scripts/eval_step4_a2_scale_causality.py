#!/usr/bin/env python3
"""A2 evaluation-only scale-wise IR residual causality audit.

Primary: F1C-I-fixed/last.pt. Replication: F1C-I-soft/last.pt.
Interventions occur strictly after recipient q_native is computed and frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal import step3_eval_utils as evu  # noqa: E402
from multimodal.causality_interventions import (  # noqa: E402
    assert_valid_shuffle_map, bijective_derangement,
)
from multimodal.raw_sample_index import CLASS_NAMES, OUT_DEFAULT  # noqa: E402
from multimodal.step4_a2_residual_interventions import (  # noqa: E402
    MASK_CONDITIONS, SCALES, ResidualCondition, build_residual_cache,
    classify_paired_effect, forward_with_residual_intervention, gain_conditions,
    mask_conditions, shuffle_conditions, tensor_sha256,
)
from multimodal.trimodal_dataset import TriModalDataset  # noqa: E402

SCHEMA = "step4-a2-scale-ir-residual-causality-v2"
EXPECTED_F1C_SUMMARY_SHA256 = "d4e64b86e221b102143bd98cc6056f8e84d7913680cad3c8c5826af4cf88942f"
FROZEN_DEPENDENCY_TARGETS = {
    "step3_eval_utils_sha256": "src/multimodal/step3_eval_utils.py",
    "model_source_sha256": "src/multimodal/step4_f1_ir_gate_model.py",
    "gate_source_sha256": "src/multimodal/reliability_gate.py",
    "trimodal_dataset_sha256": "src/multimodal/trimodal_dataset.py",
    "f0_model_source_sha256": "src/multimodal/step4_f0_model.py",
    "aux_encoder_source_sha256": "src/multimodal/aux_encoder.py",
    "feature_fusion_source_sha256": "src/multimodal/feature_fusion.py",
    "trainability_source_sha256": "src/multimodal/trainability.py",
    "causality_interventions_sha256": "src/multimodal/causality_interventions.py",
    "raw_sample_index_sha256": "src/multimodal/raw_sample_index.py",
}
RUN_SPECS = {
    "FIXED": {
        "run": "F1C-I-fixed", "group": "F1C-I-fixed",
        "aux_mode": "ir", "gate_mode": "fixed_one", "gate_module": "magnitude",
    },
    "SOFT": {
        "run": "F1C-I-soft", "group": "F1C-I-soft",
        "aux_mode": "ir", "gate_mode": "learned", "gate_module": "original",
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def state_sha256(model) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()




def verify_preexecution_audit(root: Path) -> dict:
    path = root / "reports/step4_a2/preexecution_audit.json"
    if not path.exists():
        raise RuntimeError(f"A2_PREEXECUTION_AUDIT_MISSING:{path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    if obj.get("schema") != "step4-a2-preexecution-audit-v2" or obj.get("all_passed") is not True:
        raise RuntimeError("A2_PREEXECUTION_AUDIT_NOT_PASSING")
    current = {
        "design_sha256": sha256_file(root / "docs/step4_a2/DESIGN_FREEZE.md"),
        "engine_sha256": sha256_file(root / "src/multimodal/step4_a2_residual_interventions.py"),
        "evaluator_sha256": sha256_file(root / "scripts/eval_step4_a2_scale_causality.py"),
        "tests_sha256": sha256_file(root / "tests/test_step4_a2_residual_causality.py"),
        "audit_source_sha256": sha256_file(root / "scripts/audit_step4_a2.py"),
    }
    stale = {k: {"recorded": obj.get("provenance", {}).get(k), "current": v}
             for k, v in current.items() if obj.get("provenance", {}).get(k) != v}
    if stale:
        raise RuntimeError(f"A2_PREEXECUTION_AUDIT_STALE:{stale}")
    return {"passed": True, "path": str(path.relative_to(root)),
            "sha256": sha256_file(path), "source_hashes": current}


def verify_frozen_dependency_closure(
    root: Path, project: str, contract_path: Path, summary_path: Path, summary: dict
) -> dict:
    errors = []
    summary_sha = sha256_file(summary_path)
    if summary_sha != EXPECTED_F1C_SUMMARY_SHA256:
        errors.append(
            f"F1C_SUMMARY_SHA:{summary_sha}!={EXPECTED_F1C_SUMMARY_SHA256}"
        )
    if summary.get("verdict_frozen") is not True or summary.get("decision") != "F1C_GATE_FAILED_CAUSAL_PROTOCOL":
        errors.append("F1C_SUMMARY_VERDICT")

    contract_sha = sha256_file(contract_path)
    current_sources = {key: sha256_file(root / rel)
                       for key, rel in FROZEN_DEPENDENCY_TARGETS.items()}
    current_versions = {
        "torch_version": torch.__version__,
        "ultralytics_version": __import__("ultralytics").__version__,
    }
    eval_evidence = {}
    frozen_val_ids = None
    baseline_shared = None
    for tag, spec in RUN_SPECS.items():
        run_dir = root / project / spec["run"]
        eval_path = run_dir / "eval_step4_f1_c_causality.json"
        if not eval_path.exists():
            errors.append(f"{tag}:F1C_CAUSAL_EVAL_MISSING")
            continue
        obj = json.loads(eval_path.read_text(encoding="utf-8"))
        if obj.get("schema") != "step4-f1-c-stock-validator-semantics-v1" or obj.get("group") != spec["group"]:
            errors.append(f"{tag}:F1C_CAUSAL_EVAL_IDENTITY")
            continue
        verified_key = "FIXED" if tag == "FIXED" else "ORIGSOFT"
        if summary.get("eval_provenance_verified", {}).get(verified_key, {}).get("passed") is not True:
            errors.append(f"{tag}:SUMMARY_EVAL_PROVENANCE_NOT_VERIFIED")
        prov = obj.get("provenance") or {}
        expected_last = summary["integrity"][verified_key]["observed"]["eval_provenance_current_hashes"]["last_pt_sha256"]
        if prov.get("last_pt_sha256") != expected_last:
            errors.append(f"{tag}:EVAL_LAST_PT_NOT_SUMMARY_FROZEN")
        manifest_sha = sha256_file(run_dir / "manifest.json")
        if prov.get("manifest_sha256") != manifest_sha:
            errors.append(f"{tag}:MANIFEST_STALE")
        if prov.get("contract_sha256") != contract_sha:
            errors.append(f"{tag}:CONTRACT_STALE")
        for key, current in current_sources.items():
            if prov.get(key) != current:
                errors.append(f"{tag}:SOURCE_STALE:{key}")
        for key, current in current_versions.items():
            if prov.get(key) != current:
                errors.append(f"{tag}:VERSION_STALE:{key}:{prov.get(key)}!={current}")
        val_block = (((obj.get("last.pt") or {}).get("gate_values") or {}).get("NORMAL") or {}).get("val") or {}
        ids = list(val_block.keys())
        if len(ids) != 6:
            errors.append(f"{tag}:FROZEN_VAL_IDS_COUNT:{len(ids)}")
        if frozen_val_ids is None:
            frozen_val_ids = ids
        elif ids != frozen_val_ids:
            errors.append(f"{tag}:FROZEN_VAL_IDS_MISMATCH")
        shared = {"contract_sha256": prov.get("contract_sha256"), **{k: prov.get(k) for k in FROZEN_DEPENDENCY_TARGETS}, **{k: prov.get(k) for k in current_versions}}
        if baseline_shared is None:
            baseline_shared = shared
        elif shared != baseline_shared:
            errors.append(f"{tag}:CROSS_SYSTEM_DEPENDENCY_MISMATCH")
        eval_evidence[tag] = {
            "path": str(eval_path.relative_to(root)),
            "sha256": sha256_file(eval_path),
            "manifest_sha256": manifest_sha,
            "recorded_last_pt_sha256": prov.get("last_pt_sha256"),
            "frozen_val_ids": ids,
        }
    return {
        "passed": not errors,
        "errors": errors,
        "summary_sha256": summary_sha,
        "expected_summary_sha256": EXPECTED_F1C_SUMMARY_SHA256,
        "contract_sha256": contract_sha,
        "current_source_hashes": current_sources,
        "current_versions": current_versions,
        "frozen_val_ids": frozen_val_ids or [],
        "f1c_eval_evidence": eval_evidence,
        "stock_eval_semantics_frozen": not any(
            ("SOURCE_STALE" in e or "VERSION_STALE" in e or "CONTRACT_STALE" in e or "FROZEN_VAL_IDS" in e)
            for e in errors
        ),
    }


def load_frozen_model(run_dir: Path, spec: dict, device, expected_checkpoint_sha256: str):
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    expected = {
        "schema": "step4-f1-c-manifest-v2",
        "group": spec["group"],
        "physical_run_name": spec["run"],
        "run_kind": "formal",
        "aux_mode": spec["aux_mode"],
        "gate_mode": spec["gate_mode"],
        "gate_module": spec["gate_module"],
        "expected_epochs": 80,
        "seed": 20260812,
    }
    bad = {k: {"recorded": manifest.get(k), "expected": v}
           for k, v in expected.items() if manifest.get(k) != v}
    if bad:
        raise RuntimeError(f"A2_CHECKPOINT_IDENTITY:{run_dir}:{bad}")
    ckpt_path = run_dir / "weights" / "last.pt"
    if not ckpt_path.exists():
        raise RuntimeError(f"A2_LAST_PT_MISSING:{ckpt_path}")
    actual_checkpoint_sha256 = sha256_file(ckpt_path)
    if actual_checkpoint_sha256 != expected_checkpoint_sha256:
        raise RuntimeError(
            f"A2_FROZEN_CHECKPOINT_SHA_MISMATCH:{spec['group']}:"
            f"{actual_checkpoint_sha256}!={expected_checkpoint_sha256}"
        )
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = (ck.get("ema") or ck.get("model")).float().eval().to(device)
    if getattr(model, "aux_mode", None) != spec["aux_mode"]:
        raise RuntimeError("A2_CHECKPOINT_AUX_MODE_MISMATCH")
    if getattr(model, "gate_mode", None) != spec["gate_mode"]:
        raise RuntimeError("A2_CHECKPOINT_GATE_MODE_MISMATCH")
    if getattr(model, "gate_module_kind", None) != spec["gate_module"]:
        raise RuntimeError("A2_CHECKPOINT_GATE_MODULE_MISMATCH")
    if getattr(model, "_gate_override", None) is not None:
        model.set_gate_override(None)
    return model, manifest, ckpt_path


def build_val_dataset(contract: dict):
    return TriModalDataset(contract, split="val", group="C1-I", augment=False)


def _metric_from_stats(stats_by_id: dict, ids: list[str], names: dict) -> dict:
    from ultralytics.utils.metrics import DetMetrics

    metrics = DetMetrics(names={int(k): v for k, v in names.items()})
    for sid in ids:
        metrics.update_stats(stats_by_id[sid])
    metrics.process()
    res = metrics.results_dict
    return {
        "map50": float(res["metrics/mAP50(B)"]),
        "map50_95": float(res["metrics/mAP50-95(B)"]),
        "n_images": len(ids),
    }


def collect_condition_stats(
    model, dataset, device, names, condition: ResidualCondition,
    *, donor_map: dict[str, str] | None, residual_cache: dict | None,
):
    validator = evu.make_detection_validator(model, device, names)
    stats_by_id = {}
    traces = {}
    model.eval()
    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            sid = str(sample["sample_id"])
            batch = dataset.collate_fn([sample])
            batch = evu.move_step3_batch_to_device(batch, device)
            donor_id = None
            donor_residuals = None
            if condition.kind in {"shuffle_cond", "shuffle_only"}:
                donor_id = donor_map[sid]
                donor_residuals = residual_cache[donor_id]
            output, trace = forward_with_residual_intervention(
                model, batch["img"], condition, recipient_id=sid,
                donor_id=donor_id, donor_residuals=donor_residuals,
            )
            raw = evu.extract_detection_tensor(output)
            preds = validator.postprocess(raw)
            if len(preds) != 1:
                raise RuntimeError(f"A2_EXPECTED_ONE_PREDICTION:{len(preds)}")
            pbatch = validator._prepare_batch(0, batch)
            pred = validator._prepare_pred(preds[0])
            cls_np = pbatch["cls"].detach().cpu().numpy()
            no_pred = pred["cls"].numel() == 0
            stat = validator._process_batch(pred, pbatch)
            stat.update(
                target_cls=cls_np,
                target_img=np.unique(cls_np),
                conf=(np.zeros(0, dtype=np.float32) if no_pred
                      else pred["conf"].detach().cpu().numpy()),
                pred_cls=(np.zeros(0, dtype=np.float32) if no_pred
                          else pred["cls"].detach().cpu().numpy()),
                im_name=sid,
            )
            stats_by_id[sid] = stat
            traces[sid] = trace
    ids = [str(x) for x in dataset.ids]
    full = _metric_from_stats(stats_by_id, ids, names)
    loo = {}
    for held_out in ids:
        keep = [sid for sid in ids if sid != held_out]
        loo[held_out] = _metric_from_stats(stats_by_id, keep, names)
    return {"full": full, "loo": loo, "trace": traces}


def native_equivalence_probe(model, dataset, device) -> dict:
    cond = ResidualCondition("mask", (1, 1, 1))
    rows = {}
    model.eval()
    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            sid = str(sample["sample_id"])
            batch = dataset.collate_fn([sample])
            batch = evu.move_step3_batch_to_device(batch, device)
            native = evu.extract_detection_tensor(model._predict_once(batch["img"])).detach()
            a2, trace = forward_with_residual_intervention(
                model, batch["img"], cond, recipient_id=sid,
            )
            a2 = evu.extract_detection_tensor(a2).detach()
            same = native.shape == a2.shape and torch.equal(native, a2)
            rows[sid] = {
                "bitwise_equal": bool(same),
                "native_sha256": tensor_sha256(native),
                "a2_m111_sha256": tensor_sha256(a2),
                "q_native": trace["q_native"],
            }
    return {"passed": all(r["bitwise_equal"] for r in rows.values()), "rows": rows}


def effect(a: dict, b: dict) -> dict:
    full = float(a["full"]["map50_95"] - b["full"]["map50_95"])
    folds = {
        sid: float(a["loo"][sid]["map50_95"] - b["loo"][sid]["map50_95"])
        for sid in a["loo"]
    }
    values = list(folds.values())
    return {
        "full": full,
        "loo": folds,
        "loo_median": float(statistics.median(values)),
        "positive_folds": sum(v > 0 for v in values),
        "negative_folds": sum(v < 0 for v in values),
    }


def _assert_q_freeze(results: dict, reference_q: dict, tag: str) -> dict:
    errors = []
    for condition_name, block in results.items():
        for sid, trace in block["trace"].items():
            got = trace["q_native"]
            expected = reference_q[sid]
            if got != expected:
                errors.append(f"{tag}:{condition_name}:{sid}:Q_DRIFT")
            if tag == "FIXED" and got != [1.0]:
                errors.append(f"{tag}:{condition_name}:{sid}:Q_NOT_ONE:{got}")
    return {"passed": not errors, "errors": errors}


def _assert_residual_source_trace(
    results: dict, donor_map: dict[str, str], tag: str,
) -> dict:
    errors = []
    for condition_name, block in results.items():
        for sid, trace in block["trace"].items():
            sources = trace["residual_source_ids"]
            if condition_name.startswith("SHUFFLE_"):
                target = condition_name.split("_")[1]
                for scale in SCALES:
                    expected = donor_map[sid] if scale == target else sid
                    if sources.get(scale) != expected:
                        errors.append(
                            f"{tag}:{condition_name}:{sid}:{scale}:"
                            f"SOURCE={sources.get(scale)}!={expected}"
                        )
            elif any(sources.get(scale) != sid for scale in SCALES):
                errors.append(f"{tag}:{condition_name}:{sid}:NONSHUFFLE_SOURCE_DRIFT")
    return {"passed": not errors, "errors": errors}


def _condition_set(tag: str):
    conditions = list(mask_conditions()) + list(shuffle_conditions())
    conditions += list(gain_conditions(include_native=(tag == "SOFT")))
    names = [c.name for c in conditions]
    if len(names) != len(set(names)):
        raise RuntimeError("A2_DUPLICATE_CONDITION_NAME")
    return conditions


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project", default="runs/step4_f1_c")
    p.add_argument("--contract", default=OUT_DEFAULT)
    p.add_argument("--device", default="0")
    p.add_argument("--out", default="reports/step4_a2/scale_ir_residual_causality.json")
    p.add_argument("--donor-map", default="reports/step4_a2/val_donor_map.json")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    audit_evidence = verify_preexecution_audit(ROOT)
    device = evu._as_device(a.device)
    contract_path = Path(a.contract)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))

    f1c_summary_path = ROOT / "runs/step4_f1_c/_summary_step4_f1_c.json"
    f1c_summary = json.loads(f1c_summary_path.read_text(encoding="utf-8"))
    dependency_closure = verify_frozen_dependency_closure(
        ROOT, a.project, contract_path, f1c_summary_path, f1c_summary
    )
    if not dependency_closure["passed"]:
        raise RuntimeError(f"A2_FROZEN_DEPENDENCY_CLOSURE_FAIL:{dependency_closure['errors']}")

    dataset = build_val_dataset(contract)
    val_ids = [str(x) for x in dataset.ids]
    if val_ids != dependency_closure["frozen_val_ids"]:
        raise RuntimeError(
            f"A2_VAL_SET_DRIFT:{val_ids}!={dependency_closure['frozen_val_ids']}"
        )
    names = dict(CLASS_NAMES)
    expected_checkpoint_sha = {
        "FIXED": f1c_summary["integrity"]["FIXED"]["observed"]
            ["eval_provenance_current_hashes"]["last_pt_sha256"],
        "SOFT": f1c_summary["integrity"]["ORIGSOFT"]["observed"]
            ["eval_provenance_current_hashes"]["last_pt_sha256"],
    }

    donor_path = Path(a.donor_map)
    if not donor_path.is_absolute():
        donor_path = ROOT / donor_path
    donor_path.parent.mkdir(parents=True, exist_ok=True)
    expected_donor_map = bijective_derangement(val_ids)
    if donor_path.exists():
        donor_map = json.loads(donor_path.read_text(encoding="utf-8"))
        if donor_map != expected_donor_map:
            raise RuntimeError("A2_DONOR_MAP_NOT_FROZEN_DETERMINISTIC")
    else:
        donor_map = expected_donor_map
        donor_path.write_text(json.dumps(donor_map, indent=2), encoding="utf-8")
    if not assert_valid_shuffle_map(donor_map, val_ids):
        raise RuntimeError("A2_INVALID_DONOR_MAP")

    out_path = Path(a.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    if out_path.exists() and not a.overwrite:
        raise RuntimeError(f"A2_REFUSE_OVERWRITE:{out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "schema": SCHEMA,
        "protocol": {
            "primary": "F1C-I-fixed/last.pt",
            "replication": "F1C-I-soft/last.pt",
            "val_ids": val_ids,
            "factorial_masks": list(MASK_CONDITIONS),
            "q_rule": "recipient native q before intervention; frozen thereafter",
            "shuffle_level": "post-projection residual only",
            "training": False,
        },
        "donor_map": donor_map,
        "preexecution_audit": audit_evidence,
        "frozen_dependency_closure": dependency_closure,
        "provenance": {
            "contract_sha256": sha256_file(contract_path),
            "f1c_summary_sha256": sha256_file(f1c_summary_path),
            "donor_map_sha256": sha256_file(donor_path),
            "evaluator_source_sha256": sha256_file(Path(__file__)),
            "intervention_source_sha256": sha256_file(
                ROOT / "src/multimodal/step4_a2_residual_interventions.py"),
            "design_freeze_sha256": sha256_file(ROOT / "docs/step4_a2/DESIGN_FREEZE.md"),
            "step3_eval_utils_sha256": sha256_file(ROOT / "src/multimodal/step3_eval_utils.py"),
            "f1_model_source_sha256": sha256_file(ROOT / "src/multimodal/step4_f1_ir_gate_model.py"),
            "gate_source_sha256": sha256_file(ROOT / "src/multimodal/reliability_gate.py"),
            "dataset_source_sha256": sha256_file(ROOT / "src/multimodal/trimodal_dataset.py"),
            "f0_model_source_sha256": sha256_file(ROOT / "src/multimodal/step4_f0_model.py"),
            "aux_encoder_source_sha256": sha256_file(ROOT / "src/multimodal/aux_encoder.py"),
            "feature_fusion_source_sha256": sha256_file(ROOT / "src/multimodal/feature_fusion.py"),
            "trainability_source_sha256": sha256_file(ROOT / "src/multimodal/trainability.py"),
            "causality_interventions_sha256": sha256_file(ROOT / "src/multimodal/causality_interventions.py"),
            "raw_sample_index_sha256": sha256_file(ROOT / "src/multimodal/raw_sample_index.py"),
            "preexecution_audit_sha256": audit_evidence["sha256"],
            "audit_source_sha256": sha256_file(ROOT / "scripts/audit_step4_a2.py"),
            "torch_version": torch.__version__,
            "ultralytics_version": __import__("ultralytics").__version__,
        },
        "systems": {},
        "gates": {},
    }

    for tag, spec in RUN_SPECS.items():
        run_dir = ROOT / a.project / spec["run"]
        model, manifest, ckpt_path = load_frozen_model(
            run_dir, spec, device, expected_checkpoint_sha[tag]
        )
        before_sha = state_sha256(model)
        native_probe = native_equivalence_probe(model, dataset, device)
        if not native_probe["passed"]:
            raise RuntimeError(f"A2_NATIVE_EQUIVALENCE_FAIL:{tag}")
        reference_q = {
            sid: row["q_native"] for sid, row in native_probe["rows"].items()
        }

        residual_cache = build_residual_cache(model, dataset, device)
        conditions = _condition_set(tag)
        condition_results = {}
        for condition in conditions:
            condition_results[condition.name] = collect_condition_stats(
                model, dataset, device, names, condition,
                donor_map=(donor_map if condition.kind.startswith("shuffle") else None),
                residual_cache=(residual_cache if condition.kind.startswith("shuffle") else None),
            )
            print(f"[{tag}] {condition.name}: "
                  f"{condition_results[condition.name]['full']['map50_95']:.6f}")

        q_gate = _assert_q_freeze(condition_results, reference_q, tag)
        if not q_gate["passed"]:
            raise RuntimeError(f"A2_Q_FREEZE_FAIL:{q_gate['errors'][:8]}")
        source_gate = _assert_residual_source_trace(condition_results, donor_map, tag)
        if not source_gate["passed"]:
            raise RuntimeError(f"A2_RESIDUAL_SOURCE_FAIL:{source_gate['errors'][:8]}")
        after_sha = state_sha256(model)
        if before_sha != after_sha:
            raise RuntimeError(f"A2_PARAMETER_MUTATION:{tag}:{before_sha}!={after_sha}")

        result["systems"][tag] = {
            "role": "primary" if tag == "FIXED" else "replication",
            "run_dir": str(run_dir.relative_to(ROOT)),
            "checkpoint": "last.pt",
            "checkpoint_sha256": sha256_file(ckpt_path),
            "manifest_sha256": sha256_file(run_dir / "manifest.json"),
            "manifest_identity": {
                k: manifest.get(k) for k in (
                    "group", "aux_mode", "gate_mode", "gate_module",
                    "expected_epochs", "seed",
                )
            },
            "state_sha256_before": before_sha,
            "state_sha256_after": after_sha,
            "native_equivalence": native_probe,
            "q_freeze": q_gate,
            "residual_source_trace": source_gate,
            "conditions": condition_results,
        }

    # Derived effects. All are computed from already-collected condition stats.
    scale_mask = {"P3": "M100", "P4": "M010", "P5": "M001"}
    drop_mask = {"P3": "M011", "P4": "M101", "P5": "M110"}
    derived = {}
    for tag in RUN_SPECS:
        c = result["systems"][tag]["conditions"]
        d = {"standalone": {}, "drop": {}, "conditional_paired": {},
             "standalone_paired": {}, "pair_interactions": {}}
        for scale in SCALES:
            d["standalone"][scale] = effect(c[scale_mask[scale]], c["M000"])
            d["drop"][scale] = effect(c["M111"], c[drop_mask[scale]])
            d["conditional_paired"][scale] = effect(
                c["M111"], c[f"SHUFFLE_{scale}_COND"])
            d["standalone_paired"][scale] = effect(
                c[scale_mask[scale]], c[f"SHUFFLE_{scale}_ONLY"])
        d["pair_interactions"] = {
            "P3_P4_full": (
                c["M110"]["full"]["map50_95"] - c["M100"]["full"]["map50_95"]
                - c["M010"]["full"]["map50_95"] + c["M000"]["full"]["map50_95"]
            ),
            "P3_P5_full": (
                c["M101"]["full"]["map50_95"] - c["M100"]["full"]["map50_95"]
                - c["M001"]["full"]["map50_95"] + c["M000"]["full"]["map50_95"]
            ),
            "P4_P5_full": (
                c["M011"]["full"]["map50_95"] - c["M010"]["full"]["map50_95"]
                - c["M001"]["full"]["map50_95"] + c["M000"]["full"]["map50_95"]
            ),
        }
        derived[tag] = d
    result["derived_effects"] = derived

    result["pairedness_labels"] = {}
    for axis in ("conditional_paired", "standalone_paired"):
        result["pairedness_labels"][axis] = {
            scale: classify_paired_effect(
                derived["FIXED"][axis][scale], derived["SOFT"][axis][scale]
            )
            for scale in SCALES
        }

    required_provenance = {
        "contract_sha256", "f1c_summary_sha256", "donor_map_sha256",
        "evaluator_source_sha256", "intervention_source_sha256",
        "design_freeze_sha256", "step3_eval_utils_sha256",
        "f1_model_source_sha256", "gate_source_sha256", "dataset_source_sha256",
        "f0_model_source_sha256", "aux_encoder_source_sha256",
        "feature_fusion_source_sha256", "trainability_source_sha256",
        "causality_interventions_sha256", "raw_sample_index_sha256",
        "preexecution_audit_sha256", "audit_source_sha256",
        "torch_version", "ultralytics_version",
    }
    g9 = (audit_evidence["passed"] and dependency_closure["passed"]
          and required_provenance.issubset(result["provenance"])
          and all(bool(result["provenance"].get(k)) for k in required_provenance)
          and all(bool(result["systems"][tag].get("checkpoint_sha256"))
                  and bool(result["systems"][tag].get("manifest_sha256")) for tag in RUN_SPECS))
    result["gates"] = {
        "G1_checkpoint_identity": (dependency_closure["passed"] and all(
            result["systems"][tag]["checkpoint_sha256"] == expected_checkpoint_sha[tag]
            for tag in RUN_SPECS
        )),
        "G2_eval_only_state_unchanged": all(
            result["systems"][tag]["state_sha256_before"]
            == result["systems"][tag]["state_sha256_after"] for tag in RUN_SPECS),
        "G3_native_equivalence": all(
            result["systems"][tag]["native_equivalence"]["passed"] for tag in RUN_SPECS),
        "G4_q_freeze": all(result["systems"][tag]["q_freeze"]["passed"] for tag in RUN_SPECS),
        "G5_post_gate_residual_only": all(
            result["systems"][tag]["residual_source_trace"]["passed"]
            for tag in RUN_SPECS
        ),
        "G6_donor_valid": (donor_map == expected_donor_map
                           and assert_valid_shuffle_map(donor_map, val_ids)),
        "G7_factorial_complete": list(MASK_CONDITIONS) == [c.name for c in mask_conditions()],
        "G8_stock_validator_semantics": dependency_closure["stock_eval_semantics_frozen"],
        "G9_provenance_complete": g9,
    }
    result["all_gates_passed"] = all(result["gates"].values())
    if not result["all_gates_passed"]:
        raise RuntimeError(f"A2_ABORT:{result['gates']}")

    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("->", out_path)


if __name__ == "__main__":
    main()
