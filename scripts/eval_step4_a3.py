#!/usr/bin/env python3
"""A3 evaluation-only RGB-IR Spatial / Semantic Agreement Audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal import step3_eval_utils as evu  # noqa: E402
from multimodal.raw_sample_index import CLASS_NAMES, OUT_DEFAULT  # noqa: E402
from multimodal.step4_a3_common import (  # noqa: E402
    SCALES, ap_effect, build_residual_cache_no_gate, classify_ap_effect,
    classify_sample_metric, energy_map, extract_recipient_features,
    forward_with_custom_residuals, git_blob_sha1, infer_feature_stride,
    state_sha256, tensor_sha256,
)
from multimodal.step4_a3_generic_bias import (  # noqa: E402
    build_generic_components, classify_generic_labels,
)
from multimodal.step4_a3_registration import (  # noqa: E402
    cross_fitted_median_shifts, estimate_registration_shift,
    raw_shift_to_feature_cells,
)
from multimodal.step4_a3_semantic import semantic_row, summarize_semantic_rows  # noqa: E402
from multimodal.step4_a3_spatial import spatial_row, summarize_spatial_rows  # noqa: E402
from multimodal.trimodal_dataset import TriModalDataset  # noqa: E402

SCHEMA = "step4-a3-summary-v1"
EXPECTED_A2_RESULT_SHA256 = "756093358153c5e203f485dce96e0f2a5e91881fb6c6e4b49c036cbfdc6d1c6b"
EXPECTED_A2_DONOR_MAP_SHA256 = "c5cd8e852663eae3243bc5e4c263c6f2c26c7b2faa977ae3b60cb5c1ea122af5"
EXPECTED_MODALITY_PREPROCESS_GIT_BLOB_SHA1 = "ed3a52150eedee18c60f163401dc64a198398662"
SOURCE_PATHS = {
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
RUN_TAGS = ("FIXED", "SOFT")
KEEP_ONLY = {"P3": "M100", "P4": "M010", "P5": "M001"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def json_load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def commit_json_bundle(payloads: dict[Path, dict]):
    """Best-effort transactional publication: write all .tmp files, then replace."""
    temps = []
    try:
        for path, obj in payloads.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
            temps.append((tmp, path))
        for tmp, path in temps:
            tmp.replace(path)
    except Exception:
        for tmp, _ in temps:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
        raise


def verify_preexecution_audit(root: Path) -> dict:
    path = root / "reports/step4_a3/preexecution_audit.json"
    if not path.exists():
        raise RuntimeError(f"A3_PREEXECUTION_AUDIT_MISSING:{path}")
    obj = json_load(path)
    if obj.get("schema") != "step4-a3-preexecution-audit-v1" or obj.get("all_passed") is not True:
        raise RuntimeError("A3_PREEXECUTION_AUDIT_NOT_PASSING")
    targets = {
        "design_sha256": root / "docs/step4_a3/DESIGN_FREEZE.md",
        "common_sha256": root / "src/multimodal/step4_a3_common.py",
        "registration_sha256": root / "src/multimodal/step4_a3_registration.py",
        "spatial_sha256": root / "src/multimodal/step4_a3_spatial.py",
        "semantic_sha256": root / "src/multimodal/step4_a3_semantic.py",
        "generic_bias_sha256": root / "src/multimodal/step4_a3_generic_bias.py",
        "evaluator_sha256": root / "scripts/eval_step4_a3.py",
        "tests_sha256": root / "tests/test_step4_a3.py",
        "audit_source_sha256": root / "scripts/audit_step4_a3.py",
    }
    current = {k: sha256_file(v) for k, v in targets.items()}
    stale = {k: {"recorded": obj.get("provenance", {}).get(k), "current": v}
             for k, v in current.items() if obj.get("provenance", {}).get(k) != v}
    if stale:
        raise RuntimeError(f"A3_PREEXECUTION_AUDIT_STALE:{stale}")
    return {"passed": True, "path": str(path.relative_to(root)),
            "sha256": sha256_file(path), "source_hashes": current}


def verify_upstream_and_dependencies(root: Path, contract_path: Path) -> tuple[dict, dict, dict]:
    a2_path = root / "reports/step4_a2/scale_ir_residual_causality.json"
    donor_path = root / "reports/step4_a2/val_donor_map.json"
    if sha256_file(a2_path) != EXPECTED_A2_RESULT_SHA256:
        raise RuntimeError("A3_UPSTREAM_FREEZE_FAIL:A2_RESULT_SHA")
    if sha256_file(donor_path) != EXPECTED_A2_DONOR_MAP_SHA256:
        raise RuntimeError("A3_UPSTREAM_FREEZE_FAIL:A2_DONOR_SHA")
    a2 = json_load(a2_path)
    donor_file = json_load(donor_path)
    if a2.get("schema") != "step4-a2-scale-ir-residual-causality-v2" or a2.get("all_gates_passed") is not True:
        raise RuntimeError("A3_UPSTREAM_FREEZE_FAIL:A2_RESULT_NOT_ACCEPTABLE")
    if donor_file != a2.get("donor_map"):
        raise RuntimeError("A3_DONOR_MAP_DRIFT:CONTENT")
    if a2.get("provenance", {}).get("donor_map_sha256") != EXPECTED_A2_DONOR_MAP_SHA256:
        raise RuntimeError("A3_DONOR_MAP_DRIFT:A2_PROVENANCE")

    errors = []
    closure = a2.get("frozen_dependency_closure") or {}
    if closure.get("passed") is not True or closure.get("stock_eval_semantics_frozen") is not True:
        errors.append("A2_DEPENDENCY_CLOSURE_NOT_PASSING")
    if sha256_file(contract_path) != a2.get("provenance", {}).get("contract_sha256"):
        errors.append("CONTRACT_DRIFT")
    current_sources = {k: sha256_file(root / rel) for k, rel in SOURCE_PATHS.items()}
    expected_sources = closure.get("current_source_hashes") or {}
    for key, current in current_sources.items():
        if expected_sources.get(key) != current:
            errors.append(f"SOURCE_DRIFT:{key}")
    versions = {
        "torch_version": torch.__version__,
        "ultralytics_version": __import__("ultralytics").__version__,
    }
    for key, current in versions.items():
        if closure.get("current_versions", {}).get(key) != current:
            errors.append(f"VERSION_DRIFT:{key}")
    mp_path = root / "src/multimodal/modality_preprocess.py"
    mp_blob = git_blob_sha1(mp_path)
    if mp_blob != EXPECTED_MODALITY_PREPROCESS_GIT_BLOB_SHA1:
        errors.append(f"MODALITY_PREPROCESS_DRIFT:{mp_blob}")
    f1c_summary = root / "runs/step4_f1_c/_summary_step4_f1_c.json"
    if sha256_file(f1c_summary) != a2.get("provenance", {}).get("f1c_summary_sha256"):
        errors.append("F1C_SUMMARY_DRIFT")
    f1_obj = json_load(f1c_summary)
    if f1_obj.get("verdict_frozen") is not True or f1_obj.get("decision") != "F1C_GATE_FAILED_CAUSAL_PROTOCOL":
        errors.append("F1C_SUMMARY_VERDICT_DRIFT")
    if errors:
        raise RuntimeError(f"A3_FROZEN_DEPENDENCY_CLOSURE_FAIL:{errors}")
    evidence = {
        "passed": True,
        "a2_result_sha256": EXPECTED_A2_RESULT_SHA256,
        "a2_donor_map_sha256": EXPECTED_A2_DONOR_MAP_SHA256,
        "contract_sha256": sha256_file(contract_path),
        "current_source_hashes": current_sources,
        "current_versions": versions,
        "modality_preprocess_git_blob_sha1": mp_blob,
        "modality_preprocess_sha256": sha256_file(mp_path),
        "f1c_summary_sha256": sha256_file(f1c_summary),
        "stock_eval_semantics_frozen": True,
    }
    return a2, donor_file, evidence


def load_model(root: Path, project: str, tag: str, a2: dict, device):
    sysrec = a2["systems"][tag]
    run_dir = root / project / Path(sysrec["run_dir"]).name
    manifest_path = run_dir / "manifest.json"
    ckpt_path = run_dir / "weights/last.pt"
    if sha256_file(ckpt_path) != sysrec["checkpoint_sha256"]:
        raise RuntimeError(f"A3_CHECKPOINT_SHA_MISMATCH:{tag}")
    if sha256_file(manifest_path) != sysrec["manifest_sha256"]:
        raise RuntimeError(f"A3_MANIFEST_SHA_MISMATCH:{tag}")
    manifest = json_load(manifest_path)
    for key, expected in sysrec["manifest_identity"].items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"A3_CHECKPOINT_IDENTITY:{tag}:{key}")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = (ck.get("ema") or ck.get("model")).float().eval().to(device)
    if getattr(model, "_gate_override", None) is not None:
        model.set_gate_override(None)
    return model, run_dir, ckpt_path, manifest_path


def build_dataset(contract: dict, expected_ids: list[str]):
    ds = TriModalDataset(contract, split="val", group="C1-I", augment=False)
    if list(ds.ids) != list(expected_ids):
        raise RuntimeError(f"A3_VAL_SET_DRIFT:{list(ds.ids)}!={expected_ids}")
    return ds


def metric_from_stats(stats_by_id: dict, ids: list[str], names: dict) -> dict:
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


def collect_ap_condition(model, dataset, device, names, forward_fn):
    validator = evu.make_detection_validator(model, device, names)
    stats, traces = {}, {}
    model.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            sid = str(sample["sample_id"])
            batch = dataset.collate_fn([sample])
            batch = evu.move_step3_batch_to_device(batch, device)
            output, trace = forward_fn(sid, sample, batch)
            raw = evu.extract_detection_tensor(output)
            preds = validator.postprocess(raw)
            if len(preds) != 1:
                raise RuntimeError(f"A3_EXPECTED_ONE_PREDICTION:{len(preds)}")
            pbatch = validator._prepare_batch(0, batch)
            pred = validator._prepare_pred(preds[0])
            cls_np = pbatch["cls"].detach().cpu().numpy()
            no_pred = pred["cls"].numel() == 0
            stat = validator._process_batch(pred, pbatch)
            stat.update(
                target_cls=cls_np,
                target_img=np.unique(cls_np),
                conf=np.zeros(0, dtype=np.float32) if no_pred else pred["conf"].detach().cpu().numpy(),
                pred_cls=np.zeros(0, dtype=np.float32) if no_pred else pred["cls"].detach().cpu().numpy(),
                im_name=sid,
            )
            stats[sid], traces[sid] = stat, trace
    ids = [str(x) for x in dataset.ids]
    full = metric_from_stats(stats, ids, names)
    loo = {held: metric_from_stats(stats, [sid for sid in ids if sid != held], names)
           for held in ids}
    return {"full": full, "loo": loo, "trace": traces}


def native_probe(model, dataset, device, a2_system: dict):
    rows, qref = {}, {}
    model.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            sid = str(sample["sample_id"])
            batch = dataset.collate_fn([sample])
            batch = evu.move_step3_batch_to_device(batch, device)
            native = evu.extract_detection_tensor(model._predict_once(batch["img"])).detach()
            a3out, tr = forward_with_custom_residuals(
                model, batch["img"], recipient_id=sid, active_scales=SCALES,
                condition_name="A3_NATIVE",
            )
            a3raw = evu.extract_detection_tensor(a3out).detach()
            expected = a2_system["native_equivalence"]["rows"][sid]["native_sha256"]
            passed = (torch.equal(native, a3raw)
                      and tensor_sha256(native) == expected
                      and tensor_sha256(a3raw) == expected)
            if not passed:
                raise RuntimeError(f"A3_NATIVE_EQUIVALENCE_FAIL:{sid}")
            rows[sid] = {
                "bitwise_equal": True,
                "native_sha256": tensor_sha256(native),
                "a3_native_sha256": tensor_sha256(a3raw),
                "a2_expected_sha256": expected,
                "q_native": tr["q_native"],
            }
            qref[sid] = tr["q_native"]
    return {"passed": True, "rows": rows}, qref


def validate_q_trace(tag: str, qref: dict, result_blocks: list[dict]):
    errors = []
    for block in result_blocks:
        for sid, tr in (block.get("trace") or {}).items():
            if tr["q_native"] != qref[sid]:
                errors.append(f"{sid}:{tr['condition']}")
            if tag == "FIXED" and tr["q_native"] != [1.0]:
                errors.append(f"FIXED_NOT_ONE:{sid}")
    if errors:
        raise RuntimeError(f"A3_Q_FREEZE_FAIL:{tag}:{errors[:10]}")
    return True


def validate_shift_trace(result: dict, target: str, context: str, ids: list[str]):
    for sid, tr in result["trace"].items():
        if "|SHIFT(" not in tr["residual_source_ids"][target]:
            raise RuntimeError(f"A3_RESIDUAL_INTERVENTION_SEMANTICS_FAIL:SHIFT:{sid}:{target}")
        expected_active = set(SCALES if context == "conditional" else [target])
        if set(tr["active_scales"]) != expected_active:
            raise RuntimeError(f"A3_RESIDUAL_INTERVENTION_SEMANTICS_FAIL:ACTIVE:{sid}")
        for s in SCALES:
            if s != target and tr["residual_source_ids"][s] != sid:
                raise RuntimeError(f"A3_RESIDUAL_INTERVENTION_SEMANTICS_FAIL:OTHER:{sid}:{s}")
    return True


def run_registration(model, dataset, device, names, tag, a2, raw_report, qref):
    systems = {}
    qblocks = []
    for scale in SCALES:
        systems[scale] = {}
        # Verify stride for every recipient and use it to map its cross-fit raw shift.
        stride_by_sid = {}
        with torch.no_grad():
            for i in range(len(dataset)):
                sample = dataset[i]
                sid = str(sample["sample_id"])
                batch = dataset.collate_fn([sample])
                img = batch["img"].to(device).float()
                f = extract_recipient_features(model, img)
                stride_by_sid[sid] = infer_feature_stride(f.input_hw, f.rgb[scale])
        for context in ("standalone", "conditional"):
            def forward_fn(sid, sample, batch, scale=scale, context=context):
                med = raw_report["cross_fitted"][sid]["median_shift"]
                cell = raw_shift_to_feature_cells(med["dx"], med["dy"], stride_by_sid[sid])
                return forward_with_custom_residuals(
                    model, batch["img"], recipient_id=sid,
                    active_scales=[scale] if context == "standalone" else SCALES,
                    shifts={scale: cell},
                    condition_name=f"SHIFT_{scale}_{context.upper()}",
                )
            result = collect_ap_condition(model, dataset, device, names, forward_fn)
            validate_shift_trace(result, scale, context, list(dataset.ids))
            qblocks.append(result)
            baseline_name = KEEP_ONLY[scale] if context == "standalone" else "M111"
            baseline = a2["systems"][tag]["conditions"][baseline_name]
            effect = ap_effect(result, baseline)
            systems[scale][context] = {
                "baseline_a2_condition": baseline_name,
                "shifted": result,
                "effect": effect,
                "feature_stride_by_sample": stride_by_sid,
            }
    validate_q_trace(tag, qref, qblocks)
    return systems, qblocks


def build_raw_registration(dataset) -> dict:
    raw = {}
    for i in range(len(dataset)):
        sample = dataset[i]
        sid = str(sample["sample_id"])
        s = estimate_registration_shift(sample)
        raw[sid] = s
    cross = cross_fitted_median_shifts(raw)
    # Runtime leakage gate.
    for sid, row in cross.items():
        if sid in row["train_ids_for_shift"] or len(row["train_ids_for_shift"]) != len(raw) - 1:
            raise RuntimeError(f"A3_REGISTRATION_LEAKAGE:{sid}")
    return {
        "estimator": "Sobel gradient + normalized phase correlation; common valid-content crop",
        "raw_shifts": {sid: {"dx": s.dx, "dy": s.dy, "phase_response": s.phase_response}
                       for sid, s in raw.items()},
        "cross_fitted": cross,
    }


def run_spatial_semantic(model, dataset, device, donor_map: dict):
    gate_calls = {"n": 0}
    handle = model.reliability_gate.register_forward_hook(
        lambda module, inputs, output: gate_calls.__setitem__("n", gate_calls["n"] + 1)
    )
    donor_cache = build_residual_cache_no_gate(model, dataset, device)
    handle.remove()
    if gate_calls["n"] != 0:
        raise RuntimeError(f"A3_RESIDUAL_INTERVENTION_SEMANTICS_FAIL:DONOR_CACHE_GATE:{gate_calls['n']}")

    spatial, semantic = {s: {} for s in SCALES}, {s: {} for s in SCALES}
    model.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            sid = str(sample["sample_id"])
            donor = donor_map[sid]
            batch = dataset.collate_fn([sample])
            img = batch["img"].to(device).float()
            f = extract_recipient_features(model, img)
            for scale in SCALES:
                native = f.residual[scale].detach().cpu()
                rgb = f.rgb[scale].detach().cpu()
                dres = donor_cache[donor][scale]
                spatial[scale][sid] = spatial_row(rgb, native, dres)
                try:
                    semantic[scale][sid] = semantic_row(sample["bboxes"], native, dres)
                except RuntimeError as e:
                    if "SEMANTIC_MASK_DEGENERATE" not in str(e):
                        raise
                    semantic[scale][sid] = {"valid": False, "error": "SEMANTIC_MASK_DEGENERATE"}
    spatial_summary = {s: summarize_spatial_rows(spatial[s]) for s in SCALES}
    semantic_summary = {s: summarize_semantic_rows(semantic[s], expected_n=len(dataset)) for s in SCALES}
    return spatial_summary, semantic_summary, donor_cache


def run_generic(model, dataset, device, names, tag, a2, qref, residual_cache):
    out = {}
    qblocks = []
    no_self_evidence = {}
    for scale in SCALES:
        comp_results = {}
        no_self_evidence[scale] = {}
        for component in ("LOO_MEAN", "NATIVE_DC", "NATIVE_AC", "LOO_MEAN_DC"):
            donor_sets = {}
            def forward_fn(sid, sample, batch, component=component, scale=scale):
                by_id = {k: v[scale] for k, v in residual_cache.items()}
                components = build_generic_components(by_id, sid)
                tensor, donors = components[component]
                if donors is not None:
                    if sid in donors or len(donors) != len(dataset) - 1:
                        raise RuntimeError(f"A3_LOO_MEAN_SELF_LEAKAGE:{scale}:{component}:{sid}")
                    donor_sets[sid] = donors
                    source = f"{component}[{','.join(donors)}]"
                else:
                    donor_sets[sid] = []
                    source = f"{component}[{sid}]"
                return forward_with_custom_residuals(
                    model, batch["img"], recipient_id=sid, active_scales=[scale],
                    replacements={scale: tensor}, source_ids={scale: source},
                    condition_name=f"{component}_{scale}",
                )
            result = collect_ap_condition(model, dataset, device, names, forward_fn)
            for rsid, tr in result["trace"].items():
                if set(tr["active_scales"]) != {scale}:
                    raise RuntimeError(
                        f"A3_RESIDUAL_INTERVENTION_SEMANTICS_FAIL:GENERIC_ACTIVE:{rsid}:{scale}"
                    )
                if not tr["residual_source_ids"][scale].startswith(component + "["):
                    raise RuntimeError(
                        f"A3_RESIDUAL_INTERVENTION_SEMANTICS_FAIL:GENERIC_SOURCE:{rsid}:{scale}:{component}"
                    )
                for other in SCALES:
                    if other != scale and tr["residual_source_ids"][other] != rsid:
                        raise RuntimeError(
                            f"A3_RESIDUAL_INTERVENTION_SEMANTICS_FAIL:GENERIC_OTHER:{rsid}:{other}"
                        )
            comp_results[component] = result
            no_self_evidence[scale][component] = donor_sets
            qblocks.append(result)

        zero = a2["systems"][tag]["conditions"]["M000"]
        native = a2["systems"][tag]["conditions"][KEEP_ONLY[scale]]
        e = {
            "native_minus_mean": ap_effect(native, comp_results["LOO_MEAN"]),
            "U_mean": ap_effect(comp_results["LOO_MEAN"], zero),
            "U_dc": ap_effect(comp_results["NATIVE_DC"], zero),
            "U_ac": ap_effect(comp_results["NATIVE_AC"], zero),
            "U_meanDC": ap_effect(comp_results["LOO_MEAN_DC"], zero),
        }
        out[scale] = {
            "a2_baselines": {"zero": "M000", "native": KEEP_ONLY[scale],
                             "donor": f"SHUFFLE_{scale}_ONLY"},
            "new_conditions": comp_results,
            "effects_by_system": e,
        }
    validate_q_trace(tag, qref, qblocks)
    return out, no_self_evidence, qblocks


def cross_system_labels(registration, spatial, semantic, generic):
    reg_labels, spatial_labels, semantic_labels, generic_labels = {}, {}, {}, {}
    for scale in SCALES:
        reg_labels[scale] = {}
        for context in ("standalone", "conditional"):
            reg_labels[scale][context] = classify_ap_effect(
                registration["FIXED"][scale][context]["effect"],
                registration["SOFT"][scale][context]["effect"],
                pos_label="STRONG_POSITIVE_RESCUE",
                neg_label="STRONG_NEGATIVE_RESCUE",
            )
        spatial_labels[scale] = classify_sample_metric(
            spatial["FIXED"][scale]["delta_summary"],
            spatial["SOFT"][scale]["delta_summary"],
        )
        semantic_labels[scale] = classify_sample_metric(
            semantic["FIXED"][scale]["delta_summary"],
            semantic["SOFT"][scale]["delta_summary"],
        )

        effects = {}
        for key in ("native_minus_mean", "U_mean", "U_dc", "U_ac", "U_meanDC"):
            p = generic["FIXED"][scale]["effects_by_system"][key]
            r = generic["SOFT"][scale]["effects_by_system"][key]
            effects[key] = {"primary": p, "replication": r,
                            "label": classify_ap_effect(p, r)}
        generic_labels[scale] = {
            "effects": effects,
            **classify_generic_labels(effects),
        }
    return reg_labels, spatial_labels, semantic_labels, generic_labels


def mechanism_matrix(reg_labels, spatial_labels, semantic_labels, generic_labels):
    return {
        scale: {
            "registration_rescue": reg_labels[scale],
            "spatial_recipient_specific": spatial_labels[scale],
            "semantic_recipient_specific": semantic_labels[scale],
            "generic_component": generic_labels[scale]["generic_component"],
            "generic_dc": generic_labels[scale]["generic_dc"],
            "spatial_ac": generic_labels[scale]["spatial_ac"],
        }
        for scale in SCALES
    }


def decision_branches(matrix: dict) -> dict:
    reg = [s for s, row in matrix.items()
           if "STRONG_POSITIVE_RESCUE" in row["registration_rescue"].values()]
    spatial_bad = [s for s, row in matrix.items()
                   if row["spatial_recipient_specific"] == "STRONG_DONOR_FAVORED"]
    semantic_bad = [s for s, row in matrix.items()
                    if row["semantic_recipient_specific"] == "STRONG_DONOR_FAVORED"]
    generic = [s for s, row in matrix.items()
               if row["generic_component"] == "GENERIC_COMPONENT_SUPPORTED"
               or row["generic_dc"] == "GENERIC_DC_SUPPORTED"]
    spatial_ac = [s for s, row in matrix.items()
                  if row["spatial_ac"] == "SPATIAL_AC_SUPPORTED"]
    return {
        "registration_implicated_scales": reg,
        "spatial_donor_favored_scales": spatial_bad,
        "semantic_donor_favored_scales": semantic_bad,
        "generic_bias_supported_scales": generic,
        "spatial_ac_supported_scales": spatial_ac,
        "A3_DIAGNOSIS_INCONCLUSIVE": not bool(
            reg or spatial_bad or semantic_bad or generic or spatial_ac
        ),
    }


def check_post_projection_reports(registration: dict, generic: dict) -> bool:
    for tag in RUN_TAGS:
        for scale in SCALES:
            for context in ("standalone", "conditional"):
                block = registration[tag][scale][context]["shifted"]
                expected_active = set(SCALES if context == "conditional" else [scale])
                for sid, tr in block["trace"].items():
                    if set(tr["active_scales"]) != expected_active:
                        return False
                    if "|SHIFT(" not in tr["residual_source_ids"][scale]:
                        return False
                    if any(tr["residual_source_ids"][s] != sid for s in SCALES if s != scale):
                        return False
            for component, block in generic[tag][scale]["new_conditions"].items():
                for sid, tr in block["trace"].items():
                    if set(tr["active_scales"]) != {scale}:
                        return False
                    if not tr["residual_source_ids"][scale].startswith(component + "["):
                        return False
                    if any(tr["residual_source_ids"][s] != sid for s in SCALES if s != scale):
                        return False
    return True


def validate_provenance(prov: dict):
    required = {
        "design_sha256", "evaluator_sha256", "common_sha256", "registration_sha256",
        "spatial_sha256", "semantic_sha256", "generic_bias_sha256",
        "a2_result_sha256", "a2_donor_map_sha256", "f1c_summary_sha256",
        "contract_sha256", "torch_version", "ultralytics_version",
        "modality_preprocess_sha256", "modality_preprocess_git_blob_sha1",
        "FIXED_last_pt_sha256", "SOFT_last_pt_sha256",
        "FIXED_manifest_sha256", "SOFT_manifest_sha256",
    }
    missing = sorted(k for k in required if not prov.get(k))
    if missing:
        raise RuntimeError(f"A3_PROVENANCE_INCOMPLETE:{missing}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="runs/step4_f1_c")
    ap.add_argument("--contract", default=OUT_DEFAULT)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out-dir", default="reports/step4_a3")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    out_dir = ROOT / a.out_dir
    summary_path = out_dir / "a3_summary.json"
    evaluator_outputs = [
        out_dir / "raw_registration.json",
        out_dir / "registration_rescue.json",
        out_dir / "spatial_correspondence.json",
        out_dir / "semantic_agreement.json",
        out_dir / "generic_residual_bias.json",
        summary_path,
    ]
    existing_outputs = [str(p) for p in evaluator_outputs if p.exists()]
    if existing_outputs and not a.overwrite:
        raise RuntimeError(f"A3_REFUSE_OVERWRITE:{existing_outputs}")

    audit = verify_preexecution_audit(ROOT)
    contract_path = Path(a.contract)
    contract = json_load(contract_path)
    a2, donor_map, dependency = verify_upstream_and_dependencies(ROOT, contract_path)
    expected_ids = list(a2["protocol"]["val_ids"])
    dataset = build_dataset(contract, expected_ids)
    devarg = str(a.device)
    if devarg == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
    elif devarg.startswith("cuda:"):
        device = torch.device(devarg)
    else:
        device = torch.device(f"cuda:{devarg}")
    names = {int(k): v for k, v in CLASS_NAMES.items()} if isinstance(CLASS_NAMES, dict) else CLASS_NAMES

    raw_registration = build_raw_registration(dataset)

    models, identities = {}, {}
    native_evidence, qrefs = {}, {}
    state_before = {}
    for tag in RUN_TAGS:
        model, run_dir, ckpt_path, manifest_path = load_model(ROOT, a.project, tag, a2, device)
        models[tag] = model
        identities[tag] = {
            "run_dir": str(run_dir.relative_to(ROOT)),
            "last_pt": str(ckpt_path.relative_to(ROOT)),
            "last_pt_sha256": sha256_file(ckpt_path),
            "manifest_sha256": sha256_file(manifest_path),
        }
        state_before[tag] = state_sha256(model)
        native_evidence[tag], qrefs[tag] = native_probe(model, dataset, device, a2["systems"][tag])

    registration = {}
    spatial, semantic, generic = {}, {}, {}
    all_q_blocks = {tag: [] for tag in RUN_TAGS}
    loo_no_self = {}
    for tag in RUN_TAGS:
        model = models[tag]
        registration[tag], qblocks = run_registration(
            model, dataset, device, names, tag, a2, raw_registration, qrefs[tag]
        )
        all_q_blocks[tag].extend(qblocks)
        spatial[tag], semantic[tag], residual_cache = run_spatial_semantic(
            model, dataset, device, donor_map
        )
        generic[tag], loo_no_self[tag], qblocks = run_generic(
            model, dataset, device, names, tag, a2, qrefs[tag], residual_cache
        )
        all_q_blocks[tag].extend(qblocks)

    reg_labels, spatial_labels, semantic_labels, generic_labels = cross_system_labels(
        registration, spatial, semantic, generic
    )
    matrix = mechanism_matrix(reg_labels, spatial_labels, semantic_labels, generic_labels)
    branches = decision_branches(matrix)

    state_after = {tag: state_sha256(models[tag]) for tag in RUN_TAGS}
    if any(state_after[t] != state_before[t] for t in RUN_TAGS):
        raise RuntimeError("A3_PARAMETER_MUTATION")

    # G5 q freeze rechecked globally.
    q_freeze_status = {
        tag: bool(validate_q_trace(tag, qrefs[tag], all_q_blocks[tag]))
        for tag in RUN_TAGS
    }

    # G10 no-self recheck from persisted donor-set evidence.
    loo_no_self_ok = True
    for tag in RUN_TAGS:
        for scale, comps in loo_no_self[tag].items():
            for comp, rows in comps.items():
                for sid, donors in rows.items():
                    if donors and sid in donors:
                        loo_no_self_ok = False
                        raise RuntimeError(f"A3_LOO_MEAN_SELF_LEAKAGE:{tag}:{scale}:{comp}:{sid}")

    post_projection_ok = check_post_projection_reports(registration, generic)
    if not post_projection_ok:
        raise RuntimeError("A3_RESIDUAL_INTERVENTION_SEMANTICS_FAIL:REPORT_RECHECK")

    provenance = {
        "design_sha256": sha256_file(ROOT / "docs/step4_a3/DESIGN_FREEZE.md"),
        "evaluator_sha256": sha256_file(ROOT / "scripts/eval_step4_a3.py"),
        "common_sha256": sha256_file(ROOT / "src/multimodal/step4_a3_common.py"),
        "registration_sha256": sha256_file(ROOT / "src/multimodal/step4_a3_registration.py"),
        "spatial_sha256": sha256_file(ROOT / "src/multimodal/step4_a3_spatial.py"),
        "semantic_sha256": sha256_file(ROOT / "src/multimodal/step4_a3_semantic.py"),
        "generic_bias_sha256": sha256_file(ROOT / "src/multimodal/step4_a3_generic_bias.py"),
        "a2_result_sha256": EXPECTED_A2_RESULT_SHA256,
        "a2_donor_map_sha256": EXPECTED_A2_DONOR_MAP_SHA256,
        "f1c_summary_sha256": dependency["f1c_summary_sha256"],
        "contract_sha256": dependency["contract_sha256"],
        "torch_version": dependency["current_versions"]["torch_version"],
        "ultralytics_version": dependency["current_versions"]["ultralytics_version"],
        "modality_preprocess_sha256": dependency["modality_preprocess_sha256"],
        "modality_preprocess_git_blob_sha1": dependency["modality_preprocess_git_blob_sha1"],
        "FIXED_last_pt_sha256": identities["FIXED"]["last_pt_sha256"],
        "SOFT_last_pt_sha256": identities["SOFT"]["last_pt_sha256"],
        "FIXED_manifest_sha256": identities["FIXED"]["manifest_sha256"],
        "SOFT_manifest_sha256": identities["SOFT"]["manifest_sha256"],
        "val_ids": expected_ids,
        "preexecution_audit_sha256": audit["sha256"],
        "dependency_source_hashes": dependency["current_source_hashes"],
        "registration_estimator": raw_registration["estimator"],
        "semantic_mask": "normalized final-letterbox xywh -> union feature-cell mask",
    }
    provenance_ok = bool(validate_provenance(provenance))

    upstream_ok = (
        a2.get("schema") == "step4-a2-scale-ir-residual-causality-v2"
        and a2.get("all_gates_passed") is True
        and donor_map == a2.get("donor_map")
        and provenance["a2_result_sha256"] == EXPECTED_A2_RESULT_SHA256
        and provenance["a2_donor_map_sha256"] == EXPECTED_A2_DONOR_MAP_SHA256
    )
    gates = {
        "G1_upstream": upstream_ok,
        "G2_checkpoint_dependency": dependency["passed"],
        "G3_eval_only_state_unchanged": all(state_before[t] == state_after[t] for t in RUN_TAGS),
        "G4_native_equivalence": all(native_evidence[t]["passed"] for t in RUN_TAGS),
        "G5_q_freeze": all(q_freeze_status.values()),
        "G6_donor_freeze": donor_map == a2["donor_map"],
        "G7_registration_crossfit": all(
            sid not in row["train_ids_for_shift"] and len(row["train_ids_for_shift"]) == 5
            for sid, row in raw_registration["cross_fitted"].items()
        ),
        "G8_post_projection_intervention": post_projection_ok,
        "G9_semantic_mask": all(semantic[t][s]["valid_count"] >= 5 for t in RUN_TAGS for s in SCALES),
        "G10_loo_mean_no_self": loo_no_self_ok,
        "G11_stock_eval": dependency["stock_eval_semantics_frozen"],
        "G12_provenance": provenance_ok,
    }
    if not all(gates.values()):
        raise RuntimeError(f"A3_ABORT:{gates}")

    summary = {
        "schema": SCHEMA,
        "protocol": {
            "evaluation_only": True,
            "primary": "F1C-I-fixed/last.pt",
            "replication": "F1C-I-soft/last.pt",
            "val_ids": expected_ids,
            "donor_map_sha256": EXPECTED_A2_DONOR_MAP_SHA256,
            "q_rule": "recipient untouched native q before all interventions",
            "registration_rule": "raw-space LOO cross-fitted shift only; feature best-shift descriptive only",
        },
        "preexecution_audit": audit,
        "frozen_dependency_closure": dependency,
        "systems": {
            t: {
                **identities[t],
                "state_sha256_before": state_before[t],
                "state_sha256_after": state_after[t],
                "native_equivalence": native_evidence[t],
            } for t in RUN_TAGS
        },
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "mechanism_matrix": matrix,
        "registration_rescue_labels": reg_labels,
        "spatial_labels": spatial_labels,
        "semantic_labels": semantic_labels,
        "generic_bias_labels": generic_labels,
        "decision_branches": branches,
        "provenance": provenance,
        "interpretation_discipline": {
            "fixed_primary_soft_replication": True,
            "correlation_is_not_registration_causality": True,
            "residual_utility_is_not_paired_semantic_value": True,
            "generic_mean_utility_is_not_multimodal_information_use": True,
            "no_post_result_threshold_edits": True,
        },
    }
    commit_json_bundle({
        out_dir / "raw_registration.json": raw_registration,
        out_dir / "registration_rescue.json": {
            "schema": "step4-a3-registration-rescue-v1",
            "raw_registration_file": "raw_registration.json",
            "systems": registration,
            "labels": reg_labels,
        },
        out_dir / "spatial_correspondence.json": {
            "schema": "step4-a3-spatial-v1",
            "systems": spatial,
            "labels": spatial_labels,
        },
        out_dir / "semantic_agreement.json": {
            "schema": "step4-a3-semantic-v1",
            "systems": semantic,
            "labels": semantic_labels,
        },
        out_dir / "generic_residual_bias.json": {
            "schema": "step4-a3-generic-bias-v1",
            "systems": generic,
            "labels": generic_labels,
            "loo_no_self_evidence": loo_no_self,
        },
        summary_path: summary,
    })
    print(json.dumps({
        "schema": SCHEMA,
        "all_gates_passed": summary["all_gates_passed"],
        "gates": gates,
        "mechanism_matrix": matrix,
        "decision_branches": branches,
        "summary": str(summary_path),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"A3_ABORT:{type(exc).__name__}:{exc}", file=sys.stderr)
        raise
