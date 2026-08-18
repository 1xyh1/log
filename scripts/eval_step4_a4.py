#!/usr/bin/env python3
"""A4 evaluation-only Residual DC/AC Paired Causality Audit."""
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
    SCALES, build_residual_cache_no_gate, forward_with_custom_residuals,
    git_blob_sha1, state_sha256, tensor_sha256,
)
from multimodal.step4_a4_content_mask import sample_meta  # noqa: E402
from multimodal.step4_a4_dc_ac import (  # noqa: E402
    decompose_all, decompose_content, validate_component_trace,
)
from multimodal.step4_a4_decision import (  # noqa: E402
    FACTORIAL_CELLS, apply_content_diagnostic_veto, classify_ap_effect,
    content_diagnostic_interpretation, factorial_effects, joint_p5_decision,
)
from multimodal.trimodal_dataset import TriModalDataset  # noqa: E402

SCHEMA = "step4-a4-summary-v1"
EXPECTED_A2_RESULT_SHA256 = "756093358153c5e203f485dce96e0f2a5e91881fb6c6e4b49c036cbfdc6d1c6b"
EXPECTED_A3_SUMMARY_SHA256 = "121dacc0ed50f5d24a8108ea3710e981c3c0314210729c80ed339652ea579839"
EXPECTED_A3_SUMMARY_CANONICAL_LF_SHA256 = "3523cb526d7a0fde3b0f0f121f73f29326aa88167bf6ad60d0505d7fed50d9ed"
EXPECTED_DONOR_MAP_SHA256 = "c5cd8e852663eae3243bc5e4c263c6f2c26c7b2faa977ae3b60cb5c1ea122af5"
EXPECTED_MODALITY_PREPROCESS_GIT_BLOB_SHA1 = "ed3a52150eedee18c60f163401dc64a198398662"
RUN_TAGS = ("FIXED", "SOFT")
KEEP_ONLY = {"P3": "M100", "P4": "M010", "P5": "M001"}
SCALE_BIT = {"P3": 0, "P4": 1, "P5": 2}
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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_lf_sha256(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def json_load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def commit_json_bundle(payloads: dict[Path, dict]):
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
    path = root / "reports/step4_a4/preexecution_audit.json"
    if not path.exists():
        raise RuntimeError(f"A4_PREEXECUTION_AUDIT_MISSING:{path}")
    obj = json_load(path)
    if obj.get("schema") != "step4-a4-preexecution-audit-v1" or obj.get("all_passed") is not True:
        raise RuntimeError("A4_PREEXECUTION_AUDIT_NOT_PASSING")
    targets = {
        "design_sha256": root / "docs/step4_a4/DESIGN_FREEZE.md",
        "dc_ac_sha256": root / "src/multimodal/step4_a4_dc_ac.py",
        "content_mask_sha256": root / "src/multimodal/step4_a4_content_mask.py",
        "decision_sha256": root / "src/multimodal/step4_a4_decision.py",
        "evaluator_sha256": root / "scripts/eval_step4_a4.py",
        "tests_sha256": root / "tests/test_step4_a4.py",
        "audit_source_sha256": root / "scripts/audit_step4_a4.py",
    }
    current = {k: sha256_file(v) for k, v in targets.items()}
    stale = {
        k: {"recorded": obj.get("provenance", {}).get(k), "current": v}
        for k, v in current.items()
        if obj.get("provenance", {}).get(k) != v
    }
    if stale:
        raise RuntimeError(f"A4_PREEXECUTION_AUDIT_STALE:{stale}")
    return {
        "passed": True,
        "path": str(path.relative_to(root)),
        "sha256": sha256_file(path),
        "source_hashes": current,
    }


def verify_upstream_and_dependencies(root: Path, contract_path: Path):
    a2_path = root / "reports/step4_a2/scale_ir_residual_causality.json"
    donor_path = root / "reports/step4_a2/val_donor_map.json"
    a3_path = root / "reports/step4_a3/a3_summary.json"
    if sha256_file(a2_path) != EXPECTED_A2_RESULT_SHA256:
        raise RuntimeError("A4_UPSTREAM_FREEZE_FAIL:A2_RESULT_SHA")
    if sha256_file(donor_path) != EXPECTED_DONOR_MAP_SHA256:
        raise RuntimeError("A4_UPSTREAM_FREEZE_FAIL:DONOR_SHA")
    if sha256_file(a3_path) != EXPECTED_A3_SUMMARY_SHA256:
        raise RuntimeError("A4_UPSTREAM_FREEZE_FAIL:A3_SUMMARY_RAW_SHA")
    if canonical_lf_sha256(a3_path) != EXPECTED_A3_SUMMARY_CANONICAL_LF_SHA256:
        raise RuntimeError("A4_UPSTREAM_FREEZE_FAIL:A3_SUMMARY_CANONICAL_SHA")

    a2, donor, a3 = json_load(a2_path), json_load(donor_path), json_load(a3_path)
    if a2.get("schema") != "step4-a2-scale-ir-residual-causality-v2" or a2.get("all_gates_passed") is not True:
        raise RuntimeError("A4_UPSTREAM_FREEZE_FAIL:A2_NOT_ACCEPTABLE")
    if a3.get("schema") != "step4-a3-summary-v1" or a3.get("all_gates_passed") is not True:
        raise RuntimeError("A4_UPSTREAM_FREEZE_FAIL:A3_NOT_ACCEPTABLE")
    if not a3.get("gates") or not all(bool(v) for v in a3["gates"].values()):
        raise RuntimeError("A4_UPSTREAM_FREEZE_FAIL:A3_GATES")
    if donor != a2.get("donor_map"):
        raise RuntimeError("A4_DONOR_MAP_DRIFT:A2_CONTENT")
    if a3.get("protocol", {}).get("donor_map_sha256") != EXPECTED_DONOR_MAP_SHA256:
        raise RuntimeError("A4_DONOR_MAP_DRIFT:A3_PROTOCOL")
    if a3.get("provenance", {}).get("a2_result_sha256") != EXPECTED_A2_RESULT_SHA256:
        raise RuntimeError("A4_UPSTREAM_FREEZE_FAIL:A3_A2_PROVENANCE")

    errors = []
    if sha256_file(contract_path) != a3.get("provenance", {}).get("contract_sha256"):
        errors.append("CONTRACT_DRIFT")
    current_sources = {k: sha256_file(root / rel) for k, rel in SOURCE_PATHS.items()}
    expected_sources = a3.get("provenance", {}).get("dependency_source_hashes") or {}
    for key, current in current_sources.items():
        if expected_sources.get(key) != current:
            errors.append(f"SOURCE_DRIFT:{key}")
    a3_common = root / "src/multimodal/step4_a3_common.py"
    if sha256_file(a3_common) != a3.get("provenance", {}).get("common_sha256"):
        errors.append("A3_COMMON_DRIFT")
    versions = {
        "torch_version": torch.__version__,
        "ultralytics_version": __import__("ultralytics").__version__,
    }
    for key, current in versions.items():
        if a3.get("provenance", {}).get(key) != current:
            errors.append(f"VERSION_DRIFT:{key}")
    mp_path = root / "src/multimodal/modality_preprocess.py"
    mp_blob = git_blob_sha1(mp_path)
    if mp_blob != EXPECTED_MODALITY_PREPROCESS_GIT_BLOB_SHA1:
        errors.append(f"MODALITY_PREPROCESS_BLOB_DRIFT:{mp_blob}")
    if sha256_file(mp_path) != a3.get("provenance", {}).get("modality_preprocess_sha256"):
        errors.append("MODALITY_PREPROCESS_SHA_DRIFT")
    f1c_summary = root / "runs/step4_f1_c/_summary_step4_f1_c.json"
    if sha256_file(f1c_summary) != a3.get("provenance", {}).get("f1c_summary_sha256"):
        errors.append("F1C_SUMMARY_DRIFT")
    f1_obj = json_load(f1c_summary)
    if f1_obj.get("verdict_frozen") is not True or f1_obj.get("decision") != "F1C_GATE_FAILED_CAUSAL_PROTOCOL":
        errors.append("F1C_SUMMARY_VERDICT_DRIFT")
    if errors:
        raise RuntimeError(f"A4_FROZEN_DEPENDENCY_CLOSURE_FAIL:{errors}")

    evidence = {
        "passed": True,
        "a2_result_sha256": EXPECTED_A2_RESULT_SHA256,
        "a3_summary_raw_sha256": EXPECTED_A3_SUMMARY_SHA256,
        "a3_summary_canonical_lf_sha256": EXPECTED_A3_SUMMARY_CANONICAL_LF_SHA256,
        "a2_donor_map_sha256": EXPECTED_DONOR_MAP_SHA256,
        "contract_sha256": sha256_file(contract_path),
        "current_source_hashes": current_sources,
        "a3_common_sha256": sha256_file(a3_common),
        "current_versions": versions,
        "modality_preprocess_git_blob_sha1": mp_blob,
        "modality_preprocess_sha256": sha256_file(mp_path),
        "f1c_summary_sha256": sha256_file(f1c_summary),
        "stock_eval_semantics_frozen": bool(a3.get("frozen_dependency_closure", {}).get("stock_eval_semantics_frozen")),
    }
    if evidence["stock_eval_semantics_frozen"] is not True:
        raise RuntimeError("A4_FROZEN_DEPENDENCY_CLOSURE_FAIL:STOCK_EVAL")
    return a2, a3, donor, evidence


def load_model(root: Path, project: str, tag: str, a2: dict, device):
    sysrec = a2["systems"][tag]
    run_dir = root / project / Path(sysrec["run_dir"]).name
    manifest_path = run_dir / "manifest.json"
    ckpt_path = run_dir / "weights/last.pt"
    if sha256_file(ckpt_path) != sysrec["checkpoint_sha256"]:
        raise RuntimeError(f"A4_CHECKPOINT_SHA_MISMATCH:{tag}")
    if sha256_file(manifest_path) != sysrec["manifest_sha256"]:
        raise RuntimeError(f"A4_MANIFEST_SHA_MISMATCH:{tag}")
    manifest = json_load(manifest_path)
    for key, expected in sysrec["manifest_identity"].items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"A4_CHECKPOINT_IDENTITY:{tag}:{key}")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = (ck.get("ema") or ck.get("model")).float().eval().to(device)
    if getattr(model, "_gate_override", None) is not None:
        model.set_gate_override(None)
    return model, run_dir, ckpt_path, manifest_path


def build_dataset(contract: dict, expected_ids: list[str]):
    ds = TriModalDataset(contract, split="val", group="C1-I", augment=False)
    if list(ds.ids) != list(expected_ids):
        raise RuntimeError(f"A4_VAL_SET_DRIFT:{list(ds.ids)}!={expected_ids}")
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
            raw = evu.extract_detection_tensor(output).detach()
            trace = dict(trace)
            trace["detection_sha256"] = tensor_sha256(raw)
            preds = validator.postprocess(raw)
            if len(preds) != 1:
                raise RuntimeError(f"A4_EXPECTED_ONE_PREDICTION:{len(preds)}")
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
    loo = {held: metric_from_stats(stats, [sid for sid in ids if sid != held], names) for held in ids}
    return {"full": full, "loo": loo, "trace": traces}


def ap_effect(new_result: dict, baseline_result: dict) -> dict:
    full = float(new_result["full"]["map50_95"] - baseline_result["full"]["map50_95"])
    loo = {
        sid: float(new_result["loo"][sid]["map50_95"] - baseline_result["loo"][sid]["map50_95"])
        for sid in baseline_result["loo"]
    }
    vals = list(loo.values())
    return {
        "full": full,
        "loo": loo,
        "loo_median": float(np.median(vals)),
        "positive_folds": int(sum(v > 0 for v in vals)),
        "negative_folds": int(sum(v < 0 for v in vals)),
        "zero_folds": int(sum(v == 0 for v in vals)),
    }


def native_probe(model, dataset, device, a2_system: dict, a3_system: dict):
    rows, qref = {}, {}
    model.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            sid = str(sample["sample_id"])
            batch = dataset.collate_fn([sample])
            batch = evu.move_step3_batch_to_device(batch, device)
            native = evu.extract_detection_tensor(model._predict_once(batch["img"])).detach()
            a4out, tr = forward_with_custom_residuals(
                model, batch["img"], recipient_id=sid, active_scales=SCALES,
                condition_name="A4_NATIVE_FULL",
            )
            custom = evu.extract_detection_tensor(a4out).detach()
            expected_a3 = a3_system["native_equivalence"]["rows"][sid]["native_sha256"]
            expected_a2 = a2_system["native_equivalence"]["rows"][sid]["native_sha256"]
            sh_native, sh_custom = tensor_sha256(native), tensor_sha256(custom)
            passed = torch.equal(native, custom) and sh_native == sh_custom == expected_a3 == expected_a2
            if not passed:
                raise RuntimeError(f"A4_NATIVE_EQUIVALENCE_FAIL:{sid}")
            rows[sid] = {
                "bitwise_equal": True,
                "native_sha256": sh_native,
                "a4_native_sha256": sh_custom,
                "a3_expected_sha256": expected_a3,
                "a2_expected_sha256": expected_a2,
                "q_native": tr["q_native"],
            }
            qref[sid] = tr["q_native"]
    return {"passed": True, "rows": rows}, qref


def sample_meta_cache(dataset) -> dict:
    out = {}
    for i in range(len(dataset)):
        s = dataset[i]
        out[str(s["sample_id"])] = sample_meta(s)
    return out


def residual_cache_with_gate_guard(model, dataset, device):
    calls = {"n": 0}
    handle = model.reliability_gate.register_forward_hook(
        lambda module, inputs, output: calls.__setitem__("n", calls["n"] + 1)
    )
    cache = build_residual_cache_no_gate(model, dataset, device)
    handle.remove()
    if calls["n"] != 0:
        raise RuntimeError(f"A4_RESIDUAL_INTERVENTION_SEMANTICS_FAIL:DONOR_CACHE_GATE:{calls['n']}")
    return cache, {"gate_calls": calls["n"], "passed": True}


def make_component(cache: dict, metas: dict, source_id: str, scale: str, mode: str):
    residual = cache[source_id][scale]
    if mode == "AC_ALL":
        tensor, evidence = decompose_all(residual, source_id=source_id)
    elif mode == "AC_CONTENT":
        tensor, evidence = decompose_content(
            residual,
            source_id=source_id,
            content_mask_source_id=source_id,
            meta=metas[source_id],
        )
    else:
        raise ValueError(mode)
    evidence = {**evidence, "scale": scale}
    return tensor, evidence


def forward_component(model, batch, *, sid: str, scale: str, context: str,
                      tensor: torch.Tensor, evidence: dict, role: str, mode: str):
    active = [scale] if context == "standalone" else SCALES
    out, tr = forward_with_custom_residuals(
        model, batch["img"], recipient_id=sid, active_scales=active,
        replacements={scale: tensor}, source_ids={scale: f"{mode}[{evidence['residual_source_id']}]"},
        condition_name=f"{mode}_{role.upper()}_{scale}_{context.upper()}",
    )
    tr["a4_role"] = role
    tr["a4_target_scale"] = scale
    tr["a4_context"] = context
    tr["a4_component_trace"] = {scale: evidence}
    return out, tr


def validate_q_trace(tag: str, qref: dict, blocks: list[dict]) -> bool:
    errors = []
    for block in blocks:
        for sid, tr in (block.get("trace") or {}).items():
            if tr.get("q_native") != qref[sid]:
                errors.append(f"{sid}:{tr.get('condition')}")
            if tag == "FIXED" and tr.get("q_native") != [1.0]:
                errors.append(f"FIXED_NOT_ONE:{sid}")
    if errors:
        raise RuntimeError(f"A4_Q_FREEZE_FAIL:{tag}:{errors[:10]}")
    return True


def validate_ac_block(block: dict, *, donor_map: dict, mode: str) -> dict:
    all_ok = True
    donor_self_ok = True
    coverage_ok = True
    provenance_ok = True
    post_projection_ok = True
    for sid, tr in block["trace"].items():
        scale = tr["a4_target_scale"]
        context = tr["a4_context"]
        role = tr["a4_role"]
        expected_active = set([scale] if context == "standalone" else SCALES)
        if set(tr.get("active_scales", [])) != expected_active:
            post_projection_ok = False
        for other in SCALES:
            if other != scale and tr["residual_source_ids"].get(other) != sid:
                post_projection_ok = False
        comp = tr.get("a4_component_trace", {}).get(scale) or {}
        if comp.get("mode") != mode or not validate_component_trace(comp):
            all_ok = False
        expected_source = sid if role == "native" else donor_map[sid]
        if comp.get("residual_source_id") != expected_source or comp.get("mean_source_id") != expected_source:
            if role == "donor":
                donor_self_ok = False
            else:
                all_ok = False
        if role == "donor" and comp.get("mean_source_id") != donor_map[sid]:
            donor_self_ok = False
        if mode == "AC_CONTENT":
            if comp.get("content_mask_source_id") != expected_source:
                if role == "donor":
                    donor_self_ok = False
                else:
                    all_ok = False
            mask = comp.get("content_mask") or {}
            if mask.get("source") != "ori_shape+ratio_pad":
                provenance_ok = False
            if float(mask.get("coverage_sum", 0)) <= 0:
                coverage_ok = False
    return {
        "decomposition_ok": all_ok,
        "donor_self_centering_ok": donor_self_ok,
        "content_coverage_ok": coverage_ok,
        "content_mask_provenance_ok": provenance_ok,
        "post_projection_ok": post_projection_ok,
    }


def run_ac_pair_rescue(model, dataset, device, names, tag, a2, donor_map, qref,
                       cache, metas, mode: str):
    out, qblocks = {}, []
    validation_rows = []
    for scale in SCALES:
        out[scale] = {}
        for context in ("standalone", "conditional"):
            def native_forward(sid, sample, batch, scale=scale, context=context):
                tensor, ev = make_component(cache, metas, sid, scale, mode)
                return forward_component(
                    model, batch, sid=sid, scale=scale, context=context,
                    tensor=tensor, evidence=ev, role="native", mode=mode,
                )
            def donor_forward(sid, sample, batch, scale=scale, context=context):
                donor = donor_map[sid]
                tensor, ev = make_component(cache, metas, donor, scale, mode)
                return forward_component(
                    model, batch, sid=sid, scale=scale, context=context,
                    tensor=tensor, evidence=ev, role="donor", mode=mode,
                )
            native = collect_ap_condition(model, dataset, device, names, native_forward)
            donor = collect_ap_condition(model, dataset, device, names, donor_forward)
            qblocks.extend([native, donor])
            validation_rows.extend([
                validate_ac_block(native, donor_map=donor_map, mode=mode),
                validate_ac_block(donor, donor_map=donor_map, mode=mode),
            ])
            baseline_name = KEEP_ONLY[scale] if context == "standalone" else "M111"
            baseline = a2["systems"][tag]["conditions"][baseline_name]
            out[scale][context] = {
                "mode": mode,
                "baseline_full_native_a2_condition": baseline_name,
                "native_ac": native,
                "donor_ac": donor,
                "paired_effect_native_minus_donor": ap_effect(native, donor),
                "centering_rescue_native_ac_minus_full_native": ap_effect(native, baseline),
            }
    validate_q_trace(tag, qref, qblocks)
    aggregate = {
        key: all(row[key] for row in validation_rows)
        for key in validation_rows[0]
    }
    return out, qblocks, aggregate


def run_factorial(model, dataset, device, names, tag, qref, cache, metas, a2_system, a3_system):
    cells, qblocks = {}, []
    for cell in FACTORIAL_CELLS:
        bits = cell[1:]
        centered = {SCALES[i] for i, b in enumerate(bits) if b == "1"}
        def forward_fn(sid, sample, batch, centered=centered, cell=cell):
            replacements, sources, comps = {}, {}, {}
            for scale in centered:
                tensor, ev = make_component(cache, metas, sid, scale, "AC_ALL")
                replacements[scale] = tensor
                sources[scale] = f"AC_ALL[{sid}]"
                comps[scale] = ev
            out, tr = forward_with_custom_residuals(
                model, batch["img"], recipient_id=sid, active_scales=SCALES,
                replacements=replacements, source_ids=sources,
                condition_name=cell,
            )
            tr["a4_factorial_cell"] = cell
            tr["a4_centered_scales"] = sorted(centered)
            tr["a4_component_trace"] = comps
            return out, tr
        result = collect_ap_condition(model, dataset, device, names, forward_fn)
        qblocks.append(result)
        cells[cell] = result
    validate_q_trace(tag, qref, qblocks)

    # Runtime factorial completeness + C000 raw/metric identity.
    if set(cells) != set(FACTORIAL_CELLS):
        raise RuntimeError("A4_FACTORIAL_INCOMPLETE:CELLS")
    for sid, tr in cells["C000"]["trace"].items():
        expected = a3_system["native_equivalence"]["rows"][sid]["native_sha256"]
        if tr["detection_sha256"] != expected:
            raise RuntimeError(f"A4_FACTORIAL_INCOMPLETE:C000_NATIVE:{sid}")
        if tr.get("a4_component_trace"):
            raise RuntimeError(f"A4_FACTORIAL_INCOMPLETE:C000_COMPONENT:{sid}")
    a2_m111 = a2_system["conditions"]["M111"]
    if cells["C000"]["full"]["map50_95"] != a2_m111["full"]["map50_95"]:
        raise RuntimeError("A4_FACTORIAL_INCOMPLETE:C000_A2_M111_FULL")
    for sid in cells["C000"]["loo"]:
        if cells["C000"]["loo"][sid]["map50_95"] != a2_m111["loo"][sid]["map50_95"]:
            raise RuntimeError(f"A4_FACTORIAL_INCOMPLETE:C000_A2_M111_LOO:{sid}")
    for cell, result in cells.items():
        expected_centered = {SCALES[i] for i, b in enumerate(cell[1:]) if b == "1"}
        for sid, tr in result["trace"].items():
            if set(tr.get("a4_centered_scales", [])) != expected_centered:
                raise RuntimeError(f"A4_FACTORIAL_INCOMPLETE:{cell}:{sid}:BITS")
            if set(tr.get("active_scales", [])) != set(SCALES):
                raise RuntimeError(f"A4_RESIDUAL_INTERVENTION_SEMANTICS_FAIL:{cell}:{sid}:ACTIVE")
            comps = tr.get("a4_component_trace", {})
            if set(comps) != expected_centered:
                raise RuntimeError(f"A4_FACTORIAL_INCOMPLETE:{cell}:{sid}:COMPONENTS")
            if not all(validate_component_trace(v) and v.get("mode") == "AC_ALL" for v in comps.values()):
                raise RuntimeError(f"A4_DC_AC_DECOMPOSITION_FAIL:{cell}:{sid}")
    return {"cells": cells, "effects": factorial_effects(cells)}, qblocks


def cross_system_labels(primary: dict):
    paired, rescue = {}, {}
    for scale in SCALES:
        paired[scale], rescue[scale] = {}, {}
        for context in ("standalone", "conditional"):
            pf = primary["FIXED"][scale][context]["paired_effect_native_minus_donor"]
            pr = primary["SOFT"][scale][context]["paired_effect_native_minus_donor"]
            paired[scale][context] = classify_ap_effect(pf, pr)
            rf = primary["FIXED"][scale][context]["centering_rescue_native_ac_minus_full_native"]
            rr = primary["SOFT"][scale][context]["centering_rescue_native_ac_minus_full_native"]
            rescue[scale][context] = classify_ap_effect(
                rf, rr,
                pos_label="STRONG_POSITIVE_RESCUE",
                neg_label="STRONG_NEGATIVE_RESCUE",
            )
    return paired, rescue


def content_labels(content: dict):
    paired, rescue = {}, {}
    for scale in SCALES:
        paired[scale], rescue[scale] = {}, {}
        for context in ("standalone", "conditional"):
            pf = content["FIXED"][scale][context]["paired_effect_native_minus_donor"]
            pr = content["SOFT"][scale][context]["paired_effect_native_minus_donor"]
            paired[scale][context] = classify_ap_effect(pf, pr)
            rf = content["FIXED"][scale][context]["centering_rescue_native_ac_minus_full_native"]
            rr = content["SOFT"][scale][context]["centering_rescue_native_ac_minus_full_native"]
            rescue[scale][context] = classify_ap_effect(
                rf, rr,
                pos_label="STRONG_POSITIVE_RESCUE",
                neg_label="STRONG_NEGATIVE_RESCUE",
            )
    return paired, rescue


def validate_provenance(prov: dict) -> bool:
    required = {
        "design_sha256", "evaluator_sha256", "dc_ac_sha256", "content_mask_sha256",
        "decision_sha256", "tests_sha256", "audit_source_sha256",
        "a2_result_sha256", "a3_summary_raw_sha256", "a3_summary_canonical_lf_sha256",
        "a2_donor_map_sha256", "f1c_summary_sha256", "contract_sha256",
        "torch_version", "ultralytics_version", "modality_preprocess_sha256",
        "modality_preprocess_git_blob_sha1", "a3_common_sha256",
        "FIXED_last_pt_sha256", "SOFT_last_pt_sha256",
        "FIXED_manifest_sha256", "SOFT_manifest_sha256",
        "preexecution_audit_sha256", "dc_definition_version", "content_coverage_definition",
    }
    missing = sorted(k for k in required if not prov.get(k))
    if missing:
        raise RuntimeError(f"A4_PROVENANCE_INCOMPLETE:{missing}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="runs/step4_f1_c")
    ap.add_argument("--contract", default=OUT_DEFAULT)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out-dir", default="reports/step4_a4")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    out_dir = ROOT / a.out_dir
    summary_path = out_dir / "a4_summary.json"
    evaluator_outputs = [
        out_dir / "ac_paired_standalone.json",
        out_dir / "ac_paired_conditional.json",
        out_dir / "centering_rescue.json",
        out_dir / "dc_removal_factorial.json",
        out_dir / "content_dc_diagnostic.json",
        summary_path,
    ]
    existing = [str(p) for p in evaluator_outputs if p.exists()]
    if existing and not a.overwrite:
        raise RuntimeError(f"A4_REFUSE_OVERWRITE:{existing}")

    audit = verify_preexecution_audit(ROOT)
    contract_path = Path(a.contract)
    contract = json_load(contract_path)
    a2, a3, donor_map, dependency = verify_upstream_and_dependencies(ROOT, contract_path)
    expected_ids = list(a3["protocol"]["val_ids"])
    if expected_ids != list(a2["protocol"]["val_ids"]):
        raise RuntimeError("A4_UPSTREAM_FREEZE_FAIL:VAL_IDS_A2_A3")
    dataset = build_dataset(contract, expected_ids)
    metas = sample_meta_cache(dataset)

    devarg = str(a.device)
    if devarg == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
    elif devarg.startswith("cuda:"):
        device = torch.device(devarg)
    else:
        device = torch.device(f"cuda:{devarg}")
    names = {int(k): v for k, v in CLASS_NAMES.items()} if isinstance(CLASS_NAMES, dict) else CLASS_NAMES

    models, identities, native_evidence, qrefs, state_before = {}, {}, {}, {}, {}
    residual_caches, cache_gate_guard = {}, {}
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
        native_evidence[tag], qrefs[tag] = native_probe(
            model, dataset, device, a2["systems"][tag], a3["systems"][tag]
        )
        residual_caches[tag], cache_gate_guard[tag] = residual_cache_with_gate_guard(model, dataset, device)

    primary, content, factorial = {}, {}, {}
    primary_valid, content_valid = {}, {}
    all_q_blocks = {t: [] for t in RUN_TAGS}
    for tag in RUN_TAGS:
        primary[tag], qblocks, primary_valid[tag] = run_ac_pair_rescue(
            models[tag], dataset, device, names, tag, a2, donor_map, qrefs[tag],
            residual_caches[tag], metas, "AC_ALL"
        )
        all_q_blocks[tag].extend(qblocks)
        factorial[tag], qblocks = run_factorial(
            models[tag], dataset, device, names, tag, qrefs[tag], residual_caches[tag], metas,
            a2["systems"][tag], a3["systems"][tag]
        )
        all_q_blocks[tag].extend(qblocks)
        content[tag], qblocks, content_valid[tag] = run_ac_pair_rescue(
            models[tag], dataset, device, names, tag, a2, donor_map, qrefs[tag],
            residual_caches[tag], metas, "AC_CONTENT"
        )
        all_q_blocks[tag].extend(qblocks)
        # Diagnostic comparison to primary AC_ALL, same condition identities.
        for scale in SCALES:
            for context in ("standalone", "conditional"):
                content[tag][scale][context]["native_content_minus_native_all"] = ap_effect(
                    content[tag][scale][context]["native_ac"],
                    primary[tag][scale][context]["native_ac"],
                )

    paired_labels, rescue_labels = cross_system_labels(primary)
    content_paired_labels, content_rescue_labels = content_labels(content)

    factorial_cross_labels = {}
    for effect in ("R3", "R4", "R5"):
        factorial_cross_labels[effect] = classify_ap_effect(
            factorial["FIXED"]["effects"][effect],
            factorial["SOFT"]["effects"][effect],
            pos_label="STRONG_POSITIVE_RESCUE",
            neg_label="STRONG_NEGATIVE_RESCUE",
        )

    p5_primary_joint = joint_p5_decision(paired_labels["P5"], rescue_labels["P5"])
    p5_joint = apply_content_diagnostic_veto(p5_primary_joint, content_rescue_labels["P5"])
    content_interpretation = {
        scale: {
            ctx: content_diagnostic_interpretation(
                rescue_labels[scale][ctx], content_rescue_labels[scale][ctx]
            )
            for ctx in ("standalone", "conditional")
        }
        for scale in SCALES
    }

    state_after = {tag: state_sha256(models[tag]) for tag in RUN_TAGS}
    if any(state_after[t] != state_before[t] for t in RUN_TAGS):
        raise RuntimeError("A4_PARAMETER_MUTATION")
    q_freeze = {tag: validate_q_trace(tag, qrefs[tag], all_q_blocks[tag]) for tag in RUN_TAGS}

    # Aggregate runtime evidence for G7-G11.
    donor_self_ok = all(primary_valid[t]["donor_self_centering_ok"] for t in RUN_TAGS) and all(
        content_valid[t]["donor_self_centering_ok"] for t in RUN_TAGS
    )
    dc_ac_ok = all(primary_valid[t]["decomposition_ok"] for t in RUN_TAGS) and all(
        content_valid[t]["decomposition_ok"] for t in RUN_TAGS
    )
    content_prov_ok = all(content_valid[t]["content_mask_provenance_ok"] for t in RUN_TAGS)
    content_coverage_ok = all(content_valid[t]["content_coverage_ok"] for t in RUN_TAGS)
    post_projection_ok = all(primary_valid[t]["post_projection_ok"] for t in RUN_TAGS) and all(
        content_valid[t]["post_projection_ok"] for t in RUN_TAGS
    )
    factorial_ok = all(set(factorial[t]["cells"]) == set(FACTORIAL_CELLS) for t in RUN_TAGS)

    if not donor_self_ok:
        raise RuntimeError("A4_DONOR_AC_MEAN_SOURCE_FAIL")
    if not dc_ac_ok:
        raise RuntimeError("A4_DC_AC_DECOMPOSITION_FAIL")
    if not content_prov_ok:
        raise RuntimeError("A4_CONTENT_MASK_PROVENANCE_FAIL")
    if not content_coverage_ok:
        raise RuntimeError("A4_CONTENT_DC_COVERAGE_FAIL")
    if not post_projection_ok:
        raise RuntimeError("A4_RESIDUAL_INTERVENTION_SEMANTICS_FAIL")
    if not factorial_ok:
        raise RuntimeError("A4_FACTORIAL_INCOMPLETE")

    provenance = {
        "design_sha256": sha256_file(ROOT / "docs/step4_a4/DESIGN_FREEZE.md"),
        "evaluator_sha256": sha256_file(ROOT / "scripts/eval_step4_a4.py"),
        "dc_ac_sha256": sha256_file(ROOT / "src/multimodal/step4_a4_dc_ac.py"),
        "content_mask_sha256": sha256_file(ROOT / "src/multimodal/step4_a4_content_mask.py"),
        "decision_sha256": sha256_file(ROOT / "src/multimodal/step4_a4_decision.py"),
        "tests_sha256": sha256_file(ROOT / "tests/test_step4_a4.py"),
        "audit_source_sha256": sha256_file(ROOT / "scripts/audit_step4_a4.py"),
        "a2_result_sha256": dependency["a2_result_sha256"],
        "a3_summary_raw_sha256": dependency["a3_summary_raw_sha256"],
        "a3_summary_canonical_lf_sha256": dependency["a3_summary_canonical_lf_sha256"],
        "a2_donor_map_sha256": dependency["a2_donor_map_sha256"],
        "f1c_summary_sha256": dependency["f1c_summary_sha256"],
        "contract_sha256": dependency["contract_sha256"],
        "torch_version": dependency["current_versions"]["torch_version"],
        "ultralytics_version": dependency["current_versions"]["ultralytics_version"],
        "modality_preprocess_sha256": dependency["modality_preprocess_sha256"],
        "modality_preprocess_git_blob_sha1": dependency["modality_preprocess_git_blob_sha1"],
        "a3_common_sha256": dependency["a3_common_sha256"],
        "FIXED_last_pt_sha256": identities["FIXED"]["last_pt_sha256"],
        "SOFT_last_pt_sha256": identities["SOFT"]["last_pt_sha256"],
        "FIXED_manifest_sha256": identities["FIXED"]["manifest_sha256"],
        "SOFT_manifest_sha256": identities["SOFT"]["manifest_sha256"],
        "preexecution_audit_sha256": audit["sha256"],
        "dependency_source_hashes": dependency["current_source_hashes"],
        "val_ids": expected_ids,
        "dc_definition_version": "A4_DC_ALL_V1=mean_full_feature_HW;AC=residual-DC",
        "content_coverage_definition": "adaptive_avg_pool2d(binary_letterbox_content_mask)",
    }
    provenance_ok = validate_provenance(provenance)

    gates = {
        "G1_upstream_freeze": (
            a2.get("all_gates_passed") is True
            and a3.get("all_gates_passed") is True
            and all(bool(v) for v in a3["gates"].values())
            and donor_map == a2.get("donor_map")
        ),
        "G2_frozen_dependency_closure": dependency["passed"],
        "G3_eval_only_state_unchanged": all(state_before[t] == state_after[t] for t in RUN_TAGS),
        "G4_native_equivalence": all(native_evidence[t]["passed"] for t in RUN_TAGS),
        "G5_q_freeze": all(q_freeze.values()),
        "G6_donor_freeze": donor_map == a2.get("donor_map") and a3["protocol"]["donor_map_sha256"] == EXPECTED_DONOR_MAP_SHA256,
        "G7_donor_ac_self_centering": donor_self_ok,
        "G8_full_map_dc_ac_semantics": dc_ac_ok,
        "G9_content_mask_provenance": content_prov_ok,
        "G10_content_dc_coverage": content_coverage_ok,
        "G11_post_projection_intervention": post_projection_ok and all(cache_gate_guard[t]["passed"] for t in RUN_TAGS),
        "G12_factorial_completeness": factorial_ok,
        "G13_stock_eval": dependency["stock_eval_semantics_frozen"],
        "G14_provenance_complete": bool(provenance_ok),
    }
    if not all(gates.values()):
        raise RuntimeError(f"A4_ABORT:{gates}")

    interpretation = {
        "ac_utility_is_not_ac_paired_causality": True,
        "centering_rescue_is_not_paired_restoration": True,
        "dc_harm_is_not_projection_bias_parameter_harm": True,
        "ac_content_is_diagnostic_only": True,
        "fixed_primary_soft_replication": True,
        "p5_primary_p3_secondary_p4_control": True,
        "no_post_result_threshold_edits": True,
    }

    summary = {
        "schema": SCHEMA,
        "protocol": {
            "evaluation_only": True,
            "primary_system": "F1C-I-fixed/last.pt",
            "replication_system": "F1C-I-soft/last.pt",
            "primary_scale": "P5",
            "secondary_scale": "P3",
            "control_scale": "P4",
            "val_ids": expected_ids,
            "donor_map_sha256": EXPECTED_DONOR_MAP_SHA256,
            "q_rule": "recipient untouched native q before all post-projection DC/AC interventions",
            "primary_dc": "DC_ALL=mean over full projected residual HxW",
            "diagnostic_dc": "DC_CONTENT=fractional letterbox-content-weighted mean",
        },
        "preexecution_audit": audit,
        "frozen_dependency_closure": dependency,
        "systems": {
            tag: {
                **identities[tag],
                "state_sha256_before": state_before[tag],
                "state_sha256_after": state_after[tag],
                "native_equivalence": native_evidence[tag],
                "residual_cache_no_gate": cache_gate_guard[tag],
            }
            for tag in RUN_TAGS
        },
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "pairedness_labels_ac_all": paired_labels,
        "centering_rescue_labels_ac_all": rescue_labels,
        "factorial_main_effect_labels": factorial_cross_labels,
        "content_diagnostic": {
            "pairedness_labels_ac_content": content_paired_labels,
            "centering_rescue_labels_ac_content": content_rescue_labels,
            "all_vs_content_interpretation": content_interpretation,
            "training_go_allowed": False,
        },
        "joint_decision": {
            "P5": p5_joint,
            "training_go": bool(p5_joint["training_go"]),
            "decision_source": "AC_ALL pairedness + AC_ALL centering rescue only",
        },
        "provenance": provenance,
        "interpretation_discipline": interpretation,
    }

    standalone_report = {
        "schema": "step4-a4-ac-paired-standalone-v1",
        "systems": {tag: {s: primary[tag][s]["standalone"] for s in SCALES} for tag in RUN_TAGS},
        "labels": {s: paired_labels[s]["standalone"] for s in SCALES},
    }
    conditional_report = {
        "schema": "step4-a4-ac-paired-conditional-v1",
        "systems": {tag: {s: primary[tag][s]["conditional"] for s in SCALES} for tag in RUN_TAGS},
        "labels": {s: paired_labels[s]["conditional"] for s in SCALES},
    }
    rescue_report = {
        "schema": "step4-a4-centering-rescue-v1",
        "systems": {
            tag: {
                s: {
                    ctx: primary[tag][s][ctx]["centering_rescue_native_ac_minus_full_native"]
                    for ctx in ("standalone", "conditional")
                }
                for s in SCALES
            }
            for tag in RUN_TAGS
        },
        "labels": rescue_labels,
    }
    factorial_report = {
        "schema": "step4-a4-dc-removal-factorial-v1",
        "systems": factorial,
        "main_effect_labels": factorial_cross_labels,
        "bit_order": ["P3", "P4", "P5"],
        "bit_semantics": {"0": "FULL native", "1": "AC_ALL native"},
    }
    content_report = {
        "schema": "step4-a4-content-dc-diagnostic-v1",
        "diagnostic_only": True,
        "systems": content,
        "pairedness_labels": content_paired_labels,
        "centering_rescue_labels": content_rescue_labels,
        "all_vs_content_interpretation": content_interpretation,
        "training_go_allowed": False,
    }

    commit_json_bundle({
        out_dir / "ac_paired_standalone.json": standalone_report,
        out_dir / "ac_paired_conditional.json": conditional_report,
        out_dir / "centering_rescue.json": rescue_report,
        out_dir / "dc_removal_factorial.json": factorial_report,
        out_dir / "content_dc_diagnostic.json": content_report,
        summary_path: summary,
    })
    print(json.dumps({
        "schema": SCHEMA,
        "all_gates_passed": True,
        "gates": gates,
        "pairedness_labels_ac_all": paired_labels,
        "centering_rescue_labels_ac_all": rescue_labels,
        "joint_decision": summary["joint_decision"],
        "summary": str(summary_path),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"A4_ABORT:{type(exc).__name__}:{exc}", file=sys.stderr)
        raise
