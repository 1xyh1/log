"""Pure closeout helpers for the frozen F1-C protocol."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HISTORICAL_B1_SOFT_LAST = 0.304028

def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _check_sha(prov: dict, key: str, path: Path, errors: list[str]) -> None:
    current = sha256_file(path) if path.exists() else None
    if prov.get(key) != current:
        errors.append(f"STALE_PROVENANCE:{key}")

def verify_causal_eval_provenance(
    root: Path, run_dir: Path, obj: dict, contract_path: Path
) -> dict:
    errors: list[str] = []
    prov = obj.get("provenance") or {}
    targets = {
        "results_sha256": run_dir / "results.csv",
        "args_sha256": run_dir / "args.yaml",
        "last_pt_sha256": run_dir / "weights/last.pt",
        "best_pt_sha256": run_dir / "weights/best.pt",
        "manifest_sha256": run_dir / "manifest.json",
        "contract_sha256": contract_path,
        "evaluator_source_sha256": root / "scripts/eval_step4_f1_c_causality.py",
        "model_source_sha256": root / "src/multimodal/step4_f1_ir_gate_model.py",
        "gate_source_sha256": root / "src/multimodal/reliability_gate.py",
        "step3_eval_utils_sha256": root / "src/multimodal/step3_eval_utils.py",
        "trimodal_dataset_sha256": root / "src/multimodal/trimodal_dataset.py",
        "f0_model_source_sha256": root / "src/multimodal/step4_f0_model.py",
        "aux_encoder_source_sha256": root / "src/multimodal/aux_encoder.py",
        "feature_fusion_source_sha256": root / "src/multimodal/feature_fusion.py",
        "trainability_source_sha256": root / "src/multimodal/trainability.py",
        "causality_interventions_sha256": root / "src/multimodal/causality_interventions.py",
        "raw_sample_index_sha256": root / "src/multimodal/raw_sample_index.py",
    }
    for key, path in targets.items():
        _check_sha(prov, key, path, errors)
    shuffle = prov.get("shuffle_map_sha256") or {}
    for split in ("train", "val", "all17"):
        path = run_dir / f"shuffle_map_{split}.json"
        current = sha256_file(path) if path.exists() else None
        if shuffle.get(split) != current:
            errors.append(f"STALE_PROVENANCE:shuffle_map_{split}")
    return {"passed": not errors, "errors": errors}

def verify_loo_provenance(
    root: Path, project: Path, obj: dict, run_dirs: dict[str, Path],
    contract_path: Path,
) -> dict:
    errors: list[str] = []
    prov = obj.get("provenance") or {}
    checkpoint = obj.get("checkpoint")
    if checkpoint not in {"last.pt", "best.pt"}:
        errors.append("LOO_CHECKPOINT_INVALID")
        return {"passed": False, "errors": errors}
    for tag, run_dir in run_dirs.items():
        _check_sha(prov, f"{tag}_ckpt_sha256", run_dir / "weights" / checkpoint, errors)
        _check_sha(prov, f"{tag}_manifest_sha256", run_dir / "manifest.json", errors)
        _check_sha(
            prov, f"{tag}_eval_sha256",
            run_dir / "eval_step4_f1_c_causality.json", errors,
        )
    _check_sha(prov, "contract_sha256", contract_path, errors)
    _check_sha(prov, "loo_source_sha256", root / "scripts/step4_f1_c_loo.py", errors)
    _check_sha(prov, "eval_core_sha256", root / "src/multimodal/step3_eval_utils.py", errors)
    _check_sha(prov, "dataset_source_sha256", root / "src/multimodal/trimodal_dataset.py", errors)
    _check_sha(prov, "model_source_sha256", root / "src/multimodal/step4_f1_ir_gate_model.py", errors)
    _check_sha(prov, "gate_source_sha256", root / "src/multimodal/reliability_gate.py", errors)
    _check_sha(
        prov, "causality_interventions_sha256",
        root / "src/multimodal/causality_interventions.py", errors,
    )
    for tag in ("FIXED", "MAGSOFT", "ORIGSOFT"):
        _check_sha(
            prov, f"{tag}_shuffle_map_val_sha256",
            run_dirs[tag] / "shuffle_map_val.json", errors,
        )
    recorded_runs = obj.get("runs") or {}
    expected_runs = {tag: str(path) for tag, path in run_dirs.items()}
    if recorded_runs != expected_runs:
        errors.append("LOO_RUN_PATH_IDENTITY")
    return {"passed": not errors, "errors": errors}

def verify_quality_provenance(
    root: Path, obj: dict, mag_dir: Path, fixed_dir: Path, orig_dir: Path,
    contract_path: Path,
) -> dict:
    errors: list[str] = []
    prov = obj.get("provenance") or {}
    checkpoint = obj.get("checkpoint")
    if checkpoint not in {"last.pt", "best.pt"}:
        errors.append("QUALITY_CHECKPOINT_INVALID")
        return {"passed": False, "errors": errors}
    targets = {
        "checkpoint_sha256": mag_dir / "weights" / checkpoint,
        "fixed_checkpoint_sha256": fixed_dir / "weights" / checkpoint,
        "original_checkpoint_sha256": orig_dir / "weights" / checkpoint,
        "magsoft_manifest_sha256": mag_dir / "manifest.json",
        "fixed_manifest_sha256": fixed_dir / "manifest.json",
        "original_manifest_sha256": orig_dir / "manifest.json",
        "contract_sha256": contract_path,
        "script_sha256": root / "scripts/eval_step4_f1_c_quality.py",
        "interventions_source_sha256": root / "src/multimodal/step4_f1_interventions.py",
        "evaluator_core_sha256": root / "src/multimodal/step3_eval_utils.py",
        "model_source_sha256": root / "src/multimodal/step4_f1_ir_gate_model.py",
        "gate_source_sha256": root / "src/multimodal/reliability_gate.py",
        "dataset_source_sha256": root / "src/multimodal/trimodal_dataset.py",
    }
    for key, path in targets.items():
        _check_sha(prov, key, path, errors)
    return {"passed": not errors, "errors": errors}

def verify_posthoc_provenance(
    root: Path, obj: dict, mag_dir: Path, c0_dir: Path, contract_path: Path
) -> dict:
    errors: list[str] = []
    prov = obj.get("provenance") or {}
    targets = {
        "soft_last_pt_sha256": mag_dir / "weights/last.pt",
        "c0_last_pt_sha256": c0_dir / "weights/last.pt",
        "soft_manifest_sha256": mag_dir / "manifest.json",
        "c0_manifest_sha256": c0_dir / "manifest.json",
        "soft_update_gate_sha256": mag_dir / "step4_update_gate.json",
        "contract_sha256": contract_path,
        "script_sha256": root / "scripts/audit_step4_f1_c_posthoc.py",
        "model_source_sha256": root / "src/multimodal/step4_f1_ir_gate_model.py",
        "gate_source_sha256": root / "src/multimodal/reliability_gate.py",
        "dataset_source_sha256": root / "src/multimodal/trimodal_dataset.py",
        "eval_core_sha256": root / "src/multimodal/step3_eval_utils.py",
    }
    for key, path in targets.items():
        _check_sha(prov, key, path, errors)
    return {"passed": not errors, "errors": errors}

def frozen_promotion_decision(
    *,
    c0: float, fixed: float, normal: float, zero: float, shuffle: float,
    origsoft: float, loo_c0_ok: bool, loo_fixed_ok: bool, loo_orig_ok: bool,
    macro_soft: float, macro_qclean: float, worst4_soft: float,
    worst4_qclean: float, learned_minus_qclean_pos: int,
) -> dict:
    """Apply only the conditions registered in DESIGN_FREEZE.

    Macro/worst-4 comparisons against FIXED and ORIGSOFT remain diagnostics.
    They are not silently promoted to hard criteria after registration.
    """
    primary = (
        normal > c0 and normal > fixed and normal > zero
        and normal > shuffle and normal > origsoft
    )
    loo = loo_c0_ok and loo_fixed_ok and loo_orig_ok
    historical = normal > HISTORICAL_B1_SOFT_LAST
    macro_qclean_pass = macro_soft > macro_qclean
    worst4_qclean_pass = worst4_soft > worst4_qclean
    adaptive = learned_minus_qclean_pos >= 9

    full = (primary and loo and historical and macro_qclean_pass
            and worst4_qclean_pass and adaptive)
    if full:
        decision = "PROMOTE_F1C_MAGNITUDE_GATE_CONFIRM_ONE_SEED"
        next_step = "one confirmation seed; keep Depth out"
    elif (
        normal > c0 and normal > zero and normal > shuffle
        and normal > origsoft and loo_c0_ok and loo_orig_ok
    ):
        decision = "F1C_MAGNITUDE_HELPED_BUT_NOT_FULLY_PROMOTED"
        next_step = "inspect the registered failed axis; no new modules"
    elif normal > c0 and normal > zero and normal > shuffle and loo_c0_ok:
        decision = "F1C_IR_COMPLEMENTARY_MAGNITUDE_NOT_BETTER_THAN_ORIGINAL"
        next_step = "keep the simpler original gate unless magnitude wins"
    else:
        decision = "F1C_GATE_FAILED_CAUSAL_PROTOCOL"
        next_step = "stop; inspect intervention signs"
    return {
        "primary_pass": primary,
        "loo_pass": loo,
        "beats_historical_b1_soft": historical,
        "macro_vs_own_qclean_pass": macro_qclean_pass,
        "worst4_vs_own_qclean_pass": worst4_qclean_pass,
        "adaptive_pass": adaptive,
        "full_promotion_pass": full,
        "decision": decision,
        "next_step": next_step,
    }
