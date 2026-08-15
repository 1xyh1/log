"""Step 4-F0 closeout shared gates — torch-free on purpose.

Single source of truth consumed by scripts/step4_loo.py (producer),
scripts/summarize_step4.py (consumer) and tests/test_step4_closeout.py
(adversarial tests).  Keeping this module torch-free is a HARD constraint:
the lightweight test fixtures depend on it.

Freezing discipline:
- DEPENDENCY_SOURCES is the static import transitive closure of the LOO
  execution path.  EXPANDING OR NARROWING THIS LIST invalidates every existing
  LOO JSON (all become RECORDED_SHA_MISSING) — that is BY DESIGN: changing the
  list means the execution semantics changed, which forces a LOO rerun.
- validate_loo_payload compares with EXACT `==`.  Deliberate: JSON roundtrip
  is bit-exact for doubles, so an epsilon would only whitewash micro-tampering
  and has no principled size.  Do NOT replace with math.isclose.
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from multimodal.causality_interventions import assert_valid_shuffle_map
from multimodal.run_integrity import sha256_file

# LOO execution-path local-module import closure.  Keys feed provenance names
# `dep_<key>_sha256`; values are repo-relative paths.
DEPENDENCY_SOURCES = {
    "step4_closeout": "src/multimodal/step4_closeout.py",
    "step3_eval_utils": "src/multimodal/step3_eval_utils.py",
    "trimodal_dataset": "src/multimodal/trimodal_dataset.py",
    "step4_f0_model": "src/multimodal/step4_f0_model.py",
    "aux_encoder": "src/multimodal/aux_encoder.py",
    "feature_fusion": "src/multimodal/feature_fusion.py",
    "trainability": "src/multimodal/trainability.py",
    "causality_interventions": "src/multimodal/causality_interventions.py",
    "run_integrity": "src/multimodal/run_integrity.py",
    "raw_sample_index": "src/multimodal/raw_sample_index.py",
    "modality_preprocess": "src/multimodal/modality_preprocess.py",
    "early_fusion_yolo26": "src/multimodal/early_fusion_yolo26.py",
}

# The six causal delta series: key -> (tag, variant, base_tag, base_variant).
DELTA_SPECS = {
    "IR_minus_C0": ("IR", "NORMAL", "C0", "NORMAL"),
    "D_minus_C0": ("D", "NORMAL", "C0", "NORMAL"),
    "IR_N_minus_Z": ("IR", "NORMAL", "IR", "ZERO-AUX"),
    "IR_N_minus_S": ("IR", "NORMAL", "IR", "SHUFFLE"),
    "D_N_minus_Z": ("D", "NORMAL", "D", "ZERO-AUX"),
    "D_N_minus_S": ("D", "NORMAL", "D", "SHUFFLE"),
}

LOO_SCHEMA = "step4-loo-v2"


def resolve_dep_targets(repo_root: Path | None = None) -> dict[str, Path]:
    """Absolute-path resolution of DEPENDENCY_SOURCES (repo_root injectable
    for tests; defaults to the repository that contains this module)."""
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    return {name: root / rel for name, rel in DEPENDENCY_SOURCES.items()}


def _finite_in_range(values: list) -> bool:
    """mAP50-95 lives in [0, 1]; inf == inf would defeat exact-equality
    comparison, so finiteness is checked explicitly."""
    for v in values:
        if not isinstance(v, (int, float)) or not math.isfinite(v) \
                or not (0.0 <= v <= 1.0):
            return False
    return True


def compute_deltas(folds: dict, val_ids: list[str]) -> dict:
    """Single implementation of the six LOO delta series.  The producer
    (step4_loo.py) writes JSON with THIS function; validate_loo_payload
    recomputes with THIS function and demands exact equality."""
    out = {}
    for key, (tag, variant, base_tag, base_variant) in DELTA_SPECS.items():
        full = round(folds["full"][tag][variant]
                     - folds["full"][base_tag][base_variant], 6)
        per_fold = {f: round(folds[f][tag][variant]
                             - folds[f][base_tag][base_variant], 6)
                    for f in val_ids}
        vals = list(per_fold.values())
        out[key] = {
            "full": full,
            "per_fold": per_fold,
            "positive_folds": sum(1 for x in vals if x > 0),
            "n_folds": len(vals),
            "median": round(statistics.median(vals), 6) if vals else None,
            "min": round(min(vals), 6),
            "max": round(max(vals), 6),
        }
    return out


def validate_loo_payload(loo: dict) -> dict:
    """Recompute every delta from the raw folds and demand exact equality with
    the stored deltas block.  Also enforces fold-key identity, value range,
    and the C0 copy_of_normal invariant."""
    errors: list[str] = []
    val_ids = list(loo.get("val_ids") or [])
    folds = loo.get("folds") or {}

    if not val_ids:
        errors.append("VAL_IDS_MISSING")
    if set(folds.keys()) != (set(val_ids) | {"full"}):
        errors.append("FOLD_KEYS_DO_NOT_MATCH_VAL_IDS")

    for fk, group in folds.items():
        if not isinstance(group, dict):
            errors.append(f"FOLD_NOT_DICT:{fk}")
            continue
        for tag, variants in group.items():
            if tag not in ("C0", "IR", "D") or not isinstance(variants, dict):
                errors.append(f"FOLD_STRUCTURE_BAD:{fk}/{tag}")
                continue
            for vk in ("NORMAL", "ZERO-AUX", "SHUFFLE"):
                if vk in variants and not _finite_in_range([variants[vk]]):
                    errors.append(f"VALUE_OUT_OF_RANGE:{fk}/{tag}/{vk}")
        # C0 copy_of_normal invariant: ZERO-AUX/SHUFFLE equal NORMAL by
        # construction; IR/D must be marked as independently evaluated.
        if "C0" in group:
            g0 = group["C0"]
            if g0.get("copy_of_normal") is not True:
                errors.append(f"C0_COPY_FLAG_INVALID:{fk}")
            if g0.get("ZERO-AUX") != g0.get("NORMAL") \
                    or g0.get("SHUFFLE") != g0.get("NORMAL"):
                errors.append(f"C0_VARIANTS_NOT_EQUAL:{fk}")
        for tag in ("IR", "D"):
            if tag in group and group[tag].get("copy_of_normal") is not False:
                errors.append(f"{tag}_COPY_FLAG_INVALID:{fk}")

    recomputed = None
    try:
        recomputed = compute_deltas(folds, val_ids)
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"DELTA_RECOMPUTE_FAILED:{type(exc).__name__}:{exc}")
    if recomputed is not None:
        stored = loo.get("deltas")
        if not isinstance(stored, dict):
            errors.append("DELTAS_MISSING")
        elif set(stored.keys()) != set(recomputed.keys()):
            errors.append("DELTAS_KEY_SET_MISMATCH")
        else:
            for key in recomputed:
                if stored[key] != recomputed[key]:
                    errors.append(f"DELTA_MISMATCH:{key}")
    return {"errors": errors, "passed": not errors, "recomputed": recomputed}


def g8_check(run_dirs: dict[str, Path], expected_epochs: int) -> dict:
    """Actual-yield G8 closeout gate: per-row expected==actual order/flip,
    flag all-true, strict positional epoch continuity, cross-group agreement,
    byte-identical traces.  Returns a report dict; never raises."""
    traces: dict[str, list] = {}
    errors: list[str] = []
    for g, rd in run_dirs.items():
        fp = rd / "step4_g8_trace.jsonl"
        if not fp.exists():
            errors.append(f"TRACE_MISSING:{g}")
            traces[g] = []
            continue
        traces[g] = [json.loads(x)
                     for x in fp.read_text(encoding="utf-8").splitlines()
                     if x.strip()]

    row_counts = {g: len(t) for g, t in traces.items()}
    for g, n in row_counts.items():
        if n != expected_epochs:
            errors.append(f"ROW_COUNT_MISMATCH:{g}:{n}!={expected_epochs}")
    n = min(row_counts.values()) if row_counts else 0

    epoch_bad, exp_act_bad, flag_bad, mismatches = [], [], [], []
    for e in range(n):
        for g in traces:
            row = traces[g][e]
            if row.get("epoch") != e:
                epoch_bad.append(f"{g}:e{e}")
            if row.get("expected_order_sha256") != row.get("actual_order_sha256") \
                    or row.get("expected_flip_sha256") != row.get("actual_flip_sha256"):
                exp_act_bad.append(f"{g}:e{e}")
            if row.get("actual_matches_expected") is not True:
                flag_bad.append(f"{g}:e{e}")
        orders = {traces[g][e].get("actual_order_sha256") for g in traces}
        flips = {traces[g][e].get("actual_flip_sha256") for g in traces}
        if None in orders or len(orders) != 1 or None in flips or len(flips) != 1:
            mismatches.append(e)

    all_actual = all(
        "actual_order_sha256" in r and "actual_flip_sha256" in r
        for t in traces.values() for r in t)
    file_shas = {}
    for g, rd in run_dirs.items():
        fp = rd / "step4_g8_trace.jsonl"
        if fp.exists():
            file_shas[g] = sha256_file(fp)
    byte_identical = (len(file_shas) == len(run_dirs)
                      and len(set(file_shas.values())) == 1)
    return {
        "epochs_compared": n,
        "row_counts": row_counts,
        "actual_yield_fields_present_all_epochs": all_actual,
        "epoch_position_continuous": not epoch_bad,
        "epoch_position_errors": epoch_bad,
        "expected_equals_actual_all_rows": not exp_act_bad,
        "expected_actual_mismatch_rows": exp_act_bad,
        "actual_matches_expected_flag_all_true": not flag_bad,
        "flag_false_rows": flag_bad,
        "order_and_flip_hashes_match": not mismatches,
        "mismatched_epochs": mismatches,
        "trace_files_byte_identical": byte_identical,
        "trace_file_sha256": file_shas,
        "errors": errors,
        "passed": bool(not errors and all_actual and not epoch_bad
                       and not exp_act_bad and not flag_bad and not mismatches
                       and byte_identical),
    }


def load_validated_shuffle_maps(run_dirs: dict[str, Path],
                                val_ids: list[str]) -> dict[str, dict]:
    """Each group uses its OWN shuffle_map_val.json; both re-validated as
    bijective no-self derangements and asserted equal (they derive from the
    same deterministic derangement of val6)."""
    maps: dict[str, dict] = {}
    for tag in ("IR", "D"):
        fp = run_dirs[tag] / "shuffle_map_val.json"
        m = json.loads(fp.read_text(encoding="utf-8"))
        if not assert_valid_shuffle_map(m, val_ids):
            raise RuntimeError(f"INVALID_SHUFFLE_MAP:{tag}")
        maps[tag] = m
    if maps["IR"] != maps["D"]:
        raise RuntimeError("IR_AND_D_SHUFFLE_MAPS_DIFFER")
    return maps


def loo_provenance_check(loo: dict, run_dirs: dict[str, Path],
                         contract_path: Path, loo_script_path: Path,
                         evals: dict[str, dict],
                         dep_targets: dict[str, Path] | None = None) -> dict:
    """Re-verify LOO file identity before consuming it (avoid a stale LOO).

    Checks: schema/checkpoint declaration, five recorded provenance SHA
    (3x last.pt / contract / LOO script source), the full dependency SHA set,
    IR/D shuffle map SHA, groups-path cross-check, and cross-consistency with
    the eval JSON provenance for the same checkpoints.
    """
    checks = {
        "schema": {"recorded": loo.get("schema"), "expected": LOO_SCHEMA,
                   "match": loo.get("schema") == LOO_SCHEMA},
        "checkpoint": {"recorded": loo.get("checkpoint"), "expected": "last.pt",
                       "match": loo.get("checkpoint") == "last.pt"},
    }
    targets = {
        "C0_last_pt_sha256": run_dirs["C0"] / "weights" / "last.pt",
        "IR_last_pt_sha256": run_dirs["IR"] / "weights" / "last.pt",
        "D_last_pt_sha256": run_dirs["D"] / "weights" / "last.pt",
        "contract_sha256": contract_path,
        "loo_source_sha256": loo_script_path,
    }
    deps = dep_targets if dep_targets is not None else resolve_dep_targets()
    for name, fp in deps.items():
        targets[f"dep_{name}_sha256"] = fp
    targets["ir_shuffle_map_val_sha256"] = run_dirs["IR"] / "shuffle_map_val.json"
    targets["d_shuffle_map_val_sha256"] = run_dirs["D"] / "shuffle_map_val.json"

    prov = loo.get("provenance") or {}
    for key, fp in targets.items():
        if key not in prov:
            checks[key] = {"recorded": None, "current": None, "match": False,
                           "error": "RECORDED_SHA_MISSING"}
            continue
        if not fp.exists():
            checks[key] = {"recorded": prov[key], "current": None,
                           "match": False, "error": "TARGET_FILE_MISSING"}
            continue
        cur = sha256_file(fp)
        checks[key] = {"recorded": prov[key], "current": cur,
                       "match": prov[key] == cur}

    # groups paths must resolve to the same physical run dirs actually used
    loo_groups = loo.get("groups")
    norm_loo = ({tag: str(Path(v).resolve()) for tag, v in loo_groups.items()}
                if isinstance(loo_groups, dict) else {})
    expected_groups = {tag: str(Path(rd).resolve()) for tag, rd in run_dirs.items()}
    checks["groups_paths"] = {"recorded": norm_loo, "expected": expected_groups,
                              "match": norm_loo == expected_groups}

    # cross-consistency: LOO checkpoint SHAs must equal the eval JSON provenance
    for tag, key in (("C0", "C0_last_pt_sha256"), ("IR", "IR_last_pt_sha256"),
                     ("D", "D_last_pt_sha256")):
        eval_sha = evals[tag].get("provenance", {}).get("last_pt_sha256")
        checks[f"cross_eval_{key}"] = {
            "recorded": prov.get(key), "eval_provenance": eval_sha,
            "match": bool(prov.get(key) and prov.get(key) == eval_sha)}
    return checks
