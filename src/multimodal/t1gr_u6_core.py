"""Pure contracts and decision logic for the T1-U6 G0--G3 server suite."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

from .t1gr_g_core import SEEDS, sha256_json
from .t1gr_secure_io import sha256_file

SCHEMA_SPEC = "t1gr-u6-design-freeze-v1"
SCHEMA_VIEW = "t1gr-u6-view-private-v1"
SCHEMA_VIEW_PUBLIC = "t1gr-u6-view-public-v1"
SCHEMA_PREFLIGHT = "t1gr-u6-server-preflight-public-v1"
SCHEMA_RUN = "t1gr-u6-server-run-public-v1"
SCHEMA_SMOKE_AUDIT = "t1gr-u6-server-smoke-audit-public-v1"
SCHEMA_RESULTS = "t1gr-u6-server-results-public-v1"
SCHEMA_EVAL_AUDIT = "t1gr-u6-server-eval-public-v1"
SCHEMA_CROSS_SEED = "t1gr-u6-server-cross-seed-public-v1"
SCHEMA_SUMMARY = "t1gr-u6-server-summary-public-v1"

ARMS = ("G0-N", "G1-P", "G2-S", "G3-D")
EPOCHS = 80
SMOKE_EPOCHS = 1
CHANNEL_SEMANTICS = ("R", "G", "B", "IR_scalar", "log_depth", "valid_mask")
ZERO_IR = "__ZERO_IR__"

ARM_POLICIES: dict[str, dict[str, Any]] = {
    "G0-N": {
        "train_ir": "ZERO_IR",
        "dev_native_ir": "ZERO_IR",
        "depth": False,
        "meaning": "RGB control in a physical six-channel model",
    },
    "G1-P": {
        "train_ir": "CORRECT_PAIRED_IR",
        "dev_native_ir": "CORRECT_PAIRED_IR",
        "depth": False,
        "meaning": "correctly paired IR",
    },
    "G2-S": {
        "train_ir": "BALANCED_FULLY_WRONG_IR",
        "dev_native_ir": "CORRECT_PAIRED_IR",
        "depth": False,
        "meaning": "wrong-IR training control, evaluated with deployable paired IR",
    },
    "G3-D": {
        "train_ir": "CORRECT_PAIRED_IR",
        "dev_native_ir": "CORRECT_PAIRED_IR",
        "depth": True,
        "meaning": "correctly paired IR plus qualified Depth and validity mask",
    },
}

SOURCE_FILES = (
    "config/t1gr_u6_design.frozen.json",
    "src/multimodal/t1gr_u6_core.py",
    "src/multimodal/t1gr_u6_model.py",
    "src/multimodal/t1gr_u6_dataset.py",
    "src/multimodal/t1gr_u6_runtime.py",
    "scripts/t1gr_u6_build_view.py",
    "scripts/t1gr_u6_server_preflight.py",
    "scripts/t1gr_u6_server_run_one.py",
    "scripts/t1gr_u6_server_run_lane.py",
    "scripts/t1gr_u6_server_parallel.py",
    "scripts/t1gr_u6_server_smoke_audit.py",
    "scripts/t1gr_u6_server_eval.py",
    "scripts/t1gr_u6_server_summarize.py",
    "tests/test_t1gr_u6_server.py",
    "docs/step4_t1gr/U6_G0_G3_DESIGN.md",
    "T1GR_U6_AUDIT.md",
    "T1GR_U6_SERVER_PACKAGE_README.md",
)


def payload_sha256(obj: Mapping[str, Any]) -> str:
    base = dict(obj)
    base.pop("payload_sha256", None)
    base.pop("request_fingerprint", None)
    return sha256_json(base)


def payload_ok(obj: Mapping[str, Any]) -> bool:
    claimed = obj.get("payload_sha256")
    return isinstance(claimed, str) and claimed == payload_sha256(obj)


def read_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"T1GR_U6_JSON_OBJECT_REQUIRED:{Path(path).name}")
    return value


def arm_policy(arm: str) -> dict[str, Any]:
    if arm not in ARM_POLICIES:
        raise ValueError(f"T1GR_U6_UNKNOWN_ARM:{arm}")
    return dict(ARM_POLICIES[arm])


def validate_spec(spec: Mapping[str, Any]) -> dict:
    if spec.get("schema") != SCHEMA_SPEC or spec.get("status") != "FROZEN_BEFORE_SERVER_PREFLIGHT":
        raise ValueError("T1GR_U6_SPEC_SCHEMA_OR_STATUS_FAIL")
    if not payload_ok(spec):
        raise ValueError("T1GR_U6_SPEC_PAYLOAD_FAIL")
    if tuple(spec.get("seeds") or ()) != tuple(SEEDS):
        raise ValueError("T1GR_U6_SEED_DRIFT")
    arms = spec.get("arms") or {}
    if tuple(arms) != ARMS:
        raise ValueError("T1GR_U6_ARM_DRIFT")
    for arm in ARMS:
        expected = ARM_POLICIES[arm]
        actual = arms.get(arm) or {}
        if (
            actual.get("train_ir") != expected["train_ir"]
            or actual.get("dev_native_ir") != expected["dev_native_ir"]
            or actual.get("depth_enabled") is not expected["depth"]
        ):
            raise ValueError(f"T1GR_U6_ARM_POLICY_DRIFT:{arm}")
    channels = spec.get("channel_contract") or {}
    if channels.get("channels") != 6 or tuple(channels.get("model_after_format") or ()) != CHANNEL_SEMANTICS:
        raise ValueError("T1GR_U6_CHANNEL_CONTRACT_DRIFT")
    model = spec.get("model") or {}
    if (
        model.get("physical_first_conv_in_channels") != 6
        or model.get("first_conv_initialization") != "[W_R,W_G,W_B,0,0,0] from the same seeded E5-v2 reference"
        or model.get("loss") != "Ultralytics E2ELoss"
        or model.get("new_attention_or_gate") is not False
        or model.get("same_seed_complete_initial_state_across_four_arms_required") is not True
    ):
        raise ValueError("T1GR_U6_MODEL_CONTRACT_DRIFT")
    legacy = spec.get("legacy_g3_adjudication") or {}
    if (
        legacy.get("old_g3_fe_merged") is not False
        or legacy.get("reason") != "NOT_REPRESENTABLE_IN_PURE_EARLY_FUSION_WITHOUT_CHANGING_THE_QUESTION"
        or legacy.get("new_g3_id") != "G3-D"
    ):
        raise ValueError("T1GR_U6_LEGACY_G3_MISLABEL_RISK")
    depth = spec.get("depth_policy") or {}
    metric = depth.get("metric_png") or {}
    unknown = depth.get("unknown_scale_jpg") or {}
    if (
        metric.get("required_storage") != "uint16 HxW"
        or metric.get("valid_mm_min") != 300
        or metric.get("valid_mm_max") != 19999
        or unknown.get("treatment") != "QUARANTINE_AS_MISSING"
        or unknown.get("log_depth") != 0
        or unknown.get("valid_mask") != 0
        or unknown.get("millimeter_reconstruction_forbidden") is not True
    ):
        raise ValueError("T1GR_U6_DEPTH_POLICY_DRIFT")
    training = spec.get("training") or {}
    expected_training = {
        "optimizer": "MuSGD",
        "epochs": 80,
        "smoke_epochs": 1,
        "batch": 4,
        "nbs": 64,
        "imgsz": 640,
        "lr0": 0.01,
        "momentum": 0.9,
        "primary_checkpoint": "last.pt",
        "best_checkpoint_role": "DIAGNOSTIC_ONLY",
    }
    if any(training.get(key) != value for key, value in expected_training.items()):
        raise ValueError("T1GR_U6_TRAINING_DRIFT")
    if training.get("run_count") != 12 or training.get("server_lane_count") != 3:
        raise ValueError("T1GR_U6_RUN_MATRIX_DRIFT")
    authority = spec.get("authority") or {}
    if (
        authority.get("server_preflight_entry_authorized") is not True
        or authority.get("smoke_training_authorized_before_preflight") is not False
        or authority.get("formal_training_authorized_before_smoke_audit") is not False
        or authority.get("legacy_primary_g_suite_mutation_authorized") is not False
        or authority.get("final_holdout_open_authorized") is not False
    ):
        raise ValueError("T1GR_U6_AUTHORITY_DRIFT")
    if launch_rows() != spec.get("launch_rows"):
        raise ValueError("T1GR_U6_LAUNCH_ORDER_DRIFT")
    return {"arms": list(ARMS), "seeds": list(SEEDS), "run_count": len(ARMS) * len(SEEDS)}


def launch_rows() -> list[dict]:
    # Four arms and three seeds cannot form a complete 4x4 Latin square.
    # This cyclic incomplete Latin rectangle rotates the first three positions;
    # every arm appears once per seed and misses exactly one lane position.
    order = {
        20260812: ("G0-N", "G1-P", "G2-S", "G3-D"),
        20260813: ("G1-P", "G2-S", "G3-D", "G0-N"),
        20260814: ("G2-S", "G3-D", "G0-N", "G1-P"),
    }
    rows: list[dict] = []
    for seed in SEEDS:
        for lane_position, arm in enumerate(order[int(seed)]):
            rows.append({
                "position": len(rows),
                "lane_position": lane_position,
                "seed": int(seed),
                "arm": arm,
            })
    expected = {(int(seed), arm) for seed in SEEDS for arm in ARMS}
    if len(rows) != 12 or {(row["seed"], row["arm"]) for row in rows} != expected:
        raise RuntimeError("T1GR_U6_LAUNCH_MATRIX_INTERNAL_FAIL")
    return rows


def run_name(mode: str, seed: int, arm: str) -> str:
    return f"T1GR_U6_SERVER_{mode.upper()}_S{int(seed)}_{arm.replace('-', '_')}"


def run_report_rel(mode: str, seed: int, arm: str) -> str:
    safe = arm.lower().replace("-", "_")
    return f"reports/step4_t1gr/t1gr_u6_server_{mode}_s{int(seed)}_{safe}_public.json"


def implementation_source_hashes(repo: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for rel in SOURCE_FILES:
        path = Path(repo) / rel
        if not path.is_file():
            raise RuntimeError(f"T1GR_U6_SOURCE_FILE_MISSING:{rel}")
        out[rel] = sha256_file(path)
    return out


def atomic_private_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw = json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    if os.name != "nt":
        os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def encode_depth_array(raw: Any, suffix: str) -> tuple[Any, Any, str]:
    """Encode only qualified metric PNG; unknown-scale JPEG stays missing."""
    import numpy as np

    value = np.asarray(raw)
    ext = str(suffix).lower()
    if ext == ".png":
        if value.dtype != np.uint16 or value.ndim != 2:
            raise ValueError("T1GR_U6_METRIC_PNG_STORAGE_FAIL")
        valid = (value >= 300) & (value <= 19999)
        encoded = np.zeros(value.shape, dtype=np.uint8)
        if bool(valid.any()):
            clipped = value[valid].astype(np.float64)
            scaled = (np.log(clipped) - math.log(300.0)) / (math.log(19999.0) - math.log(300.0))
            encoded[valid] = np.rint(np.clip(scaled, 0.0, 1.0) * 255.0).astype(np.uint8)
        mask = valid.astype(np.uint8) * np.uint8(255)
        encoded[mask == 0] = 0
        return encoded, mask, "METRIC_UINT16_PNG"
    if ext in {".jpg", ".jpeg"}:
        if value.dtype != np.uint8 or value.ndim not in {2, 3}:
            raise ValueError("T1GR_U6_UNKNOWN_JPG_STORAGE_FAIL")
        shape = value.shape[:2]
        return np.zeros(shape, dtype=np.uint8), np.zeros(shape, dtype=np.uint8), "UNKNOWN_SCALE_JPG_QUARANTINED"
    raise ValueError("T1GR_U6_DEPTH_EXTENSION_FORBIDDEN")


def raw_array_sha256(array: Any) -> str:
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _metric(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("T1GR_U6_METRIC_TYPE_FAIL")
    out = float(value)
    if not math.isfinite(out) or not 0.0 <= out <= 1.0:
        raise ValueError("T1GR_U6_METRIC_RANGE_FAIL")
    return out


def _mean(values: list[float]) -> float:
    return sum(values) / float(len(values))


def _median3(values: list[float]) -> float:
    if len(values) != 3:
        raise ValueError("T1GR_U6_THREE_SEEDS_REQUIRED")
    return sorted(values)[1]


def _contrast(indexed: Mapping[tuple[int, str], Mapping[str, Any]], left: str, right: str, field: str) -> dict[str, float]:
    return {
        str(seed): float(indexed[(int(seed), left)][field]) - float(indexed[(int(seed), right)][field])
        for seed in SEEDS
    }


def summarize_results(results: Mapping[str, Any]) -> tuple[dict, dict]:
    """Apply predeclared DEV-only operational rules without opening holdout."""
    if results.get("schema") != SCHEMA_RESULTS or results.get("final_holdout_accessed") is not False:
        raise ValueError("T1GR_U6_RESULTS_HEADER_FAIL")
    rows = results.get("rows")
    if not isinstance(rows, list) or len(rows) != 12:
        raise ValueError("T1GR_U6_RESULTS_MATRIX_FAIL")
    indexed: dict[tuple[int, str], dict[str, Any]] = {}
    candidate_lofo = {"G0-N", "G1-P", "G3-D"}
    for raw in rows:
        seed, arm = int(raw.get("seed", -1)), str(raw.get("arm", ""))
        key = (seed, arm)
        if seed not in SEEDS or arm not in ARMS or key in indexed:
            raise ValueError("T1GR_U6_RESULT_KEY_FAIL")
        lofo_raw = raw.get("lofo_native_map50_95")
        if arm in candidate_lofo:
            if not isinstance(lofo_raw, dict) or set(lofo_raw) != {f"fold_{i}" for i in range(5)}:
                raise ValueError("T1GR_U6_LOFO_KEYS_FAIL")
            lofo = {fold: _metric(value) for fold, value in lofo_raw.items()}
        elif lofo_raw is not None:
            raise ValueError("T1GR_U6_G2_LOFO_MUST_BE_NULL")
        else:
            lofo = None
        domains_raw = raw.get("depth_domain_native_map50_95")
        if arm in {"G1-P", "G3-D"}:
            if not isinstance(domains_raw, dict) or set(domains_raw) != {"metric_png", "unknown_jpg"}:
                raise ValueError("T1GR_U6_DEPTH_DOMAIN_KEYS_FAIL")
            domains = {name: _metric(value) for name, value in domains_raw.items()}
        elif domains_raw is not None:
            raise ValueError("T1GR_U6_NONDEPTH_DOMAIN_FIELD_MUST_BE_NULL")
        else:
            domains = None
        wrong = raw.get("wrong_ir_zero_depth_map50_95")
        if arm in {"G1-P", "G2-S"}:
            wrong = _metric(wrong)
        elif wrong is not None:
            raise ValueError("T1GR_U6_WRONG_IR_FIELD_DRIFT")
        indexed[key] = {
            **dict(raw),
            "native_map50_95": _metric(raw.get("native_map50_95")),
            "rgb_only_map50_95": _metric(raw.get("rgb_only_map50_95")),
            "paired_ir_zero_depth_map50_95": _metric(raw.get("paired_ir_zero_depth_map50_95")),
            "wrong_ir_zero_depth_map50_95": wrong,
            "lofo_native_map50_95": lofo,
            "depth_domain_native_map50_95": domains,
        }
    expected = {(int(seed), arm) for seed in SEEDS for arm in ARMS}
    if set(indexed) != expected:
        raise ValueError("T1GR_U6_RESULT_MATRIX_INCOMPLETE")

    ir_native = _contrast(indexed, "G1-P", "G0-N", "native_map50_95")
    ir_common_zero = _contrast(indexed, "G1-P", "G0-N", "rgb_only_map50_95")
    correct_vs_wrong_common_zero = _contrast(indexed, "G1-P", "G2-S", "rgb_only_map50_95")
    wrong_vs_rgb_native = _contrast(indexed, "G2-S", "G0-N", "native_map50_95")
    depth_native = _contrast(indexed, "G3-D", "G1-P", "native_map50_95")
    depth_dependency = {
        str(seed): indexed[(int(seed), "G3-D")]["native_map50_95"]
        - indexed[(int(seed), "G3-D")]["paired_ir_zero_depth_map50_95"]
        for seed in SEEDS
    }
    depth_training_transfer = {
        str(seed): indexed[(int(seed), "G3-D")]["paired_ir_zero_depth_map50_95"]
        - indexed[(int(seed), "G1-P")]["native_map50_95"]
        for seed in SEEDS
    }
    ir_dependency = {
        str(seed): indexed[(int(seed), "G1-P")]["native_map50_95"]
        - indexed[(int(seed), "G1-P")]["rgb_only_map50_95"]
        for seed in SEEDS
    }
    g1_pairing_sensitivity = {
        str(seed): indexed[(int(seed), "G1-P")]["native_map50_95"]
        - indexed[(int(seed), "G1-P")]["wrong_ir_zero_depth_map50_95"]
        for seed in SEEDS
    }
    g2_pairing_sensitivity = {
        str(seed): indexed[(int(seed), "G2-S")]["native_map50_95"]
        - indexed[(int(seed), "G2-S")]["wrong_ir_zero_depth_map50_95"]
        for seed in SEEDS
    }

    ir_lofo, depth_lofo = {}, {}
    for seed in SEEDS:
        g0 = indexed[(int(seed), "G0-N")]["lofo_native_map50_95"]
        g1 = indexed[(int(seed), "G1-P")]["lofo_native_map50_95"]
        g3 = indexed[(int(seed), "G3-D")]["lofo_native_map50_95"]
        ir_lofo[str(seed)] = {fold: g1[fold] - g0[fold] for fold in sorted(g1)}
        depth_lofo[str(seed)] = {fold: g3[fold] - g1[fold] for fold in sorted(g3)}

    def rule(deltas: Mapping[str, float], dependency: Mapping[str, float], lofo: Mapping[str, Mapping[str, float]]) -> dict:
        values = list(deltas.values())
        dependency_values = list(dependency.values())
        lofo_values = [value for by_seed in lofo.values() for value in by_seed.values()]
        evidence = {
            "positive_seed_count": sum(value > 0.0 for value in values),
            "mean_delta": _mean(values),
            "median_delta": _median3(values),
            "worst_seed_delta": min(values),
            "dependency_positive_seed_count": sum(value > 0.0 for value in dependency_values),
            "positive_lofo_count": sum(value > 0.0 for value in lofo_values),
            "lofo_total": len(lofo_values),
        }
        evidence["eligible"] = bool(
            evidence["positive_seed_count"] >= 2
            and evidence["mean_delta"] > 0.0
            and evidence["median_delta"] > 0.0
            and evidence["worst_seed_delta"] >= -0.01
            and evidence["dependency_positive_seed_count"] >= 2
            and evidence["positive_lofo_count"] >= 9
        )
        return evidence

    ir_rule = rule(ir_native, ir_dependency, ir_lofo)
    depth_rule = rule(depth_native, depth_dependency, depth_lofo)
    selected = "G3-D" if depth_rule["eligible"] else "G1-P" if ir_rule["eligible"] else "G0-N"
    means = {
        arm: _mean([indexed[(int(seed), arm)]["native_map50_95"] for seed in SEEDS])
        for arm in ARMS
    }
    medians = {
        arm: _median3([indexed[(int(seed), arm)]["native_map50_95"] for seed in SEEDS])
        for arm in ARMS
    }
    ranking_order = sorted(ARMS, key=lambda arm: (medians[arm], means[arm]), reverse=True)
    ranking = {f"rank_{index + 1}": arm for index, arm in enumerate(ranking_order)}
    g2_warning = bool(means["G2-S"] > means[selected])

    cross = {
        "schema": SCHEMA_CROSS_SEED,
        "native_mean_by_arm": means,
        "native_median_by_arm": medians,
        "native_rank_by_median_then_mean": ranking,
        "contrasts": {
            "g1_native_minus_g0_native": ir_native,
            "g1_rgb_only_minus_g0_rgb_only": ir_common_zero,
            "g1_rgb_only_minus_g2_rgb_only": correct_vs_wrong_common_zero,
            "g2_native_minus_g0_native": wrong_vs_rgb_native,
            "g3_native_minus_g1_native": depth_native,
            "g3_native_minus_g3_paired_ir_zero_depth": depth_dependency,
            "g3_paired_ir_zero_depth_minus_g1_native": depth_training_transfer,
            "g1_native_minus_g1_rgb_only": ir_dependency,
            "g1_paired_ir_minus_wrong_ir_at_inference": g1_pairing_sensitivity,
            "g2_paired_ir_minus_wrong_ir_at_inference": g2_pairing_sensitivity,
        },
        "lofo_contrasts": {
            "g1_minus_g0": ir_lofo,
            "g3_minus_g1": depth_lofo,
        },
        "operational_rules": {
            "ir_eligibility": ir_rule,
            "depth_eligibility": depth_rule,
        },
        "wrong_ir_control_warning": g2_warning,
        "interpretation_boundary": (
            "DEV-only operational evidence. G2-S is a diagnostic control and is never auto-selected; "
            "no statistical-significance, universal-causality, or FINAL-HOLDOUT claim."
        ),
        "final_holdout_accessed": False,
    }
    cross["payload_sha256"] = payload_sha256(cross)
    summary = {
        "schema": SCHEMA_SUMMARY,
        "competition_recommendation": selected,
        "recommendation_input": ARM_POLICIES[selected]["meaning"],
        "ir_eligible": ir_rule["eligible"],
        "depth_eligible": depth_rule["eligible"],
        "wrong_ir_control_outperforms_recommendation_mean": g2_warning,
        "manual_review_required": g2_warning,
        "native_rank_by_median_then_mean": ranking,
        "cross_seed_payload_sha256": cross["payload_sha256"],
        "dev_only": True,
        "legacy_g3_fe_included": False,
        "final_holdout_open_authorized": False,
        "production_go": False,
    }
    summary["payload_sha256"] = payload_sha256(summary)
    return cross, summary
