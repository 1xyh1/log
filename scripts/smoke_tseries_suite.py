#!/usr/bin/env python3
"""Run the three matched one-epoch T-series pretraining smokes."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal.raw_sample_index import OUT_DEFAULT  # noqa: E402
from multimodal.tseries_core import TREATMENTS, optimizer_snapshots_equivalent, sha256_file  # noqa: E402

SCHEMA = "step4-tseries-pretraining-smoke-v1"

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", default="runs/step4_tseries_smoke")
    p.add_argument("--contract", default=OUT_DEFAULT)
    p.add_argument("--data", default="D:/pycharm/Python Develop/YOLO_1/v031_step1_rgb_sample/dataset.yaml")
    p.add_argument("--base-checkpoint", default="E:/odin/yolo26s.pt")
    p.add_argument("--device", default="0")
    p.add_argument("--out", default="reports/step4_tseries/pretraining_smoke.json")
    p.add_argument("--static-audit", default="reports/step4_tseries/pretraining_static_audit.json")
    p.add_argument("--python", default=sys.executable)
    a = p.parse_args()

    static_audit_path = ROOT / a.static_audit
    if not static_audit_path.exists():
        raise RuntimeError(f"T_SERIES_STATIC_AUDIT_MISSING:{static_audit_path}")
    static_audit = load(static_audit_path)
    if (
        static_audit.get("schema") != "step4-tseries-pretraining-audit-v1"
        or static_audit.get("phase") != "static"
        or static_audit.get("all_passed") is not True
    ):
        raise RuntimeError("T_SERIES_STATIC_AUDIT_NOT_PASSING")
    source_map = {
        "design_sha256": "docs/step4_tseries/TRAINING_DESIGN_FREEZE.md",
        "model_sha256": "src/multimodal/tseries_p5_model.py",
        "core_sha256": "src/multimodal/tseries_core.py",
        "runtime_sha256": "src/multimodal/tseries_runtime.py",
        "runner_sha256": "scripts/run_tseries.py",
        "suite_sha256": "scripts/smoke_tseries_suite.py",
        "audit_sha256": "scripts/audit_tseries.py",
        "posttrain_eval_sha256": "scripts/eval_tseries_posttrain.py",
        "paired_eval_sha256": "scripts/eval_tseries_paired.py",
        "formal_suite_sha256": "scripts/run_tseries_formal_suite.py",
        "summary_sha256": "scripts/summarize_tseries.py",
        "implementation_adjudication_sha256": "docs/step4_tseries/IMPLEMENTATION_ADJUDICATION.md",
        "tests_sha256": "tests/test_tseries.py",
        "readme_sha256": "T_SERIES_README.md",
    }
    stale = {
        key: {"recorded": static_audit.get("source_hashes", {}).get(key),
              "current": sha256_file(ROOT / rel)}
        for key, rel in source_map.items()
        if static_audit.get("source_hashes", {}).get(key) != sha256_file(ROOT / rel)
    }
    if stale:
        raise RuntimeError(f"T_SERIES_STATIC_AUDIT_STALE:{stale}")

    project = Path(a.project).resolve()
    project.mkdir(parents=True, exist_ok=True)
    manifests = {}
    runs = {}

    for treatment in ("T0-N", "T1-F", "T2-A"):
        before = set(p.name for p in project.iterdir() if p.is_dir())
        cmd = [
            a.python, str(ROOT / "scripts/run_tseries.py"),
            "--treatment", treatment,
            "--run-kind", "smoke",
            "--epochs", "1",
            "--batch", "4",
            "--seed", "20260812",
            "--project", str(project),
            "--contract", str(a.contract),
            "--data", str(a.data),
            "--base-checkpoint", str(a.base_checkpoint),
            "--device", str(a.device),
        ]
        print("RUN", " ".join(cmd))
        subprocess.run(cmd, check=True)
        after_dirs = [p for p in project.iterdir() if p.is_dir() and p.name not in before]
        if len(after_dirs) != 1:
            raise RuntimeError(f"T_SERIES_SMOKE_RUN_DISCOVERY_FAIL:{treatment}:{after_dirs}")
        run = after_dirs[0]
        runs[treatment] = run
        manifests[treatment] = load(run / "manifest.json")

    ids = {t: manifests[t]["initial_identity"] for t in manifests}
    complete_shas = {t: ids[t]["complete_model_state_sha256"] for t in ids}
    keys = {t: ids[t]["state_dict_keys"] for t in ids}
    req = {t: ids[t]["requires_grad"] for t in ids}
    pred_sha = {t: manifests[t]["prediction_probe"]["raw_prediction_sha256"] for t in manifests}
    protocol = {t: manifests[t]["protocol_hash"] for t in manifests}
    bn = {
        t: (
            manifests[t]["aux_bn_stats_policy"],
            manifests[t]["rgb_bn_stats_policy"],
            manifests[t]["tail_bn_stats_policy"],
        )
        for t in manifests
    }

    optimizer_equal = (
        optimizer_snapshots_equivalent(manifests["T0-N"]["optimizer"], manifests["T1-F"]["optimizer"])
        and optimizer_snapshots_equivalent(manifests["T0-N"]["optimizer"], manifests["T2-A"]["optimizer"])
    )
    traces = {t: read_jsonl(runs[t] / "tseries_data_order.jsonl") for t in runs}
    order_equal = len({traces[t][0]["sample_order_sha256"] for t in traces}) == 1
    flip_equal = len({traces[t][0]["flip_schedule_sha256"] for t in traces}) == 1

    t2 = manifests["T2-A"]
    t2_mech = read_jsonl(runs["T2-A"] / "tseries_mechanism.jsonl")
    t2_centered_ok = bool(
        t2_mech
        and float(t2_mech[-1]["post_center_channel_mean_abs_max"]) <= 1e-6
        and int(t2_mech[-1].get("t2_bias_zero_guard_calls", 0)) > 0
    )
    t2_bias_safe = bool(
        t2["t2_bias_unchanged"]
        and float(t2["optimizer"]["proj_bias_weight_decay"]) == 0.0
        and t2_centered_ok
    )

    gates = {
        "G2_identical_model_class": len({m["model_class"] for m in manifests.values()}) == 1,
        "G3_no_reliability_gate": all(m["prediction_probe"]["no_reliability_gate_attribute"] for m in manifests.values()),
        "G4_p5_only_topology": all(
            m["prediction_probe"]["forward_trace"]["p3_direct_injection_count"] == 0
            and m["prediction_probe"]["forward_trace"]["p4_direct_injection_count"] == 0
            and m["prediction_probe"]["forward_trace"]["p5_direct_injection_count"] == 1
            for m in manifests.values()
        ),
        "G5_neck_handoff": all(
            m["prediction_probe"]["forward_trace"]["y10_is_fused5_object"]
            and m["prediction_probe"]["forward_trace"]["x_is_y10_object"]
            for m in manifests.values()
        ),
        "G6_matched_initial_state": len(set(complete_shas.values())) == 1 and len({json.dumps(v) for v in keys.values()}) == 1,
        "G7_epoch0_prediction_equivalence": len(set(pred_sha.values())) == 1,
        "G8_zero_init_projection": all(m["prediction_probe"]["zero_init"] for m in manifests.values()),
        "G9_trainability_map_matched": len({json.dumps(v, sort_keys=True) for v in req.values()}) == 1,
        "G10_optimizer_groups_matched": optimizer_equal,
        "G11_t0_null_loss_graph": manifests["T0-N"]["gradient_probe"]["t0_aux_proj_disconnected"],
        "G12_t0_no_silent_optimizer_update": (
            manifests["T0-N"]["t0_aux_params_unchanged"]
            and manifests["T0-N"]["t0_p5_params_unchanged"]
        ),
        "G13_t2_bias_cancellation": t2_centered_ok,
        "G14_t2_bias_optimizer_safety": t2_bias_safe,
        "G15_bn_policy_matched": len(set(bn.values())) == 1,
        "G17_rng_data_order_closure": order_equal and flip_equal,
        "G18_protocol_equality": len(set(protocol.values())) == 1,
    }
    # G1 and G16 are repo/upstream checkpoint gates rechecked by audit_tseries.py.
    report = {
        "schema": SCHEMA,
        "runs": {t: str(r.relative_to(ROOT) if ROOT in r.parents else r) for t, r in runs.items()},
        "manifests": manifests,
        "gates": gates,
        "all_dynamic_gates_passed": all(gates.values()),
        "static_audit": {
            "path": str(static_audit_path.relative_to(ROOT)),
            "sha256": sha256_file(static_audit_path),
        },
        "source_hashes": {
            key: sha256_file(ROOT / rel) for key, rel in source_map.items()
        },
    }
    if not report["all_dynamic_gates_passed"]:
        raise RuntimeError(f"T_SERIES_PRETRAIN_SMOKE_FAIL:{gates}")
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"all_dynamic_gates_passed": True, "out": str(out)}, indent=2))

if __name__ == "__main__":
    main()
