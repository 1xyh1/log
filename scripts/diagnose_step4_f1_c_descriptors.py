#!/usr/bin/env python3
"""F1-C-A1: magnitude-aware gate-input descriptor audit (NO training).

The reviewer's exploratory finding: aux_rel_energy (P5) tracks corruption
severity (identity 0.057 -> noise 0.247) while learned q barely moves
(0.494-0.505), Spearman(energy, sweep-optimal q) ~ -0.85.  This script audits
candidate gate-input descriptors against the q-sweep evidence:

    * current LayerNorm(GAP(A)) norm (the actual gate input)
    * raw GAP(A) L2 magnitude (no LayerNorm)
    * per-scale log-RMS(A_i)
    * RGB-relative energy ||P_i(A_i)|| / ||R_i||
    * projected residual energy ||P_i(A_i)||
    * per-scale spatial cosine (secondary)

For each descriptor x condition we compute the Spearman rank correlation with
the sweep-optimal q (the q that maximizes AP in the per-condition scan).  This
is exploratory evidence, not a promotion criterion (same val6, discrete q
grid, conditions not independent).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal import step3_eval_utils as evu  # noqa: E402
from multimodal.raw_sample_index import OUT_DEFAULT  # noqa: E402
from multimodal.run_integrity import inspect_step3_run  # noqa: E402
from multimodal.step4_f1_interventions import IRCorruptionDatasetView  # noqa: E402
from multimodal.trimodal_dataset import TriModalDataset  # noqa: E402

SCALES = {4: "P3", 6: "P4", 10: "P5"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _spearman(x: list, y: list) -> float:
    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        for pos, i in enumerate(order):
            out[i] = float(pos)
        return out

    rx, ry = ranks(x), ranks(y)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx)
                    * sum((b - my) ** 2 for b in ry))
    return float(num / den) if den > 0 else float("nan")


def _descriptors(model, img) -> dict:
    rgb, aux = model._split_input(img)
    y = [None] * (len(model.rgb_backbone) + len(model.tail))
    x = rgb
    for m in model.rgb_backbone:
        x = m(x)
        y[m.i] = x
    a3, a4, a5 = model.aux_encoder(aux)
    out = {}
    for layer, a in ((4, a3), (6, a4), (10, a5)):
        r = y[layer]
        p_a = model.fusions[str(layer)].proj(a)
        gap_a = torch.nn.functional.adaptive_avg_pool2d(a, 1).flatten(1)
        ln_gap = torch.nn.functional.layer_norm(
            gap_a, (gap_a.shape[1],))
        rms = float(a.pow(2).mean().sqrt())
        out[SCALES[layer]] = {
            "ln_gap_norm": float(ln_gap.norm()),
            "raw_gap_magnitude": float(gap_a.norm()),
            "log_rms": math.log(rms + 1e-9),
            "rgb_relative_energy": float(p_a.norm() / (r.norm() + 1e-12)),
            "proj_energy": float(p_a.norm()),
            "spatial_cos_mean": float(torch.nn.functional.cosine_similarity(
                r.flatten(2), p_a.flatten(2), dim=-1).mean()),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project", default="runs/step4_f1_b_corruption")
    p.add_argument("--soft-run", default="B1-I-soft")
    p.add_argument("--contract", default=OUT_DEFAULT)
    p.add_argument("--device", default="0")
    p.add_argument("--expected-epochs", type=int, default=80)
    p.add_argument("--checkpoint", choices=["last.pt", "best.pt"],
                   default="last.pt")
    a = p.parse_args()
    device = evu._as_device(a.device)
    run_dir = Path(a.project) / a.soft_run
    integrity = inspect_step3_run(
        run_dir, a.expected_epochs, require_weights=True,
        trace_name="step4_g8_trace.jsonl", growth_name="step4_growth.jsonl",
        eval_name="eval_step4_f1_b_causality.json")
    if not integrity.to_dict()["passed"]:
        raise RuntimeError("REFUSE_DESCRIPTOR_AUDIT_INCOHERENT_RUN")

    quality_path = run_dir / f"eval_step4_f1_b_quality_{a.checkpoint.removesuffix('.pt')}.json"
    if not quality_path.exists():
        raise RuntimeError(f"QUALITY_EVIDENCE_MISSING: {quality_path}")
    quality = json.loads(quality_path.read_text(encoding="utf-8"))

    contract = json.loads(Path(a.contract).read_text(encoding="utf-8"))
    base = TriModalDataset(contract, split="val", group="C1-I", augment=False)
    ck = torch.load(run_dir / "weights" / a.checkpoint, map_location="cpu",
                    weights_only=False)
    model = (ck.get("ema") or ck.get("model")).float().eval().to(device)
    model.requires_grad_(False)

    conditions = [("identity", 0.0), ("zero", 1.0)]
    for kind in ("blur", "contrast", "noise", "shift"):
        conditions.extend((kind, v) for v in (0.25, 0.50, 0.75, 1.0))

    per_condition = {}
    with torch.no_grad():
        for kind, severity in conditions:
            ds = IRCorruptionDatasetView(base, kind=kind, severity=severity)
            means = {scale: {key: [] for key in (
                "ln_gap_norm", "raw_gap_magnitude", "log_rms",
                "rgb_relative_energy", "proj_energy", "spatial_cos_mean")}
                for scale in SCALES.values()}
            for idx in range(len(ds)):
                sample = ds[idx]
                batch = ds.collate_fn([sample])
                batch = evu.move_step3_batch_to_device(batch, device)
                desc = _descriptors(model, batch["img"])
                for scale in SCALES.values():
                    for key in means[scale]:
                        means[scale][key].append(desc[scale][key])
            per_condition[f"{kind}:{severity:.2f}"] = {
                scale: {key: statistics.mean(vals) for key, vals in d.items()}
                for scale, d in means.items()
            }

    # sweep-optimal q per condition (AP-maximizing q in the recorded scan)
    sweep_best_q = {}
    for key, row in quality["conditions"].items():
        scan = row.get("scan_overrides") or {}
        best = max(scan.items(), key=lambda kv: kv[1]["map50_95"])
        sweep_best_q[key] = float(best[0])

    # Spearman per descriptor x scale vs sweep-optimal q
    cond_keys = [f"{k}:{s:.2f}" for k, s in conditions]
    correlations = {}
    for scale in SCALES.values():
        for desc_key in ("ln_gap_norm", "raw_gap_magnitude", "log_rms",
                         "rgb_relative_energy", "proj_energy",
                         "spatial_cos_mean"):
            xs = [per_condition[k][scale][desc_key] for k in cond_keys]
            ys = [sweep_best_q[k] for k in cond_keys]
            correlations[f"{scale}_{desc_key}"] = _spearman(xs, ys)

    report = {
        "schema": "step4-f1-c-descriptor-audit-v1",
        "checkpoint": a.checkpoint,
        "provenance": {
            "checkpoint_sha256": _sha(run_dir / "weights" / a.checkpoint),
            "quality_sha256": _sha(quality_path),
            "contract_sha256": _sha(Path(a.contract)),
            "script_sha256": _sha(Path(__file__)),
            "model_source_sha256": _sha(
                ROOT / "src" / "multimodal" / "step4_f1_ir_gate_model.py"),
            "gate_source_sha256": _sha(
                ROOT / "src" / "multimodal" / "reliability_gate.py"),
            "dataset_source_sha256": _sha(
                ROOT / "src" / "multimodal" / "trimodal_dataset.py"),
            "corruption_view_sha256": _sha(
                ROOT / "src" / "multimodal" / "step4_f1_interventions.py"),
        },
        "sweep_best_q": sweep_best_q,
        "per_condition_descriptors": per_condition,
        "spearman_vs_sweep_optimal_q": correlations,
        "caveats": ("exploratory evidence only: same val6, discrete q grid, "
                    "conditions not independent; NOT a promotion criterion"),
    }
    out_dir = ROOT / "reports" / "step4_f1_c_agreement"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"descriptor_audit_{a.checkpoint.removesuffix('.pt')}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(json.dumps({k: round(v, 4) for k, v in correlations.items()},
                     indent=2, ensure_ascii=False))
    print("->", out)


if __name__ == "__main__":
    main()
