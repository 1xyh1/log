#!/usr/bin/env python3
"""Step 3-A gates G1/G2/G3: data contract, group split, float input semantics.

Runs the contract builder, then samples real batches through TriModalDataset for
each group and validates the FINAL model input (B,6,640,640) semantics.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal import modality_preprocess as mp  # noqa: E402
from multimodal.raw_sample_index import build_contract  # noqa: E402
from multimodal.trimodal_dataset import GROUPS, TriModalDataset  # noqa: E402


def check_g3(group: str, contract: dict) -> dict:
    ds = TriModalDataset(contract, split="all17", group=group, augment=False)
    samples = [ds[i] for i in range(len(ds))]
    img = torch.stack([torch.as_tensor(s["img"]) for s in samples])  # (17, 6, 640, 640)
    out = {"group": group, "n_images": len(samples), "shape": list(img.shape),
           "finite": bool(torch.isfinite(img).all())}
    rgb, i, d, m = img[:, 0:3], img[:, 3], img[:, 4], img[:, 5]
    out["rgb_range"] = [float(rgb.min()), float(rgb.max())]
    out["rgb_pad_value_present"] = bool((rgb - 114 / 255).abs().min() < 1e-4)
    out["i_range"] = [float(i.min()), float(i.max())]
    out["d_range"] = [float(d.min()), float(d.max())]
    out["m_unique"] = sorted(float(v) for v in torch.unique(m))[:8]
    out["m_binary"] = bool(set(torch.unique(m).tolist()) <= {0.0, 1.0})
    out["m0_implies_d0"] = bool((d[m == 0].abs().max() < 1e-6))
    ok = (out["finite"] and out["rgb_range"][0] >= 0 and out["rgb_range"][1] <= 1
          and out["i_range"][0] >= 0 and out["i_range"][1] <= 1
          and out["d_range"][0] >= 0 and out["d_range"][1] <= 1
          and out["m_binary"] and out["m0_implies_d0"])
    # aux pad-band strictly zero (letterbox geometry from each sample's ori_shape;
    # all 17 share 1080x1920 - verified per sample)
    for idx, s in enumerate(samples):
        h, w = s["ori_shape"]
        r, left, top, new_unpad = mp.letterbox_geometry(h, w, 640)
        band = np.ones((640, 640), dtype=bool)
        band[top:top + new_unpad[0], left:left + new_unpad[1]] = False
        for ch, name in ((3, "I"), (4, "D"), (5, "M")):
            v = float(img[idx, ch][band].abs().max())
            out.setdefault(f"aux_pad_{name}_max_abs", 0.0)
            out[f"aux_pad_{name}_max_abs"] = max(out[f"aux_pad_{name}_max_abs"], v)
            ok = ok and v == 0.0
    if group == "C0-N":
        aux_zero = bool(i.abs().max() == 0 and d.abs().max() == 0 and m.abs().max() == 0)
        out["aux_strictly_zero"] = aux_zero
        ok = ok and aux_zero
    if group == "C1-I":
        out["i_nonzero"] = bool(i.abs().max() > 0)
        out["d_m_zero"] = bool(d.abs().max() == 0 and m.abs().max() == 0)
        ok = ok and out["i_nonzero"] and out["d_m_zero"]
    if group == "C2-D":
        out["i_zero"] = bool(i.abs().max() == 0)
        out["d_nonzero"] = bool(d.abs().max() > 0)
        out["m_nonzero"] = bool(m.abs().max() > 0)
        ok = ok and out["i_zero"] and out["d_nonzero"] and out["m_nonzero"]
    out["passed"] = bool(ok)
    return out


def main():
    contract = build_contract()
    g1 = {
        "raw": contract["raw_multimodal_groups"],
        "usable": contract["usable_metric_multimodal_groups"],
        "excluded": contract["excluded_groups"],
        "classes_present": contract["classes_present"],
        "classes_absent": contract["classes_absent"],
        "class_support": contract["class_support"],
        "format_gate_passed": contract["format_gate_passed"],
        "passed": (contract["raw_multimodal_groups"] == 18
                   and contract["usable_metric_multimodal_groups"] == 17
                   and contract["classes_absent"] == [1, 7, 10]
                   and contract["format_gate_passed"]),
    }
    g2 = {
        "grouping_rule": contract["grouping_rule"],
        "train_groups": contract["train_groups"],
        "val_groups": contract["val_groups"],
        "intersection": contract["group_intersection"],
        "split_relations": contract["split_relations"],
        "passed": contract["split_gate_passed"],
    }
    g3 = {g: check_g3(g, contract) for g in GROUPS}
    report = {"G1_data_contract": g1, "G2_split": g2, "G3_float_input_semantics": g3,
              "all_passed": bool(g1["passed"] and g2["passed"]
                                 and all(v["passed"] for v in g3.values()))}
    out = Path("reports") / "step3_gates_g1_g2_g3.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "G3_float_input_semantics"},
                     indent=2, ensure_ascii=False))
    for g, v in g3.items():
        print(f"G3 {g}: passed={v['passed']} shape={v['shape']} m_unique={v['m_unique']} "
              f"rgb_pad114={v['rgb_pad_value_present']}")
    print("ALL PASSED:", report["all_passed"], "->", out)
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
