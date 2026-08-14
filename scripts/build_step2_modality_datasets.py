#!/usr/bin/env python3
"""Step 2 dataset builder: IR-only / Depth-only for the 17-image probe (final).

Standalone (no src import, plain cv2/numpy/torch via ultralytics loader gate).
Split = the ACTUAL Step-1 split on disk (v031_step1_rgb_sample/images/{train,val});
the repo splits/*.txt files are deliberately NOT used (found to differ 2026-08-13).

Encodings (frozen by advisor protocol, reviewer decision 2026-08-13):
    B1-A/B1-B IR : 3ch infrared PNG -> per-pixel MEDIAN -> single scalar thermal I
                   -> repeat to 3ch (B1-A) / grayscale loader (B1-B, channels:1)
                   median (not mean) auto-rejects isolated single-channel anomalies
    B2-A Depth   : uint16 mm -> fixed physical log map (300..19999) ->
                   model tensor [logD, logD, valid_mask]
                   disk BGR write [mask, logD, logD] (Ultralytics flips BGR->RGB)
    B2-B Depth   : model tensor [logD, valid_mask, 0]
                   disk BGR write [0, mask, logD]
Gates:
    IR robust redundancy gate (two-level; legacy global max_abs_diff gate was
        rejected by reviewer as over-sensitive to isolated pixels):
        A. per-image mean pairwise abs diff <= 6
        B. frac pixels with max pairwise diff > 50 <= 0.001
        max_abs_diff / p99 / p99.9 kept as diagnostic fields only.
        Verdict for 17 samples: PASS_WITH_LOCAL_CHANNEL_ARTIFACT
        (000005_010 has a 0.034% localized single-channel saturation area;
         exact sensor/ISP mechanism unknown; image kept, no patching).
    loader tensor gate: real YOLODataset tensor, finite / 0<=mask<=1 / channel order
        (mask may be SOFT after INTER_LINEAR resize + letterbox pad=114; raw PNG mask
         is strictly binary - two-level definition per protocol)
Labels are copied from the Step-1 dataset (identical to what B0-* trained on);
zip labels byte-compared and reported.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

D_MIN, D_MAX = 300.0, 19999.0  # mm, fixed physical scale
IR_MEAN_PAIRWISE_GATE = 6.0
IR_FRAC_GT50_GATE = 0.001


def convert_ir(src: Path, dst: Path) -> dict:
    """Per-pixel median collapse of the 3 (near-redundant) IR channels."""
    im = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if im is None or im.ndim != 3:
        raise RuntimeError(f"IR read failed: {src}")
    im16 = im.astype(np.int16)
    pair_diffs = np.abs(np.stack([im16[..., i] - im16[..., j]
                                  for i, j in ((0, 1), (0, 2), (1, 2))]))
    max_pairwise = pair_diffs.max(axis=0)  # per-pixel worst of the 3 pairs
    gray = np.rint(np.median(im.astype(np.float32), axis=2)).clip(0, 255).astype(np.uint8)
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if not cv2.imwrite(str(dst), out):
        raise OSError(f"failed to write {dst}")
    hi = max_pairwise > 50
    if hi.any():
        dom = pair_diffs[:, hi].argmax(axis=0)  # which pair dominates each high-diff pixel
        component_frac = float(np.bincount(dom, minlength=3).max() / hi.sum())
    else:
        component_frac = 0.0
    return {"shape": im.shape[:2],
            "max_abs_diff": int(max_pairwise.max()),
            "p99_abs_diff": float(np.percentile(max_pairwise, 99)),
            "p99_9_abs_diff": float(np.percentile(max_pairwise, 99.9)),
            "mean_pairwise_abs_diff": float(pair_diffs.mean()),
            "frac_pixels_pairwise_gt_50": float(hi.mean()),
            "largest_high_diff_component_fraction": component_frac}


def encode_depth(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """uint16 mm -> (logd_u8, mask_u8). Fixed physical log map, invalid -> logD=0 & mask=0."""
    valid = (d >= D_MIN) & (d <= D_MAX)
    vals = np.clip(d.astype(np.float32), D_MIN, D_MAX)
    logd = np.zeros(d.shape, dtype=np.float32)
    logd[valid] = (np.log(vals[valid]) - math.log(D_MIN)) / (math.log(D_MAX) - math.log(D_MIN))
    return np.rint(logd * 255.0).astype(np.uint8), (valid.astype(np.uint8) * 255)


def depth_quant_audit(d: np.ndarray, logd_u8: np.ndarray, valid: np.ndarray) -> dict:
    """P1 diagnostic metadata: 8-bit round-trip error back in mm."""
    z_true = d[valid].astype(np.float32)
    q = np.exp(logd_u8[valid].astype(np.float32) / 255.0 *
               (math.log(D_MAX) - math.log(D_MIN)) + math.log(D_MIN))
    err = np.abs(q - z_true)
    rel = err / np.maximum(z_true, 1.0)
    return {
        "levels_used": int(len(np.unique(logd_u8[valid]))),
        "mae_mm": float(err.mean()),
        "p95_abs_error_mm": float(np.percentile(err, 95)),
        "max_abs_error_mm": float(err.max()),
        "relative_mae": float(rel.mean()),
        "valid_depth_fraction": float(valid.mean()),
    }


def loader_gate(images_dir: Path, names: dict, channels: int, tag: str, mode: str = "identical3") -> dict:
    """Real Ultralytics YOLODataset tensor gate (augment=False: letterbox resize only).

    mode: identical3 (all channels equal), c01_equal (C0==C1; C2 = soft mask),
          c2_zero (C2 content 0 + pad 114; C0/C1 free), none (single channel).
    """
    from ultralytics.data import YOLODataset
    from ultralytics.utils import DEFAULT_CFG
    ds = YOLODataset(img_path=str(images_dir), imgsz=640, cache=False, augment=False,
                     hyp=DEFAULT_CFG, batch_size=1,
                     data={"names": names, "channels": channels}, task="detect")
    item = ds[0]
    im = item["img"]  # batch_size=1: no batch dim -> (C, 640, 640) uint8 pre-normalization
    if im.ndim == 2:
        im = im[None]
    out = {"tag": tag, "channels": channels, "mode": mode, "shape": list(im.shape),
           "finite": bool(torch.isfinite(im.float()).all()), "min": int(im.min()),
           "max": int(im.max())}
    ok = out["finite"] and out["min"] >= 0 and out["max"] <= 255
    if channels == 3:
        out["c0_c1_max_abs_diff"] = int((im[0].int() - im[1].int()).abs().max())
        out["c2_unique"] = sorted(int(v) for v in torch.unique(im[2]))[:8]
        out["c2_pad_value_present"] = bool((im[2].int() - 114).abs().min() == 0)
        if mode == "identical3":
            out["c0_c2_max_abs_diff"] = int((im[0].int() - im[2].int()).abs().max())
            ok = ok and out["c0_c1_max_abs_diff"] < 1 and out["c0_c2_max_abs_diff"] < 1
        elif mode == "c01_equal":
            ok = ok and out["c0_c1_max_abs_diff"] < 1  # [D,D,M]
        elif mode == "c2_zero":
            ok = ok and all(v in (0, 114) for v in out["c2_unique"]) \
                and out["c2_pad_value_present"]  # [D,M,0]
        else:
            raise ValueError(mode)
    if channels == 1:
        out["single_channel"] = True
    out["passed"] = bool(ok)
    if not out["passed"]:
        raise RuntimeError(f"loader gate FAILED for {tag}: {out}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="D:/pycharm/Python Develop/YOLO_1/sample_multimodal")
    p.add_argument("--split-src", default="D:/pycharm/Python Develop/YOLO_1/v031_step1_rgb_sample")
    p.add_argument("--out-ir", default="D:/pycharm/Python Develop/YOLO_1/v031_step2_ir_sample")
    p.add_argument("--out-depth", default="D:/pycharm/Python Develop/YOLO_1/v031_step2_depth_sample")
    p.add_argument("--out-depth2", default="D:/pycharm/Python Develop/YOLO_1/v031_step2_depth2_sample")
    a = p.parse_args()

    raw, split_src = Path(a.raw), Path(a.split_src)
    out_ir, out_depth, out_depth2 = Path(a.out_ir), Path(a.out_depth), Path(a.out_depth2)
    step1_yaml = yaml.safe_load((split_src / "dataset.yaml").read_text(encoding="utf-8"))
    names = step1_yaml["names"]

    report = {
        "d_min_mm": D_MIN, "d_max_mm": D_MAX,
        "depth_encoding": "fixed_log_mm_v1",
        "depth_invalid_value": 0,
        "disk_channel_order_bgr_b2a": ["valid_mask", "log_depth", "log_depth"],
        "model_channel_order_rgb_b2a": ["log_depth", "log_depth", "valid_mask"],
        "disk_channel_order_bgr_b2b": ["0", "valid_mask", "log_depth"],
        "model_channel_order_rgb_b2b": ["log_depth", "valid_mask", "0"],
        "ir_collapse": "per_pixel_median (sample-stage robust scalarization; NOT the final IR representation)",
        "ir_redundancy_gate_version": "sample_probe_v1",
        "ir_gate": {"mean_pairwise_abs_diff_max": 6.0,
                    "high_diff_threshold": 50,
                    "high_diff_fraction_max": 0.001,
                    "legacy_max_abs_diff_gate_2": "RETIRED / over-sensitive to isolated pixels",
                    "note": "thresholds frozen from the 17-image sample probe; re-derive on official 2000-image data"},
        "ir_audit_verdict": "PASS_WITH_LOCAL_CHANNEL_ARTIFACT",
        "ir_artifact_note": "000005_010: 0.034% pixels with pairwise diff>50 (localized single-channel "
                            "saturation/artifact; exact sensor/ISP mechanism unknown; CFA demosaic "
                            "clipping is a possible but unproven mechanism). Image kept, not patched; "
                            "median collapse absorbs it. Anomaly does not cover the image's only GT "
                            "(class 2 animal) - diagnostic aid only, NOT a PASS condition.",
        "raw_mask_binary": True,
        "loader_resize_interpolation": "INTER_LINEAR",
        "letterbox_padding_value": 114,
        "eval_tensor_mask_binary": False,
        "training_augmented_mask_may_be_soft": True,
        "note": "fixed log map holds only for original valid pixels; letterbox/mosaic "
                "padding regions carry no physical depth semantics (accepted for probe).",
        "splits": {}, "label_mismatches": [], "loader_gates": [],
    }

    for split in ("train", "val"):
        ids = sorted(p.stem for p in (split_src / "images" / split).glob("*.png"))
        report["splits"][split] = {"n": len(ids), "ids": ids}
        for out_root in (out_ir, out_depth, out_depth2):
            (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
            (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for sid in ids:
            # IR
            ir_meta = convert_ir(raw / "infrared" / f"{sid}.png",
                                 out_ir / "images" / split / f"{sid}.png")
            report["splits"][split].setdefault("ir", {})[sid] = ir_meta
            # Depth (both encodings from one read)
            d = cv2.imread(str(raw / "depth" / f"{sid}.png"), cv2.IMREAD_UNCHANGED)
            if d is None or d.ndim != 2 or d.dtype != np.uint16:
                raise RuntimeError(f"depth not uint16 HxW: {sid} ({d.dtype if d is not None else None})")
            logd_u8, mask_u8 = encode_depth(d)
            valid = (d >= D_MIN) & (d <= D_MAX)
            for out_root, disk_img in ((out_depth, cv2.merge([mask_u8, logd_u8, logd_u8])),
                                       (out_depth2, cv2.merge([np.zeros_like(logd_u8), mask_u8, logd_u8]))):
                if not cv2.imwrite(str(out_root / "images" / split / f"{sid}.png"), disk_img):
                    raise OSError(f"failed to write depth for {sid}")
            d_meta = depth_quant_audit(d, logd_u8, valid)
            d_meta.update({"shape": d.shape, "dmin_mm": int(d.min()), "dmax_mm": int(d.max()),
                           "n_zero": int((d == 0).sum()), "raw_mask_values": sorted(int(v) for v in np.unique(mask_u8))})
            report["splits"][split].setdefault("depth", {})[sid] = d_meta
            # labels: copy from Step-1 dataset; compare zip labels
            lab_step1 = split_src / "labels" / split / f"{sid}.txt"
            if not lab_step1.is_file():
                raise RuntimeError(f"missing step1 label: {lab_step1}")
            for out_root in (out_ir, out_depth, out_depth2):
                shutil.copy2(lab_step1, out_root / "labels" / split / f"{sid}.txt")
            lab_zip = raw / "labels" / f"{sid}.txt"
            if lab_zip.is_file() and lab_zip.read_bytes() != lab_step1.read_bytes():
                report["label_mismatches"].append(sid)

    # ---- IR robust redundancy gate (two-level; legacy global max-diff gate rejected) ----
    all_ir = [m for s in report["splits"].values() for m in s.get("ir", {}).values()]
    ir_fails = []
    for m in all_ir:
        a_ok = m["mean_pairwise_abs_diff"] <= IR_MEAN_PAIRWISE_GATE
        b_ok = m["frac_pixels_pairwise_gt_50"] <= IR_FRAC_GT50_GATE
        m["gate_pass"] = bool(a_ok and b_ok)
        if not m["gate_pass"]:
            ir_fails.append(m)
    if ir_fails:
        raise RuntimeError(f"IR robust gate FAILED on {len(ir_fails)} image(s): {ir_fails}")

    # ---- dataset yamls ----
    def write_yaml(root: Path, channels: int | None, name: str):
        body = "train: images/train\nval: images/val\n"
        if channels is not None:
            body += f"channels: {channels}\n"
        body += "names:\n" + "\n".join(f"  {k}: {v}" for k, v in names.items()) + "\n"
        (root / name).write_text(body, encoding="utf-8")

    write_yaml(out_ir, None, "dataset.yaml")          # B1-A (3ch default)
    write_yaml(out_ir, 1, "dataset_1ch.yaml")          # B1-B (grayscale loader)
    write_yaml(out_depth, None, "dataset.yaml")        # B2-A
    write_yaml(out_depth2, None, "dataset.yaml")       # B2-B (3ch compatibility)

    # ---- loader tensor gates (real dataloader, augment off) ----
    report["loader_gates"].append(loader_gate(out_ir / "images" / "train", names, 3, "B1-A ir 3ch", "identical3"))
    report["loader_gates"].append(loader_gate(out_ir / "images" / "train", names, 1, "B1-B ir 1ch", "none"))
    report["loader_gates"].append(loader_gate(out_depth / "images" / "train", names, 3, "B2-A depth [D,D,M]", "c01_equal"))
    report["loader_gates"].append(loader_gate(out_depth2 / "images" / "train", names, 3, "B2-B depth [D,M,0]", "c2_zero"))

    out_json = Path(a.out_depth).parent / "v031_step2_datasets_report.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("ir_collapse", "ir_gate", "ir_audit_verdict", "label_mismatches")}, indent=2))
    for g_ in report["loader_gates"]:
        print(f"loader gate {g_['tag']}: passed={g_['passed']} shape={g_['shape']} "
              f"finite={g_['finite']} range=[{g_['min']:.4f},{g_['max']:.4f}]")
    print(f"report -> {out_json}")


if __name__ == "__main__":
    main()
