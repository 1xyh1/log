#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmod_qaf.inference import (
    TriModalInferenceDataset,
    collate_inference,
    preprocess_config_from_checkpoint,
    restore_boxes_from_letterbox,
)
from mmod_qaf.model import load_training_checkpoint
from mmod_qaf.submission import make_submission_zip, validate_submission_dir


def parse_output(output):
    if isinstance(output, tuple):
        output = output[0]
    if output.ndim != 3 or output.shape[-1] != 6:
        raise ValueError(f"Expected end-to-end [B,N,6], got {tuple(output.shape)}")
    return output.detach().float().cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--base-weights", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Explicit inference-size override; by default use the size saved in the training checkpoint",
    )
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--max-det", type=int, default=100)
    args = p.parse_args()

    device = torch.device(args.device)
    model, checkpoint = load_training_checkpoint(args.checkpoint, args.base_weights, device=device)
    preprocess = preprocess_config_from_checkpoint(checkpoint, imgsz_override=args.imgsz)
    model.eval(); model.set_head_attr(max_det=args.max_det)
    print(
        "[preprocess] "
        f"imgsz={preprocess.imgsz} ir_mode={preprocess.ir_mode} "
        f"depth_mm=[{preprocess.depth_min_mm},{preprocess.depth_max_mm}] "
        f"depth_resize={preprocess.depth_resize}"
    )
    ds = TriModalInferenceDataset(
        args.data,
        imgsz=preprocess.imgsz,
        ir_mode=preprocess.ir_mode,
        depth_min_mm=preprocess.depth_min_mm,
        depth_max_mm=preprocess.depth_max_mm,
        depth_resize=preprocess.depth_resize,
    )
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, collate_fn=collate_inference)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch in dl:
            pred = parse_output(
                model.predict_with_quality(batch["img"].to(device), batch["quality"].to(device))
            )
            for bi, sid in enumerate(batch["sample_id"]):
                h0, w0 = batch["ori_shape"][bi]
                rows = pred[bi]
                rows = rows[np.isfinite(rows).all(axis=1)]
                rows = rows[rows[:, 4] >= args.conf]
                rows = rows[np.argsort(-rows[:, 4])[: args.max_det]]
                rows = rows.copy()
                rows[:, :4] = restore_boxes_from_letterbox(rows[:, :4], (h0, w0), batch["ratio_pad"][bi])
                rows = rows[(rows[:, 5] >= 0) & (rows[:, 5] < 12) & (rows[:, 5] == np.floor(rows[:, 5]))]
                output_lines = []
                for x1, y1, x2, y2, conf, cls in rows:
                    if x2 <= x1 or y2 <= y1:
                        continue
                    cx, cy = (x1 + x2) / 2 / w0, (y1 + y2) / 2 / h0
                    bw, bh = (x2 - x1) / w0, (y2 - y1) / h0
                    output_lines.append(f"{int(cls)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {conf:.6f}")
                (out_dir / f"{sid}.txt").write_text("\n".join(output_lines) + ("\n" if output_lines else ""), encoding="utf-8")

    errors = validate_submission_dir(out_dir, [sid for sid, *_ in ds.records], max_det=args.max_det)
    if errors:
        raise RuntimeError("Submission validation failed:\n" + "\n".join(errors[:50]))
    zip_path = out_dir.with_suffix(".zip")
    make_submission_zip(out_dir, zip_path)
    print(f"submission ready: {zip_path}")

if __name__ == "__main__":
    main()
