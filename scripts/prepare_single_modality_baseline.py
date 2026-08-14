#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mmod_qaf.constants import CLASS_NAMES
from mmod_qaf.data import discover_samples


def read_ids(path: str | Path) -> set[str]:
    return {x.strip() for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()}


def convert_image(record, mode: str, out_path: Path, dmin=300, dmax=19999):
    if mode == "rgb":
        shutil.copy2(record.rgb, out_path)
        return
    if mode == "ir":
        image = cv2.imread(str(record.infrared), cv2.IMREAD_COLOR)
        gray = np.rint(image.astype(np.float32).mean(axis=2)).astype(np.uint8)
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif mode == "depth":
        depth = cv2.imread(str(record.depth), cv2.IMREAD_UNCHANGED)
        if depth.dtype != np.uint16 or depth.ndim != 2:
            raise ValueError(f"Invalid metric depth: {record.depth}")
        valid = (depth >= dmin) & (depth <= dmax)
        encoded = np.zeros(depth.shape, dtype=np.float32)
        values = np.clip(depth.astype(np.float32), dmin, dmax)
        encoded[valid] = (np.log(values[valid]) - math.log(dmin)) / (math.log(dmax) - math.log(dmin))
        gray = np.rint(encoded * 255).astype(np.uint8)
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        raise ValueError(mode)
    if not cv2.imwrite(str(out_path), image):
        raise OSError(f"Failed to write {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--train-ids", required=True)
    p.add_argument("--val-ids", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=("rgb", "ir", "depth"), required=True)
    args = p.parse_args()
    records = {r.sample_id: r for r in discover_samples(args.data, invalid_depth_policy="skip")}
    split_ids = {"train": read_ids(args.train_ids), "val": read_ids(args.val_ids)}
    out = Path(args.out)
    for split, ids in split_ids.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for sid in sorted(ids):
            r = records[sid]
            convert_image(r, args.mode, out / "images" / split / f"{sid}.png")
            shutil.copy2(r.label, out / "labels" / split / f"{sid}.txt")
    dataset_yaml = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
    }
    (out / "dataset.yaml").write_text(yaml.safe_dump(dataset_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(out / "dataset.yaml")

if __name__ == "__main__":
    main()
