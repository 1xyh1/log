#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmod_qaf.train_loop import train_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train tri-modal YOLO26 QAF model")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    print(train_from_config(cfg))


if __name__ == "__main__":
    main()
