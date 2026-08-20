#!/usr/bin/env python3
"""Fail-closed T1-GR training entry.

This design-stage bundle intentionally cannot train.  Keeping an explicit entry
point prevents an old T-series runner from being mistaken for the formal T1-GR
implementation.
"""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "formal"), required=True)
    ap.add_argument("--arm", choices=("G0-N", "G1-P", "G2-S"), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.parse_args()
    raise RuntimeError(
        "T1GR_G_TRAINING_NOT_AUTHORIZED:design entry is approved, but the "
        "aligned multimodal loader and one-epoch matched-arm smoke have not "
        "been audited; do not substitute the old T-series runner"
    )


if __name__ == "__main__":
    main()

