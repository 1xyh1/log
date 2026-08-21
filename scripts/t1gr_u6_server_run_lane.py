#!/usr/bin/env python3
"""Run four U6 arms sequentially for one seed on one visible GPU."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal.t1gr_u6_core import launch_rows  # noqa: E402
from multimodal.t1gr_g_core import SEEDS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--device", required=True, help="physical CUDA device exposed to this seed lane")
    parser.add_argument("--u6-view-manifest", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    arms = [row["arm"] for row in launch_rows() if row["seed"] == int(args.seed)]
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(args.device)
    completed = []
    for arm in arms:
        command = [
            sys.executable,
            str(ROOT / "scripts/t1gr_u6_server_run_one.py"),
            "--mode", args.mode,
            "--arm", arm,
            "--seed", str(args.seed),
            "--u6-view-manifest", args.u6_view_manifest,
            "--base-checkpoint", args.base_checkpoint,
            "--run-root", args.run_root,
        ]
        process = subprocess.run(command, cwd=str(ROOT), env=environment, check=False)
        if process.returncode != 0:
            print(json.dumps({"status": "FAIL", "seed": args.seed, "arm": arm, "returncode": process.returncode}), file=sys.stderr)
            raise SystemExit(process.returncode)
        completed.append(arm)
    print(json.dumps({"status": "PASS", "mode": args.mode, "seed": args.seed, "physical_device": str(args.device), "completed_arms": completed}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
