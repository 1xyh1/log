#!/usr/bin/env python3
"""Launch independent U6 seed lanes in parallel on distinct GPUs."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal.t1gr_g_core import SEEDS  # noqa: E402


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--devices", required=True, help="comma-separated distinct physical CUDA IDs")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument("--u6-view-manifest", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    devices = _csv(args.devices)
    seeds = [int(value) for value in _csv(args.seeds)]
    if not seeds or any(seed not in SEEDS for seed in seeds) or len(seeds) != len(set(seeds)):
        raise SystemExit("invalid --seeds")
    if len(devices) != len(seeds) or len(devices) != len(set(devices)):
        raise SystemExit("provide exactly one distinct --devices entry per selected seed")
    processes = []
    for seed, device in zip(seeds, devices):
        command = [
            sys.executable,
            str(ROOT / "scripts/t1gr_u6_server_run_lane.py"),
            "--mode", args.mode,
            "--seed", str(seed),
            "--device", device,
            "--u6-view-manifest", args.u6_view_manifest,
            "--base-checkpoint", args.base_checkpoint,
            "--run-root", args.run_root,
        ]
        processes.append((seed, device, subprocess.Popen(command, cwd=str(ROOT))))
    failures = []
    for seed, device, process in processes:
        code = process.wait()
        if code != 0:
            failures.append({"seed": seed, "device": device, "returncode": code})
    if failures:
        print(json.dumps({"status": "FAIL", "failures": failures}, ensure_ascii=False, indent=2), file=sys.stderr)
        raise SystemExit(2)
    print(json.dumps({"status": "PASS", "mode": args.mode, "lanes": [{"seed": seed, "device": device} for seed, device, _ in processes]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
