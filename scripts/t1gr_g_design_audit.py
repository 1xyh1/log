#!/usr/bin/env python3
"""Audit E5 v2 provenance and the additive T1-GR design freeze."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal.t1gr_g_core import (  # noqa: E402
    SCHEMA_DESIGN_AUDIT, payload_ok, payload_sha256, validate_design,
)


def read_json(path: Path) -> dict:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"T1GR_G_JSON_READ_FAIL:{path.name}") from exc
    if not isinstance(obj, dict):
        raise RuntimeError(f"T1GR_G_JSON_OBJECT_REQUIRED:{path.name}")
    return obj


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(obj, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True)
    ap.add_argument("--e5-recipe", required=True)
    ap.add_argument("--e5-final", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    design_path = Path(args.design).resolve()
    recipe_path = Path(args.e5_recipe).resolve()
    final_path = Path(args.e5_final).resolve()
    out_path = Path(args.out).resolve()
    design = read_json(design_path)
    recipe = read_json(recipe_path)
    final = read_json(final_path)
    design_info = validate_design(design)

    upstream = design["upstream"]
    checks = {
        "design_payload_valid": payload_ok(design),
        "e5_recipe_schema": recipe.get("schema") == upstream["e5_recipe_schema"],
        "e5_recipe_payload_valid": payload_ok(recipe),
        "e5_recipe_payload_pin": recipe.get("payload_sha256") == upstream["e5_recipe_payload_sha256"],
        "e5_recipe_self_pin": recipe.get("recipe_sha256_self") == upstream["e5_recipe_sha256_self"],
        "e5_final_schema": final.get("schema") == upstream["e5_final_schema"],
        "e5_final_payload_valid": payload_ok(final),
        "e5_final_payload_pin": final.get("payload_sha256") == upstream["e5_final_payload_sha256"],
        "e5_gate_passed": final.get("e5_gate_passed") is True,
        "e5_baseline_accepted_dev_only": final.get("step1_baseline_status") == "ACCEPTED_DEV_ONLY",
        "e5_dev_metric_pin": final.get("step1_dev_map50_95") == upstream["e5_dev_map50_95"],
        "design_entry_authorized": final.get("t1gr_design_entry_authorized") is True,
        "multiseed_training_still_blocked": final.get("t1gr_multiseed_training_authorized") is False,
        "holdout_still_blocked": final.get("final_holdout_open_authorized") is False,
        "base_checkpoint_pin": recipe.get("base_checkpoint_sha256") == upstream["base_checkpoint_sha256"],
        "train_commitment_pin": (recipe.get("ids_commitments") or {}).get("train") == upstream["train_ids_commitment"],
        "dev_commitment_pin": (recipe.get("ids_commitments") or {}).get("dev") == upstream["dev_ids_commitment"],
        "holdout_commitment_pin": (recipe.get("ids_commitments") or {}).get("final_holdout") == upstream["final_holdout_ids_commitment"],
        "sample_count_pin": recipe.get("sample_counts") == design.get("sample_counts"),
        "explicit_musgd_pin": (recipe.get("train_args") or {}).get("optimizer") == "MuSGD",
        "last_checkpoint_primary": (design.get("training") or {}).get("checkpoint_primary") == "last.pt",
        "dev_max_det_100": (design.get("evaluation") or {}).get("max_det") == 100,
        "final_holdout_not_an_input": all("holdout" not in p.name.lower() for p in (design_path, recipe_path, final_path)),
    }
    passed = all(checks.values())
    report = {
        "schema": SCHEMA_DESIGN_AUDIT,
        "design_freeze_passed": passed,
        "passed_count": sum(bool(v) for v in checks.values()),
        "total_count": len(checks),
        "checks": checks,
        "design": design_info,
        "implementation_entry_authorized": passed,
        "smoke_training_authorized": False,
        "multiseed_training_authorized": False,
        "final_holdout_open_authorized": False,
        "depth_go": False,
        "production_go": False,
        "next_action": "implement and audit a one-epoch matched-arm smoke; FINAL HOLDOUT remains sealed",
    }
    report["payload_sha256"] = payload_sha256(report)
    atomic_json(out_path, report)
    print(json.dumps({
        "status": "PASS" if passed else "FAIL",
        "out": str(out_path),
        "passed_count": report["passed_count"],
        "total_count": report["total_count"],
        "multiseed_training_authorized": False,
        "final_holdout_open_authorized": False,
    }, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

