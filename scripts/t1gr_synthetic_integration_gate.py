#!/usr/bin/env python3
"""Synthetic P0/P1 integration gate for T1-GR E2-E5 v2. No GPU/Ultralytics training required."""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from multimodal.t1gr_e2e5 import (  # noqa: E402
    SCHEMA_STEP1_RECIPE, SCHEMA_VIEW_MANIFEST, canonical_ids_sha, sha256_file,
)


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def make_dataset(root: Path, *, bad_depth: bool = False, rare_class: bool = False, n: int = 6) -> Path:
    for d in ("rgb", "ir", "depth", "labels"):
        (root / d).mkdir(parents=True, exist_ok=True)
    for i in range(n):
        sid = f"g{i:02d}_f000"
        rgb = np.full((12, 16, 3), 10 + i, np.uint8)
        ir = np.full((12, 16, 3), 30 + i, np.uint8)
        if bad_depth and i == 2:
            dep = np.full((12, 16, 3), 40, np.uint8)
        else:
            dep = np.full((12, 16), 1000 + i, np.uint16)
        cv2.imwrite(str(root / "rgb" / f"{sid}.png"), rgb)
        cv2.imwrite(str(root / "ir" / f"{sid}.png"), ir)
        cv2.imwrite(str(root / "depth" / f"{sid}.png"), dep)
        cls = 1 if rare_class and i == 0 else 0
        (root / "labels" / f"{sid}.txt").write_text(f"{cls} 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    return root


def layout_for(root: Path, *, coverage_all: bool = False) -> dict:
    minv = 1 if coverage_all else 0
    return {
        "schema": "t1gr-layout-spec-v2",
        "dataset_root": str(root),
        "expected_sample_count": 6,
        "modalities": {
            "rgb": {"dir": "rgb", "extensions": [".png"]},
            "ir": {"dir": "ir", "extensions": [".png"]},
            "depth": {"dir": "depth", "extensions": [".png"]},
            "labels": {"dir": "labels", "extensions": [".txt"]},
        },
        "sample_id": {"mode": "stem", "regex": None, "regex_group": None},
        "label_format": {"type": "yolo_xywh_normalized_detect", "exact_fields": 5, "edge_tolerance": 1e-6, "num_classes": 2, "class_names": ["a", "b"]},
        "format_expectations": {
            "rgb": {"allowed_dtypes": ["uint8"], "allowed_ndim": [3], "allowed_channels": [3], "height": 12, "width": 16},
            "ir": {"allowed_dtypes": ["uint8"], "allowed_ndim": [3], "allowed_channels": [3], "height": 12, "width": 16},
            "depth": {"allowed_dtypes": ["uint16"], "allowed_ndim": [2], "allowed_channels": [1], "height": 12, "width": 16},
            "require_cross_modal_hw_match": True,
        },
        "group_rule": {"type": "regex", "description": "synthetic group", "regex": "^(g\\d+)_", "regex_group": 1, "metadata_file": None, "metadata_id_field": None, "metadata_group_field": None, "parent_level": None},
        "split_policy": {
            "train_fraction": 0.5, "dev_fraction": 0.25, "final_holdout_fraction": 0.25, "split_seed": 7,
            "objective_weights": {"samples": 1.0, "class_images": 2.0, "class_boxes": 1.0},
            "coverage_policy": {
                "min_image_count_by_split": {"train": minv, "dev": minv, "final_holdout": minv},
                "min_box_count_by_split": {"train": minv, "dev": minv, "final_holdout": minv},
                "exempt_classes": [],
            },
        },
    }


def import_runner():
    p = ROOT / "scripts/t1gr_run_step1_baseline.py"
    spec = importlib.util.spec_from_file_location("t1gr_runner_test", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_fake_view_and_recipe(tmp: Path):
    view = tmp / "view"; (view / "images/train").mkdir(parents=True); (view / "images/val").mkdir(parents=True)
    (view / "labels/train").mkdir(parents=True); (view / "labels/val").mkdir(parents=True)
    train_id, dev_id, hold_id = "train0", "dev0", "hold0"
    for sid, split in ((train_id, "train"), (dev_id, "val")):
        img = view / "images" / split / f"{sid}.png"; lab = view / "labels" / split / f"{sid}.txt"
        img.write_bytes((sid + "-image").encode()); lab.write_text("0 0.5 0.5 0.2 0.2\n")
    yaml = view / "dataset.yaml"; yaml.write_text(f"path: {view.as_posix()}\ntrain: images/train\nval: images/val\nnc: 1\nnames:\n  0: a\n")
    ck = tmp / "base.pt"; ck.write_bytes(b"checkpoint-v1")
    recipe = tmp / "recipe.json"
    recipe_obj = {
        "schema": SCHEMA_STEP1_RECIPE,
        "private_contract_sha256": "contractsha",
        "split_manifest_private_sha256": "splitsha",
        "split_ids_sha256": {"train": canonical_ids_sha([train_id]), "dev": canonical_ids_sha([dev_id]), "final_holdout": canonical_ids_sha([hold_id])},
        "base_checkpoint": str(ck), "base_checkpoint_sha256": sha256_file(ck),
        "freeze_timestamp_utc": "2026-08-19T00:00:00+00:00",
    }
    recipe.write_text(json.dumps(recipe_obj), encoding="utf-8")
    mappings=[]
    for sid, split in ((train_id,"train"),(dev_id,"val")):
        img=view/"images"/split/f"{sid}.png"; lab=view/"labels"/split/f"{sid}.txt"
        mappings.append({"sample_id":sid,"split":split,"rgb":{"dest":str(img),"sha256":sha256_file(img)},"label":{"dest":str(lab),"sha256":sha256_file(lab)}})
    mf = tmp / "view_manifest.json"
    mf_obj = {
        "schema": SCHEMA_VIEW_MANIFEST, "view_root": str(view), "recipe_sha256": sha256_file(recipe),
        "private_contract_sha256":"contractsha","split_manifest_private_sha256":"splitsha",
        "dataset_yaml":str(yaml),"dataset_yaml_sha256":sha256_file(yaml),
        "train_ids":[train_id],"dev_ids":[dev_id],"train_ids_sha256":canonical_ids_sha([train_id]),"dev_ids_sha256":canonical_ids_sha([dev_id]),
        "final_holdout_ids_sha256":canonical_ids_sha([hold_id]),"final_holdout_intersection_count":0,"final_holdout_excluded_by_actual_id_set":True,
        "mappings":mappings,"mapping_count":2,
    }
    mf.write_text(json.dumps(mf_obj), encoding="utf-8")
    return recipe, mf, ck, view, hold_id


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="reports/step4_t1gr/synthetic_integration_gate.json"); args=ap.parse_args()
    results = {}
    with tempfile.TemporaryDirectory(prefix="t1gr_v2_gate_") as td:
        t = Path(td)
        # P0 format gate: bad depth MUST fail formal contract.
        ds = make_dataset(t / "bad_depth", bad_depth=True)
        spec = t / "bad_layout.json"; spec.write_text(json.dumps(layout_for(ds)), encoding="utf-8")
        cp = run([sys.executable, str(ROOT/"scripts/t1gr_build_contract.py"), "--layout-spec", str(spec), "--private-out", str(t/"bad_private.json"), "--public-out", str(t/"bad_public.json")])
        results["bad_depth_contract_fails"] = cp.returncode == 2 and "contract_gate_passed" in cp.stdout

        # Formal full hash is unconditional: good contract must contain hash on every paired file.
        ds2 = make_dataset(t / "good")
        sp2 = t / "good_layout.json"; sp2.write_text(json.dumps(layout_for(ds2)), encoding="utf-8")
        private2, public2 = t/"good_private.json", t/"good_public.json"
        cp2 = run([sys.executable, str(ROOT/"scripts/t1gr_build_contract.py"), "--layout-spec", str(sp2), "--private-out", str(private2), "--public-out", str(public2)])
        c2 = json.loads(private2.read_text()) if private2.exists() else {}
        results["formal_full_hash_unconditional"] = cp2.returncode == 0 and c2.get("full_hash_mode") is True and all("sha256" in c2["file_meta"][sid][m] for sid in c2.get("paired_ids",[]) for m in ("rgb","ir","depth","labels"))

        # P1 coverage: rare class in one group cannot satisfy 3-way required coverage -> split proposal fail.
        ds3 = make_dataset(t / "rare", rare_class=True)
        l3 = layout_for(ds3, coverage_all=True); sp3=t/"rare_layout.json"; sp3.write_text(json.dumps(l3),encoding="utf-8")
        priv3,pub3=t/"rare_private.json",t/"rare_public.json"
        cp3=run([sys.executable,str(ROOT/"scripts/t1gr_build_contract.py"),"--layout-spec",str(sp3),"--private-out",str(priv3),"--public-out",str(pub3)])
        prop3=t/"rare_proposal.json"
        pp3=run([sys.executable,str(ROOT/"scripts/t1gr_propose_split.py"),"--private-contract",str(priv3),"--out-private",str(prop3)])
        pobj=json.loads(prop3.read_text()) if prop3.exists() else {}
        results["rare_class_coverage_blocks_split"] = cp3.returncode==0 and pp3.returncode==2 and pobj.get("proposal_gate_passed") is False and (not pobj.get("class_group_feasibility",{}).get("passed",True) or not pobj.get("class_coverage_audit",{}).get("passed",True))

        # P0 access control: runner accepts no arbitrary --data; forged extra holdout file also invalidates pinned view.
        runner = import_runner(); recipe,mf,ck,view,hold_id=make_fake_view_and_recipe(t/"runner")
        runner.preflight(recipe,mf)  # baseline good evidence
        extra=view/"images"/"val"/f"{hold_id}.png"; extra.write_bytes(b"holdout")
        try:
            runner.preflight(recipe,mf); forged_failed=False
        except RuntimeError:
            forged_failed=True
        results["forged_holdout_in_view_fails"] = forged_failed
        cli=run([sys.executable,str(ROOT/"scripts/t1gr_run_step1_baseline.py"),"--recipe",str(recipe),"--view-manifest",str(mf),"--data",str(view/"dataset.yaml")])
        results["arbitrary_data_cli_rejected"] = cli.returncode != 0 and "unrecognized arguments: --data" in cli.stdout
        extra.unlink()

        # P0 checkpoint pin: same path, changed bytes MUST fail before importing Ultralytics.
        runner.preflight(recipe,mf)
        ck.write_bytes(b"checkpoint-v2-mutated")
        try:
            runner.preflight(recipe,mf); ck_failed=False
        except RuntimeError as e:
            ck_failed="BASE_CHECKPOINT_SHA_DRIFT" in str(e)
        results["checkpoint_same_path_content_change_fails"] = ck_failed

    all_passed = all(results.values())
    report={"schema":"t1gr-e2-e5-synthetic-integration-gate-v2","all_passed":all_passed,"cases":results}
    out=ROOT/args.out; out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists(): raise RuntimeError(f"REFUSE_OVERWRITE:{out}")
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(report,indent=2,ensure_ascii=False))
    raise SystemExit(0 if all_passed else 2)


if __name__=="__main__":
    main()
