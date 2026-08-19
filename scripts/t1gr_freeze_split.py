#!/usr/bin/env python3
"""Freeze private split truth outside repo and publish only non-ID commitments."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from multimodal.t1gr_e2e5 import (  # noqa: E402
    SCHEMA_SPLIT_FREEZE_PRIVATE, SCHEMA_SPLIT_FREEZE_PUBLIC, SCHEMA_SPLIT_PRIVATE,
    SPLITS, canonical_ids_sha, require_outside_repo, sha256_file, sha256_json, utc_now_iso,
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def git_head(repo: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposal-private", required=True)
    ap.add_argument("--split-manifest-private", required=True)
    ap.add_argument("--sealed-holdout-out", required=True)
    ap.add_argument("--public-out", default="reports/step4_t1gr/split_freeze_public.json")
    ap.add_argument("--repo-root", default=str(ROOT))
    args = ap.parse_args()

    repo = Path(args.repo_root)
    prop_path = Path(args.proposal_private)
    private_out = Path(args.split_manifest_private)
    sealed_out = Path(args.sealed_holdout_out)
    public_out = ROOT / args.public_out
    for p, code in (
        (prop_path, "PRIVATE_PROPOSAL_MUST_BE_OUTSIDE_REPO"),
        (private_out, "PRIVATE_SPLIT_MANIFEST_MUST_BE_OUTSIDE_REPO"),
        (sealed_out, "SEALED_HOLDOUT_MUST_BE_OUTSIDE_REPO"),
    ):
        require_outside_repo(p, repo, code)
    if private_out.exists() or sealed_out.exists() or public_out.exists():
        raise RuntimeError("REFUSE_OVERWRITE_SPLIT_FREEZE")

    prop = load(prop_path)
    if prop.get("schema") != SCHEMA_SPLIT_PRIVATE:
        raise RuntimeError("SPLIT_PROPOSAL_SCHEMA_FAIL")
    if prop.get("proposal_gate_passed") is not True:
        raise RuntimeError("SPLIT_PROPOSAL_GATE_NOT_PASS")
    gates = prop.get("gates") or {}
    if not gates or not all(gates.values()):
        raise RuntimeError(f"SPLIT_PROPOSAL_GATES_NOT_ALL_PASS:{gates}")

    freeze_time = utc_now_iso()
    freeze_commit = git_head(repo)
    ids = {s: sorted(map(str, prop[f"{s}_ids"])) for s in SPLITS}
    id_hashes = {s: canonical_ids_sha(ids[s]) for s in SPLITS}
    group_hashes = {s: sha256_json(sorted(map(str, prop["groups"][s]))) for s in SPLITS}

    private_obj = {
        "schema": SCHEMA_SPLIT_FREEZE_PRIVATE,
        "freeze_timestamp_utc": freeze_time,
        "freeze_repo_commit": freeze_commit,
        "proposal_sha256": sha256_file(prop_path),
        "private_contract_sha256": prop["private_contract_sha256"],
        "split_seed": prop["split_policy"]["split_seed"],
        "train_ids": ids["train"],
        "dev_ids": ids["dev"],
        "final_holdout_ids": ids["final_holdout"],
        "ids_sha256": id_hashes,
        "groups": prop["groups"],
        "group_sha256": group_hashes,
        "class_support": prop["class_support"],
        "proposal_gate_passed": True,
        "training_access_policy": {
            "train": "ALLOWED",
            "dev": "ALLOWED_FOR_MONITORING_AND_BASELINE_EVAL",
            "final_holdout": "FORBIDDEN_UNTIL_T1GR_FINAL_ADJUDICATION",
        },
    }
    private_out.parent.mkdir(parents=True, exist_ok=True)
    private_out.write_text(json.dumps(private_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    sealed_obj = {
        "schema": "t1gr-final-holdout-sealed-v2",
        "freeze_timestamp_utc": freeze_time,
        "split_manifest_private_sha256": sha256_file(private_out),
        "final_holdout_ids": ids["final_holdout"],
        "count": len(ids["final_holdout"]),
        "ids_sha256": id_hashes["final_holdout"],
        "open_policy": "DO_NOT_OPEN_UNTIL_T1GR_FINAL_ADJUDICATION",
    }
    sealed_out.parent.mkdir(parents=True, exist_ok=True)
    sealed_out.write_text(json.dumps(sealed_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    public_obj = {
        "schema": SCHEMA_SPLIT_FREEZE_PUBLIC,
        "freeze_timestamp_utc": freeze_time,
        "freeze_repo_commit": freeze_commit,
        "proposal_sha256": sha256_file(prop_path),
        "split_manifest_private_sha256": sha256_file(private_out),
        "private_contract_sha256": prop["private_contract_sha256"],
        "sample_counts": {s: len(ids[s]) for s in SPLITS},
        "ids_sha256": id_hashes,
        "group_counts": {s: len(prop["groups"][s]) for s in SPLITS},
        "group_sha256": group_hashes,
        "class_support": prop["class_support"],
        "proposal_gate_passed": True,
        "contains_any_sample_ids": False,
        "final_holdout_ids_exposed": False,
        "evidence_model": "repo nondisclosure + harness access seal; not a claim of human secrecy from the raw dataset",
        "training_precedes_freeze_claim": "NOT_ASSERTED_HERE; future runner must prove freeze_timestamp < training_start_timestamp",
    }
    # Fail closed if accidental ID-bearing keys are ever added.
    forbidden_keys = {"train_ids", "dev_ids", "final_holdout_ids", "paired_ids"}
    if forbidden_keys & set(public_obj):
        raise RuntimeError("PUBLIC_FREEZE_EXPOSES_SAMPLE_IDS")
    public_out.parent.mkdir(parents=True, exist_ok=True)
    public_out.write_text(json.dumps(public_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "private_split_manifest": str(private_out),
        "sealed_holdout": str(sealed_out),
        "public_freeze": str(public_out),
        "freeze_timestamp_utc": freeze_time,
        "sample_counts": public_obj["sample_counts"],
        "ids_sha256": id_hashes,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
