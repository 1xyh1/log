#!/usr/bin/env python3
"""Private group-aware class-stratified split proposal. Never writes holdout IDs into repo."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from multimodal.t1gr_e2e5 import (  # noqa: E402
    SCHEMA_CONTRACT_PRIVATE, SCHEMA_SPLIT_PRIVATE, SPLITS,
    canonical_ids_sha, class_stats_for_ids, classify_overlap, coverage_audit,
    cross_split_duplicate_audit, group_stats_from_contract, group_stratified_split,
    require_outside_repo, sha256_file, split_sample_overlap, utc_now_iso,
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def complete_policy(policy: dict) -> tuple[bool, list[str]]:
    missing = []
    for k in ("train_fraction", "dev_fraction", "final_holdout_fraction", "split_seed"):
        if policy.get(k) is None:
            missing.append(k)
    cp = policy.get("coverage_policy") or {}
    for block in ("min_image_count_by_split", "min_box_count_by_split"):
        b = cp.get(block) or {}
        for s in SPLITS:
            if b.get(s) is None:
                missing.append(f"coverage_policy.{block}.{s}")
    return not missing, missing


def group_class_feasibility(group_stats: dict, n_classes: int, coverage_policy: dict) -> dict:
    exemptions = {int(x["class_id"]): str(x.get("reason") or "") for x in coverage_policy.get("exempt_classes") or []}
    required_splits = [s for s in SPLITS if int((coverage_policy.get("min_image_count_by_split") or {}).get(s, 0)) > 0]
    rows = []
    for c in range(n_classes):
        groups_with_class = [g for g, st in group_stats.items() if st["image_counts"][c] > 0]
        feasible_by_group_count = len(groups_with_class) >= len(required_splits)
        rows.append({
            "class_id": c,
            "groups_with_class_count": len(groups_with_class),
            "required_split_count": len(required_splits),
            "feasible_by_group_count": feasible_by_group_count,
            "explicitly_exempt": c in exemptions,
            "exemption_reason": exemptions.get(c),
        })
    impossible_unexempt = [r for r in rows if not r["feasible_by_group_count"] and not r["explicitly_exempt"]]
    return {"rows": rows, "impossible_unexempt": impossible_unexempt, "passed": not impossible_unexempt}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--private-contract", required=True)
    ap.add_argument("--out-private", required=True)
    ap.add_argument("--repo-root", default=str(ROOT))
    args = ap.parse_args()

    contract_path = Path(args.private_contract)
    out = Path(args.out_private)
    require_outside_repo(contract_path, Path(args.repo_root), "PRIVATE_CONTRACT_MUST_BE_OUTSIDE_REPO")
    require_outside_repo(out, Path(args.repo_root), "PRIVATE_SPLIT_PROPOSAL_MUST_BE_OUTSIDE_REPO")
    if out.exists():
        raise RuntimeError(f"REFUSE_OVERWRITE:{out}")

    c = load(contract_path)
    if c.get("schema") != SCHEMA_CONTRACT_PRIVATE:
        raise RuntimeError("PRIVATE_CONTRACT_SCHEMA_FAIL")
    if c.get("contract_gate_passed") is not True or c.get("full_hash_mode") is not True:
        raise RuntimeError("PRIVATE_CONTRACT_NOT_FORMAL_PASS")
    gv = c.get("group_rule_validation") or {}
    if gv.get("passed") is not True:
        raise RuntimeError(f"GROUP_RULE_VALIDATION_NOT_PASS:{gv}")

    spec = c["layout_spec"]
    policy = spec.get("split_policy") or {}
    policy_ok, missing_policy = complete_policy(policy)
    if not policy_ok:
        raise RuntimeError(f"SPLIT_POLICY_UNRESOLVED:{missing_policy}")

    ids = [str(x) for x in c["paired_ids"]]
    grouping = {str(k): str(v) for k, v in c["group_map"].items()}
    if set(ids) != set(grouping):
        raise RuntimeError("GROUP_MAP_ID_COVERAGE_FAIL")
    g2i = defaultdict(list)
    for sid, g in grouping.items():
        g2i[g].append(sid)
    if len(g2i) < 3:
        raise RuntimeError("NEED_AT_LEAST_THREE_GROUPS")

    n_classes = int(spec["label_format"]["num_classes"])
    per_id = c["label_stats"]
    gstats = group_stats_from_contract(g2i, per_id, n_classes)
    fractions = {s: float(policy[f"{s}_fraction"]) for s in SPLITS}
    split_groups = group_stratified_split(
        g2i, gstats, fractions, int(policy["split_seed"]),
        weights=policy.get("objective_weights") or {},
    )
    samples = {s: sorted(sid for g in split_groups[s] for sid in g2i[g]) for s in SPLITS}
    group_overlap = classify_overlap(split_groups)
    sample_overlap = split_sample_overlap(samples)
    nonempty = all(samples[s] and split_groups[s] for s in SPLITS)
    union_complete = set().union(*(set(samples[s]) for s in SPLITS)) == set(ids)

    support = {s: class_stats_for_ids(per_id, samples[s], n_classes) for s in SPLITS}
    coverage_policy = policy["coverage_policy"]
    feasibility = group_class_feasibility(gstats, n_classes, coverage_policy)
    coverage = coverage_audit(support, coverage_policy, n_classes)

    split_of = {sid: s for s in SPLITS for sid in samples[s]}
    duplicates = cross_split_duplicate_audit(c.get("duplicate_groups_by_kind") or {}, split_of)

    target_counts = {s: len(ids) * fractions[s] for s in SPLITS}
    size_error = {s: len(samples[s]) - target_counts[s] for s in SPLITS}
    gates = {
        "policy_complete": policy_ok,
        "group_rule_validated": True,
        "three_splits_nonempty": nonempty,
        "group_overlap_empty": group_overlap["passed"],
        "sample_overlap_empty": sample_overlap["passed"],
        "split_union_equals_contract": union_complete,
        "class_coverage_feasible_or_exempt": feasibility["passed"],
        "class_coverage_passed": coverage["passed"],
        "cross_split_exact_duplicate_leakage_empty": duplicates["passed"],
    }
    proposal_gate = all(gates.values())

    report = {
        "schema": SCHEMA_SPLIT_PRIVATE,
        "created_at_utc": utc_now_iso(),
        "private_contract_sha256": sha256_file(contract_path),
        "paired_ids_sha256": c["paired_ids_sha256"],
        "group_rule": spec["group_rule"],
        "split_policy": policy,
        "group_count": len(g2i),
        "group_stats": gstats,
        "groups": split_groups,
        "sample_counts": {s: len(samples[s]) for s in SPLITS},
        "target_sample_counts": target_counts,
        "sample_count_error": size_error,
        "train_ids": samples["train"],
        "dev_ids": samples["dev"],
        "final_holdout_ids": samples["final_holdout"],
        "ids_sha256": {s: canonical_ids_sha(samples[s]) for s in SPLITS},
        "group_overlap_audit": group_overlap,
        "sample_overlap_audit": sample_overlap,
        "union_complete": union_complete,
        "class_support": support,
        "class_group_feasibility": feasibility,
        "class_coverage_audit": coverage,
        "cross_split_duplicate_audit": duplicates,
        "gates": gates,
        "proposal_gate_passed": proposal_gate,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "out_private": str(out),
        "counts": report["sample_counts"],
        "group_counts": {s: len(split_groups[s]) for s in SPLITS},
        "proposal_gate_passed": proposal_gate,
        "failed_gates": [k for k, v in gates.items() if not v],
        "coverage_failures": len(coverage["failures"]),
    }, ensure_ascii=False, indent=2))
    if not proposal_gate:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
