"""Pure schedule and adjudication helpers for T1-TR."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Mapping, Sequence

FORMAL_SEED = 20260812
FORMAL_EPOCHS = 80
FORMAL_BATCH = 4
U2_RUN_NAME = "U2-S_P5_FULL_BALANCED_SHUFFLED_seed20260812"

T1_MANIFEST_SHA256 = "081afec392d96ee2d570a3424e5f015f05ee308297daed8900ece5584c707312"
T1_LAST_SHA256 = "8380e21504fabd0d8c3715398739bbb0bed5aaafd9c822dfc14c9503af2daeee"
T0_MANIFEST_SHA256 = "99c98b741ff3599223a26c0726f8cf7e702a9582d039e1dfe27c4c4af00b0f67"
T0_LAST_SHA256 = "a977dbe19e81bde06a14d635656a47034bf55d186f091a3352dd73b17e40a496"
T1S_SUMMARY_RAW_SHA256 = "a38881cb019764242f3e34560c0be4a6d364aad36d7cb7496978526caf2e98f2"
T1S_ZERO_VAL6_MAP5095 = 0.29596085371085373
T0_ZERO_VAL6_MAP5095 = 0.26443877551020406
T1S_ACCEPTED_COMMIT = "7c86a87a0ca61e6ebc8299b4f3b35dc997f3d46d"

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def sha256_json(obj) -> str:
    return sha256_bytes(json.dumps(
        obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8"))

def epoch_shift(epoch: int, n: int) -> int:
    if n < 2:
        raise ValueError("need at least 2 ids")
    return 1 + (int(epoch) % (n - 1))

def balanced_derangement_map(ids: Sequence[str], epoch: int) -> dict[str, str]:
    ids = tuple(str(x) for x in ids)
    if len(set(ids)) != len(ids):
        raise ValueError("ids must be unique")
    shift = epoch_shift(epoch, len(ids))
    return {
        sid: ids[(i + shift) % len(ids)]
        for i, sid in enumerate(ids)
    }

def verify_epoch_mapping(ids: Sequence[str], epoch: int, mapping: Mapping[str, str]) -> dict:
    ids = tuple(str(x) for x in ids)
    expected = balanced_derangement_map(ids, epoch)
    got = {str(k): str(v) for k, v in mapping.items()}
    values = list(got.values())
    return {
        "epoch": int(epoch),
        "shift": epoch_shift(epoch, len(ids)),
        "exact_expected": got == expected,
        "keys_exact": set(got) == set(ids),
        "values_bijection": len(values) == len(set(values)) and set(values) == set(ids),
        "self_matches": sum(got.get(sid) == sid for sid in ids),
        "mapping_sha256": sha256_json(got),
        "passed": (
            got == expected
            and set(got) == set(ids)
            and len(values) == len(set(values))
            and set(values) == set(ids)
            and all(got[sid] != sid for sid in ids)
        ),
    }

def schedule_balance(ids: Sequence[str], epochs: int = FORMAL_EPOCHS) -> dict:
    ids = tuple(str(x) for x in ids)
    counts = Counter()
    shift_counts = Counter()
    for epoch in range(int(epochs)):
        mapping = balanced_derangement_map(ids, epoch)
        shift_counts[epoch_shift(epoch, len(ids))] += 1
        for rec, donor in mapping.items():
            counts[(rec, donor)] += 1
    nonself = {
        f"{r}->{d}": counts[(r, d)]
        for r in ids for d in ids if r != d
    }
    self_counts = {
        f"{r}->{r}": counts[(r, r)]
        for r in ids
    }
    expected_each = epochs // (len(ids) - 1) if epochs % (len(ids)-1) == 0 else None
    passed = (
        expected_each is not None
        and all(v == expected_each for v in nonself.values())
        and all(v == 0 for v in self_counts.values())
        and set(shift_counts) == set(range(1, len(ids)))
        and all(v == expected_each for v in shift_counts.values())
    )
    return {
        "n_ids": len(ids),
        "epochs": int(epochs),
        "expected_each_nonself_pair": expected_each,
        "shift_counts": dict(sorted(shift_counts.items())),
        "self_counts": self_counts,
        "nonself_counts": nonself,
        "passed": passed,
    }

def contrast_label(new: Mapping[str, float], base: Mapping[str, float]) -> dict:
    if set(new) != set(base):
        raise ValueError("endpoint keys mismatch")
    deltas = {k: float(new[k]) - float(base[k]) for k in new}
    vals = list(deltas.values())
    if all(v > 0 for v in vals):
        label = "STABLE_POSITIVE"
    elif all(v < 0 for v in vals):
        label = "STABLE_NEGATIVE"
    elif all(v == 0 for v in vals):
        label = "EXACT_TIE"
    else:
        label = "MIXED"
    return {"deltas": deltas, "label": label}

def decide_training_source_specificity(
    u0: Mapping[str, float],
    u1: Mapping[str, float],
    u2: Mapping[str, float],
) -> dict:
    p = contrast_label(u1, u0)  # paired - null
    s = contrast_label(u2, u0)  # shuffled - null
    q = contrast_label(u1, u2)  # paired - shuffled

    if q["label"] == "STABLE_NEGATIVE":
        branch = "SHUFFLED_TRAINING_OUTPERFORMS_PAIRED"
        replication = False
    elif p["label"] == "STABLE_POSITIVE" and q["label"] == "STABLE_POSITIVE":
        branch = "PAIRED_TRAINING_SOURCE_SPECIFICITY_SUPPORTED_SINGLE_SEED"
        replication = True
    elif (
        p["label"] == "STABLE_POSITIVE"
        and s["label"] == "STABLE_POSITIVE"
        and q["label"] in {"MIXED", "EXACT_TIE"}
    ):
        branch = "GENERIC_TRAINING_REGULARIZATION_SOURCE_IDENTITY_UNPROVEN"
        replication = False
    elif s["label"] == "STABLE_POSITIVE" and q["label"] == "MIXED":
        branch = "SHUFFLED_TRAINING_HAS_GAIN_PAIRED_ADVANTAGE_INCONCLUSIVE"
        replication = False
    elif p["label"] == "STABLE_POSITIVE":
        branch = "PAIRED_TRAINING_ADVANTAGE_INCONCLUSIVE"
        replication = False
    else:
        branch = "TRAINING_TREATMENT_GAIN_NOT_STABLE"
        replication = False

    return {
        "branch": branch,
        "replication_seed_go": replication,
        "depth_go": False,
        "production_go": False,
        "contrasts": {
            "U1_minus_U0": p,
            "U2_minus_U0": s,
            "U1_minus_U2": q,
        },
    }
