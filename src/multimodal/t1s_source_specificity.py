"""Pure combinatorics/statistics helpers for T1-S source-specificity audit."""
from __future__ import annotations

from itertools import permutations
import statistics
from typing import Mapping, Sequence

VAL6_IDS = (
    "000003_013_00000085",
    "000004_013_00000081",
    "000004_014_00000001",
    "000016",
    "000016_001_00000001",
    "000016_042_suppl_00000164",
)

EXPECTED_DERANGEMENTS = 265
ALPHA = 0.05


def generate_derangements(ids: Sequence[str] = VAL6_IDS) -> list[tuple[str, ...]]:
    ids = tuple(str(x) for x in ids)
    return [
        tuple(p)
        for p in permutations(ids)
        if all(src != rec for rec, src in zip(ids, p))
    ]


def mapping_dict(ids: Sequence[str], perm: Sequence[str]) -> dict[str, str]:
    ids = tuple(str(x) for x in ids)
    perm = tuple(str(x) for x in perm)
    if len(ids) != len(perm) or set(ids) != set(perm):
        raise ValueError("mapping must be a permutation of ids")
    return dict(zip(ids, perm))


def is_derangement(mapping: Mapping[str, str], ids: Sequence[str] = VAL6_IDS) -> bool:
    ids = tuple(str(x) for x in ids)
    return (
        set(mapping) == set(ids)
        and set(map(str, mapping.values())) == set(ids)
        and all(str(mapping[r]) != r for r in ids)
    )


def fixed_donor_index(
    donor_map: Mapping[str, str],
    derangements: Sequence[Sequence[str]],
    ids: Sequence[str] = VAL6_IDS,
) -> int:
    target = tuple(str(donor_map[r]) for r in ids)
    for i, perm in enumerate(derangements):
        if tuple(map(str, perm)) == target:
            return i
    raise ValueError("fixed donor map is not present in derangements")


def distribution_summary(values: Sequence[float]) -> dict:
    xs = [float(x) for x in values]
    if not xs:
        raise ValueError("empty distribution")
    q = statistics.quantiles(xs, n=4, method="inclusive")
    return {
        "n": len(xs),
        "min": min(xs),
        "q1": q[0],
        "median": statistics.median(xs),
        "q3": q[2],
        "max": max(xs),
        "mean": statistics.fmean(xs),
    }


def rank_and_percentile(value: float, values: Sequence[float]) -> dict:
    x = float(value)
    vals = [float(v) for v in values]
    greater = sum(v > x for v in vals)
    equal = sum(v == x for v in vals)
    lower = sum(v < x for v in vals)
    return {
        "greater": greater,
        "equal": equal,
        "lower": lower,
        "descending_rank_min": 1 + greater,
        "descending_rank_max": greater + equal + 1,
        "strict_percentile_vs_distribution": 0.0 if not vals else lower / len(vals),
    }


def exact_identity_randomization(identity: float, derangement_values: Sequence[float]) -> dict:
    vals = [float(v) for v in derangement_values]
    if len(vals) != EXPECTED_DERANGEMENTS:
        raise ValueError(f"expected {EXPECTED_DERANGEMENTS} derangements")
    ge = sum(v >= float(identity) for v in vals)
    p = (1 + ge) / (1 + len(vals))
    return {
        "alpha": ALPHA,
        "count_derangements_ge_identity": ge,
        "denominator": 1 + len(vals),
        "p_one_sided": p,
        "significant": p <= ALPHA,
    }


def decide_source_specificity(
    identity: float,
    zero: float,
    derangement_values: Sequence[float],
) -> dict:
    dsum = distribution_summary(derangement_values)
    rand = exact_identity_randomization(identity, derangement_values)
    i = float(identity)
    z = float(zero)
    med = float(dsum["median"])

    if med > i:
        branch = "WRONG_SOURCE_TYPICALLY_OUTPERFORMS_NATIVE"
        replication = False
        next_step = "source-binding/channel-semantic/spatial-correspondence audit"
    elif z >= i:
        branch = "INFERENCE_RESIDUAL_NOT_SUPPORTED_TRAINING_DYNAMICS_CANDIDATE"
        replication = False
        next_step = "training-time-IR / inference-RGB ablation design"
    elif i > z and rand["significant"]:
        branch = "PAIRED_SOURCE_SPECIFICITY_SUPPORTED_SINGLE_SEED"
        replication = True
        next_step = "T1 replication seed design"
    elif i > z and med > z and not rand["significant"]:
        branch = "GENERIC_RESIDUAL_BENEFIT_SOURCE_IDENTITY_UNPROVEN"
        replication = False
        next_step = "generic residual / representation / training-dynamics audit"
    else:
        branch = "SOURCE_SPECIFICITY_INCONCLUSIVE"
        replication = False
        next_step = "expand diagnostic/data evidence"

    return {
        "branch": branch,
        "replication_seed_go": replication,
        "depth_go": False,
        "production_go": False,
        "next_step": next_step,
        "identity_minus_zero": i - z,
        "identity_minus_derangement_median": i - med,
        "derangement_median_minus_zero": med - z,
        "randomization": rand,
    }


def verify_exact_derangement_family(ids: Sequence[str] = VAL6_IDS) -> dict:
    ders = generate_derangements(ids)
    unique = len(set(ders))
    return {
        "count": len(ders),
        "unique_count": unique,
        "all_valid": all(
            all(src != rec for rec, src in zip(ids, perm))
            and set(perm) == set(ids)
            for perm in ders
        ),
        "passed": len(ders) == EXPECTED_DERANGEMENTS
        and unique == EXPECTED_DERANGEMENTS,
    }
