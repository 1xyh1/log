"""Torch-free statistics contract for the F1-C descriptor audit.

The q sweep is deliberately small and often tied.  Keep rank handling and
target construction here so the diagnostic script and adversarial tests use
one implementation.
"""
from __future__ import annotations

import math
import statistics


Q_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
SCAN_IDENTIFIABLE_RANGE = 5e-3


def average_ranks(values: list[float]) -> list[float]:
    """Return zero-based average ranks, assigning equal values equal ranks."""
    if not values:
        return []
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("rank inputs must be finite")
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        value = values[order[start]]
        while stop < len(order) and values[order[stop]] == value:
            stop += 1
        average = (start + stop - 1) / 2.0
        for position in range(start, stop):
            ranks[order[position]] = average
        start = stop
    return ranks


def pearson(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    if not all(math.isfinite(float(v)) for v in [*x, *y]):
        return float("nan")
    mx, my = statistics.mean(x), statistics.mean(y)
    numerator = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denominator = math.sqrt(
        sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)
    )
    return float(numerator / denominator) if denominator > 0 else float("nan")


def spearman(x: list[float], y: list[float]) -> float:
    """Tie-aware Spearman correlation (Pearson correlation of average ranks)."""
    return pearson(average_ranks(x), average_ranks(y))


def scan_targets(scan: dict) -> dict:
    """Validate a five-point q scan and derive exploratory/continuous targets."""
    parsed = {float(q): float(row["map50_95"]) for q, row in scan.items()}
    if set(parsed) != set(Q_GRID):
        raise ValueError(f"q scan must contain exactly {Q_GRID}, got {sorted(parsed)}")
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0
               for value in parsed.values()):
        raise ValueError("q scan metrics must be finite and in [0,1]")

    ordered = sorted(parsed.items(), key=lambda item: (-item[1], item[0]))
    best_q, best_ap = ordered[0]
    second_ap = ordered[1][1]
    scan_range = max(parsed.values()) - min(parsed.values())
    return {
        "best_q": best_q,
        "best_ap": best_ap,
        "second_best_ap": second_ap,
        "best_minus_second_margin": best_ap - second_ap,
        "scan_range": scan_range,
        "identifiable_for_best_q": scan_range >= SCAN_IDENTIFIABLE_RANGE,
        "q0_minus_q1_map50_95": parsed[0.0] - parsed[1.0],
        "scan": {f"{q:.2f}": parsed[q] for q in Q_GRID},
    }


def correlation_report(descriptor: list[float], targets: list[dict],
                       families: list[str]) -> dict:
    """Correlate one descriptor with both q* and the continuous q0-q1 target."""
    if not (len(descriptor) == len(targets) == len(families)):
        raise ValueError("descriptor, targets and families must have equal length")
    best_q = [row["best_q"] for row in targets]
    utility = [row["q0_minus_q1_map50_95"] for row in targets]
    identifiable = [i for i, row in enumerate(targets)
                    if row["identifiable_for_best_q"]]

    def optional(value: float) -> float | None:
        return value if math.isfinite(value) else None

    leave_one_family_out = {}
    for family in sorted(set(families)):
        keep = [i for i, value in enumerate(families) if value != family]
        leave_one_family_out[family] = {
            "n": len(keep),
            "spearman_vs_q0_minus_q1": optional(spearman(
                [descriptor[i] for i in keep], [utility[i] for i in keep])),
            "pearson_vs_q0_minus_q1": optional(pearson(
                [descriptor[i] for i in keep], [utility[i] for i in keep])),
        }

    return {
        "n_all": len(descriptor),
        "spearman_vs_best_q_all_exploratory": optional(
            spearman(descriptor, best_q)),
        "n_identifiable_for_best_q": len(identifiable),
        "spearman_vs_best_q_identifiable_only": optional(spearman(
            [descriptor[i] for i in identifiable],
            [best_q[i] for i in identifiable],
        )),
        "spearman_vs_q0_minus_q1": optional(spearman(descriptor, utility)),
        "pearson_vs_q0_minus_q1": optional(pearson(descriptor, utility)),
        "leave_one_corruption_family_out": leave_one_family_out,
    }
