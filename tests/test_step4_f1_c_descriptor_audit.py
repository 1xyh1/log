from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal.step4_f1_c_descriptor_audit import (
    SCAN_IDENTIFIABLE_RANGE,
    average_ranks,
    correlation_report,
    scan_targets,
    spearman,
)


def _scan(values):
    return {
        f"{q:.2f}": {"map50_95": value}
        for q, value in zip((0.0, 0.25, 0.5, 0.75, 1.0), values)
    }


def test_average_ranks_assigns_equal_values_equal_average_rank():
    assert average_ranks([4.0, 1.0, 1.0, 9.0]) == [2.0, 0.5, 0.5, 3.0]
    assert spearman([1, 1, 2, 3], [10, 10, 20, 30]) == pytest.approx(1.0)


def test_tie_aware_spearman_is_permutation_invariant():
    x = [1, 1, 2, 3, 3]
    y = [5, 5, 4, 1, 1]
    order = [3, 0, 4, 2, 1]
    assert spearman(x, y) == pytest.approx(
        spearman([x[i] for i in order], [y[i] for i in order])
    )


def test_scan_targets_records_range_margin_utility_and_tie_break():
    row = scan_targets(_scan([0.30, 0.31, 0.31, 0.29, 0.28]))
    assert row["best_q"] == 0.25
    assert row["best_minus_second_margin"] == pytest.approx(0.0)
    assert row["scan_range"] == pytest.approx(0.03)
    assert row["q0_minus_q1_map50_95"] == pytest.approx(0.02)
    assert row["identifiable_for_best_q"]


def test_weak_scan_is_not_identifiable_for_hard_best_q():
    delta = SCAN_IDENTIFIABLE_RANGE / 4
    row = scan_targets(_scan([0.30, 0.30 + delta, 0.30, 0.30, 0.30]))
    assert not row["identifiable_for_best_q"]


def test_scan_requires_exact_finite_q_grid():
    scan = _scan([0.1] * 5)
    scan.pop("1.00")
    with pytest.raises(ValueError):
        scan_targets(scan)
    with pytest.raises(ValueError):
        scan_targets(_scan([0.1, 0.2, math.nan, 0.3, 0.4]))


def test_correlation_report_contains_continuous_and_family_holdout_axes():
    # scans chosen so q0-q1 utility is STRICTLY decreasing (0.5, 0.4, -0.4,
    # -0.5) — with ties the standard tie-aware Spearman is mathematically
    # capped below 1.0 (e.g. 0.8944), which is not what this test pins.
    targets = [
        scan_targets(_scan([0.9, 0.7, 0.6, 0.5, 0.4])),
        scan_targets(_scan([0.7, 0.6, 0.5, 0.4, 0.3])),
        scan_targets(_scan([0.4, 0.5, 0.6, 0.7, 0.8])),
        scan_targets(_scan([0.3, 0.4, 0.5, 0.6, 0.8])),
    ]
    report = correlation_report(
        [4.0, 3.0, 2.0, 1.0], targets, ["noise", "noise", "blur", "blur"]
    )
    assert report["n_identifiable_for_best_q"] == 4
    assert report["spearman_vs_q0_minus_q1"] == pytest.approx(1.0)
    assert set(report["leave_one_corruption_family_out"]) == {"blur", "noise"}
