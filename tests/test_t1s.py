from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal.t1s_source_specificity import (  # noqa: E402
    ALPHA,
    EXPECTED_DERANGEMENTS,
    VAL6_IDS,
    decide_source_specificity,
    distribution_summary,
    exact_identity_randomization,
    fixed_donor_index,
    generate_derangements,
    is_derangement,
    mapping_dict,
    rank_and_percentile,
    verify_exact_derangement_family,
)


def test_exact_derangement_count_265():
    ders = generate_derangements()
    assert len(ders) == 265
    assert len(set(ders)) == 265


def test_every_derangement_has_no_fixed_point():
    for perm in generate_derangements():
        assert all(a != b for a, b in zip(VAL6_IDS, perm))
        assert set(perm) == set(VAL6_IDS)


def test_verify_family():
    got = verify_exact_derangement_family()
    assert got == {
        "count": 265,
        "unique_count": 265,
        "all_valid": True,
        "passed": True,
    }


def test_mapping_dict_requires_permutation():
    with pytest.raises(ValueError):
        mapping_dict(VAL6_IDS, [VAL6_IDS[0]] * 6)


def test_is_derangement_false_for_identity():
    assert not is_derangement(dict(zip(VAL6_IDS, VAL6_IDS)))


def test_is_derangement_true_for_rotation():
    rot = VAL6_IDS[1:] + VAL6_IDS[:1]
    assert is_derangement(dict(zip(VAL6_IDS, rot)))


def test_fixed_donor_index_finds_rotation():
    ders = generate_derangements()
    rot = VAL6_IDS[1:] + VAL6_IDS[:1]
    donor = dict(zip(VAL6_IDS, rot))
    idx = fixed_donor_index(donor, ders)
    assert ders[idx] == rot


def test_distribution_summary():
    s = distribution_summary([1, 2, 3, 4, 5])
    assert s["n"] == 5
    assert s["min"] == 1
    assert s["median"] == 3
    assert s["max"] == 5
    assert s["mean"] == 3


def test_rank_and_percentile():
    r = rank_and_percentile(3, [1, 2, 3, 4, 5])
    assert r["greater"] == 2
    assert r["equal"] == 1
    assert r["lower"] == 2
    assert r["descending_rank_min"] == 3
    assert r["descending_rank_max"] == 4
    assert r["strict_percentile_vs_distribution"] == pytest.approx(0.4)


def test_exact_randomization_minimum_p():
    r = exact_identity_randomization(10.0, [0.0] * 265)
    assert r["count_derangements_ge_identity"] == 0
    assert r["p_one_sided"] == pytest.approx(1 / 266)
    assert r["significant"] is True


def test_exact_randomization_alpha_frozen():
    assert ALPHA == 0.05
    vals = [1.0] * 13 + [0.0] * 252
    r = exact_identity_randomization(1.0, vals)
    assert r["p_one_sided"] == pytest.approx(14 / 266)
    assert r["significant"] is False


def test_exact_randomization_rejects_wrong_count():
    with pytest.raises(ValueError):
        exact_identity_randomization(1.0, [0.0] * 264)


def test_decision_wrong_source_priority():
    d = [2.0] * 265
    out = decide_source_specificity(identity=1.0, zero=0.0, derangement_values=d)
    assert out["branch"] == "WRONG_SOURCE_TYPICALLY_OUTPERFORMS_NATIVE"
    assert out["replication_seed_go"] is False


def test_decision_zero_priority_over_source_specificity():
    d = [0.0] * 265
    out = decide_source_specificity(identity=1.0, zero=1.0, derangement_values=d)
    assert out["branch"] == "INFERENCE_RESIDUAL_NOT_SUPPORTED_TRAINING_DYNAMICS_CANDIDATE"


def test_decision_source_specificity_supported():
    d = [0.5] * 265
    out = decide_source_specificity(identity=1.0, zero=0.4, derangement_values=d)
    assert out["branch"] == "PAIRED_SOURCE_SPECIFICITY_SUPPORTED_SINGLE_SEED"
    assert out["replication_seed_go"] is True
    assert out["depth_go"] is False
    assert out["production_go"] is False


def test_decision_generic_residual():
    # 20 derangements >= identity prevents p<=.05; median stays above ZERO.
    d = [1.1] * 20 + [0.8] * 245
    out = decide_source_specificity(identity=1.0, zero=0.7, derangement_values=d)
    assert out["branch"] == "GENERIC_RESIDUAL_BENEFIT_SOURCE_IDENTITY_UNPROVEN"
    assert out["replication_seed_go"] is False


def test_decision_inconclusive():
    d = [0.2] * 265
    out = decide_source_specificity(identity=0.5, zero=0.4, derangement_values=d)
    # identity is significant in this setup, so make a non-significant but median <= zero case.
    d = [0.6] * 20 + [0.3] * 245
    out = decide_source_specificity(identity=0.5, zero=0.4, derangement_values=d)
    assert out["branch"] == "SOURCE_SPECIFICITY_INCONCLUSIVE"


@pytest.mark.parametrize(
    "needle",
    [
        "6 recipients × 6 sources = 36",
        "!6 = 265",
        "T1S_NATIVE_ANCHOR_FAIL",
        "T1S_FIXED_DONOR_ANCHOR_FAIL",
        "alpha = 0.05",
        "NO model training",
        "NO Depth",
        "ZERO residual condition",
        "PAIRED_SOURCE_SPECIFICITY_SUPPORTED_SINGLE_SEED",
    ],
)
def test_design_contains_frozen_contract(needle):
    text = (ROOT / "docs/step4_t1s/DESIGN_FREEZE.md").read_text(encoding="utf-8")
    assert needle in text


@pytest.mark.parametrize(
    "needle",
    [
        "T1S_NATIVE_ANCHOR_FAIL",
        "T1S_FIXED_DONOR_ANCHOR_FAIL",
        "EXPECTED_DERANGEMENTS",
        "fixed_donor_index",
        "zero_forward",
        "assemble_metric(matrix_stats",
        "T1S_REFUSE_OVERWRITE",
        'RUN_NAMES["T1-F"]',
    ],
)
def test_evaluator_contains_hard_contract(needle):
    text = (ROOT / "scripts/eval_t1s_source_specificity.py").read_text(encoding="utf-8")
    assert needle in text


def test_evaluator_has_no_training_entrypoint():
    text = (ROOT / "scripts/eval_t1s_source_specificity.py").read_text(encoding="utf-8")
    assert ".train(" not in text
    assert "DetectionTrainer" not in text
    assert "run_tseries.py" not in text


def test_evaluator_does_not_use_t2():
    text = (ROOT / "scripts/eval_t1s_source_specificity.py").read_text(encoding="utf-8")
    assert 'RUN_NAMES["T2-A"]' not in text


def test_static_audit_package_only(tmp_path):
    out = tmp_path / "audit.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/audit_t1s.py"),
            "--phase", "static",
            "--out", str(out),
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    obj = json.loads(out.read_text(encoding="utf-8"))
    assert obj["schema"] == "step4-t1s-preexecution-audit-v1"
    assert obj["phase"] == "static"
    assert obj["all_passed"] is True
    assert obj["failed"] == []
