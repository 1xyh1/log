"""F1-B corruption schedule adversarial tests (torch-free).

Frozen contract under test (reviewer 2026-08-16):
  * clean 0.50 / zero 0.125 / noise 0.125 / blur 0.125 / contrast 0.125
  * severity uniform {0.25, 0.50, 0.75, 1.00}; zero fixed 1.0; shift excluded
  * every decision derives from SHA256(seed|epoch|sample_id|field)
  * noise fields depend on epoch
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal.step4_f1_b_corruption import (  # noqa: E402
    KIND_PROBS, NOISE_SIGMA, SEVERITIES, TRAIN_KINDS, ZERO_SEVERITY,
    apply_schedule_to_plane, sample_schedule, schedule_for_epoch,
    schedule_sha256)

SEED = 20260812
IDS = ["000001", "000002", "000003", "000004", "000005", "000006"]


class TestFrozenSchedule:
    def test_kind_probs_sum_to_one(self):
        total = sum(prob for _, prob in KIND_PROBS)
        assert abs(total - 1.0) < 1e-9

    def test_kind_probs_values(self):
        assert dict(KIND_PROBS) == {"clean": 0.50, "zero": 0.125,
                                    "noise": 0.125, "blur": 0.125,
                                    "contrast": 0.125}

    def test_shift_excluded_from_training(self):
        assert "shift" not in TRAIN_KINDS

    def test_zero_severity_fixed_one(self):
        for epoch in range(3):
            for sid in IDS:
                s = sample_schedule(SEED, epoch, sid)
                if s["kind"] == "zero":
                    assert s["severity"] == 1.0

    def test_severity_uniform_support(self):
        seen = set()
        for epoch in range(50):
            for sid in IDS:
                s = sample_schedule(SEED, epoch, sid)
                if s["kind"] in ("noise", "blur", "contrast"):
                    seen.add(s["severity"])
                    assert s["severity"] in SEVERITIES
        assert seen == set(SEVERITIES)  # all four levels occur


class TestDeterminism:
    def test_same_inputs_same_schedule(self):
        assert sample_schedule(SEED, 3, "000001") == sample_schedule(SEED, 3, "000001")

    def test_epoch_changes_schedule(self):
        rows = [sample_schedule(SEED, e, "000001") for e in range(20)]
        kinds = {r["kind"] for r in rows}
        assert len(kinds) > 1  # over epochs the kind varies

    def test_schedule_sha_stable_and_unique_per_epoch(self):
        shas = [schedule_sha256(SEED, e, IDS) for e in range(5)]
        assert len(set(shas)) == len(shas)
        assert schedule_sha256(SEED, 0, IDS) == schedule_sha256(SEED, 0, IDS)

    def test_schedule_rows_sorted_by_sample_id(self):
        rows = schedule_for_epoch(SEED, 1, IDS)
        assert [r["sample_id"] for r in rows] == sorted(IDS)

    def test_empirical_kind_frequencies_near_preregistered(self):
        counts = {}
        total = 0
        for epoch in range(200):
            for sid in IDS:
                s = sample_schedule(SEED, epoch, sid)
                counts[s["kind"]] = counts.get(s["kind"], 0) + 1
                total += 1
        for kind, prob in KIND_PROBS:
            ratio = counts.get(kind, 0) / total
            # 1200 draws: allow generous tolerance but catch gross drift
            assert abs(ratio - prob) < 0.05, f"{kind}: {ratio} vs {prob}"


class TestApplySemantics:
    def _plane(self):
        rng = np.random.default_rng(0)
        return rng.uniform(0.1, 0.9, size=(32, 32)).astype(np.float32)

    def test_clean_is_identity(self):
        plane = self._plane()
        sched = {"sample_id": "000001", "kind": "clean", "severity": 0.0}
        out = apply_schedule_to_plane(plane, sched, seed=SEED, epoch=0)
        assert np.array_equal(out, plane)

    def test_zero_drops_plane(self):
        plane = self._plane()
        sched = {"sample_id": "000001", "kind": "zero", "severity": 1.0}
        out = apply_schedule_to_plane(plane, sched, seed=SEED, epoch=0)
        assert float(out.max()) == 0.0

    def test_noise_differs_across_epochs(self):
        plane = self._plane()
        sched = {"sample_id": "000001", "kind": "noise", "severity": 0.5}
        out0 = apply_schedule_to_plane(plane, sched, seed=SEED, epoch=0)
        out1 = apply_schedule_to_plane(plane, sched, seed=SEED, epoch=1)
        assert not np.array_equal(out0, out1)  # epoch enters the noise field

    def test_noise_respects_content_mask(self):
        plane = self._plane()
        mask = np.zeros_like(plane, dtype=bool)
        mask[4:12, 4:12] = True
        sched = {"sample_id": "000001", "kind": "noise", "severity": 1.0}
        out = apply_schedule_to_plane(plane, sched, seed=SEED, epoch=0,
                                      content_mask=mask)
        assert np.all(out[~mask] == 0.0)
        assert not np.array_equal(out[mask], plane[mask])

    def test_blur_reduces_gradients(self):
        rng = np.random.default_rng(1)
        plane = rng.normal(0.5, 0.2, size=(32, 32)).astype(np.float32)
        plane = np.clip(plane, 0.0, 1.0)
        sched = {"sample_id": "000001", "kind": "blur", "severity": 1.0}
        out = apply_schedule_to_plane(plane, sched, seed=SEED, epoch=0)
        from numpy import gradient
        g_in = float(np.abs(gradient(plane)).mean())
        g_out = float(np.abs(gradient(out)).mean())
        assert g_out < g_in

    def test_contrast_compresses_toward_median(self):
        plane = self._plane()
        sched = {"sample_id": "000001", "kind": "contrast", "severity": 1.0}
        out = apply_schedule_to_plane(plane, sched, seed=SEED, epoch=0)
        med = float(np.median(plane))
        assert float(np.abs(out - med).max()) < 1e-6

    def test_shift_rejected_in_training(self):
        plane = self._plane()
        sched = {"sample_id": "000001", "kind": "shift", "severity": 0.5}
        with pytest.raises(ValueError):
            apply_schedule_to_plane(plane, sched, seed=SEED, epoch=0)

    def test_output_range_and_dtype(self):
        plane = self._plane()
        for kind in ("zero", "noise", "blur", "contrast"):
            sched = {"sample_id": "000001", "kind": kind, "severity": 0.75}
            out = apply_schedule_to_plane(plane, sched, seed=SEED, epoch=2)
            assert out.dtype == np.float32
            assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
