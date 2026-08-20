from __future__ import annotations

import json
import sys
import unittest
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal.t1gr_g_core import (  # noqa: E402
    ARMS, SEEDS, balanced_component_folds, balanced_wrong_map,
    contrast_label, payload_sha256, schedule_summary, summarize_results,
    validate_design, verify_epoch_map,
)


def fake_results(g0, g1, g2, *, fold_mode="same"):
    values = {"G0-N": g0, "G1-P": g1, "G2-S": g2}
    rows = []
    for arm in ARMS:
        for seed in SEEDS:
            v = float(values[arm][seed])
            folds = {f"fold_{i}": v for i in range(5)}
            if fold_mode == "g1_unstable" and arm == "G1-P" and seed == SEEDS[0]:
                folds["fold_0"] = float(g0[seed]) - 0.01
                folds["fold_1"] = float(g0[seed]) - 0.01
            rows.append({
                "seed": seed,
                "arm": arm,
                "dev_map50_95": v,
                "lofo_map50_95": folds,
                "run_manifest_sha256": "a" * 64,
                "last_checkpoint_sha256": "b" * 64,
            })
    return {
        "schema": "t1gr-g-per-seed-results-v1",
        "final_holdout_accessed": False,
        "metric": "mAP50-95",
        "checkpoint": "last.pt",
        "max_det": 100,
        "rows": rows,
    }


class TestT1GRG(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.design_path = ROOT / "config/t1gr_g_design.frozen.json"
        cls.design = json.loads(cls.design_path.read_text(encoding="utf-8"))

    def test_design_is_valid_and_self_hashed(self):
        info = validate_design(self.design)
        self.assertEqual(info["n_runs"], 9)
        self.assertEqual(self.design["payload_sha256"], payload_sha256(self.design))

    def test_authority_is_fail_closed(self):
        auth = self.design["authority"]
        self.assertFalse(auth["smoke_training_authorized"])
        self.assertFalse(auth["multiseed_training_authorized"])
        self.assertFalse(auth["final_holdout_open_authorized"])

    def test_design_drift_fails(self):
        changed = deepcopy(self.design)
        changed["training"]["optimizer"] = "AdamW"
        changed["payload_sha256"] = payload_sha256(changed)
        with self.assertRaisesRegex(ValueError, "E5_RECIPE_DRIFT"):
            validate_design(changed)

    def test_balanced_wrong_mapping(self):
        ids = [f"id_{i:04d}" for i in range(101)]
        for seed in SEEDS:
            for epoch in (0, 1, 79):
                mapping = balanced_wrong_map(ids, seed, epoch)
                check = verify_epoch_map(ids, seed, epoch, mapping)
                self.assertTrue(check["passed"])
                self.assertEqual(check["self_matches"], 0)
                self.assertEqual(check["donor_min_uses"], 1)
                self.assertEqual(check["donor_max_uses"], 1)

    def test_formal_size_schedule_summary(self):
        ids = [f"formal_{i:04d}" for i in range(1504)]
        row = schedule_summary(ids, SEEDS[0], 80)
        self.assertTrue(row["passed"])
        self.assertEqual(row["self_pair_count"], 0)
        self.assertEqual(row["recipient_distinct_donor_min"], 80)
        self.assertEqual(row["recipient_distinct_donor_max"], 80)

    def test_seed_changes_wrong_schedule(self):
        ids = [f"id_{i}" for i in range(20)]
        self.assertNotEqual(
            balanced_wrong_map(ids, SEEDS[0], 0),
            balanced_wrong_map(ids, SEEDS[1], 0),
        )

    def test_component_folds_do_not_split_components(self):
        component_by_id = {f"id_{i}": f"component_{i // 2}" for i in range(20)}
        folds = balanced_component_folds(component_by_id)
        for i in range(0, 20, 2):
            self.assertEqual(folds[f"id_{i}"], folds[f"id_{i + 1}"])
        self.assertEqual(set(folds.values()), set(range(5)))

    def test_contrast_labels(self):
        base = {seed: 0.30 for seed in SEEDS}
        positive = {seed: 0.31 for seed in SEEDS}
        self.assertEqual(contrast_label(positive, base)["label"], "STABLE_POSITIVE")
        mixed = dict(positive)
        mixed[SEEDS[0]] = 0.29
        self.assertEqual(contrast_label(mixed, base)["label"], "MIXED")

    def test_supported_branch(self):
        g0 = {seed: 0.30 for seed in SEEDS}
        g2 = {seed: 0.31 for seed in SEEDS}
        g1 = {seed: 0.33 for seed in SEEDS}
        _, summary = summarize_results(fake_results(g0, g1, g2))
        self.assertEqual(summary["decision"], "PAIRED_TRAINING_GENERALIZATION_SUPPORTED")
        self.assertFalse(summary["final_holdout_open_authorized"])

    def test_fold_instability_blocks_supported_branch(self):
        g0 = {seed: 0.30 for seed in SEEDS}
        g2 = {seed: 0.31 for seed in SEEDS}
        g1 = {seed: 0.33 for seed in SEEDS}
        _, summary = summarize_results(fake_results(g0, g1, g2, fold_mode="g1_unstable"))
        self.assertEqual(summary["decision"], "INCONCLUSIVE_REPLICATION")

    def test_generic_branch(self):
        g0 = {seed: 0.30 for seed in SEEDS}
        g1 = {SEEDS[0]: 0.32, SEEDS[1]: 0.33, SEEDS[2]: 0.32}
        g2 = {SEEDS[0]: 0.33, SEEDS[1]: 0.32, SEEDS[2]: 0.32}
        _, summary = summarize_results(fake_results(g0, g1, g2))
        self.assertEqual(
            summary["decision"],
            "GENERIC_TRAINING_BENEFIT_SOURCE_IDENTITY_NOT_ESTABLISHED",
        )

    def test_wrong_or_tie_fails_specificity(self):
        g0 = {seed: 0.30 for seed in SEEDS}
        g1 = {seed: 0.32 for seed in SEEDS}
        g2 = {seed: 0.33 for seed in SEEDS}
        _, summary = summarize_results(fake_results(g0, g1, g2))
        self.assertEqual(summary["decision"], "PAIRED_SOURCE_SPECIFICITY_FAILED")

    def test_no_treatment_gain_is_no_transfer(self):
        g0 = {seed: 0.33 for seed in SEEDS}
        g1 = {seed: 0.31 for seed in SEEDS}
        g2 = {seed: 0.32 for seed in SEEDS}
        _, summary = summarize_results(fake_results(g0, g1, g2))
        self.assertEqual(summary["decision"], "SMALL_SAMPLE_SIGNAL_DID_NOT_TRANSFER")

    def test_holdout_claim_is_rejected(self):
        g0 = {seed: 0.30 for seed in SEEDS}
        g1 = {seed: 0.32 for seed in SEEDS}
        g2 = {seed: 0.31 for seed in SEEDS}
        results = fake_results(g0, g1, g2)
        results["final_holdout_accessed"] = True
        with self.assertRaisesRegex(ValueError, "HOLDOUT_ACCESS_CLAIM"):
            summarize_results(results)


if __name__ == "__main__":
    unittest.main()
