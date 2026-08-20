from __future__ import annotations

import ast
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
DELIVERABLES = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.append(str(DELIVERABLES / "T1GR_G_design_freeze_bundle_v1" / "src"))

from multimodal.t1gr_g_core import ARMS, SEEDS, balanced_wrong_map, sha256_json  # noqa: E402
from multimodal.t1gr_g_impl_core import (  # noqa: E402
    ZERO_IR,
    fast_source_for_recipient,
    finite_metric,
    parse_component_map,
    payload_ok,
    source_for_recipient,
    source_schedule_index,
    trace_epoch_summary,
    validate_impl_spec,
)


class ImplementationCoreTests(unittest.TestCase):
    def setUp(self):
        self.spec_path = ROOT / "config/t1gr_g_implementation_spec.frozen.json"
        self.spec = json.loads(self.spec_path.read_text(encoding="utf-8"))

    def test_frozen_spec_integrity_and_exact_recipe(self):
        self.assertTrue(payload_ok(self.spec))
        info = validate_impl_spec(self.spec)
        self.assertEqual(info["formal_runs"], 9)
        self.assertEqual(self.spec["training"]["optimizer"], "MuSGD")
        self.assertEqual(self.spec["training"]["epochs"], 80)
        self.assertEqual(self.spec["training"]["batch"], 4)
        self.assertEqual(self.spec["training"]["imgsz"], 640)
        self.assertEqual(self.spec["training"]["lr0"], 0.01)
        self.assertEqual(self.spec["training"]["momentum"], 0.9)
        self.assertNotIn("AdamW", json.dumps(self.spec["training"], sort_keys=True))

    def test_source_rules_and_zero_ir_dev(self):
        ids = tuple(f"x{i:03d}" for i in range(11))
        for arm in ("G0-N", "G1-P"):
            self.assertEqual(
                source_for_recipient(ids, arm=arm, seed=SEEDS[0], epoch=7, recipient=ids[3], split="train"),
                ids[3],
            )
        wrong = source_for_recipient(ids, arm="G2-S", seed=SEEDS[0], epoch=7, recipient=ids[3], split="train")
        self.assertNotEqual(wrong, ids[3])
        for arm in ARMS:
            self.assertEqual(
                source_for_recipient(ids, arm=arm, seed=SEEDS[0], epoch=7, recipient=ids[3], split="dev"),
                ZERO_IR,
            )

    def test_fast_g2_mapping_matches_frozen_reference(self):
        ids = tuple(f"id{i:04d}" for i in range(37))
        for seed in SEEDS:
            ordered, position = source_schedule_index(ids, seed=seed)
            for epoch in (0, 1, 35, 79):
                reference = balanced_wrong_map(ids, seed, epoch)
                actual = {
                    sid: fast_source_for_recipient(ordered, position, epoch=epoch, recipient=sid)
                    for sid in ids
                }
                self.assertEqual(actual, reference)

    def test_runtime_trace_requires_exact_anchor_coverage(self):
        ids = tuple(f"id{i:02d}" for i in range(13))
        mapping = balanced_wrong_map(ids, SEEDS[1], 4)
        rows = [
            {"recipient": sid, "donor": mapping[sid], "role": "anchor", "epoch": 4}
            for sid in ids
        ]
        rows.append({"recipient": ids[0], "donor": mapping[ids[0]], "role": "mosaic_aux", "epoch": 4})
        summary = trace_epoch_summary(rows, ids, arm="G2-S", seed=SEEDS[1], epoch=4)
        self.assertTrue(summary["source_condition_passed"])
        self.assertEqual(summary["anchor_count"], len(ids))
        broken = rows[:-2] + [rows[-1]]
        self.assertFalse(trace_epoch_summary(broken, ids, arm="G2-S", seed=SEEDS[1], epoch=4)["source_condition_passed"])

    def test_component_map_variants_and_metric_validation(self):
        ids = ["a", "b", "c"]
        self.assertEqual(parse_component_map({"component_by_id": {"a": "x", "b": "x", "c": "y"}}, ids)["c"], "y")
        self.assertEqual(parse_component_map({"assignments": [
            {"sample_id": "a", "component_id": "x"},
            {"sample_id": "b", "component_id": "x"},
            {"sample_id": "c", "component_id": "y"},
        ]}, ids)["a"], "x")
        with self.assertRaises(ValueError):
            parse_component_map({"component_by_id": {"a": "x", "b": "x", "c": "y", "extra": "z"}}, ids)
        self.assertEqual(finite_metric(0.3519), 0.3519)
        for bad in (-0.1, 1.1, float("inf"), True):
            with self.assertRaises(ValueError):
                finite_metric(bad)


class StaticSafetyTests(unittest.TestCase):
    def source(self, relative: str) -> str:
        return (ROOT / relative).read_text(encoding="utf-8")

    def test_joint_four_channel_loader_and_recipient_rng_are_explicit(self):
        source = self.source("src/multimodal/t1gr_g_dataset.py")
        self.assertIn("np.concatenate((visible, ir[:, :, None]), axis=2)", source)
        self.assertIn("T1GR_AUG_V1", source)
        self.assertIn("with scoped_rng(draw_seed)", source)
        self.assertIn("set_albumentations_seed(self.transforms, draw_seed)", source)
        self.assertIn("T1GRVisibleHSV", source)
        self.assertIn("T1GRVisibleAlbumentations", source)
        self.assertIn("T1GRFormat", source)
        self.assertIn("class T1GRLetterBox", source)
        self.assertIn("padded[:, :, 3] = 0", source)

    def test_epoch_worker_reset_is_synchronous(self):
        source = self.source("src/multimodal/t1gr_g_runtime.py")
        self.assertIn("old._shutdown_workers()", source)
        self.assertIn("reset_for_epoch", source)
        self.assertIn("class RecipientEpochSampler", source)
        self.assertIn("T1GR_ORDER_V1", source)
        self.assertIn("workers=expected_workers", source)
        self.assertIn("split = \"train\" if mode == \"train\" else \"dev\"", source)

    def test_model_uses_e2e_loss_and_trainable_rgb(self):
        source = self.source("src/multimodal/t1gr_g_model.py")
        self.assertIn("E2ELoss(self)", source)
        self.assertIn("freeze_rgb_backbone=False", source)
        self.assertIn("nn.Module.train(self, mode)", source)
        self.assertIn("zeros_like(infrared)", source)
        self.assertNotIn("repeat(1, 2", source)
        self.assertNotIn("freeze_rgb_backbone=True", source)

    def test_formal_entry_requires_smoke_authority_and_suite_state(self):
        run_one = self.source("scripts/t1gr_g_run_one.py")
        suite = self.source("src/multimodal/t1gr_g_suite.py")
        self.assertIn("multiseed_training_authorized", run_one)
        self.assertIn("T1GR_G_SUITE_REQUEST_OUT_OF_ORDER", run_one)
        self.assertIn("subprocess.run(command", suite)
        self.assertNotIn("shell=True", suite)
        self.assertIn("selective_rerun_authorized", self.source("scripts/t1gr_g_smoke_audit.py"))

    def test_eval_is_last_primary_zero_ir_dev_only(self):
        source = self.source("scripts/t1gr_g_eval_suite.py")
        self.assertIn('"checkpoint": "last.pt"', source)
        self.assertIn('"inference_ir": "ZERO_IR"', source)
        self.assertIn('"final_holdout_accessed": False', source)
        self.assertIn("balanced_component_folds", source)
        self.assertNotIn("best.pt", source)

    def test_all_python_files_parse(self):
        for path in sorted(ROOT.rglob("*.py")):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_frozen_seed_arm_matrix(self):
        self.assertEqual(len(ARMS) * len(SEEDS), 9)
        self.assertEqual(sha256_json(list(SEEDS)), sha256_json([20260812, 20260813, 20260814]))


if __name__ == "__main__":
    unittest.main()
