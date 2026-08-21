from __future__ import annotations

import ast
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal.t1gr_u6_core import (  # noqa: E402
    ARMS,
    ARM_POLICIES,
    CHANNEL_SEMANTICS,
    SCHEMA_RESULTS,
    SOURCE_FILES,
    encode_depth_array,
    implementation_source_hashes,
    launch_rows,
    payload_ok,
    summarize_results,
    validate_spec,
)
from multimodal.t1gr_g_core import SEEDS  # noqa: E402
from multimodal.t1gr_secure_io import assert_public_safe  # noqa: E402


class T1GRU6ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads((ROOT / "config/t1gr_u6_design.frozen.json").read_text(encoding="utf-8"))

    def test_frozen_spec(self):
        info = validate_spec(self.spec)
        self.assertTrue(payload_ok(self.spec))
        self.assertEqual(info["run_count"], 12)
        self.assertFalse(self.spec["authority"]["final_holdout_open_authorized"])
        self.assertFalse(self.spec["authority"]["legacy_primary_g_suite_mutation_authorized"])
        self.assertTrue(self.spec["upstream"]["resplit_forbidden"])

    def test_arm_semantics(self):
        self.assertEqual(ARMS, ("G0-N", "G1-P", "G2-S", "G3-D"))
        self.assertEqual(ARM_POLICIES["G0-N"]["train_ir"], "ZERO_IR")
        self.assertEqual(ARM_POLICIES["G1-P"]["train_ir"], "CORRECT_PAIRED_IR")
        self.assertEqual(ARM_POLICIES["G2-S"]["train_ir"], "BALANCED_FULLY_WRONG_IR")
        self.assertTrue(ARM_POLICIES["G3-D"]["depth"])

    def test_legacy_g3_not_mislabeled(self):
        ruling = self.spec["legacy_g3_adjudication"]
        self.assertFalse(ruling["old_g3_fe_merged"])
        self.assertEqual(ruling["new_g3_id"], "G3-D")
        self.assertIn("NOT_REPRESENTABLE", ruling["reason"])

    def test_channel_contract(self):
        self.assertEqual(tuple(self.spec["channel_contract"]["model_after_format"]), CHANNEL_SEMANTICS)
        self.assertEqual(self.spec["model"]["physical_first_conv_in_channels"], 6)
        self.assertEqual(
            self.spec["model"]["first_conv_initialization"],
            "[W_R,W_G,W_B,0,0,0] from the same seeded E5-v2 reference",
        )

    def test_launch_matrix_is_honestly_incomplete(self):
        rows = launch_rows()
        self.assertEqual(len(rows), 12)
        self.assertEqual(
            {(row["seed"], row["arm"]) for row in rows},
            {(int(seed), arm) for seed in SEEDS for arm in ARMS},
        )
        positions = {arm: [] for arm in ARMS}
        for seed in SEEDS:
            lane = [row for row in rows if row["seed"] == seed]
            self.assertEqual({row["lane_position"] for row in lane}, {0, 1, 2, 3})
            for row in lane:
                positions[row["arm"]].append(row["lane_position"])
        self.assertTrue(all(len(set(value)) == 3 for value in positions.values()))
        self.assertFalse(self.spec["launch_design"]["full_four_position_balance_possible_with_three_seeds"])

    def test_metric_png_encoding(self):
        raw = np.array([[0, 299, 300], [1000, 19999, 20000]], dtype=np.uint16)
        depth, mask, kind = encode_depth_array(raw, ".png")
        self.assertEqual(kind, "METRIC_UINT16_PNG")
        self.assertEqual(depth.dtype, np.uint8)
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(set(np.unique(mask)), {0, 255})
        self.assertEqual(mask[0, 2], 255)
        self.assertEqual(depth[0, 2], 0)
        self.assertEqual(depth[1, 1], 255)
        self.assertTrue(np.all(depth[mask == 0] == 0))

    def test_unknown_jpg_is_missing(self):
        raw = np.full((7, 9, 3), 123, dtype=np.uint8)
        depth, mask, kind = encode_depth_array(raw, ".jpg")
        self.assertEqual(kind, "UNKNOWN_SCALE_JPG_QUARANTINED")
        self.assertEqual(int(np.count_nonzero(depth)), 0)
        self.assertEqual(int(np.count_nonzero(mask)), 0)

    def test_no_fake_depth_conversion(self):
        with self.assertRaises(ValueError):
            encode_depth_array(np.ones((4, 4), dtype=np.uint8), ".png")
        with self.assertRaises(ValueError):
            encode_depth_array(np.ones((4, 4), dtype=np.uint16), ".bmp")

    def _results(self, *, ir_gain: float, depth_gain: float, g2_native: float = 0.305) -> dict:
        rows = []
        for index, seed in enumerate(SEEDS):
            jitter = 0.002 * index
            g0 = 0.30 + jitter
            g1 = g0 + ir_gain
            g3 = g1 + depth_gain
            values = {
                "G0-N": (g0, g0, g0),
                "G1-P": (g1, g1 - max(ir_gain, 0.0) * 0.6, g1),
                "G2-S": (g2_native + jitter, g0 + 0.001, g2_native + jitter),
                "G3-D": (g3, g1 + depth_gain * 0.2, g1 + depth_gain * 0.2),
            }
            for arm in ARMS:
                native, rgb_only, paired = values[arm]
                lofo = None
                if arm in {"G0-N", "G1-P", "G3-D"}:
                    base = {"G0-N": g0, "G1-P": g1, "G3-D": g3}[arm]
                    lofo = {f"fold_{fold}": base - 0.001 * fold for fold in range(5)}
                domains = None
                if arm in {"G1-P", "G3-D"}:
                    domains = {"metric_png": native, "unknown_jpg": rgb_only}
                rows.append({
                    "seed": int(seed),
                    "arm": arm,
                    "native_map50_95": native,
                    "rgb_only_map50_95": rgb_only,
                    "paired_ir_zero_depth_map50_95": paired,
                    "wrong_ir_zero_depth_map50_95": g0 - 0.002 if arm in {"G1-P", "G2-S"} else None,
                    "lofo_native_map50_95": lofo,
                    "depth_domain_native_map50_95": domains,
                })
        return {"schema": SCHEMA_RESULTS, "rows": rows, "final_holdout_accessed": False}

    def test_positive_depth_selector(self):
        cross, summary = summarize_results(self._results(ir_gain=0.02, depth_gain=0.02))
        self.assertEqual(summary["competition_recommendation"], "G3-D")
        self.assertTrue(summary["ir_eligible"])
        self.assertTrue(summary["depth_eligible"])
        self.assertEqual(cross["operational_rules"]["depth_eligibility"]["positive_lofo_count"], 15)
        self.assertFalse(cross["final_holdout_accessed"])
        assert_public_safe(cross)
        assert_public_safe(summary)

    def test_ir_fallback_selector(self):
        _, summary = summarize_results(self._results(ir_gain=0.02, depth_gain=-0.01))
        self.assertEqual(summary["competition_recommendation"], "G1-P")
        self.assertTrue(summary["ir_eligible"])
        self.assertFalse(summary["depth_eligible"])

    def test_rgb_fallback_selector(self):
        _, summary = summarize_results(self._results(ir_gain=-0.01, depth_gain=-0.01))
        self.assertEqual(summary["competition_recommendation"], "G0-N")

    def test_wrong_ir_control_warning(self):
        _, summary = summarize_results(self._results(ir_gain=0.01, depth_gain=-0.01, g2_native=0.36))
        self.assertTrue(summary["wrong_ir_control_outperforms_recommendation_mean"])
        self.assertTrue(summary["manual_review_required"])

    def test_source_closure(self):
        hashes = implementation_source_hashes(ROOT)
        self.assertEqual(set(hashes), set(SOURCE_FILES))
        self.assertTrue(all(len(value) == 64 for value in hashes.values()))

    def test_all_extension_python_parses(self):
        for rel in SOURCE_FILES:
            if rel.endswith(".py"):
                ast.parse((ROOT / rel).read_text(encoding="utf-8"), filename=rel)

    def test_model_is_e5_descendant_not_old_p5(self):
        source = (ROOT / "src/multimodal/t1gr_u6_model.py").read_text(encoding="utf-8")
        self.assertIn("build_seeded_model", source)
        self.assertIn("expand_first_conv_to_six", source)
        self.assertNotIn("TSeriesP5Model", source)
        self.assertNotIn("t1gr_g_model", source)

    def test_four_arm_identity_rejects_g3_drift(self):
        path = ROOT / "src/multimodal/t1gr_u6_model.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "assert_same_seed_arm_identity"
        )
        namespace = {"ARMS": ARMS}
        module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
        exec(compile(module, filename=str(path), mode="exec"), namespace)
        check = namespace["assert_same_seed_arm_identity"]
        common = {
            "seed": int(SEEDS[0]),
            "model_class": "DetectionModel",
            "reference_initial_state_sha256": "a" * 64,
            "complete_initial_state_sha256": "b" * 64,
            "state_dict_keys_sha256": "c" * 64,
            "trainable_parameter_count": 1,
            "total_parameter_count": 1,
            "stem": {"physical_in_channels": 6},
            "physical_head_nc": 12,
            "end2end": True,
            "loss_class": "E2ELoss",
        }
        rows = [{**common, "arm": arm} for arm in ARMS]
        self.assertTrue(check(rows)["all_identity_fields_equal"])
        rows[-1] = {**rows[-1], "complete_initial_state_sha256": "d" * 64}
        with self.assertRaisesRegex(RuntimeError, "T1GR_U6_INITIAL_IDENTITY_DRIFT"):
            check(rows)

    def test_dataset_geometry_and_io_policy(self):
        source = (ROOT / "src/multimodal/t1gr_u6_dataset.py").read_text(encoding="utf-8")
        self.assertIn("resize_depth_valid", source)
        self.assertIn("warp_depth_valid", source)
        self.assertIn("should_decode = enabled and expected_kind ==", source)
        self.assertIn("fast_source_for_recipient", source)

    def test_runner_enforces_lane_predecessors(self):
        source = (ROOT / "scripts/t1gr_u6_server_run_one.py").read_text(encoding="utf-8")
        self.assertIn("_assert_lane_predecessors", source)
        self.assertIn("T1GR_U6_LANE_PREDECESSOR_NOT_COMPLETE", source)

    def test_evaluator_freezes_complete_matrix(self):
        source = (ROOT / "scripts/t1gr_u6_server_eval.py").read_text(encoding="utf-8")
        self.assertIn('"expected_evaluation_count": 90', source)
        self.assertIn("WRONG_DIAGNOSTIC_ARMS", source)
        self.assertIn("LOFO_ARMS", source)

    def test_legacy_primary_g_files_not_in_overlay_closure(self):
        self.assertFalse(any(rel.startswith("src/multimodal/t1gr_g_") for rel in SOURCE_FILES))
        self.assertFalse(any(rel.startswith("scripts/t1gr_g_") for rel in SOURCE_FILES))


if __name__ == "__main__":
    unittest.main()
