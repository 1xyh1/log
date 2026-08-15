"""Step 4-F0 closeout adversarial tests (reviewer-mandated, torch-free).

Attacks covered:
  * LOO payload tampering (deltas edited while provenance SHA stays legal)
  * 79-row / epoch-gap / false-flag / byte-divergent G8 traces
  * stale dependency SHAs in LOO provenance
  * invalid shuffle maps (non-bijective / self-match / missing keys / IR!=D)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal.causality_interventions import bijective_derangement  # noqa: E402
from multimodal.run_integrity import sha256_file  # noqa: E402
from multimodal.step4_closeout import (  # noqa: E402
    DEPENDENCY_SOURCES, LOO_SCHEMA, compute_deltas, g8_check,
    load_validated_shuffle_maps, loo_provenance_check, validate_loo_payload)

VAL_IDS = ["000001", "000002", "000003", "000004", "000005", "000006"]
GROUPS = ("C0", "IR", "D")
VARIANTS = ("NORMAL", "ZERO-AUX", "SHUFFLE")


# ---------------------------------------------------------------------------
# LOO payload fixtures
# ---------------------------------------------------------------------------

def _make_valid_loo(val_ids=VAL_IDS):
    folds = {}
    for fk in ["full", *val_ids]:
        folds[fk] = {
            "C0": {"NORMAL": 0.30, "ZERO-AUX": 0.30, "SHUFFLE": 0.30,
                   "copy_of_normal": True},
            "IR": {"NORMAL": 0.31, "ZERO-AUX": 0.28, "SHUFFLE": 0.30,
                   "copy_of_normal": False},
            "D": {"NORMAL": 0.29, "ZERO-AUX": 0.28, "SHUFFLE": 0.30,
                   "copy_of_normal": False},
        }
    return {"schema": LOO_SCHEMA, "checkpoint": "last.pt",
            "val_ids": list(val_ids), "folds": folds,
            "deltas": compute_deltas(folds, list(val_ids))}


class TestValidateLooPayload:
    def test_valid_payload_passes(self):
        res = validate_loo_payload(_make_valid_loo())
        assert res["passed"], res["errors"]

    def test_hand_computed_semantics(self):
        """Independent small-integer fixture pins the recompute semantics."""
        folds = {
            "full": {"C0": {"NORMAL": 0.20, "ZERO-AUX": 0.20, "SHUFFLE": 0.20,
                            "copy_of_normal": True},
                     "IR": {"NORMAL": 0.30, "ZERO-AUX": 0.28, "SHUFFLE": 0.29,
                            "copy_of_normal": False}},
            "000001": {"C0": {"NORMAL": 0.10, "ZERO-AUX": 0.10, "SHUFFLE": 0.10,
                              "copy_of_normal": True},
                       "IR": {"NORMAL": 0.20, "ZERO-AUX": 0.18, "SHUFFLE": 0.19,
                              "copy_of_normal": False}},
            "000002": {"C0": {"NORMAL": 0.30, "ZERO-AUX": 0.30, "SHUFFLE": 0.30,
                              "copy_of_normal": True},
                       "IR": {"NORMAL": 0.40, "ZERO-AUX": 0.38, "SHUFFLE": 0.39,
                              "copy_of_normal": False}},
        }
        loo = {"schema": LOO_SCHEMA, "checkpoint": "last.pt",
               "val_ids": ["000001", "000002"], "folds": folds,
               "deltas": compute_deltas(folds, ["000001", "000002"])}
        assert validate_loo_payload(loo)["passed"]
        d = loo["deltas"]["IR_minus_C0"]
        assert d["per_fold"] == {"000001": 0.10, "000002": 0.10}
        assert d["full"] == 0.10
        assert d["positive_folds"] == 2 and d["n_folds"] == 2
        assert d["median"] == 0.10 and d["min"] == 0.10 and d["max"] == 0.10

    def _tampered(self, mutate):
        loo = _make_valid_loo()
        mutate(loo)
        return loo

    @pytest.mark.parametrize("name", [
        "per_fold_value", "positive_folds_only", "full_value", "deleted_fold",
        "renamed_fold_key", "extra_delta_key", "nan_inject", "inf_inject",
        "c0_copy_flag", "c0_variants_unequal"])
    def test_tampering_rejected(self, name):
        def mutate(loo):
            d = loo["deltas"]["IR_minus_C0"]
            if name == "per_fold_value":
                f0 = loo["val_ids"][0]
                d["per_fold"][f0] += 0.001
            elif name == "positive_folds_only":
                d["positive_folds"] += 1
            elif name == "full_value":
                d["full"] += 0.001
            elif name == "deleted_fold":
                loo["folds"].pop(loo["val_ids"][0])
            elif name == "renamed_fold_key":
                loo["folds"]["hacked"] = loo["folds"].pop(loo["val_ids"][0])
            elif name == "extra_delta_key":
                loo["deltas"]["HACKED_minus_C0"] = dict(d)
            elif name == "nan_inject":
                loo["folds"]["full"]["IR"]["NORMAL"] = float("nan")
            elif name == "inf_inject":
                loo["folds"]["full"]["IR"]["NORMAL"] = float("inf")
            elif name == "c0_copy_flag":
                loo["folds"]["full"]["C0"]["copy_of_normal"] = False
            elif name == "c0_variants_unequal":
                loo["folds"]["full"]["C0"]["ZERO-AUX"] = 0.25

        res = validate_loo_payload(self._tampered(mutate))
        assert not res["passed"], f"{name} must be rejected, got pass"

    def test_consistent_tampering_passes_this_layer(self):
        """Fold AND deltas edited together passes payload validation by design;
        that attack surface is closed one layer up by loo_file_sha256."""
        loo = _make_valid_loo()
        loo["folds"]["full"]["IR"]["NORMAL"] = 0.40
        loo["deltas"] = compute_deltas(loo["folds"], loo["val_ids"])
        assert validate_loo_payload(loo)["passed"]


# ---------------------------------------------------------------------------
# G8 trace fixtures
# ---------------------------------------------------------------------------

def _g8_row(e, *, exp_order=None, act_order=None, exp_flip=None, act_flip=None,
            flag=True, epoch=None):
    o = f"order{e:064x}"
    f = f"flip{e:064x}"
    return {"epoch": e if epoch is None else epoch, "n_samples": 11,
            "sample_order_sha256": o, "flip_schedule_sha256": f,
            "expected_order_sha256": o if exp_order is None else exp_order,
            "actual_order_sha256": o if act_order is None else act_order,
            "expected_flip_sha256": f if exp_flip is None else exp_flip,
            "actual_flip_sha256": f if act_flip is None else act_flip,
            "actual_matches_expected": flag, "batch": 4}


def _write_trace(rd: Path, rows):
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "step4_g8_trace.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _make_run_dirs(tmp_path, rows):
    dirs = {g: tmp_path / g for g in GROUPS}
    for rd in dirs.values():
        _write_trace(rd, rows)
    return dirs


class TestG8Check:
    N_EPOCHS = 80

    def _rows(self, n=None):
        return [_g8_row(e) for e in range(n or self.N_EPOCHS)]

    def test_valid_traces_pass(self, tmp_path):
        dirs = _make_run_dirs(tmp_path, self._rows())
        res = g8_check(dirs, self.N_EPOCHS)
        assert res["passed"], res

    def test_79_rows_rejected(self, tmp_path):
        dirs = _make_run_dirs(tmp_path, self._rows(79))
        res = g8_check(dirs, self.N_EPOCHS)
        assert not res["passed"]
        assert any("ROW_COUNT_MISMATCH" in e for e in res["errors"])

    def test_epoch_gap_rejected(self, tmp_path):
        rows = self._rows()
        rows[40]["epoch"] = 41  # duplicate, breaks positional continuity
        dirs = _make_run_dirs(tmp_path, rows)
        res = g8_check(dirs, self.N_EPOCHS)
        assert not res["passed"]
        assert res["epoch_position_errors"]

    def test_expected_actual_mismatch_rejected(self, tmp_path):
        rows = self._rows()
        rows[10]["actual_order_sha256"] = "f" * 64
        dirs = _make_run_dirs(tmp_path, rows)
        res = g8_check(dirs, self.N_EPOCHS)
        assert not res["passed"]
        assert res["expected_actual_mismatch_rows"]

    def test_false_flag_rejected(self, tmp_path):
        rows = self._rows()
        rows[10]["actual_matches_expected"] = False
        dirs = _make_run_dirs(tmp_path, rows)
        res = g8_check(dirs, self.N_EPOCHS)
        assert not res["passed"]
        assert res["flag_false_rows"]

    def test_byte_divergent_trace_rejected(self, tmp_path):
        dirs = _make_run_dirs(tmp_path, self._rows())
        # same semantic rows, one extra blank line -> bytes differ
        fp = dirs["IR"] / "step4_g8_trace.jsonl"
        fp.write_text(fp.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        res = g8_check(dirs, self.N_EPOCHS)
        assert not res["passed"]
        assert not res["trace_files_byte_identical"]

    def test_missing_trace_rejected(self, tmp_path):
        dirs = _make_run_dirs(tmp_path, self._rows())
        (dirs["D"] / "step4_g8_trace.jsonl").unlink()
        res = g8_check(dirs, self.N_EPOCHS)
        assert not res["passed"]
        assert any("TRACE_MISSING" in e for e in res["errors"])


# ---------------------------------------------------------------------------
# LOO provenance fixtures
# ---------------------------------------------------------------------------

def _make_loo_prov_fixture(tmp_path, *, break_key=None, break_groups=False):
    dirs = {}
    for tag in GROUPS:
        rd = tmp_path / tag
        (rd / "weights").mkdir(parents=True)
        (rd / "weights" / "last.pt").write_bytes(b"placeholder-weights")
        (rd / "shuffle_map_val.json").write_text("{}", encoding="utf-8")
        dirs[tag] = rd
    deps = {name: tmp_path / f"dep_{name}.py" for name in DEPENDENCY_SOURCES}
    for fp in deps.values():
        fp.write_text(f"# stub {fp.name}\n", encoding="utf-8")
    contract = tmp_path / "contract.json"
    contract.write_text("{}", encoding="utf-8")
    loo_script = tmp_path / "step4_loo.py"
    loo_script.write_text("# stub\n", encoding="utf-8")

    prov = {f"{tag}_last_pt_sha256":
            sha256_file(dirs[tag] / "weights" / "last.pt") for tag in GROUPS}
    prov["contract_sha256"] = sha256_file(contract)
    prov["loo_source_sha256"] = sha256_file(loo_script)
    prov.update({f"dep_{n}_sha256": sha256_file(fp) for n, fp in deps.items()})
    prov["ir_shuffle_map_val_sha256"] = sha256_file(
        dirs["IR"] / "shuffle_map_val.json")
    prov["d_shuffle_map_val_sha256"] = sha256_file(
        dirs["D"] / "shuffle_map_val.json")

    loo = {"schema": LOO_SCHEMA, "checkpoint": "last.pt", "provenance": prov,
           "groups": {tag: str(rd) for tag, rd in dirs.items()}}
    if break_key:
        loo["provenance"][break_key] = "0" * 64
    if break_groups:
        loo["groups"]["IR"] = str(tmp_path / "elsewhere")
    evals = {tag: {"provenance": {"last_pt_sha256": prov[f"{tag}_last_pt_sha256"]}}
             for tag in GROUPS}
    return loo, dirs, contract, loo_script, deps, evals


class TestLooProvenanceCheck:
    def test_valid_fixture_passes(self, tmp_path):
        loo, dirs, contract, script, deps, evals = _make_loo_prov_fixture(tmp_path)
        checks = loo_provenance_check(loo, dirs, contract, script, evals,
                                      dep_targets=deps)
        assert all(v["match"] for v in checks.values()), checks

    def test_stale_dependency_sha_rejected(self, tmp_path):
        loo, dirs, contract, script, deps, evals = _make_loo_prov_fixture(
            tmp_path, break_key="dep_step3_eval_utils_sha256")
        checks = loo_provenance_check(loo, dirs, contract, script, evals,
                                      dep_targets=deps)
        assert not checks["dep_step3_eval_utils_sha256"]["match"]

    def test_missing_recorded_sha_rejected(self, tmp_path):
        loo, dirs, contract, script, deps, evals = _make_loo_prov_fixture(tmp_path)
        del loo["provenance"]["contract_sha256"]
        checks = loo_provenance_check(loo, dirs, contract, script, evals,
                                      dep_targets=deps)
        assert not checks["contract_sha256"]["match"]

    def test_groups_path_mismatch_rejected(self, tmp_path):
        loo, dirs, contract, script, deps, evals = _make_loo_prov_fixture(
            tmp_path, break_groups=True)
        checks = loo_provenance_check(loo, dirs, contract, script, evals,
                                      dep_targets=deps)
        assert not checks["groups_paths"]["match"]

    def test_wrong_schema_rejected(self, tmp_path):
        loo, dirs, contract, script, deps, evals = _make_loo_prov_fixture(tmp_path)
        loo["schema"] = "step4-loo-v1"  # legacy LOO must be structurally refused
        checks = loo_provenance_check(loo, dirs, contract, script, evals,
                                      dep_targets=deps)
        assert not checks["schema"]["match"]


# ---------------------------------------------------------------------------
# Shuffle map validation
# ---------------------------------------------------------------------------

def _make_map_dirs(tmp_path, ir_map=None, d_map=None, ids=VAL_IDS):
    dirs = {}
    for tag, m in (("IR", ir_map), ("D", d_map)):
        rd = tmp_path / tag
        rd.mkdir(parents=True, exist_ok=True)
        (rd / "shuffle_map_val.json").write_text(
            json.dumps(m if m is not None else bijective_derangement(ids)),
            encoding="utf-8")
        dirs[tag] = rd
    return dirs


class TestShuffleMaps:
    def test_valid_maps_pass(self, tmp_path):
        dirs = _make_map_dirs(tmp_path)
        maps = load_validated_shuffle_maps(dirs, VAL_IDS)
        assert maps["IR"] == maps["D"]

    def test_non_bijective_rejected(self, tmp_path):
        bad = bijective_derangement(VAL_IDS)
        bad[VAL_IDS[0]] = bad[VAL_IDS[1]]  # duplicate value
        dirs = _make_map_dirs(tmp_path, ir_map=bad, d_map=bad)
        with pytest.raises(RuntimeError, match="INVALID_SHUFFLE_MAP"):
            load_validated_shuffle_maps(dirs, VAL_IDS)

    def test_self_match_rejected(self, tmp_path):
        bad = bijective_derangement(VAL_IDS)
        bad[VAL_IDS[0]] = VAL_IDS[0]
        dirs = _make_map_dirs(tmp_path, ir_map=bad, d_map=bad)
        with pytest.raises(RuntimeError, match="INVALID_SHUFFLE_MAP"):
            load_validated_shuffle_maps(dirs, VAL_IDS)

    def test_missing_key_rejected(self, tmp_path):
        bad = bijective_derangement(VAL_IDS)
        bad.pop(VAL_IDS[0])
        dirs = _make_map_dirs(tmp_path, ir_map=bad, d_map=bad)
        with pytest.raises(RuntimeError, match="INVALID_SHUFFLE_MAP"):
            load_validated_shuffle_maps(dirs, VAL_IDS)

    def test_ir_d_divergence_rejected(self, tmp_path):
        dirs = _make_map_dirs(tmp_path, ir_map=bijective_derangement(VAL_IDS),
                              d_map=bijective_derangement(list(reversed(VAL_IDS))))
        with pytest.raises(RuntimeError, match="IR_AND_D_SHUFFLE_MAPS_DIFFER"):
            load_validated_shuffle_maps(dirs, VAL_IDS)
