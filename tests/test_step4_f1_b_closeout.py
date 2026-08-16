"""F1-B closeout adversarial tests (torch-free).

The reviewer's attack: edit one record's kind (noise -> blur), recompute the
records SHA and kind_counts to stay internally consistent, and the old
rejudge_g9() still passed.  The hardened version must detect it via the
per-sample sample_schedule comparison and the recomputed actual-schedule SHA.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from multimodal.step4_f1_b_corruption import (  # noqa: E402
    sample_schedule, schedule_sha256)

from summarize_step4_f1_b import _sha_json, rejudge_g9  # noqa: E402

SEED = 20260812
IDS = ["000001", "000002", "000003", "000004", "000005", "000006",
       "000007", "000008", "000009", "000010", "000011"]
GROUPS = {"C0": "zero", "FIXED": "ir", "SOFT": "ir"}


def _make_records(epoch: int, tag: str = "SOFT") -> list[dict]:
    rows = []
    for sid in IDS:
        s = sample_schedule(SEED, epoch, sid)
        if tag == "C0":
            # corruption not applied for C0: IR bytes never change
            after = "a" * 64
        else:
            after = "a" * 64 if s["kind"] == "clean" else "b" * 64
        rows.append({
            "epoch": epoch,
            "sample_id": sid,
            "kind": s["kind"],
            "severity": s["severity"],
            "ir_sha_before": "a" * 64,
            "ir_sha_after": after,
            "rgb_unchanged": True,
            "depth_unchanged": True,
            "labels_bboxes_same_object": True,
        })
    return rows


def _make_fixture(tmp_path, mutate=None, tag="SOFT"):
    rd = tmp_path / tag
    rd.mkdir(parents=True)
    epoch = 0
    records = _make_records(epoch, tag)
    if mutate:
        records = mutate(records)
    rows = [{"sample_id": r["sample_id"], "kind": r["kind"],
             "severity": r["severity"]} for r in records]
    actual_sha = _sha_json(sorted(rows, key=lambda r: r["sample_id"]))
    counts = {}
    for r in records:
        counts[r["kind"]] = counts.get(r["kind"], 0) + 1
    trace_row = {
        "epoch": epoch,
        "n_samples": len(records),
        "expected_schedule_sha256": schedule_sha256(SEED, epoch, IDS),
        "actual_schedule_sha256": actual_sha,
        "expected_matches_actual": True,
        "records_sha256": _sha_json(sorted(records,
                                           key=lambda r: r["sample_id"])),
        "ir_changed_for_corrupted_only": True,
        "rgb_depth_labels_bboxes_unchanged": True,
        "kind_counts": counts,
        "batch": 4,
    }
    (rd / "step4_b1_g9_trace.jsonl").write_text(
        json.dumps(trace_row) + "\n", encoding="utf-8")
    (rd / "step4_b1_g9_records.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    contract = {"train_ids": list(IDS)}
    return {tag: rd}, contract


class TestRejudgeG9:
    def test_valid_fixture_passes(self, tmp_path):
        runs, contract = _make_fixture(tmp_path)
        res = rejudge_g9(runs, 1, SEED, contract)
        assert res["passed"], res["errors"]

    def test_tampered_kind_detected(self, tmp_path):
        """Reviewer attack: noise -> blur with recomputed SHA/counts."""

        def mutate(records):
            for r in records:
                if r["kind"] == "noise":
                    r["kind"] = "blur"
                    r["severity"] = 0.25
                    break
            return records

        runs, contract = _make_fixture(tmp_path, mutate=mutate)
        res = rejudge_g9(runs, 1, SEED, contract)
        assert not res["passed"]
        assert any("G9_SCHEDULE_MISMATCH" in e for e in res["errors"]), res["errors"]

    def test_missing_sample_detected(self, tmp_path):
        runs, contract = _make_fixture(
            tmp_path, mutate=lambda recs: recs[:-1])
        res = rejudge_g9(runs, 1, SEED, contract)
        assert not res["passed"]
        assert any("G9_ID_SET_INCOMPLETE" in e or "G9_RECORDS_COUNT" in e
                   for e in res["errors"])

    def test_duplicate_sample_detected(self, tmp_path):
        def mutate(records):
            records.append(dict(records[0]))
            return records

        runs, contract = _make_fixture(tmp_path, mutate=mutate)
        res = rejudge_g9(runs, 1, SEED, contract)
        assert not res["passed"]
        assert any("G9_ID_DUPLICATES" in e for e in res["errors"])

    def test_c0_ir_changed_detected(self, tmp_path):
        def mutate(records):
            for r in records:
                if r["kind"] == "clean":
                    r["ir_sha_after"] = "c" * 64
                    break
            return records

        runs, contract = _make_fixture(tmp_path, mutate=mutate, tag="C0")
        res = rejudge_g9(runs, 1, SEED, contract)
        assert not res["passed"]
        assert any("G9_C0_IR_CHANGED" in e for e in res["errors"])

    def test_ir_semantics_violation_detected(self, tmp_path):
        def mutate(records):
            for r in records:
                if r["kind"] == "noise":
                    r["ir_sha_after"] = r["ir_sha_before"]  # should differ
                    break
            return records

        runs, contract = _make_fixture(tmp_path, mutate=mutate)
        res = rejudge_g9(runs, 1, SEED, contract)
        assert not res["passed"]
        assert any("G9_IR_SEMANTICS" in e for e in res["errors"])
