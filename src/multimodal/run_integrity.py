"""Step 3 run-integrity checks.

The goal is to prevent a formal run from being evaluated after a smoke/debug run has
silently overwritten part of the directory.  This module intentionally checks the
artifacts that define experiment identity, not model quality.
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonl_count(path: Path) -> int | None:
    if not path.exists():
        return None
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _csv_epoch_info(path: Path) -> tuple[int | None, float | None]:
    if not path.exists():
        return None, None
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return 0, None
    raw = rows[-1].get("epoch")
    try:
        last_epoch = float(raw) if raw not in (None, "") else None
    except ValueError:
        last_epoch = None
    return len(rows), last_epoch


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - runtime dependency in project env
        raise RuntimeError("PyYAML is required to inspect args.yaml") from exc
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    return obj if isinstance(obj, dict) else {}


@dataclass
class RunIntegrityReport:
    run_dir: str
    expected_epochs: int
    passed: bool
    errors: list[str]
    warnings: list[str]
    observed: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_step3_run(
    run_dir: str | Path,
    expected_epochs: int = 80,
    *,
    require_weights: bool = True,
    check_eval_provenance: bool = True,
) -> RunIntegrityReport:
    """Inspect one Step-3 run directory.

    A formal run is coherent when args/results/G8/kernel-growth all agree on the same
    epoch budget.  Existing legacy eval JSON without provenance is a warning, not an
    error; a provenance block that exists but no longer matches current files is an
    error (stale/mixed artifacts).
    """
    p = Path(run_dir)
    errors: list[str] = []
    warnings: list[str] = []
    observed: dict[str, Any] = {"exists": p.exists()}

    if not p.exists():
        errors.append("RUN_DIR_MISSING")
        return RunIntegrityReport(str(p), expected_epochs, False, errors, warnings, observed)

    args = _read_yaml(p / "args.yaml")
    args_epochs = args.get("epochs")
    observed["args_epochs"] = args_epochs
    observed["args_name"] = args.get("name")
    observed["args_exist_ok"] = args.get("exist_ok")
    try:
        if int(args_epochs) != int(expected_epochs):
            errors.append(f"ARGS_EPOCHS_MISMATCH:{args_epochs}!={expected_epochs}")
    except (TypeError, ValueError):
        errors.append("ARGS_EPOCHS_MISSING_OR_INVALID")

    row_count, last_epoch = _csv_epoch_info(p / "results.csv")
    observed["results_rows"] = row_count
    observed["results_last_epoch"] = last_epoch
    if row_count is None:
        errors.append("RESULTS_CSV_MISSING")
    elif row_count != expected_epochs:
        errors.append(f"RESULTS_ROW_COUNT_MISMATCH:{row_count}!={expected_epochs}")

    # Ultralytics writes human-visible epoch numbers 1..N to results.csv in this run.
    if last_epoch is not None and abs(last_epoch - expected_epochs) > 1e-6:
        errors.append(f"RESULTS_LAST_EPOCH_MISMATCH:{last_epoch}!={expected_epochs}")

    g8_count = _jsonl_count(p / "step3_g8_trace.jsonl")
    growth_count = _jsonl_count(p / "step3_kernel_growth.jsonl")
    observed["g8_rows"] = g8_count
    observed["kernel_growth_rows"] = growth_count
    if g8_count is None:
        errors.append("G8_TRACE_MISSING")
    elif g8_count != expected_epochs:
        errors.append(f"G8_ROW_COUNT_MISMATCH:{g8_count}!={expected_epochs}")
    if growth_count is None:
        errors.append("KERNEL_GROWTH_MISSING")
    elif growth_count != expected_epochs:
        errors.append(f"KERNEL_GROWTH_ROW_COUNT_MISMATCH:{growth_count}!={expected_epochs}")

    manifest_path = p / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
            errors.append("MANIFEST_INVALID_JSON")
        observed["manifest_run_kind"] = manifest.get("run_kind")
        observed["manifest_expected_epochs"] = manifest.get("expected_epochs")
        if manifest.get("expected_epochs") is not None and int(manifest["expected_epochs"]) != expected_epochs:
            errors.append("MANIFEST_EPOCHS_MISMATCH")
        if manifest.get("run_kind") is None:
            warnings.append("LEGACY_MANIFEST_WITHOUT_RUN_KIND")
    else:
        warnings.append("MANIFEST_MISSING")

    weights = p / "weights"
    last_pt = weights / "last.pt"
    best_pt = weights / "best.pt"
    observed["last_pt_exists"] = last_pt.exists()
    observed["best_pt_exists"] = best_pt.exists()
    if require_weights:
        if not last_pt.exists():
            errors.append("LAST_PT_MISSING")
        if not best_pt.exists():
            errors.append("BEST_PT_MISSING")

    eval_path = p / "eval_step3_causality.json"
    if check_eval_provenance and eval_path.exists():
        try:
            eval_obj = json.loads(eval_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            errors.append("EVAL_JSON_INVALID")
            eval_obj = {}
        prov = eval_obj.get("provenance") if isinstance(eval_obj, dict) else None
        if not isinstance(prov, dict):
            warnings.append("LEGACY_EVAL_WITHOUT_PROVENANCE")
        else:
            current_hashes = {}
            for key, rel in (
                ("results_sha256", "results.csv"),
                ("args_sha256", "args.yaml"),
                ("last_pt_sha256", "weights/last.pt"),
                ("best_pt_sha256", "weights/best.pt"),
            ):
                fp = p / rel
                current_hashes[key] = sha256_file(fp) if fp.exists() else None
                expected_hash = prov.get(key)
                if expected_hash is not None and expected_hash != current_hashes[key]:
                    errors.append(f"STALE_EVAL_PROVENANCE:{key}")
            observed["eval_provenance_current_hashes"] = current_hashes

    passed = not errors
    return RunIntegrityReport(str(p), expected_epochs, passed, errors, warnings, observed)


def assert_step3_run_integrity(run_dir: str | Path, expected_epochs: int = 80) -> RunIntegrityReport:
    report = inspect_step3_run(run_dir, expected_epochs=expected_epochs)
    if not report.passed:
        raise RuntimeError(
            "STEP3_RUN_INTEGRITY_FAILED: " + "; ".join(report.errors) + f" | run={run_dir}"
        )
    return report
