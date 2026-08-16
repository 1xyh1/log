"""F1-B training-time deterministic IR corruption schedule (frozen 2026-08-16).

Preregistered schedule (reviewer):
    clean     0.50
    zero      0.125   (dropout, severity fixed 1.0)
    noise     0.125   (severity uniform in {0.25, 0.50, 0.75, 1.00})
    blur      0.125   (same severities)
    contrast  0.125   (same severities)
    shift stays EVALUATION-ONLY (A0 scan: best q = 1.0 on all shift levels,
    no evidence it is a "bad IR" that should be suppressed in training).

Randomness contract: every decision derives from digest bytes of
SHA256(f"{seed}|{epoch}|{sample_id}|{field}") — Python's built-in hash() is
FORBIDDEN (it is per-process salted).  Noise fields include epoch, so no two
epochs share the same noise map.  Severity selection and noise fields draw
from disjoint digest byte ranges.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

KIND_PROBS = (("clean", 0.50), ("zero", 0.125), ("noise", 0.125),
              ("blur", 0.125), ("contrast", 0.125))
TRAIN_KINDS = tuple(kind for kind, _ in KIND_PROBS)
SEVERITIES = (0.25, 0.50, 0.75, 1.00)
ZERO_SEVERITY = 1.0
NOISE_SIGMA = 0.20  # matches evaluation corrupt_ir_plane noise magnitude


def _digest(seed: int, epoch: int, sample_id: str, field: str = "schedule") -> bytes:
    payload = f"{int(seed)}|{int(epoch)}|{sample_id}|{field}".encode("utf-8")
    return hashlib.sha256(payload).digest()


def sample_schedule(seed: int, epoch: int, sample_id: str) -> dict:
    """Deterministic (kind, severity) for one sample in one epoch."""
    d = _digest(seed, epoch, sample_id)
    r = int.from_bytes(d[:8], "big") / float(2 ** 64)  # uniform in [0, 1)
    cumulative = 0.0
    chosen = "clean"
    for kind, prob in KIND_PROBS:
        cumulative += prob
        if r < cumulative:
            chosen = kind
            break
    if chosen in ("noise", "blur", "contrast"):
        idx = int.from_bytes(d[8:10], "big") % len(SEVERITIES)
        severity = float(SEVERITIES[idx])
    elif chosen == "zero":
        severity = ZERO_SEVERITY
    else:
        severity = 0.0
    return {"sample_id": sample_id, "kind": chosen, "severity": severity}


def schedule_for_epoch(seed: int, epoch: int, sample_ids) -> list[dict]:
    """Full deterministic per-epoch schedule over the (sorted) id set."""
    return [sample_schedule(seed, epoch, sid) for sid in sorted(sample_ids)]


def schedule_sha256(seed: int, epoch: int, sample_ids) -> str:
    """Canonical SHA anchor for the expected schedule of one epoch (G9).
    Normalization MUST match the runner's actual-schedule serialization:
    rows sorted by sample_id, compact separators, ensure_ascii=False."""
    payload = json.dumps(schedule_for_epoch(seed, epoch, sample_ids),
                         sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def apply_schedule_to_plane(plane: np.ndarray, schedule: dict, *, seed: int,
                            epoch: int, content_mask: np.ndarray | None = None
                            ) -> np.ndarray:
    """Apply the scheduled corruption to one IR plane (HxW float32, value
    semantics identical to evaluation corrupt_ir_plane; only the noise field
    differs: seeded from SHA256(seed|epoch|sample_id|noise))."""
    kind = schedule["kind"]
    severity = float(schedule["severity"])
    if kind not in TRAIN_KINDS:
        raise ValueError(f"training corruption kinds are {TRAIN_KINDS}, got {kind!r}")
    x = np.asarray(plane, dtype=np.float32).copy()
    mask = np.ones(x.shape, dtype=bool) if content_mask is None else np.asarray(
        content_mask, dtype=bool)
    if mask.shape != x.shape:
        raise ValueError("content mask shape mismatch")
    if kind == "clean" or severity == 0.0:
        return x
    if kind == "zero":
        x[mask] = 0.0
    elif kind == "contrast":
        values = x[mask]
        center = float(np.median(values)) if values.size else 0.0
        x[mask] = center + (values - center) * (1.0 - severity)
    elif kind == "noise":
        d = _digest(seed, epoch, schedule["sample_id"], "noise")
        rng = np.random.default_rng(int.from_bytes(d[:8], "big"))
        noise = rng.normal(0.0, NOISE_SIGMA * severity, size=x.shape).astype(np.float32)
        x[mask] += noise[mask]
    elif kind == "blur":
        import cv2
        kernel = max(3, int(round(2 + 20 * severity)))
        kernel += 1 - kernel % 2
        x = cv2.GaussianBlur(x, (kernel, kernel), sigmaX=0)
    x = np.clip(x, 0.0, 1.0).astype(np.float32, copy=False)
    x[~mask] = 0.0
    return x


def sha256_plane(plane: np.ndarray) -> str:
    """Stable SHA of an IR plane's bytes (G9 before/after anchor)."""
    data = np.ascontiguousarray(plane, dtype=np.float32)
    return hashlib.sha256(data.tobytes()).hexdigest()
