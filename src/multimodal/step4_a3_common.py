"""A3 shared evaluation-only helpers.

All A3 interventions obey the same causal order:
recipient paired aux pyramid -> q_native -> freeze q -> projected residual -> intervention.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping, Iterable

import numpy as np
import torch

from multimodal.reliability_gate import broadcast_gate

SCALES = ("P3", "P4", "P5")
TAP_BY_SCALE = {"P3": 4, "P4": 6, "P5": 10}


def tensor_sha256(t: torch.Tensor) -> str:
    x = t.detach().cpu().contiguous()
    return hashlib.sha256(x.numpy().tobytes()).hexdigest()


def state_sha256(model) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode("utf-8"))
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def git_blob_sha1(path) -> str:
    """Git blob SHA1 after canonical CRLF->LF normalization for cross-platform mirror checks."""
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def _forward_tail(model, y, x):
    for m in model.tail:
        if m.f != -1:
            x = y[m.f] if isinstance(m.f, int) else [
                x if j == -1 else y[j] for j in m.f
            ]
        x = m(x)
        if m.i in model.save:
            y[m.i] = x
    return x


@dataclass
class RecipientFeatures:
    q_native: torch.Tensor
    rgb: dict[str, torch.Tensor]
    aux: dict[str, torch.Tensor]
    residual: dict[str, torch.Tensor]
    y: list
    input_hw: tuple[int, int]


def extract_recipient_features(model, x: torch.Tensor) -> RecipientFeatures:
    """Compute untouched recipient RGB/aux features and q, with no intervention."""
    if model.training:
        raise RuntimeError("A3_EVAL_ONLY:model.training=True")
    if getattr(model, "_gate_override", None) is not None:
        raise RuntimeError("A3_REFUSE_GATE_OVERRIDE")
    if getattr(model, "aux_mode", None) != "ir":
        raise RuntimeError(f"A3_REQUIRES_IR_MODEL:{getattr(model, 'aux_mode', None)}")

    x_rgb, x_aux = model._split_input(x)
    y = [None] * (len(model.rgb_backbone) + len(model.tail))
    z = x_rgb
    for m in model.rgb_backbone:
        z = m(z)
        y[m.i] = z

    a3, a4, a5 = model.aux_encoder(x_aux)
    aux = {"P3": a3, "P4": a4, "P5": a5}

    # HARD causal boundary: q is computed before any residual intervention.
    q_native = model._effective_gate(tuple(aux[s].detach() for s in SCALES))
    rgb = {scale: y[TAP_BY_SCALE[scale]] for scale in SCALES}
    residual = {
        scale: model.fusions[str(TAP_BY_SCALE[scale])].proj(aux[scale])
        for scale in SCALES
    }
    return RecipientFeatures(
        q_native=q_native,
        rgb=rgb,
        aux=aux,
        residual=residual,
        y=y,
        input_hw=(int(x.shape[-2]), int(x.shape[-1])),
    )


def infer_feature_stride(input_hw: tuple[int, int], feature: torch.Tensor) -> int:
    ih, iw = input_hw
    fh, fw = int(feature.shape[-2]), int(feature.shape[-1])
    if ih % fh or iw % fw:
        raise RuntimeError(f"A3_FEATURE_STRIDE_NONINTEGER:{input_hw}:{(fh,fw)}")
    sh, sw = ih // fh, iw // fw
    if sh != sw:
        raise RuntimeError(f"A3_FEATURE_STRIDE_ANISOTROPIC:{sh}!={sw}")
    if sh <= 0:
        raise RuntimeError("A3_FEATURE_STRIDE_INVALID")
    return int(sh)


def zero_fill_translate(t: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    """Translate BCHW tensor without wraparound. Positive dx=right, dy=down."""
    if t.ndim != 4:
        raise ValueError("zero_fill_translate expects BCHW")
    dx, dy = int(dx), int(dy)
    out = torch.zeros_like(t)
    h, w = t.shape[-2:]
    if abs(dx) >= w or abs(dy) >= h:
        return out

    src_x0 = max(0, -dx)
    src_x1 = min(w, w - dx)
    src_y0 = max(0, -dy)
    src_y1 = min(h, h - dy)
    dst_x0 = max(0, dx)
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y0 = max(0, dy)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    out[..., dst_y0:dst_y1, dst_x0:dst_x1] = t[..., src_y0:src_y1, src_x0:src_x1]
    return out


def _same_shape(source: torch.Tensor, target: torch.Tensor, scale: str) -> torch.Tensor:
    if tuple(source.shape) != tuple(target.shape):
        raise RuntimeError(
            f"A3_RESIDUAL_SHAPE_MISMATCH:{scale}:{tuple(source.shape)}!={tuple(target.shape)}"
        )
    return source.to(device=target.device, dtype=target.dtype)


def forward_with_custom_residuals(
    model,
    x: torch.Tensor,
    *,
    recipient_id: str,
    active_scales: Iterable[str],
    replacements: Mapping[str, torch.Tensor] | None = None,
    source_ids: Mapping[str, str] | None = None,
    shifts: Mapping[str, tuple[int, int]] | None = None,
    condition_name: str,
):
    """Forward with post-projection residual replacements/shifts.

    replacements and shifts are applied only after untouched recipient q_native exists.
    Coefficients are q_native for active scales and zero otherwise.
    """
    f = extract_recipient_features(model, x)
    active = set(active_scales)
    if not active.issubset(SCALES):
        raise ValueError(f"unknown active scale:{active-set(SCALES)}")
    residuals = {s: f.residual[s] for s in SCALES}
    used_sources = {s: str(recipient_id) for s in SCALES}
    replacements = dict(replacements or {})
    source_ids = dict(source_ids or {})
    shifts = dict(shifts or {})

    for scale, value in replacements.items():
        if scale not in SCALES:
            raise ValueError(scale)
        residuals[scale] = _same_shape(value, residuals[scale], scale)
        used_sources[scale] = str(source_ids.get(scale, "CUSTOM"))

    for scale, shift in shifts.items():
        if scale not in SCALES:
            raise ValueError(scale)
        dx, dy = shift
        residuals[scale] = zero_fill_translate(residuals[scale], dx=dx, dy=dy)
        used_sources[scale] = f"{used_sources[scale]}|SHIFT({int(dx)},{int(dy)})"

    y = list(f.y)
    coeffs = {}
    for scale in SCALES:
        if scale in active:
            coeff = f.q_native
        else:
            coeff = f.q_native.new_zeros(f.q_native.shape)
        coeffs[scale] = coeff
        tap = TAP_BY_SCALE[scale]
        y[tap] = y[tap] + broadcast_gate(coeff, y[tap]) * residuals[scale]

    out = _forward_tail(model, y, y[10])
    trace = {
        "recipient_id": str(recipient_id),
        "condition": str(condition_name),
        "q_native": [float(v) for v in f.q_native.detach().cpu().reshape(-1)],
        "active_scales": sorted(active),
        "residual_source_ids": used_sources,
        "alpha": {
            s: [float(v) for v in coeffs[s].detach().cpu().reshape(-1)] for s in SCALES
        },
        "native_residual_sha256": {s: tensor_sha256(f.residual[s]) for s in SCALES},
        "used_residual_sha256": {s: tensor_sha256(residuals[s]) for s in SCALES},
        "feature_strides": {s: infer_feature_stride(f.input_hw, f.rgb[s]) for s in SCALES},
    }
    return out, trace



def project_residuals_no_gate(model, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute projected IR residuals only. Reliability gate is never called."""
    if model.training:
        raise RuntimeError("A3_EVAL_ONLY:model.training=True")
    _, x_aux = model._split_input(x)
    a3, a4, a5 = model.aux_encoder(x_aux)
    aux = {"P3": a3, "P4": a4, "P5": a5}
    return {
        scale: model.fusions[str(TAP_BY_SCALE[scale])].proj(aux[scale])
        for scale in SCALES
    }


def build_residual_cache_no_gate(model, dataset, device):
    """Cache projected residuals for donor/LOO use without calling the gate."""
    cache = {}
    model.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            sid = str(sample["sample_id"])
            batch = dataset.collate_fn([sample])
            img = batch["img"].to(device).float()
            residual = project_residuals_no_gate(model, img)
            cache[sid] = {s: residual[s].detach().cpu().clone() for s in SCALES}
    return cache


def build_residual_cache(model, dataset, device):
    """Cache native projected residuals. Gate is called only to capture recipient q."""
    cache = {}
    q = {}
    rgb = {}
    model.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            sid = str(sample["sample_id"])
            batch = dataset.collate_fn([sample])
            img = batch["img"].to(device).float()
            f = extract_recipient_features(model, img)
            cache[sid] = {s: f.residual[s].detach().cpu().clone() for s in SCALES}
            rgb[sid] = {s: f.rgb[s].detach().cpu().clone() for s in SCALES}
            q[sid] = [float(v) for v in f.q_native.detach().cpu().reshape(-1)]
    return {"residual": cache, "rgb": rgb, "q_native": q}


def zmap(t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = t.float()
    return (x - x.mean()) / (x.std(unbiased=False) + eps)


def energy_map(t: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """RMS over channel axis. Accepts CHW or BCHW; returns HW for batch=1."""
    if t.ndim == 4:
        if t.shape[0] != 1:
            raise ValueError("A3 energy_map expects batch=1")
        t = t[0]
    if t.ndim != 3:
        raise ValueError("A3 energy_map expects CHW/BCHW")
    return torch.sqrt(torch.mean(t.float() ** 2, dim=0) + eps)


def pearson_corr(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    if tuple(a.shape) != tuple(b.shape):
        raise ValueError("pearson shape mismatch")
    aa, bb = a.float().reshape(-1), b.float().reshape(-1)
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    den = torch.sqrt(torch.sum(aa * aa) * torch.sum(bb * bb))
    if float(den) <= eps:
        return 0.0
    return float(torch.sum(aa * bb) / den)


def summarize_signed(values: Mapping[str, float]) -> dict:
    xs = [float(v) for v in values.values()]
    if not xs:
        return {"mean": None, "median": None, "positive": 0, "negative": 0, "zero": 0}
    return {
        "mean": float(np.mean(xs)),
        "median": float(np.median(xs)),
        "positive": int(sum(v > 0 for v in xs)),
        "negative": int(sum(v < 0 for v in xs)),
        "zero": int(sum(v == 0 for v in xs)),
    }


def classify_sample_metric(primary: dict, replication: dict) -> str:
    """Cross-system sign label for spatial/semantic per-sample deltas."""
    if (primary["median"] > 0 and primary["positive"] >= 4
            and replication["median"] > 0):
        return "STRONG_RECIPIENT_SPECIFIC"
    if (primary["median"] < 0 and primary["negative"] >= 4
            and replication["median"] < 0):
        return "STRONG_DONOR_FAVORED"
    return "INCONCLUSIVE"


def classify_ap_effect(primary: dict, replication: dict, pos_label="STRONG_POSITIVE",
                       neg_label="STRONG_NEGATIVE") -> str:
    if (primary["full"] > 0 and primary["loo_median"] > 0
            and primary["positive_folds"] >= 4 and replication["full"] > 0):
        return pos_label
    if (primary["full"] < 0 and primary["loo_median"] < 0
            and primary["negative_folds"] >= 4 and replication["full"] < 0):
        return neg_label
    return "INCONCLUSIVE"


def ap_effect(new_result: dict, baseline_result: dict) -> dict:
    full = float(new_result["full"]["map50_95"] - baseline_result["full"]["map50_95"])
    loo = {
        sid: float(new_result["loo"][sid]["map50_95"] - baseline_result["loo"][sid]["map50_95"])
        for sid in baseline_result["loo"]
    }
    vals = list(loo.values())
    return {
        "full": full,
        "loo": loo,
        "loo_median": float(np.median(vals)),
        "positive_folds": int(sum(v > 0 for v in vals)),
        "negative_folds": int(sum(v < 0 for v in vals)),
        "zero_folds": int(sum(v == 0 for v in vals)),
    }
