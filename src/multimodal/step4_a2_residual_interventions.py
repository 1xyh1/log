"""A2 post-gate, scale-wise residual interventions (evaluation only).

The key causal invariant is ordering:
  recipient paired A3/A4/A5 -> q_native -> freeze q -> manipulate projected residuals.
Donor residuals are cached without calling the gate, so a residual shuffle can never
change the recipient gate by construction.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping

import torch

from multimodal.reliability_gate import broadcast_gate

SCALES = ("P3", "P4", "P5")
TAP_BY_SCALE = {"P3": 4, "P4": 6, "P5": 10}
MASK_CONDITIONS = (
    "M000", "M100", "M010", "M001", "M110", "M101", "M011", "M111",
)
GAIN_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)


def tensor_sha256(t: torch.Tensor) -> str:
    x = t.detach().cpu().contiguous()
    return hashlib.sha256(x.numpy().tobytes()).hexdigest()


@dataclass(frozen=True)
class ResidualCondition:
    """One identifiable A2 intervention.

    kind:
      mask: mask is a 3-bit tuple; active scale uses q_native.
      shuffle_cond: all scales active, target delta comes from donor.
      shuffle_only: only target active, target delta comes from donor.
      gain: all scales native except target uses a constant gain, or NATIVE.
    """

    kind: str
    mask: tuple[int, int, int] = (1, 1, 1)
    target_scale: str | None = None
    gain: float | None = None  # None means NATIVE for kind=gain

    def __post_init__(self) -> None:
        if self.kind not in {"mask", "shuffle_cond", "shuffle_only", "gain"}:
            raise ValueError(f"unknown A2 condition kind: {self.kind}")
        if len(self.mask) != 3 or any(v not in (0, 1) for v in self.mask):
            raise ValueError(f"mask must be 3 bits, got {self.mask}")
        if self.kind == "mask":
            if self.target_scale is not None or self.gain is not None:
                raise ValueError("mask condition cannot carry target_scale/gain")
        else:
            if self.target_scale not in SCALES:
                raise ValueError(f"target_scale must be one of {SCALES}")
        if self.kind != "gain" and self.gain is not None:
            raise ValueError("gain is only valid for kind='gain'")
        if self.kind == "gain" and self.gain is not None:
            g = float(self.gain)
            if not torch.isfinite(torch.tensor(g)) or not 0.0 <= g <= 1.0:
                raise ValueError(f"gain must be finite in [0,1], got {self.gain}")

    @property
    def name(self) -> str:
        if self.kind == "mask":
            return "M" + "".join(str(v) for v in self.mask)
        if self.kind == "shuffle_cond":
            return f"SHUFFLE_{self.target_scale}_COND"
        if self.kind == "shuffle_only":
            return f"SHUFFLE_{self.target_scale}_ONLY"
        suffix = "NATIVE" if self.gain is None else f"{float(self.gain):.2f}"
        return f"GAIN_{self.target_scale}_{suffix}"


def mask_conditions() -> tuple[ResidualCondition, ...]:
    return tuple(
        ResidualCondition("mask", tuple(int(c) for c in name[1:]))
        for name in MASK_CONDITIONS
    )


def shuffle_conditions() -> tuple[ResidualCondition, ...]:
    out = []
    for scale in SCALES:
        out.append(ResidualCondition("shuffle_cond", target_scale=scale))
        out.append(ResidualCondition("shuffle_only", target_scale=scale))
    return tuple(out)


def gain_conditions(*, include_native: bool) -> tuple[ResidualCondition, ...]:
    out = []
    for scale in SCALES:
        for value in GAIN_VALUES:
            out.append(ResidualCondition("gain", target_scale=scale, gain=value))
        if include_native:
            out.append(ResidualCondition("gain", target_scale=scale, gain=None))
    return tuple(out)



def classify_paired_effect(primary: dict, replication: dict) -> str:
    """Frozen A2 sign-stability label; no arbitrary AP margin."""
    if (primary["full"] > 0 and primary["loo_median"] > 0
            and primary["positive_folds"] >= 4 and replication["full"] > 0):
        return "STRONG_POSITIVE"
    if (primary["full"] < 0 and primary["loo_median"] < 0
            and primary["negative_folds"] >= 4 and replication["full"] < 0):
        return "STRONG_NEGATIVE"
    return "INCONCLUSIVE"


def _native_coefficients(q_native: torch.Tensor) -> dict[str, torch.Tensor]:
    return {scale: q_native for scale in SCALES}


def _constant_like(q_native: torch.Tensor, value: float) -> torch.Tensor:
    return q_native.new_full(q_native.shape, float(value))


def _condition_coefficients(
    q_native: torch.Tensor, condition: ResidualCondition,
) -> dict[str, torch.Tensor]:
    coeffs = _native_coefficients(q_native)
    if condition.kind == "mask":
        for scale, active in zip(SCALES, condition.mask):
            if not active:
                coeffs[scale] = _constant_like(q_native, 0.0)
    elif condition.kind == "shuffle_only":
        for scale in SCALES:
            if scale != condition.target_scale:
                coeffs[scale] = _constant_like(q_native, 0.0)
    elif condition.kind == "gain":
        if condition.gain is not None:
            coeffs[condition.target_scale] = _constant_like(q_native, condition.gain)
    return coeffs


def project_native_residuals(model, x_aux: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute δ3/δ4/δ5 without ever touching the reliability gate."""
    a3, a4, a5 = model.aux_encoder(x_aux)
    aux = {"P3": a3, "P4": a4, "P5": a5}
    return {
        scale: model.fusions[str(TAP_BY_SCALE[scale])].proj(aux[scale])
        for scale in SCALES
    }


def build_residual_cache(model, dataset, device) -> dict[str, dict[str, torch.Tensor]]:
    """Cache donor projected residuals on CPU. Gate is intentionally never called."""
    cache: dict[str, dict[str, torch.Tensor]] = {}
    model.eval()
    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            sid = str(sample["sample_id"])
            batch = dataset.collate_fn([sample])
            img = batch["img"].to(device).float()
            _, x_aux = model._split_input(img)
            residuals = project_native_residuals(model, x_aux)
            cache[sid] = {k: v.detach().cpu().clone() for k, v in residuals.items()}
    return cache


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


def forward_with_residual_intervention(
    model,
    x: torch.Tensor,
    condition: ResidualCondition,
    *,
    recipient_id: str,
    donor_id: str | None = None,
    donor_residuals: Mapping[str, torch.Tensor] | None = None,
):
    """Run one recipient with post-gate residual intervention.

    q_native is always computed from untouched recipient features first.  Donor
    residuals are consumed only after q is frozen.  The return trace is intended for
    A2-G4/G5 evidence and contains no trainable state.
    """
    if model.training:
        raise RuntimeError("A2_EVAL_ONLY:model.training=True")
    if getattr(model, "_gate_override", None) is not None:
        raise RuntimeError("A2_REFUSE_GATE_OVERRIDE")
    if getattr(model, "aux_mode", None) != "ir":
        raise RuntimeError(f"A2_REQUIRES_IR_MODEL: aux_mode={getattr(model, 'aux_mode', None)}")

    x_rgb, x_aux = model._split_input(x)
    y = [None] * (len(model.rgb_backbone) + len(model.tail))
    z = x_rgb
    for m in model.rgb_backbone:
        z = m(z)
        y[m.i] = z

    a3, a4, a5 = model.aux_encoder(x_aux)
    aux = {"P3": a3, "P4": a4, "P5": a5}

    # A2-G4: gate sees only untouched recipient paired features. No intervention has
    # happened yet, and no later code calls the gate again.
    q_native = model._effective_gate(tuple(aux[s].detach() for s in SCALES))
    residuals = {
        scale: model.fusions[str(TAP_BY_SCALE[scale])].proj(aux[scale])
        for scale in SCALES
    }
    native_residual_sha256 = {scale: tensor_sha256(residuals[scale]) for scale in SCALES}
    source_ids = {scale: str(recipient_id) for scale in SCALES}

    if condition.kind in {"shuffle_cond", "shuffle_only"}:
        if donor_id is None or donor_residuals is None:
            raise RuntimeError("A2_SHUFFLE_REQUIRES_DONOR")
        if str(donor_id) == str(recipient_id):
            raise RuntimeError("A2_SHUFFLE_SELF_DONOR")
        scale = condition.target_scale
        donor = donor_residuals[scale].to(
            device=residuals[scale].device, dtype=residuals[scale].dtype
        )
        if donor.shape != residuals[scale].shape:
            raise RuntimeError(
                f"A2_DONOR_SHAPE_MISMATCH:{scale}:{tuple(donor.shape)}!="
                f"{tuple(residuals[scale].shape)}"
            )
        residuals[scale] = donor
        source_ids[scale] = str(donor_id)

    coeffs = _condition_coefficients(q_native, condition)
    for scale in SCALES:
        tap = TAP_BY_SCALE[scale]
        y[tap] = y[tap] + broadcast_gate(coeffs[scale], y[tap]) * residuals[scale]
    z = y[10]  # preserve audited P5 -> neck main-chain routing
    out = _forward_tail(model, y, z)

    trace = {
        "recipient_id": str(recipient_id),
        "condition": condition.name,
        "q_native": [float(v) for v in q_native.detach().cpu().reshape(-1)],
        "residual_source_ids": source_ids,
        "alpha": {
            scale: [float(v) for v in coeffs[scale].detach().cpu().reshape(-1)]
            for scale in SCALES
        },
        "native_residual_sha256": native_residual_sha256,
        "used_residual_sha256": {
            scale: tensor_sha256(residuals[scale]) for scale in SCALES
        },
    }
    return out, trace
