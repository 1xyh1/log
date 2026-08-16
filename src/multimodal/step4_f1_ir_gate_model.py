"""Step 4-F1: RGB-primary YOLO26 with a scalar IR residual reliability gate.

F1 inherits the validated F0 data split, RGB anchor, auxiliary encoder, zero-init
projections, and P3/P4/P5 routing.  The only architectural change is:

    F_i = R_i + q * P_i(A_i)

where q is one per-image scalar predicted from the IR feature pyramid.  RGB is never
scaled or amplified.  This keeps the visible-light pretrained detector authoritative
and lets the model fall back to it when IR is unhelpful.
"""
from __future__ import annotations

import math

import torch

from multimodal.reliability_gate import (
    PyramidScalarReliabilityGate,
    broadcast_gate,
)
from multimodal.step4_f0_model import Step4F0Model


class Step4F1IRGateModel(Step4F0Model):
    """F0 plus one global task-oriented gate for the auxiliary residual.

    ``gate_mode='learned'`` is the formal F1 treatment.  ``fixed_one`` keeps the
    exact same module/state shape but forces q=1, providing an implementation-matched
    ungated residual control.  ``set_gate_override`` is reserved for post-hoc causal
    evaluation and is never used during formal training.
    """

    def __init__(self, reference, aux_encoder=None, freeze_rgb_backbone: bool = True,
                 aux_mode: str = "ir", gate_mode: str = "learned",
                 gate_hidden: int = 64):
        if aux_mode not in {"zero", "ir"}:
            raise ValueError("F1 is IR-only; Depth must not be stacked into this stage")
        if gate_mode not in {"learned", "fixed_one", "magnitude"}:
            raise ValueError(f"unknown gate_mode: {gate_mode}")
        super().__init__(reference, aux_encoder=aux_encoder,
                         freeze_rgb_backbone=freeze_rgb_backbone,
                         aux_mode=aux_mode)
        self.gate_mode = gate_mode
        if gate_mode == "magnitude":
            from multimodal.reliability_gate import MagnitudeReliabilityGate
            self.reliability_gate = MagnitudeReliabilityGate(hidden=gate_hidden)
        else:
            self.reliability_gate = PyramidScalarReliabilityGate(hidden=gate_hidden)
        self._gate_override: float | None = None
        self._last_raw_gate: torch.Tensor | None = None
        self._last_effective_gate: torch.Tensor | None = None

        # Non-persistent training diagnostics.  They do not change checkpoint state.
        self.register_buffer("_gate_sum", torch.tensor(0.0), persistent=False)
        self.register_buffer("_gate_count", torch.tensor(0, dtype=torch.long),
                             persistent=False)
        self.register_buffer("_gate_min", torch.tensor(float("inf")), persistent=False)
        self.register_buffer("_gate_max", torch.tensor(float("-inf")), persistent=False)

    def set_gate_override(self, value: float | None) -> None:
        if value is not None:
            value = float(value)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"gate override must be finite in [0,1], got {value}")
        self._gate_override = value

    def reset_gate_stats(self) -> None:
        self._gate_sum.zero_()
        self._gate_count.zero_()
        self._gate_min.fill_(float("inf"))
        self._gate_max.fill_(float("-inf"))

    def gate_stats(self) -> dict[str, float | int | None]:
        count = int(self._gate_count.item())
        if count == 0:
            return {"count": 0, "mean": None, "min": None, "max": None}
        return {
            "count": count,
            "mean": float((self._gate_sum / count).item()),
            "min": float(self._gate_min.item()),
            "max": float(self._gate_max.item()),
        }

    @property
    def last_raw_gate(self) -> torch.Tensor | None:
        return self._last_raw_gate

    @property
    def last_effective_gate(self) -> torch.Tensor | None:
        return self._last_effective_gate

    def _effective_gate(self, features) -> torch.Tensor:
        raw = self.reliability_gate(features)
        if self._gate_override is not None:
            effective = raw.new_full(raw.shape, self._gate_override)
        elif self.gate_mode == "fixed_one":
            effective = raw.new_ones(raw.shape)
        else:
            effective = raw
        self._last_raw_gate = raw.detach()
        self._last_effective_gate = effective.detach()
        if self.training:
            q = effective.detach().float()
            self._gate_sum.add_(q.sum())
            self._gate_count.add_(q.numel())
            self._gate_min.copy_(torch.minimum(self._gate_min, q.min()))
            self._gate_max.copy_(torch.maximum(self._gate_max, q.max()))
        return effective

    @staticmethod
    def _gated_residual(fusion, rgb: torch.Tensor, aux: torch.Tensor,
                        q: torch.Tensor) -> torch.Tensor:
        delta = fusion.proj(aux)
        return rgb + broadcast_gate(q, rgb) * delta

    def _forward_fused(self, x_rgb, x_aux):
        # Kept explicit to preserve the audited YOLO26 layer-index routing.  In
        # particular, x=y[10] after P5 fusion is a hard invariant.
        y = [None] * (len(self.rgb_backbone) + len(self.tail))
        x = x_rgb
        for m in self.rgb_backbone:
            x = m(x)
            y[m.i] = x
        a3, a4, a5 = self.aux_encoder(x_aux)
        # Gate inputs are detached (execution feedback 2026-08-16): the gate still
        # predicts q from the pre-projection A pyramid (per DESIGN_FREEZE), but its
        # gradient must NOT flow back into the aux encoder.  With zero aux input
        # (F1-C0) the LayerNorm gradient at zero (~1/sqrt(eps)) combined with the
        # gate loop would otherwise produce O(1e-11) "numerical dust" gradients on
        # the zero-init projections, which MuSGD's Muon branch then normalizes
        # (X /= X.norm()+eps) into O(1) updates — breaking the C0 "proj stays
        # exactly zero" invariant.  Detaching keeps the aux-encoder gradient path
        # limited to the residual (F0 semantics).
        q = self._effective_gate(tuple(f.detach() for f in (a3, a4, a5)))
        y[4] = self._gated_residual(self.fusions["4"], y[4], a3, q)
        y[6] = self._gated_residual(self.fusions["6"], y[6], a4, q)
        y[10] = self._gated_residual(self.fusions["10"], y[10], a5, q)
        x = y[10]  # CRITICAL: fused P5 must enter neck layer 11 (f=-1).
        for m in self.tail:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [
                    x if j == -1 else y[j] for j in m.f
                ]
            x = m(x)
            if m.i in self.save:
                y[m.i] = x
        return x
