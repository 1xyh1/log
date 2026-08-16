"""Task-oriented scalar reliability gate for RGB-anchor auxiliary residuals.

The gate deliberately does not scale or otherwise modify the RGB anchor.  It only
controls the magnitude of an already zero-initialized auxiliary residual:

    F_i = R_i + q * P_i(A_i),  q in (0, 1)

This is a conservative adaptation of InfraNet/QualGate to the opposite modality
priority: this project is RGB-primary and IR is auxiliary.  The module predicts one
per-image scalar from the IR feature pyramid.  It is not an image-quality score and
does not use a hand-crafted quality label.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class PyramidScalarReliabilityGate(nn.Module):
    """Predict one task-oriented reliability scalar from A3/A4/A5.

    A non-affine LayerNorm keeps the three pooled feature levels numerically
    comparable.  The final layer uses a very small random weight and zero bias, so
    q starts close to 0.5 without making the whole MLP gradient-dead.  Exact detector
    identity is still guaranteed by the zero-initialized fusion projections.
    """

    def __init__(self, channels: Sequence[int] = (256, 256, 512), hidden: int = 64,
                 final_weight_std: float = 1e-3):
        super().__init__()
        self.channels = tuple(int(c) for c in channels)
        if not self.channels or any(c <= 0 for c in self.channels):
            raise ValueError(f"invalid channels: {self.channels}")
        if hidden <= 0:
            raise ValueError(f"hidden must be positive, got {hidden}")
        total = sum(self.channels)
        self.norm = nn.LayerNorm(total, elementwise_affine=False)
        self.fc1 = nn.Linear(total, int(hidden))
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(int(hidden), 1)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=float(final_weight_std))
        nn.init.zeros_(self.fc2.bias)

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(features) != len(self.channels):
            raise ValueError(
                f"expected {len(self.channels)} pyramid levels, got {len(features)}"
            )
        pooled = []
        batch = None
        for idx, (feature, channels) in enumerate(zip(features, self.channels)):
            if feature.ndim != 4 or feature.shape[1] != channels:
                raise ValueError(
                    f"level {idx} expected BCHW with C={channels}, got "
                    f"{tuple(feature.shape)}"
                )
            if batch is None:
                batch = feature.shape[0]
            elif feature.shape[0] != batch:
                raise ValueError("pyramid levels have different batch sizes")
            pooled.append(F.adaptive_avg_pool2d(feature, 1).flatten(1))
        descriptor = self.norm(torch.cat(pooled, dim=1))
        return torch.sigmoid(self.fc2(self.act(self.fc1(descriptor))))


def broadcast_gate(q: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Validate a Bx1 scalar gate and reshape it for a BCHW residual."""
    if q.ndim != 2 or q.shape[1] != 1 or q.shape[0] != reference.shape[0]:
        raise ValueError(f"gate {tuple(q.shape)} is incompatible with {tuple(reference.shape)}")
    return q.to(device=reference.device, dtype=reference.dtype)[:, :, None, None]


def per_sample_log_rms(feature: torch.Tensor) -> torch.Tensor:
    """Per-sample log-RMS over (C,H,W), matching the A1 audit formula exactly:
    log(sqrt(mean(A^2)) + 1e-9).  NEVER reduce across the batch."""
    if feature.ndim != 4:
        raise ValueError(f"expected BCHW feature, got {tuple(feature.shape)}")
    ms = feature.pow(2).mean(dim=(1, 2, 3))
    return torch.log(ms.sqrt() + 1e-9)


class MagnitudeReliabilityGate(PyramidScalarReliabilityGate):
    """F1-C magnitude gate: the original gate plus a zero-init magnitude branch.

    Equivalent form required by the reviewer so the ORIGINAL norm/fc1/fc2
    initialization and RNG order are untouched:

        z = LayerNorm(concat(GAP(A3), GAP(A4), GAP(A5)))
        m = [logRMS(A3), logRMS(A4), logRMS(A5)]
        h = old_fc1(z) + magnitude_fc(m)     # magnitude_fc: no bias, zero-init
        q = sigmoid(old_fc2(SiLU(h)))

    With the magnitude branch at zero the outputs are bitwise identical to the
    original gate, so the model's initial detector still equals the RGB
    reference exactly.
    """

    def __init__(self, channels: Sequence[int] = (256, 256, 512),
                 hidden: int = 64, final_weight_std: float = 1e-3):
        super().__init__(channels=channels, hidden=hidden,
                         final_weight_std=final_weight_std)
        # Constructed AFTER the original layers, so the original RNG order is
        # preserved; zero-init consumes no RNG afterwards.
        self.magnitude_fc = nn.Linear(3, int(hidden), bias=False)
        nn.init.zeros_(self.magnitude_fc.weight)

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(features) != len(self.channels):
            raise ValueError(
                f"expected {len(self.channels)} pyramid levels, got {len(features)}"
            )
        pooled = []
        mags = []
        batch = None
        for idx, (feature, channels) in enumerate(zip(features, self.channels)):
            if feature.ndim != 4 or feature.shape[1] != channels:
                raise ValueError(
                    f"level {idx} expected BCHW with C={channels}, got "
                    f"{tuple(feature.shape)}"
                )
            if batch is None:
                batch = feature.shape[0]
            elif feature.shape[0] != batch:
                raise ValueError("pyramid levels have different batch sizes")
            pooled.append(F.adaptive_avg_pool2d(feature, 1).flatten(1))
            mags.append(per_sample_log_rms(feature))
        descriptor = self.norm(torch.cat(pooled, dim=1))
        magnitude = torch.stack(mags, dim=1)  # (B, 3)
        h = self.fc1(descriptor) + self.magnitude_fc(magnitude)
        return torch.sigmoid(self.fc2(self.act(h)))
