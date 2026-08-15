"""Step 4-F0: lightweight auxiliary pyramid encoders (RGB-anchor companion paths).

Frozen design (reviewer 2026-08-15): ONE shared 2-channel encoder for all groups so
F0-C0 / F0-I / F0-D share identical architecture and parameter count:
    F0-C0  aux input = [0, 0]
    F0-I   aux input = [I, 0]          (IR median scalar + zero pad)
    F0-D   aux input = [logD, validM]  (physical float pair)
Outputs A3/A4/A5 with the same channel counts as YOLO26s RGB features at P3/P4/P5
(256 / 256 / 512) so the zero-init 1x1 residual injection adds F = R + P(A).
"""
from __future__ import annotations

import torch.nn as nn


def _conv(in_ch: int, out_ch: int, stride: int = 2) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.SiLU(),
    )


class AuxPyramidEncoder2ch(nn.Module):
    """2ch aux input -> A3(256, /8), A4(256, /16), A5(512, /32).

    Lightweight (~1.1M params): shared stem then per-scale stride-2 stages.
    """

    def __init__(self, in_ch: int = 2, base: int = 64):
        super().__init__()
        self.stem = _conv(in_ch, base, 2)     # /2
        self.b1 = _conv(base, base * 2, 2)    # /4
        self.b2 = _conv(base * 2, 256, 2)     # /8  -> A3
        self.b3 = _conv(256, 256, 2)          # /16 -> A4
        self.b4 = _conv(256, 512, 2)          # /32 -> A5

    def forward(self, x):
        x = self.stem(x)
        x = self.b1(x)
        x = self.b2(x)
        a3 = x
        x = self.b3(x)
        a4 = x
        x = self.b4(x)
        a5 = x
        return a3, a4, a5


def build_aux_encoder() -> AuxPyramidEncoder2ch:
    """Single shared encoder for all F0 groups (architecture-matched control)."""
    return AuxPyramidEncoder2ch(in_ch=2)
