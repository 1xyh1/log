"""Step 4-F0 fusion primitive: zero-init residual feature injection.

F_i = R_i + P_i(A_i) with P_i = Conv2d(aux_ch, rgb_ch, 1x1, bias=True),
weight and bias initialized to EXACT zero:
    epoch-0: F == R (bitwise; no alpha scalar, no cold-start gating)
    first backward:
        fusion projection gradient > 0   (dL/dW = A * dL/dF)
        aux encoder gradient == 0        (dL/dA = W^T * dL/dF = 0 because W == 0)
    after the first projection update:
        aux encoder gradient can become > 0 (one-step self-unblocking, milder than
        alpha gating which has no self-unblocking scalar)
"""
from __future__ import annotations

import torch.nn as nn


class ZeroInitResidualFusion(nn.Module):
    def __init__(self, aux_ch: int, rgb_ch: int):
        super().__init__()
        self.proj = nn.Conv2d(aux_ch, rgb_ch, 1, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, r, a):
        return r + self.proj(a)

    def assert_zero_init(self) -> bool:
        return bool(self.proj.weight.abs().max().item() == 0.0
                    and self.proj.bias.abs().max().item() == 0.0)
