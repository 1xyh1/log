"""Step 4-F0 model: RGB anchor + aux pyramid encoder + P3/P4/P5 zero-init residual.

RGB path: the SAME 3ch O2M nc=12 YOLO26 reference used by Step 3 (pretrained,
frozen backbone layers 0..10; neck/head trainable). Aux path: shared lightweight
2ch encoder. Fusion: F_i = R_i + P_i(A_i) at layers 4/6/10 (P3/P4/P5), proj
zero-init so epoch-0 output is bitwise the RGB model.

Explicit build, no Ultralytics auto-load (its partial first-conv copy is avoided).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from multimodal.aux_encoder import build_aux_encoder
from multimodal.feature_fusion import ZeroInitResidualFusion
from multimodal.trainability import enforce_frozen_module_eval, freeze_module

RGB_TAP_LAYERS = (4, 6, 10)          # P3/P4/P5 in yolo26s backbone
RGB_TAP_CHANNELS = {4: 256, 6: 256, 10: 512}


class Step4F0Model(nn.Module):
    """Dual-input model. forward(rgb, aux) returns Detect outputs for prediction;
    forward(batch_dict) runs the training loss path (criterion on tail Detect)."""

    def __init__(self, reference, aux_encoder=None, freeze_rgb_backbone: bool = True,
                 aux_mode: str = "zero"):
        super().__init__()
        assert aux_mode in {"zero", "ir", "depth"}
        self.aux_mode = aux_mode
        self.rgb_backbone = nn.Sequential(*reference.model[:11])
        self.aux_encoder = aux_encoder or build_aux_encoder()
        self.tail = nn.Sequential(*reference.model[11:])
        self.save = reference.save
        self.yaml = dict(reference.yaml)
        self.nc = int(reference.nc)
        self.names = dict(reference.names) if hasattr(reference, "names") else None
        self.stride = reference.stride
        self.fusions = nn.ModuleDict({
            "4": ZeroInitResidualFusion(256, 256),
            "6": ZeroInitResidualFusion(256, 256),
            "10": ZeroInitResidualFusion(512, 512),
        })
        self.args = getattr(reference, "args", None)
        self.criterion = None
        if freeze_rgb_backbone:
            freeze_module(self.rgb_backbone, freeze_bn_stats=True)

    def _split_input(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map the shared 6ch batch (or 3ch/2ch gate tensors) to (rgb, aux 2ch).

        6ch [R,G,B,I,D,M] -> rgb=x[:, :3]; aux per aux_mode:
            zero  -> [0, 0]
            ir    -> [I, 0]
            depth -> [D, M]
        3ch -> (x, zeros); 2ch -> (zeros rgb, x).
        """
        c = x.shape[1]
        if c == 6:
            rgb = x[:, :3]
            if self.aux_mode == "zero":
                aux = torch.zeros(x.shape[0], 2, *x.shape[2:], device=x.device, dtype=x.dtype)
            elif self.aux_mode == "ir":
                aux = torch.cat([x[:, 3:4], torch.zeros_like(x[:, 3:4])], dim=1)
            else:
                aux = x[:, 4:6]
            return rgb, aux
        if c == 3:
            return x, torch.zeros(x.shape[0], 2, *x.shape[2:], device=x.device, dtype=x.dtype)
        if c == 2:
            return torch.zeros(x.shape[0], 3, *x.shape[2:], device=x.device, dtype=x.dtype), x
        raise ValueError(f"unsupported input channels: {c}")

    @property
    def model(self):
        # v8DetectionLoss reads model.model[-1] (Detect); property avoids double
        # registration of tail parameters under two attribute names.
        return self.tail

    def train(self, mode: bool = True):
        # stock trainer calls model.train(); keep the frozen RGB BN in eval mode.
        super().train(mode)
        enforce_frozen_module_eval(self.rgb_backbone)
        return self

    # ------------------------------------------------------------------ forward
    def _forward_fused(self, x_rgb, x_aux):
        y = [None] * (len(self.rgb_backbone) + len(self.tail))
        x = x_rgb
        for m in self.rgb_backbone:
            x = m(x)
            y[m.i] = x
        a3, a4, a5 = self.aux_encoder(x_aux)
        y[4] = self.fusions["4"](y[4], a3)
        y[6] = self.fusions["6"](y[6], a4)
        y[10] = self.fusions["10"](y[10], a5)
        # CRITICAL (reviewer P0-1): neck layer 11 has f=-1 and consumes the CURRENT x.
        # Without this, the top-down main chain starts from the pre-fusion RGB P5 and
        # fused P5 only re-enters at layer 21's [-1,10] — not the frozen design.
        x = y[10]
        for m in self.tail:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            if m.i in self.save:
                y[m.i] = x
        return x

    def _predict_once(self, x):
        """Single-input entry point matching stock DetectionModel: the 6ch batch
        (or 3ch/2ch gate tensor) is split internally. Returns Detect outputs."""
        rgb, aux = self._split_input(x)
        return self._forward_fused(rgb, aux)

    def forward(self, x, *args, **kwargs):
        if isinstance(x, dict):              # training path: model(batch)
            return self.loss(x)
        return self._predict_once(x)         # validator/gate path (augment kwarg ignored)

    def loss(self, batch, preds=None):
        if preds is None:
            preds = self._predict_once(batch["img"])
        if self.criterion is None:
            from ultralytics.utils.loss import v8DetectionLoss
            self.criterion = v8DetectionLoss(self)  # uses self.args + self.model[-1]
        return self.criterion(preds, batch)
