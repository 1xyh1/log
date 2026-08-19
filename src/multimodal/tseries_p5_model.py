"""T-series model: a single P5 direct-IR injection site, no reliability gate."""
from __future__ import annotations

import torch
import torch.nn as nn

from multimodal.aux_encoder import build_aux_encoder
from multimodal.feature_fusion import ZeroInitResidualFusion
from multimodal.trainability import enforce_frozen_module_eval, freeze_module
from multimodal.tseries_core import TREATMENTS, center_full_map, p5_mechanism_metrics, tensor_sha256

P5_TAP = 10
P5_CHANNELS = 512

class TSeriesP5Model(nn.Module):
    """Same physical module tree for T0-N/T1-F/T2-A.

    P3/P4 backbone taps receive no direct IR residual.  IR enters only at
    backbone P5 and can then propagate through the unchanged YOLO26 neck/head.

    T0 still evaluates the aux branch under no_grad so AuxEncoder BN buffers see
    the same batches/mode, but the detection loss graph is exactly RGB-only.
    """

    def __init__(
        self,
        reference,
        *,
        treatment_id: str,
        aux_encoder=None,
        freeze_rgb_backbone: bool = True,
    ):
        super().__init__()
        if treatment_id not in TREATMENTS:
            raise ValueError(f"unknown treatment_id: {treatment_id}")
        self.treatment_id = str(treatment_id)
        self.aux_mode = "ir"
        self.rgb_backbone = nn.Sequential(*reference.model[:11])
        self.aux_encoder = aux_encoder or build_aux_encoder()
        self.p5_fusion = ZeroInitResidualFusion(P5_CHANNELS, P5_CHANNELS)
        self.tail = nn.Sequential(*reference.model[11:])
        self.save = reference.save
        self.yaml = dict(reference.yaml)
        self.nc = int(reference.nc)
        self.names = dict(reference.names) if hasattr(reference, "names") else None
        self.stride = reference.stride
        self.args = getattr(reference, "args", None)
        self.criterion = None
        self._last_forward_trace: dict = {}
        self.reset_mechanism_stats()
        if freeze_rgb_backbone:
            freeze_module(self.rgb_backbone, freeze_bn_stats=True)

    @property
    def model(self):
        return self.tail

    @property
    def last_forward_trace(self) -> dict:
        return dict(self._last_forward_trace)

    def train(self, mode: bool = True):
        super().train(mode)
        enforce_frozen_module_eval(self.rgb_backbone)
        return self

    def _split_input(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c = int(x.shape[1])
        if c == 6:
            rgb = x[:, :3]
            ir = x[:, 3:4]
            aux = torch.cat([ir, torch.zeros_like(ir)], dim=1)
            return rgb, aux
        if c == 3:
            return x, torch.zeros(x.shape[0], 2, *x.shape[2:], device=x.device, dtype=x.dtype)
        if c == 2:
            return torch.zeros(x.shape[0], 3, *x.shape[2:], device=x.device, dtype=x.dtype), x
        raise ValueError(f"unsupported input channels: {c}")

    def _run_rgb_backbone(self, x_rgb: torch.Tensor):
        y = [None] * (len(self.rgb_backbone) + len(self.tail))
        x = x_rgb
        for m in self.rgb_backbone:
            x = m(x)
            y[m.i] = x
        return y

    def _run_tail(self, y, x):
        for m in self.tail:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [
                    x if j == -1 else y[j] for j in m.f
                ]
            x = m(x)
            if m.i in self.save:
                y[m.i] = x
        return x

    def _compute_delta(self, x_aux: torch.Tensor, *, loss_connected: bool) -> torch.Tensor:
        if loss_connected:
            _, _, a5 = self.aux_encoder(x_aux)
            return self.p5_fusion.proj(a5)
        # T0: same forward-mode/BN-buffer exposure, but no autograd edge to loss.
        with torch.no_grad():
            _, _, a5 = self.aux_encoder(x_aux)
            return self.p5_fusion.proj(a5)

    def _used_residual(self, delta: torch.Tensor) -> torch.Tensor:
        if self.treatment_id == "T0-N":
            return torch.zeros_like(delta)
        if self.treatment_id == "T1-F":
            return delta
        if self.treatment_id == "T2-A":
            return center_full_map(delta)
        raise AssertionError(self.treatment_id)

    def reset_mechanism_stats(self) -> None:
        self._mech_count = 0
        self._mech_sums = {
            "full_rms": 0.0,
            "dc_rms": 0.0,
            "ac_rms": 0.0,
            "dc_over_full": 0.0,
            "ac_over_full": 0.0,
            "used_rms": 0.0,
            "post_center_channel_mean_abs_max": 0.0,
        }

    def _record_mechanism(self, delta: torch.Tensor, used: torch.Tensor) -> None:
        row = p5_mechanism_metrics(delta, used)
        self._mech_count += 1
        for k, v in row.items():
            if k == "post_center_channel_mean_abs_max":
                self._mech_sums[k] = max(self._mech_sums[k], float(v))
            else:
                self._mech_sums[k] += float(v)

    def mechanism_stats(self) -> dict:
        n = int(self._mech_count)
        if n == 0:
            return {"count": 0}
        out = {"count": n}
        for k, v in self._mech_sums.items():
            out[k] = float(v) if k == "post_center_channel_mean_abs_max" else float(v) / n
        return out

    def _forward_fused(self, x_rgb: torch.Tensor, x_aux: torch.Tensor):
        y = self._run_rgb_backbone(x_rgb)
        r5 = y[P5_TAP]
        loss_connected = self.treatment_id != "T0-N"
        delta = self._compute_delta(x_aux, loss_connected=loss_connected)
        used = self._used_residual(delta)
        fused5 = r5 if self.treatment_id == "T0-N" else r5 + used

        # Hard audited handoff: fused P5 is both saved at y[10] and becomes current x.
        y[P5_TAP] = fused5
        x = y[P5_TAP]
        self._last_forward_trace = {
            "treatment_id": self.treatment_id,
            "p3_direct_injection_count": 0,
            "p4_direct_injection_count": 0,
            "p5_direct_injection_count": 1,
            "t0_loss_connected_aux": bool(loss_connected),
            "r5_sha256": tensor_sha256(r5),
            "delta5_sha256": tensor_sha256(delta),
            "used5_sha256": tensor_sha256(used),
            "fused5_sha256": tensor_sha256(fused5),
            "y10_is_fused5_object": bool(y[P5_TAP] is fused5),
            "x_is_y10_object": bool(x is y[P5_TAP]),
        }
        if self.training:
            self._record_mechanism(delta, used)
        return self._run_tail(y, x)

    def _predict_once(self, x):
        rgb, aux = self._split_input(x)
        return self._forward_fused(rgb, aux)

    def forward(self, x, *args, **kwargs):
        if isinstance(x, dict):
            return self.loss(x)
        return self._predict_once(x)

    def loss(self, batch, preds=None):
        if preds is None:
            preds = self._predict_once(batch["img"])
        if self.criterion is None:
            from ultralytics.utils.loss import v8DetectionLoss
            self.criterion = v8DetectionLoss(self)
        return self.criterion(preds, batch)

    def p5_full_and_ac_from_input(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Post-projection residuals for mechanism/paired audits."""
        _, aux = self._split_input(x)
        _, _, a5 = self.aux_encoder(aux)
        full = self.p5_fusion.proj(a5)
        return full, center_full_map(full)

    def p5_residual_from_input(self, x: torch.Tensor) -> torch.Tensor:
        full, ac = self.p5_full_and_ac_from_input(x)
        if self.treatment_id == "T1-F":
            return full
        if self.treatment_id == "T2-A":
            return ac
        raise RuntimeError("paired residual audit is defined only for T1-F/T2-A")

    def predict_with_p5_residual(self, x: torch.Tensor, residual: torch.Tensor):
        """Recipient RGB + explicit post-projection P5 residual override.

        The override is used only by post-training recipient-vs-donor evaluation.
        """
        rgb, _ = self._split_input(x)
        y = self._run_rgb_backbone(rgb)
        r5 = y[P5_TAP]
        residual = residual.to(device=r5.device, dtype=r5.dtype)
        if tuple(residual.shape) != tuple(r5.shape):
            raise RuntimeError(
                f"T_SERIES_P5_RESIDUAL_SHAPE_MISMATCH:{tuple(residual.shape)}!={tuple(r5.shape)}"
            )
        fused5 = r5 + residual
        y[P5_TAP] = fused5
        xcur = y[P5_TAP]
        return self._run_tail(y, xcur)

    def assert_zero_init(self) -> bool:
        return self.p5_fusion.assert_zero_init()
