"""Reference-guided fusion blocks for Step 4/QAF.

These modules are intentionally isolated from the Step-3 model.  Importing this file
must not alter YOLO26.  They provide unit-testable candidates derived from patterns in
RDTTrack and a mature YOLOv5 multispectral implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, *, act: bool = True):
        super().__init__()
        p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class AuxAdapter(nn.Module):
    """Small modality-specific adapter; far lighter than a second full YOLO backbone."""

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int | None = None):
        super().__init__()
        hidden = hidden_channels or max(16, out_channels // 4)
        self.net = nn.Sequential(
            ConvBNAct(in_channels, hidden, 1),
            ConvBNAct(hidden, out_channels, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IdentityConcatFusion(nn.Module):
    """Function-preserving mid-fusion baseline.

    Inputs are same-shape RGB/IR/Depth feature maps with C channels.  A 1x1 projection
    over their concatenation is initialized to [I, 0, 0], so the output is bitwise
    equal to the RGB feature at initialization while auxiliary kernel slices can receive
    gradients immediately.  This is the feature-level analogue of the Step-3 6ch stem.
    """

    def __init__(self, channels: int, n_modalities: int = 3):
        super().__init__()
        if n_modalities < 1:
            raise ValueError("n_modalities must be >=1")
        self.channels = channels
        self.n_modalities = n_modalities
        self.proj = nn.Conv2d(channels * n_modalities, channels, 1, bias=False)
        with torch.no_grad():
            self.proj.weight.zero_()
            eye = torch.eye(channels).view(channels, channels, 1, 1)
            self.proj.weight[:, :channels].copy_(eye)

    def forward(self, features: Iterable[torch.Tensor]) -> torch.Tensor:
        xs = list(features)
        if len(xs) != self.n_modalities:
            raise ValueError(f"expected {self.n_modalities} modalities, got {len(xs)}")
        base = xs[0].shape
        if any(x.shape != base for x in xs):
            raise ValueError(f"feature shape mismatch: {[tuple(x.shape) for x in xs]}")
        return self.proj(torch.cat(xs, dim=1))

    def modality_weight_norms(self) -> list[float]:
        w = self.proj.weight.detach()
        return [float(w[:, i * self.channels:(i + 1) * self.channels].norm())
                for i in range(self.n_modalities)]


class StrictOrthogonalDecorrelation(nn.Module):
    """Mathematically strict symmetric channel-vector decorrelation.

    For each spatial position, project x onto y and y onto x using scalar channel dot
    products.  This is intentionally distinct from RDTTrack's source implementation.
    """

    def __init__(self, channels: int, hidden_channels: int = 16, eps: float = 1e-6):
        super().__init__()
        self.x_proj = nn.Conv2d(channels, hidden_channels, 1, bias=False)
        self.y_proj = nn.Conv2d(channels, hidden_channels, 1, bias=False)
        self.out = nn.Conv2d(hidden_channels * 2, channels, 1, bias=False)
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))
        self.beta_logit = nn.Parameter(torch.tensor(0.0))
        self.eps = eps

    @staticmethod
    def _project(x: torch.Tensor, onto: torch.Tensor, eps: float) -> torch.Tensor:
        coef = (x * onto).sum(dim=1, keepdim=True) / (
            onto.square().sum(dim=1, keepdim=True) + eps
        )
        return coef * onto

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.shape != y.shape:
            raise ValueError(f"shape mismatch: {tuple(x.shape)} vs {tuple(y.shape)}")
        x0, y0 = self.x_proj(x), self.y_proj(y)
        alpha, beta = self.alpha_logit.sigmoid(), self.beta_logit.sigmoid()
        x_orth = x0 - alpha * self._project(x0, y0, self.eps)
        y_orth = y0 - beta * self._project(y0, x0, self.eps)
        return self.out(torch.cat([x_orth, y_orth], dim=1))


class RDTTrackStyleDecorrelation(nn.Module):
    """Independent reimplementation of the *style* of RDTTrack Depth/TIR decorrelation.

    The reference source performs elementwise products divided by a channel L2 norm;
    it is not the textbook vector projection.  This class preserves that distinction
    in its name so experiments cannot accidentally overclaim mathematical orthogonality.
    """

    def __init__(self, channels: int, hidden_channels: int = 16, eps: float = 1e-6):
        super().__init__()
        self.x_proj = nn.Conv2d(channels, hidden_channels, 1, bias=False)
        self.y_proj = nn.Conv2d(channels, hidden_channels, 1, bias=False)
        self.out = nn.Conv2d(hidden_channels * 2, channels, 1, bias=False)
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))
        self.beta_logit = nn.Parameter(torch.tensor(0.0))
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.shape != y.shape:
            raise ValueError(f"shape mismatch: {tuple(x.shape)} vs {tuple(y.shape)}")
        x0, y0 = self.x_proj(x), self.y_proj(y)
        xy = x0 * y0
        norm_y = torch.linalg.vector_norm(y0, ord=2, dim=1, keepdim=True)
        norm_x = torch.linalg.vector_norm(x0, ord=2, dim=1, keepdim=True)
        proj_x_on_y = (xy / (norm_y + self.eps)) * y0
        proj_y_on_x = (xy / (norm_x + self.eps)) * x0
        x_orth = x0 - self.alpha_logit.sigmoid() * proj_x_on_y
        y_orth = y0 - self.beta_logit.sigmoid() * proj_y_on_x
        return self.out(torch.cat([x_orth, y_orth], dim=1))


class SpatialFovea(nn.Module):
    """Small spatial softmax reweighting inspired by RDTTrack's prompt path."""

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(float(temperature)).log())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        t = self.log_temperature.exp().clamp_min(1e-4)
        flat = x.reshape(b, c, h * w)
        weights = torch.softmax(flat / t, dim=-1)
        return (weights * flat).reshape_as(x)


class ResidualPromptFusion(nn.Module):
    """RGB-anchor residual prompt fusion.

    The residual gate is initialized at zero, making the module exactly identity on RGB
    at initialization.  This protects pretrained RGB behavior.  After the gate moves
    away from zero, the prompt path learns auxiliary corrections.
    """

    def __init__(self, channels: int, hidden_channels: int = 16):
        super().__init__()
        self.rgb_reduce = nn.Conv2d(channels, hidden_channels, 1)
        self.aux_reduce = nn.Conv2d(channels, hidden_channels, 1)
        self.fovea = SpatialFovea()
        self.expand = nn.Conv2d(hidden_channels, channels, 1)
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, rgb: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        if rgb.shape != aux.shape:
            raise ValueError(f"shape mismatch: {tuple(rgb.shape)} vs {tuple(aux.shape)}")
        prompt = self.fovea(self.rgb_reduce(rgb)) + self.aux_reduce(aux)
        prompt = self.expand(F.silu(prompt))
        return rgb + torch.tanh(self.gate) * prompt


class SoftModalityGate(nn.Module):
    """Soft reliability gate for QAF; never hard-switches a modality.

    `quality_prior`, when supplied, is a BxM tensor added to learned modality logits.
    Higher quality prior monotonically increases a modality's softmax weight while
    keeping the learned visual gate in control.
    """

    def __init__(self, channels: int, n_modalities: int = 3, hidden: int = 32,
                 identity_start: bool = True):
        super().__init__()
        self.channels = channels
        self.n_modalities = n_modalities
        self.scorer = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.identity_start = identity_start
        self.residual_gate = nn.Parameter(torch.tensor(0.0 if identity_start else 4.0))

    def weights(self, features: list[torch.Tensor], quality_prior: torch.Tensor | None = None) -> torch.Tensor:
        if len(features) != self.n_modalities:
            raise ValueError(f"expected {self.n_modalities} modalities")
        pooled = torch.stack([x.mean(dim=(2, 3)) for x in features], dim=1)  # B,M,C
        logits = self.scorer(pooled).squeeze(-1)  # B,M
        if quality_prior is not None:
            if quality_prior.shape != logits.shape:
                raise ValueError(f"quality_prior {tuple(quality_prior.shape)} != {tuple(logits.shape)}")
            logits = logits + quality_prior.to(logits)
        return torch.softmax(logits, dim=1)

    def forward(self, features: list[torch.Tensor], quality_prior: torch.Tensor | None = None,
                *, return_weights: bool = False):
        base = features[0]
        if any(x.shape != base.shape for x in features):
            raise ValueError("all modality features must have identical shape")
        w = self.weights(features, quality_prior)
        mixed = sum(w[:, i, None, None, None] * x for i, x in enumerate(features))
        if self.identity_start:
            y = base + torch.tanh(self.residual_gate) * (mixed - base)
        else:
            y = mixed
        return (y, w) if return_weights else y


@dataclass(frozen=True)
class PyramidTapSpec:
    layer: int
    stride: int
    name: str


YOLO26_BACKBONE_TAPS = (
    PyramidTapSpec(4, 8, "P3"),
    PyramidTapSpec(6, 16, "P4"),
    PyramidTapSpec(10, 32, "P5"),
)


def inspect_yolo26_backbone_taps(model: nn.Module, imgsz: int = 640, in_channels: int = 3) -> dict:
    """Runtime-check official YOLO26 P3/P4/P5 tap points instead of trusting comments.

    `model` is expected to expose `model.model` like Ultralytics DetectionModel.
    Returns observed shapes/channels and fails if spatial stride is not 8/16/32.
    """
    # Accept both the YOLO wrapper (model.model = DetectionModel, .model.model = Sequential)
    # and the bare DetectionModel snapshot (model.model = Sequential).
    if hasattr(model, "model") and hasattr(model.model, "model"):
        core = model.model          # YOLO wrapper -> DetectionModel
        layers = core.model         # Sequential of layers
    elif hasattr(model, "model") and isinstance(model.model, torch.nn.Sequential):
        core = model                # bare DetectionModel
        layers = model.model
    else:
        raise TypeError("expected Ultralytics-like model with .model.model layers")

    observed = {}
    hooks = []
    for spec in YOLO26_BACKBONE_TAPS:
        def _hook(_m, _inp, out, spec=spec):
            tensor = out[0] if isinstance(out, (tuple, list)) else out
            if not torch.is_tensor(tensor):
                raise TypeError(f"tap {spec.name} returned {type(tensor)!r}")
            observed[spec.name] = tuple(tensor.shape)
        hooks.append(layers[spec.layer].register_forward_hook(_hook))

    device = next(core.parameters()).device
    dtype = next(core.parameters()).dtype
    was_training = core.training
    core.eval()
    try:
        with torch.no_grad():
            x = torch.zeros(1, in_channels, imgsz, imgsz, device=device, dtype=dtype)
            if hasattr(core, "_predict_once"):
                core._predict_once(x)
            else:
                core(x)
    finally:
        for h in hooks:
            h.remove()
        core.train(was_training)

    report = {}
    for spec in YOLO26_BACKBONE_TAPS:
        shape = observed.get(spec.name)
        if shape is None:
            raise RuntimeError(f"tap {spec.name} layer {spec.layer} was not observed")
        if len(shape) != 4:
            raise RuntimeError(f"tap {spec.name} expected BCHW, got {shape}")
        actual_stride_h = imgsz // shape[-2]
        actual_stride_w = imgsz // shape[-1]
        if (actual_stride_h, actual_stride_w) != (spec.stride, spec.stride):
            raise RuntimeError(
                f"{spec.name} layer {spec.layer} stride mismatch: "
                f"{actual_stride_h}x{actual_stride_w} != {spec.stride}"
            )
        report[spec.name] = {
            "layer": spec.layer,
            "stride": spec.stride,
            "channels": shape[1],
            "shape": list(shape),
        }
    return report
