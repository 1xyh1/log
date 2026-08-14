"""Trainability/freeze helpers for Step-4 RGB-anchor experiments."""
from __future__ import annotations

import torch
import torch.nn as nn


def set_requires_grad(module: nn.Module, value: bool) -> None:
    for p in module.parameters():
        p.requires_grad = value


def freeze_module(module: nn.Module, *, freeze_bn_stats: bool = True) -> None:
    """Freeze parameters and optionally BatchNorm running statistics."""
    set_requires_grad(module, False)
    if freeze_bn_stats:
        module.eval()
        for m in module.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()


def enforce_frozen_module_eval(module: nn.Module) -> None:
    """Call after an outer `model.train()` to keep a frozen RGB anchor truly frozen."""
    module.eval()
    for m in module.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


def train_only_modules(model: nn.Module, modules: dict[str, nn.Module]) -> list[str]:
    """Freeze everything, then enable only explicitly declared modules."""
    set_requires_grad(model, False)
    names: list[str] = []
    for label, module in modules.items():
        set_requires_grad(module, True)
        names.append(label)
    return names


def trainable_parameter_names(model: nn.Module) -> list[str]:
    return [n for n, p in model.named_parameters() if p.requires_grad]


def assert_all_finite_gradients(module: nn.Module) -> None:
    bad = []
    for name, p in module.named_parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            bad.append(name)
    if bad:
        raise RuntimeError(f"non-finite gradients: {bad}")
