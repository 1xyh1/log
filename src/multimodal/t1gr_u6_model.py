"""Six-channel E5-v2 YOLO26 model construction for T1-U6."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .t1gr_u6_core import ARMS
from .t1gr_e5_core import build_seeded_model, state_dict_sha256
from .t1gr_g_core import SEEDS, sha256_json


def first_conv(model: Any) -> nn.Conv2d:
    try:
        layer = model.model[0].conv
    except (AttributeError, IndexError, TypeError) as exc:
        raise RuntimeError("T1GR_U6_FIRST_CONV_NOT_FOUND") from exc
    if not isinstance(layer, nn.Conv2d):
        raise RuntimeError("T1GR_U6_FIRST_CONV_CLASS_FAIL")
    return layer


def expand_first_conv_to_six(model: Any) -> nn.Conv2d:
    old = first_conv(model)
    if old.in_channels != 3 or old.groups != 1:
        raise RuntimeError("T1GR_U6_REFERENCE_STEM_CONTRACT_FAIL")
    new = nn.Conv2d(
        6,
        old.out_channels,
        old.kernel_size,
        old.stride,
        old.padding,
        old.dilation,
        1,
        old.bias is not None,
        old.padding_mode,
    ).to(device=old.weight.device, dtype=old.weight.dtype)
    new.train(old.training)
    new.weight.requires_grad_(old.weight.requires_grad)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :3].copy_(old.weight)
        if old.bias is not None:
            new.bias.copy_(old.bias)
            new.bias.requires_grad_(old.bias.requires_grad)
    model.model[0].conv = new
    if not isinstance(getattr(model, "yaml", None), dict):
        raise RuntimeError("T1GR_U6_MODEL_YAML_MISSING")
    model.yaml["channels"] = 6
    return new


def stem_contract(model: Any) -> dict:
    conv = first_conv(model)
    weight = conv.weight.detach().cpu()
    auxiliary = weight[:, 3:]
    return {
        "physical_in_channels": int(conv.in_channels),
        "out_channels": int(conv.out_channels),
        "rgb_weight_sha256": state_dict_sha256({"stem.rgb": weight[:, :3].contiguous()}),
        "aux_weight_sha256": state_dict_sha256({"stem.aux": auxiliary.contiguous()}),
        "aux_weight_nonzero_count": int(torch.count_nonzero(auxiliary).item()),
        "aux_weight_max_abs": float(auxiliary.abs().max().item()),
    }


def stem_zero_aux_equivalence(model: Any, *, seed: int) -> dict:
    """Numerically check the exact first-layer RGB/zero-aux equivalence."""
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    rgb = torch.randn(2, 3, 37, 41, generator=generator, dtype=torch.float32)
    layer = first_conv(model).cpu().eval()
    x6 = torch.cat((rgb, torch.zeros(2, 3, 37, 41)), dim=1)
    with torch.no_grad():
        actual = layer(x6)
        expected = F.conv2d(
            rgb,
            layer.weight[:, :3],
            layer.bias,
            layer.stride,
            layer.padding,
            layer.dilation,
            1,
        )
    max_abs = float((actual - expected).abs().max().item())
    return {"max_abs_error": max_abs, "threshold": 1e-5, "passed": max_abs <= 1e-5}


def full_detector_zero_aux_equivalence(checkpoint: Path, e5_recipe: dict, *, seed: int) -> dict:
    """Compare the seeded 3ch detector with its 6ch zero-aux descendant."""
    recipe = deepcopy(e5_recipe)
    recipe.setdefault("train_args", {})["seed"] = int(seed)
    reference, _ = build_seeded_model(Path(checkpoint), recipe)
    expanded = deepcopy(reference)
    expand_first_conv_to_six(expanded)
    reference_state, expanded_state = reference.state_dict(), expanded.state_dict()
    nonstem = [key for key in reference_state if key != "model.0.conv.weight"]
    nonstem_equal = bool(
        set(reference_state) == set(expanded_state)
        and all(torch.equal(reference_state[key], expanded_state[key]) for key in nonstem)
    )

    def tensors(value: Any, out: list[torch.Tensor]) -> list[torch.Tensor]:
        if isinstance(value, torch.Tensor):
            out.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                tensors(item, out)
        elif isinstance(value, (list, tuple)):
            for item in value:
                tensors(item, out)
        return out

    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 303)
    rgb = torch.rand(1, 3, 160, 160, generator=generator)
    six = torch.cat((rgb, torch.zeros(1, 3, 160, 160)), dim=1)
    reference.eval()
    expanded.eval()
    with torch.no_grad():
        out3 = tensors(reference._predict_once(rgb), [])
        out6 = tensors(expanded._predict_once(six), [])
    if len(out3) != len(out6) or any(left.shape != right.shape for left, right in zip(out3, out6)):
        raise RuntimeError("T1GR_U6_EQUIVALENCE_OUTPUT_STRUCTURE_FAIL")
    max_abs = max((float((left - right).abs().max().item()) for left, right in zip(out3, out6)), default=0.0)
    passed = bool(nonstem_equal and max_abs <= 1e-5)
    return {
        "seed": int(seed),
        "nonstem_state_bitwise_equal": nonstem_equal,
        "output_tensor_count": len(out3),
        "final_detector_max_abs_error": max_abs,
        "threshold": 1e-5,
        "passed": passed,
    }


def build_t1gr_u6_model(
    checkpoint: Path,
    e5_recipe: dict,
    *,
    arm: str,
    seed: int,
) -> tuple[Any, dict]:
    if arm not in ARMS:
        raise ValueError(f"T1GR_U6_UNKNOWN_ARM:{arm}")
    if int(seed) not in SEEDS:
        raise ValueError(f"T1GR_U6_UNFROZEN_SEED:{seed}")
    recipe = deepcopy(e5_recipe)
    recipe.setdefault("train_args", {})["seed"] = int(seed)
    model, reference = build_seeded_model(Path(checkpoint), recipe)
    expand_first_conv_to_six(model)
    head = model.model[-1]
    if int(getattr(head, "nc", -1)) != 12:
        raise RuntimeError("T1GR_U6_HEAD_NC_FAIL")
    if bool(getattr(head, "end2end", getattr(model, "end2end", False))) is not True:
        raise RuntimeError("T1GR_U6_END2END_FAIL")
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("T1GR_U6_TRAINABILITY_FAIL")
    contract = stem_contract(model)
    equivalence = stem_zero_aux_equivalence(model, seed=int(seed) + 101)
    if contract["physical_in_channels"] != 6 or contract["aux_weight_nonzero_count"] != 0:
        raise RuntimeError("T1GR_U6_AUX_INITIALIZATION_FAIL")
    if not equivalence["passed"]:
        raise RuntimeError("T1GR_U6_ZERO_AUX_EQUIVALENCE_FAIL")
    state = model.state_dict()
    identity = {
        "seed": int(seed),
        "arm": arm,
        "model_class": type(model).__name__,
        "reference_initial_state_sha256": reference["model_initial_state_sha256"],
        "complete_initial_state_sha256": state_dict_sha256(state),
        "state_dict_keys_sha256": sha256_json(sorted(state)),
        "trainable_parameter_count": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
        "total_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "stem": contract,
        "rgb_zero_aux_equivalence": equivalence,
        "physical_head_nc": 12,
        "end2end": True,
        "loss_class": "E2ELoss",
        "claim": "AUDITABLE_INITIALIZATION_ONLY",
    }
    return model, identity


def assert_same_seed_arm_identity(identities: list[dict]) -> dict:
    if len(identities) != len(ARMS) or {row.get("arm") for row in identities} != set(ARMS):
        raise RuntimeError("T1GR_U6_IDENTITY_MATRIX_FAIL")
    if len({int(row.get("seed", -1)) for row in identities}) != 1:
        raise RuntimeError("T1GR_U6_IDENTITY_SEED_FAIL")
    fields = (
        "model_class",
        "reference_initial_state_sha256",
        "complete_initial_state_sha256",
        "state_dict_keys_sha256",
        "trainable_parameter_count",
        "total_parameter_count",
        "stem",
        "physical_head_nc",
        "end2end",
        "loss_class",
    )
    drift = {field: [row.get(field) for row in identities] for field in fields if identities[0].get(field) != identities[1].get(field)}
    if drift:
        raise RuntimeError(f"T1GR_U6_INITIAL_IDENTITY_DRIFT:{sorted(drift)}")
    return {
        "seed": int(identities[0]["seed"]),
        "complete_initial_state_sha256": identities[0]["complete_initial_state_sha256"],
        "same_parameter_count": True,
        "all_identity_fields_equal": True,
    }
