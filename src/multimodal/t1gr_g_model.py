"""Formal T1-GR P5-only model adapter for the E5 v2 YOLO26 end-to-end baseline."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .t1gr_e5_core import build_seeded_model, state_dict_sha256
from .t1gr_g_core import ARMS, SEEDS, sha256_json
from .tseries_p5_model import TSeriesP5Model

ARM_TO_TREATMENT = {"G0-N": "T0-N", "G1-P": "T1-F", "G2-S": "T1-F"}


class T1GRP5Model(TSeriesP5Model):
    """T-series topology with formal E5 trainability, four-channel input, and E2E loss."""

    def _split_input(self, x):
        channels = int(x.shape[1])
        if channels == 4:
            infrared = x[:, 3:4]
            return x[:, :3], torch.cat((infrared, torch.zeros_like(infrared)), dim=1)
        return super()._split_input(x)

    def train(self, mode: bool = True):
        # Bypass the old small-data class' frozen-backbone eval enforcement.
        nn.Module.train(self, mode)
        return self

    @property
    def end2end(self) -> bool:
        return bool(getattr(self.tail[-1], "end2end", False))

    @end2end.setter
    def end2end(self, value: bool) -> None:
        self.set_head_attr(end2end=bool(value))

    def set_head_attr(self, **kwargs) -> None:
        head = self.tail[-1]
        for key, value in kwargs.items():
            if not hasattr(head, key):
                raise AttributeError(f"T1GR_G_HEAD_ATTRIBUTE_MISSING:{key}")
            setattr(head, key, value)

    def init_criterion(self):
        from ultralytics.utils.loss import E2ELoss, v8DetectionLoss

        return E2ELoss(self) if self.end2end else v8DetectionLoss(self)

    def loss(self, batch, preds=None):
        if self.criterion is None:
            self.criterion = self.init_criterion()
        if preds is None:
            preds = self._predict_once(batch["img"])
        return self.criterion(preds, batch)


def _trainability(model: Any) -> dict[str, bool]:
    return {name: bool(parameter.requires_grad) for name, parameter in model.named_parameters()}


def build_t1gr_g_model(
    checkpoint: Path,
    e5_recipe: dict,
    *,
    arm: str,
    seed: int,
) -> tuple[T1GRP5Model, dict]:
    if arm not in ARMS:
        raise ValueError(f"T1GR_G_UNKNOWN_ARM:{arm}")
    if int(seed) not in SEEDS:
        raise ValueError(f"T1GR_G_UNFROZEN_SEED:{seed}")
    recipe = deepcopy(e5_recipe)
    recipe.setdefault("train_args", {})["seed"] = int(seed)
    reference, e5_initial = build_seeded_model(Path(checkpoint), recipe)
    # TSeriesP5Model reads reference.nc during construction; build_seeded_model
    # returns a stock DetectionModel without that attribute set.
    reference.nc = int(reference.model[-1].nc)

    try:
        from ultralytics.utils.torch_utils import init_seeds
    except Exception as exc:  # pragma: no cover - target environment dependency
        raise RuntimeError("T1GR_G_MODEL_IMPORT_FAIL") from exc
    init_seeds(int(seed), deterministic=bool(recipe["train_args"]["deterministic"]))
    model = T1GRP5Model(
        reference,
        treatment_id=ARM_TO_TREATMENT[arm],
        freeze_rgb_backbone=False,
    )
    model.yaml["channels"] = 4
    model.nc = 12
    class_names = recipe.get("class_names") or {}
    if set(class_names) != {str(i) for i in range(12)}:
        raise RuntimeError("T1GR_G_CLASS_MAP_FAIL")
    model.names = {int(key): str(value) for key, value in class_names.items()}
    if model.end2end is not True:
        raise RuntimeError("T1GR_G_END2END_HEAD_FAIL")
    state = model.state_dict()
    rgb_backbone_trainable = all(parameter.requires_grad for parameter in model.rgb_backbone.parameters())
    auxiliary_trainable = all(parameter.requires_grad for parameter in model.aux_encoder.parameters())
    fusion_trainable = all(parameter.requires_grad for parameter in model.p5_fusion.parameters())
    identity = {
        "seed": int(seed),
        "arm": arm,
        "model_class": type(model).__name__,
        "model_treatment_id": ARM_TO_TREATMENT[arm],
        "reference_initial_state_sha256": e5_initial["model_initial_state_sha256"],
        "complete_initial_state_sha256": state_dict_sha256(state),
        "state_dict_keys": sorted(state),
        "requires_grad": _trainability(model),
        "zero_init_residual": bool(model.assert_zero_init()),
        "rgb_backbone_frozen": not rgb_backbone_trainable,
        "rgb_backbone_all_trainable": rgb_backbone_trainable,
        "auxiliary_all_trainable": auxiliary_trainable,
        "fusion_all_trainable": fusion_trainable,
        "end2end": bool(model.end2end),
        "loss_class": "E2ELoss",
        "claim": "AUDITABLE_INITIALIZATION_ONLY",
    }
    if not identity["zero_init_residual"]:
        raise RuntimeError("T1GR_G_ZERO_INIT_FAIL")
    if not (rgb_backbone_trainable and auxiliary_trainable and fusion_trainable):
        raise RuntimeError("T1GR_G_REQUIRED_TRAINABILITY_FAIL")
    return model, identity


def assert_same_seed_arm_identity(identities: list[dict]) -> dict:
    if len(identities) != len(ARMS) or {row.get("arm") for row in identities} != set(ARMS):
        raise ValueError("T1GR_G_IDENTITY_ARM_MATRIX_FAIL")
    if len({int(row.get("seed")) for row in identities}) != 1:
        raise ValueError("T1GR_G_IDENTITY_SEED_MISMATCH")
    fields = (
        "model_class",
        "reference_initial_state_sha256",
        "complete_initial_state_sha256",
        "state_dict_keys",
        "requires_grad",
        "zero_init_residual",
        "rgb_backbone_frozen",
        "rgb_backbone_all_trainable",
        "auxiliary_all_trainable",
        "fusion_all_trainable",
        "end2end",
        "loss_class",
    )
    drift = {
        field: {row["arm"]: row.get(field) for row in identities}
        for field in fields
        if len({repr(row.get(field)) for row in identities}) != 1
    }
    if drift:
        raise RuntimeError(f"T1GR_G_INITIAL_IDENTITY_DRIFT:{drift}")
    return {
        "seed": int(identities[0]["seed"]),
        "complete_initial_state_sha256": identities[0]["complete_initial_state_sha256"],
        "reference_initial_state_sha256": identities[0]["reference_initial_state_sha256"],
        "state_dict_keys_sha256": sha256_json(identities[0]["state_dict_keys"]),
        "requires_grad_map_sha256": sha256_json(identities[0]["requires_grad"]),
        "all_identity_fields_equal": True,
        "claim": "AUDITABLE_INITIALIZATION_ONLY",
    }
