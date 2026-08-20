"""Seeded T1-GR model construction on top of the audited E5/T-series code.

The repository already contains ``t1gr_e5_core.build_seeded_model`` and
``tseries_p5_model.TSeriesP5Model``.  This additive adapter composes them while
keeping the physical module tree identical across G arms.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from .t1gr_e5_core import build_seeded_model, state_dict_sha256
from .t1gr_g_core import ARMS, SEEDS
from .tseries_p5_model import TSeriesP5Model

ARM_TO_TREATMENT = {
    "G0-N": "T0-N",
    "G1-P": "T1-F",
    "G2-S": "T1-F",
}


def _trainability(model: Any) -> dict[str, bool]:
    return {name: bool(parameter.requires_grad) for name, parameter in model.named_parameters()}


def build_t1gr_g_model(
    checkpoint: Path,
    e5_recipe: dict,
    *,
    arm: str,
    seed: int,
) -> tuple[TSeriesP5Model, dict]:
    """Build one arm without relaxing the E5 seed-before-head contract.

    The caller must build all three arms independently for a seed and compare
    the returned complete hashes, key lists, and trainability maps before any
    optimizer is constructed.
    """
    if arm not in ARMS:
        raise ValueError(f"T1GR_G_UNKNOWN_ARM:{arm}")
    if int(seed) not in SEEDS:
        raise ValueError(f"T1GR_G_UNFROZEN_SEED:{seed}")
    recipe = deepcopy(e5_recipe)
    recipe.setdefault("train_args", {})["seed"] = int(seed)

    reference, e5_initial = build_seeded_model(Path(checkpoint), recipe)

    # Reset the construction RNG before the added auxiliary modules.  This
    # produces a seed-specific but arm-independent auxiliary initialization.
    try:
        from ultralytics.utils.torch_utils import init_seeds
    except Exception as exc:  # pragma: no cover - exercised in target env
        raise RuntimeError("T1GR_G_MODEL_IMPORT_FAIL") from exc
    effective_seed = int(seed)  # RANK is required to be -1 by E5 builder.
    init_seeds(effective_seed, deterministic=bool(recipe["train_args"]["deterministic"]))

    model = TSeriesP5Model(
        reference,
        treatment_id=ARM_TO_TREATMENT[arm],
        freeze_rgb_backbone=False,
    )
    model.nc = 12
    class_names = recipe.get("class_names") or {}
    if set(class_names) != {str(i) for i in range(12)}:
        raise RuntimeError("T1GR_G_CLASS_MAP_FAIL")
    model.names = {int(key): str(value) for key, value in class_names.items()}
    state = model.state_dict()
    identity = {
        "seed": int(seed),
        "arm": arm,
        "model_treatment_id": ARM_TO_TREATMENT[arm],
        "reference_initial_state_sha256": e5_initial["model_initial_state_sha256"],
        "complete_initial_state_sha256": state_dict_sha256(state),
        "state_dict_keys": sorted(state),
        "requires_grad": _trainability(model),
        "zero_init_residual": bool(model.assert_zero_init()),
        "rgb_backbone_frozen": False,
    }
    if not identity["zero_init_residual"]:
        raise RuntimeError("T1GR_G_ZERO_INIT_FAIL")
    return model, identity


def assert_same_seed_arm_identity(identities: list[dict]) -> dict:
    if len(identities) != len(ARMS):
        raise ValueError("T1GR_G_IDENTITY_ARM_COUNT_FAIL")
    if {row.get("arm") for row in identities} != set(ARMS):
        raise ValueError("T1GR_G_IDENTITY_ARM_SET_FAIL")
    if len({int(row.get("seed")) for row in identities}) != 1:
        raise ValueError("T1GR_G_IDENTITY_SEED_MISMATCH")
    fields = (
        "reference_initial_state_sha256",
        "complete_initial_state_sha256",
        "state_dict_keys",
        "requires_grad",
        "zero_init_residual",
        "rgb_backbone_frozen",
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
        "arms": list(ARMS),
        "complete_initial_state_sha256": identities[0]["complete_initial_state_sha256"],
        "all_identity_fields_equal": True,
    }
