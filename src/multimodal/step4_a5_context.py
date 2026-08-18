"""A5 P5-centered cross-scale context construction helpers.

A5 keeps recipient RGB/GT/q fixed. P3/P4 context is always recipient-owned;
only the P5 AC residual source identity changes between recipient and donor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from multimodal.step4_a4_dc_ac import decompose_all, validate_component_trace

CONTEXT_ORDER = ("OO", "FO", "OF", "FF", "AO", "OA", "AF", "FA", "AA")
CONTEXT_STATES = {
    "OO": {"P3": "O", "P4": "O"},
    "FO": {"P3": "F", "P4": "O"},
    "OF": {"P3": "O", "P4": "F"},
    "FF": {"P3": "F", "P4": "F"},
    "AO": {"P3": "A", "P4": "O"},
    "OA": {"P3": "O", "P4": "A"},
    "AF": {"P3": "A", "P4": "F"},
    "FA": {"P3": "F", "P4": "A"},
    "AA": {"P3": "A", "P4": "A"},
}
ALLOWED_STATES = frozenset({"O", "F", "A"})


@dataclass(frozen=True)
class ContextBuild:
    context_id: str
    states: dict[str, str]
    active_scales: tuple[str, ...]
    replacements: dict[str, torch.Tensor]
    source_ids: dict[str, str]
    component_trace: dict[str, dict]
    p5_role: str
    p5_source_id: str


def validate_context_id(context_id: str) -> str:
    context_id = str(context_id)
    if context_id not in CONTEXT_STATES:
        raise ValueError(f"A5_UNKNOWN_CONTEXT:{context_id}")
    states = CONTEXT_STATES[context_id]
    if set(states) != {"P3", "P4"} or any(v not in ALLOWED_STATES for v in states.values()):
        raise RuntimeError(f"A5_CONTEXT_DEFINITION_INVALID:{context_id}:{states}")
    return context_id


def context_states(context_id: str) -> dict[str, str]:
    context_id = validate_context_id(context_id)
    return dict(CONTEXT_STATES[context_id])


def active_scales_for_context(context_id: str) -> tuple[str, ...]:
    states = context_states(context_id)
    active = [scale for scale in ("P3", "P4") if states[scale] != "O"]
    active.append("P5")
    return tuple(active)


def _residual(cache: Mapping, source_id: str, scale: str) -> torch.Tensor:
    try:
        value = cache[source_id][scale]
    except KeyError as exc:
        raise RuntimeError(f"A5_RESIDUAL_CACHE_MISSING:{source_id}:{scale}") from exc
    if not isinstance(value, torch.Tensor) or value.ndim != 4:
        raise RuntimeError(f"A5_RESIDUAL_CACHE_INVALID:{source_id}:{scale}")
    return value


def build_context(
    cache: Mapping,
    *,
    recipient_id: str,
    donor_id: str,
    context_id: str,
    p5_role: str,
) -> ContextBuild:
    """Construct the post-projection replacements for one A5 condition.

    P3/P4 are recipient-owned. State O means inactive, F means untouched recipient
    full residual (no replacement), and A means recipient AC_ALL replacement.
    P5 is always active and always AC_ALL; its source is recipient or frozen donor.
    """
    context_id = validate_context_id(context_id)
    recipient_id, donor_id = str(recipient_id), str(donor_id)
    role = str(p5_role).lower()
    if role not in {"native", "donor"}:
        raise ValueError(f"A5_BAD_P5_ROLE:{p5_role}")
    if donor_id == recipient_id:
        raise RuntimeError(f"A5_DONOR_SELF_MATCH:{recipient_id}")

    states = context_states(context_id)
    replacements: dict[str, torch.Tensor] = {}
    source_ids: dict[str, str] = {}
    components: dict[str, dict] = {}

    # Recipient-only P3/P4 context.
    for scale in ("P3", "P4"):
        state = states[scale]
        if state == "A":
            ac, evidence = decompose_all(_residual(cache, recipient_id, scale), source_id=recipient_id)
            evidence = {**evidence, "scale": scale, "a5_context_state": "A"}
            replacements[scale] = ac
            source_ids[scale] = f"AC_ALL[{recipient_id}]"
            components[scale] = evidence
        elif state in {"O", "F"}:
            # O is represented by an inactive scale; F uses the untouched recipient residual.
            pass
        else:  # pragma: no cover - validate_context_id already protects this
            raise RuntimeError(f"A5_CONTEXT_STATE_INVALID:{context_id}:{scale}:{state}")

    p5_source = recipient_id if role == "native" else donor_id
    p5_ac, p5_evidence = decompose_all(_residual(cache, p5_source, "P5"), source_id=p5_source)
    p5_evidence = {**p5_evidence, "scale": "P5", "a5_context_state": "TARGET_AC"}
    replacements["P5"] = p5_ac
    source_ids["P5"] = f"AC_ALL[{p5_source}]"
    components["P5"] = p5_evidence

    return ContextBuild(
        context_id=context_id,
        states=states,
        active_scales=active_scales_for_context(context_id),
        replacements=replacements,
        source_ids=source_ids,
        component_trace=components,
        p5_role=role,
        p5_source_id=p5_source,
    )


def validate_context_trace(
    trace: Mapping,
    *,
    recipient_id: str,
    donor_id: str,
    context_id: str,
    p5_role: str,
    tol: float = 1e-6,
) -> dict:
    """Validate A5 context semantics from runtime trace only."""
    recipient_id, donor_id = str(recipient_id), str(donor_id)
    context_id = validate_context_id(context_id)
    role = str(p5_role).lower()
    states = context_states(context_id)
    expected_p5_source = recipient_id if role == "native" else donor_id

    errors: list[str] = []
    if trace.get("recipient_id") != recipient_id:
        errors.append("RECIPIENT_ID")
    if trace.get("a5_context") != context_id:
        errors.append("CONTEXT_ID")
    if trace.get("a5_p5_role") != role:
        errors.append("P5_ROLE")
    if trace.get("a5_p5_source_id") != expected_p5_source:
        errors.append("P5_SOURCE_ID")
    if dict(trace.get("a5_context_states") or {}) != states:
        errors.append("CONTEXT_STATES")

    expected_active = set(active_scales_for_context(context_id))
    if set(trace.get("active_scales") or []) != expected_active:
        errors.append("ACTIVE_SCALES")

    residual_sources = trace.get("residual_source_ids") or {}
    components = trace.get("a5_component_trace") or {}

    # P3/P4 may never become donor-owned.
    for scale in ("P3", "P4"):
        state = states[scale]
        src = residual_sources.get(scale)
        if state == "A":
            if src != f"AC_ALL[{recipient_id}]":
                errors.append(f"{scale}_AC_SOURCE")
            comp = components.get(scale) or {}
            if (
                comp.get("residual_source_id") != recipient_id
                or comp.get("mean_source_id") != recipient_id
                or comp.get("mode") != "AC_ALL"
                or not validate_component_trace(comp, tol=tol)
            ):
                errors.append(f"{scale}_AC_COMPONENT")
        else:
            # forward_with_custom_residuals retains recipient source id even when O is inactive.
            if src != recipient_id:
                errors.append(f"{scale}_{state}_SOURCE")
            if scale in components:
                errors.append(f"{scale}_{state}_UNEXPECTED_COMPONENT")

    p5_src = residual_sources.get("P5")
    if p5_src != f"AC_ALL[{expected_p5_source}]":
        errors.append("P5_TRACE_SOURCE")
    p5_comp = components.get("P5") or {}
    if (
        p5_comp.get("mode") != "AC_ALL"
        or p5_comp.get("residual_source_id") != expected_p5_source
        or p5_comp.get("mean_source_id") != expected_p5_source
        or not validate_component_trace(p5_comp, tol=tol)
    ):
        errors.append("P5_COMPONENT")

    expected_components = {"P5"} | {s for s in ("P3", "P4") if states[s] == "A"}
    if set(components) != expected_components:
        errors.append("COMPONENT_SET")

    return {
        "passed": not errors,
        "errors": errors,
        "context_id": context_id,
        "states": states,
        "expected_active_scales": sorted(expected_active),
        "expected_p5_source_id": expected_p5_source,
    }


def validate_pair_isolation(native_trace: Mapping, donor_trace: Mapping) -> dict:
    """Prove native/donor conditions differ only in P5 AC source identity.

    The function intentionally compares provenance rather than numerical outputs.
    """
    errors: list[str] = []
    immutable_keys = (
        "recipient_id",
        "a5_context",
        "a5_context_states",
        "q_native",
        "active_scales",
        "alpha",
        "feature_strides",
    )
    for key in immutable_keys:
        if native_trace.get(key) != donor_trace.get(key):
            errors.append(f"PAIR_DIFF:{key}")

    nsrc = native_trace.get("residual_source_ids") or {}
    dsrc = donor_trace.get("residual_source_ids") or {}
    for scale in ("P3", "P4"):
        if nsrc.get(scale) != dsrc.get(scale):
            errors.append(f"PAIR_DIFF:{scale}_SOURCE")

    ncomp = native_trace.get("a5_component_trace") or {}
    dcomp = donor_trace.get("a5_component_trace") or {}
    for scale in ("P3", "P4"):
        if ncomp.get(scale) != dcomp.get(scale):
            errors.append(f"PAIR_DIFF:{scale}_COMPONENT")

    if native_trace.get("a5_p5_role") != "native":
        errors.append("NATIVE_ROLE")
    if donor_trace.get("a5_p5_role") != "donor":
        errors.append("DONOR_ROLE")
    if native_trace.get("a5_p5_source_id") == donor_trace.get("a5_p5_source_id"):
        errors.append("P5_IDENTITY_DID_NOT_CHANGE")

    return {"passed": not errors, "errors": errors}
