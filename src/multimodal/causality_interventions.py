"""Shared causality interventions for Step 3 / Step 4 SHUFFLE variants.

Frozen requirements (reviewer 2026-08-15):
    bijective mapping over the sample set (each donor used exactly once);
    no self-match (derangement);
    cross-group whenever mathematically possible;
    deterministic (same ids -> same mapping).

A full cross-group derangement exists for all three probe sets (train11 / val6 /
all17) — verified in tests; the fallback for the final element picks any non-self
donor (cross-group still holds on these sets).
"""
from __future__ import annotations

from multimodal.raw_sample_index import group_of


def _perfect_cross_group_matching(ids: list[str]) -> dict[str, str] | None:
    """Deterministic bipartite matching: every sample gets a unique cross-group donor."""
    candidates = {
        sid: sorted(d for d in ids if d != sid and group_of(d) != group_of(sid))
        for sid in ids
    }
    left = sorted(ids, key=lambda sid: (len(candidates[sid]), sid))
    donor_to_sid: dict[str, str] = {}

    def dfs(sid: str, seen: set[str]) -> bool:
        for donor in candidates[sid]:
            if donor in seen:
                continue
            seen.add(donor)
            old = donor_to_sid.get(donor)
            if old is None or dfs(old, seen):
                donor_to_sid[donor] = sid
                return True
        return False

    if not all(dfs(sid, set()) for sid in left):
        return None
    return {sid: donor for donor, sid in donor_to_sid.items()}


def bijective_derangement(ids: list[str]) -> dict[str, str]:
    ids = list(ids)
    result = _perfect_cross_group_matching(ids)
    if result is None:
        # deterministic cyclic fallback: search rotations until no self match
        ordered = sorted(ids)
        for shift in range(1, len(ordered)):
            cand = {sid: ordered[(i + shift) % len(ordered)] for i, sid in enumerate(ordered)}
            if all(s != d for s, d in cand.items()):
                result = cand
                break
    if result is None or set(result) != set(ids) or set(result.values()) != set(ids) \
            or any(s == d for s, d in result.items()):
        raise RuntimeError("cannot build a bijective no-self derangement for this id set")
    return result


def assert_valid_shuffle_map(mapping: dict[str, str], ids: list[str]) -> bool:
    """Formal SHUFFLE gate: bijection + no self-match."""
    return (set(mapping.keys()) == set(ids)
            and set(mapping.values()) == set(ids)
            and all(k != v for k, v in mapping.items())
            and len(set(mapping.values())) == len(ids))
