# E3 v1.1 → v1.2 Hardened Change Record

## Scientific logic

**UNCHANGED.**

```text
18 historical contaminated seed IDs
180 review edges -> merge all conservatively
final graph = strong ∪ review
connected components = indivisible groups
component touching historical seed -> whole component FORCE_TRAIN
```

## P0 fixed before formal execution

v1.1 split code contained:

```python
coverage = set().union(*sets) == set(stats)
```

where `sets` is a dict. Python therefore unioned the dict keys (`train`, `dev`,
`final_holdout`) instead of the three ID sets. This would make `union_equals_2000`
incorrect and force a false HOLD.

v1.2 fixes it to:

```python
coverage = set().union(*(sets[sp] for sp in SPLITS)) == set(stats)
```

The 2000-ID full synthetic integration gate explicitly requires
`union_equals_2000 == true`.

**Do not execute v1.1 formally.**

## Engineering hardening added

- input/public/private path scope guards;
- non-null schema validation;
- duplicate ID/edge/self-edge checks;
- upstream strong-component reconstruction;
- accepted `b15f74a` public/private provenance cross-check;
- historical force-ID commitment cross-check;
- frozen policy SHA hard pin;
- request fingerprint + payload SHA integrity;
- same-request idempotency;
- conflicting-request output refusal;
- concurrency lock + stale lock rules;
- atomic fsync + replace writes;
- timeout/deadline checks;
- input-change-during-run detection;
- ZIP member/path/encryption/size caps;
- public sensitive-data scanner;
- exception redaction;
- private POSIX mode 0600 where meaningful;
- synthetic 2000-ID end-to-end integration gate.
