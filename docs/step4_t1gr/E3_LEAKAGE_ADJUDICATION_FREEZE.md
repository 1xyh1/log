# E3 Leakage Adjudication Freeze — Hardened v1.2

Scientific adjudication remains unchanged from the accepted `b15f74a` evidence.

```text
historical contaminated seed IDs = 18
source strong edges              = 808
source review edges              = 180
source strong-only nontrivial components = 25
source strong-only max component = 27
```

## Final review-edge adjudication

```text
MERGE_ALL_CONSERVATIVELY
```

Final graph:

```text
strong ∪ review
```

Final E3 leakage units:

```text
connected_components(final graph)
```

Historical quarantine propagation:

```text
if final_component intersects historical_seed_18:
    entire final_component = FORCE_TRAIN
```

## Split candidate inputs

Only:

```text
formal AIC2026_Train_2000.zip
PRIVATE leakage_components_final_private.json
frozen t1gr_e3_closure_split_policy.json
```

The split generator must not consume any model metric or old val6 evidence.

## Security hardening

v1.2 adds fail-closed checks for null/missing fields, duplicate IDs/edges, concurrency,
path/permission violations, timeout/resource limits, atomic/idempotent output writes,
ZIP safety, and public-report sensitive-data scanning. These are engineering controls only;
they do not alter the scientific treatment or split objective.
