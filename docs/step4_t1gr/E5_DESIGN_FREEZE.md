# E5 Hardened Step1 RGB Baseline Design Freeze

E5 is prepared before formal E4 execution, but formal E5 entry remains blocked until:

```text
E4 split freeze seal_gate_passed = true
E4 seal verification seal_verification_passed = true
E4 seal verification e5_entry_authorized = true
```

## Access model

E5 receives only:

```text
E4 TRAIN/DEV access artifact
```

It does not accept the E4 FINAL-HOLDOUT sealed artifact as an argument anywhere.

The Step1 RGB view is built by copying only `visible + labels` for TRAIN/DEV from the
formal ZIP. FINAL-HOLDOUT members remain outside the Step1 view.

## Public/private split

Public repo evidence:
- recipe hashes and explicit hyperparameters,
- TRAIN/DEV/FINAL counts and commitments,
- sanitized view/run/eval reports,
- no sample IDs,
- no absolute paths.

Private outside-repo:
- TRAIN/DEV ID access artifact,
- view manifest containing TRAIN/DEV IDs,
- copied RGB TRAIN/DEV view,
- Ultralytics args.yaml/results/weights.

## Scientific freeze

Formal training spec must be explicit. `optimizer=auto` is forbidden.

The bundle includes both a provenance candidate and the reviewed formal freeze.
Formal E5 consumes only:

```text
config/t1gr_e5_training_spec.frozen.json
```

The frozen spec resolves Ultralytics v8.4.56 `optimizer=auto` to explicit MuSGD
for this >10000-iteration regime, with the corresponding auto-branch
`warmup_bias_lr=0.0`. Its exact SHA256 is pinned in code; editing it causes fail-closed.

## Sequence

```text
E4 PASS
→ E5 recipe freeze
→ TRAIN/DEV-only RGB view
→ preflight
→ 1-epoch smoke
→ formal 80-epoch Step1 baseline
→ DEV-only evaluation
→ E5 final audit
→ T1-GR design-entry authorization
```

FINAL HOLDOUT remains sealed throughout E5.
