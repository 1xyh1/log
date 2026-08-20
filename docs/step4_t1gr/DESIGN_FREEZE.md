# T1-GR Formal Design Freeze

Status: `FROZEN_FOR_IMPLEMENTATION_REVIEW`  
Upstream accepted baseline: E5 v2 commit `6f14392358b6fdb35b9b9e4a6e1c814a90442902`

## 1. Question

On fresh formal-data DEV evidence, does correctly paired IR during training
improve generalization over both NULL training and balanced fully-wrong IR
training?

This is a source-specificity replication.  It is not another search over old
`val6`, not a hyperparameter sweep, and not authorization to inspect FINAL
HOLDOUT.

## 2. Arms

| Arm | IR source during training | P5 residual in detection graph |
|---|---|---|
| `G0-N` | paired IR is evaluated only for matched auxiliary-buffer exposure | NULL |
| `G1-P` | correctly paired IR | FULL |
| `G2-S` | deterministic balanced fully-wrong donor IR | FULL |

The physical model tree is identical.  `G0-N` uses treatment `T0-N`; `G1-P`
and `G2-S` both use `T1-F`.  No P3/P4 injection and no reliability gate are
allowed.

## 3. Frozen seeds and order

```text
20260812: G0-N -> G1-P -> G2-S
20260813: G1-P -> G2-S -> G0-N
20260814: G2-S -> G0-N -> G1-P
```

This Latin-square order is fixed before any G result exists.  No seed may be
added, removed, replaced, rerun selectively, or chosen after seeing a metric.
A failed run may be recovered only under a separately logged, arm-blind
recovery policy.

## 4. Matching contract

Within a seed, all arms must have the same:

- pinned checkpoint and transfer set;
- physical model class and state-dict keys;
- complete initial model-state hash;
- `requires_grad` map;
- optimizer groups and hyperparameters;
- batch, `nbs`, image size, sampling, augmentation, epochs;
- checkpoint policy and DEV validation cadence.

The only treatment is training-time IR source condition.  The formal runner
must compare all recorded identities before the first optimizer step and fail
closed on any mismatch.

## 5. E5 v2 inheritance

G arms inherit the E5 v2 baseline:

```text
YOLO26s / nc=12 / end2end=true
epochs=80 / batch=4 / nbs=64 / imgsz=640
optimizer=MuSGD
lr0=0.01 / lrf=0.01 / momentum=0.9 / weight_decay=0.0005
amp=true / deterministic=true
DEV every epoch
last.pt primary; best.pt diagnostic only
DEV eval: conf=0.001 / iou=0.7 / max_det=100
```

The obsolete T-series small-data R3 optimizer profile is forbidden.  The
audited P5-only topology is reused, but `freeze_rgb_backbone=false`; all G arms
remain full-trainable descendants of the E5 mother baseline.

The multimodal loader must reproduce E5 visible-image augmentation.  All
geometry is keyed by the recipient and applied identically to its RGB, labels,
and selected IR source.  G2 substitutes only IR identity before applying that
same geometry.  Formal training stays blocked until a smoke audit proves this.

## 6. G2 schedule

For each seed, TRAIN IDs are ordered by:

```text
sha256(str(seed) || NUL || sample_id), then sample_id
```

At zero-based epoch `e`, with `N` TRAIN samples:

```text
shift = 1 + (e mod (N - 1))
donor[ids[i]] = ids[(i + shift) mod N]
```

Required facts:

- each epoch is a bijection;
- self-match count is zero;
- every donor is used once per epoch;
- for N=1,504 and 80 epochs, each recipient receives 80 distinct donors;
- every recipient/donor pair actually consumed is logged;
- no claim is made that 80 epochs exhaust all 1,503 possible donors.

## 7. Primary evidence and folds

Primary endpoint: DEV `mAP50-95` from `last.pt`, `max_det=100`.

Five sensitivity folds are built from E3 leakage components, never individual
images.  Components are hash-sorted and assigned round-robin.  Sensitivity is
five leave-one-fold-out recomputations of DEV mAP50-95.

Official contrasts:

```text
G1-P - G0-N
G1-P - G2-S
G2-S - G0-N
```

A contrast is `STABLE_POSITIVE` only if it is strictly positive for every
frozen seed.  No post-hoc epsilon or practical-equivalence margin is allowed.
The positive fold gate requires at least four of five positive LOFO deltas for
each required seed/contrast.

Decision branches are frozen in
`config/t1gr_g_design.frozen.json`.  `depth_go` and `production_go` remain false
for every DEV-only result.  FINAL HOLDOUT opening needs a separate final
adjudication artifact and is never implied by this design.

## 8. Result input schema

`reports/step4_t1gr/per_seed_results.json`:

```json
{
  "schema": "t1gr-g-per-seed-results-v1",
  "final_holdout_accessed": false,
  "metric": "mAP50-95",
  "checkpoint": "last.pt",
  "max_det": 100,
  "rows": [
    {
      "seed": 20260812,
      "arm": "G0-N",
      "dev_map50_95": 0.0,
      "lofo_map50_95": {
        "fold_0": 0.0,
        "fold_1": 0.0,
        "fold_2": 0.0,
        "fold_3": 0.0,
        "fold_4": 0.0
      },
      "run_manifest_sha256": "64 lowercase hex characters",
      "last_checkpoint_sha256": "64 lowercase hex characters"
    }
  ]
}
```

Exactly nine rows are required: one for every frozen seed/arm pair.  Metrics
must be finite numbers in `[0,1]`.  Duplicate, missing, extra, best-checkpoint,
wrong-cap, or holdout-accessed evidence fails closed.

## 9. Current authorization

```text
design entry                    authorized
implementation/preflight entry after passing design audit
smoke training                  not yet authorized
formal multi-seed training      not authorized
FINAL HOLDOUT open              not authorized
depth                           no-go
production                      no-go
```

