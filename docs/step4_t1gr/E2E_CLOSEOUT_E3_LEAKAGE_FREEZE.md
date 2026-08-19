# T1-GR E2e Closeout + E3 Leakage Audit Freeze

日期：2026-08-20

## E2e label adjudication

Formal taxonomy at `d0858542`:

```text
hard_schema_or_class_samples              0
ultralytics_8_4_56_reject_samples         0
strict_[0,1]_but_ultralytics_tolerates    3
corner_overflow_without_hard/reject       332
duplicate_row_samples                     2
```

Formal result:

```text
E2e LABEL VALIDITY
  PASS
```

Policy: official labels unchanged; no clipping, deletion, or exclusion. Derived-corner overflow is diagnostic-only under the frozen Ultralytics 8.4.56 label verifier. Duplicate rows remain raw and loader behavior is recorded.

## Historical contamination

The historical 17 images influenced A2→T1-TR. Therefore any formal-2000 sample matching old17 by the frozen Visible+IR audit is:

```text
NEVER FINAL HOLDOUT
```

Depth is excluded from the similarity audit because formal Depth has two incompatible encoding domains and T1-GR does not use Depth.

Historical classes:

```text
EXACT_BOTH_MODALITIES       -> FORCE_TRAIN
STRONG_NEAR_DUPLICATE       -> FORCE_TRAIN
REVIEW_NEAR_DUPLICATE       -> FORCE_TRAIN_CONSERVATIVE
NONE                        -> eligible for later split
```

This audit never chooses TRAIN/DEV/HOLDOUT.

## Formal2000 internal leakage graph

Only EXACT_BOTH_MODALITIES and STRONG_NEAR_DUPLICATE create graph edges. Their connected components are the minimum indivisible leakage units for E3. REVIEW_NEAR_DUPLICATE edges are reported for review but are not auto-connected.

JPG/PNG encoding class is a stratification variable, not a scene/group identity.

If there is no independent scene/sequence metadata, the final allowed claim is only:

> no exact/strong-near-duplicate leakage under the frozen visual leakage-component rule.

Do not claim true scene-independent generalization.

## Route

```text
E1                         PASS
E2 structure              PASS
E2 class map              PASS
E2 Depth encoding facts   PASS
E2 IR encoding facts      PASS
E2 label validity         PASS

E3 leakage graph          RUN NEXT
E3 split                  HOLD
E4 seal                   HOLD
E5 Step1                  HOLD
T1-GR training            HOLD
Depth                     HOLD
Production                HOLD
```
