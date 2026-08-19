# E3 Final Closeout → E4 Freeze

Reviewed E3 commit:

```text
9835262acd8a23aa86ff7076909abbceb18060dc
```

## E3 final adjudication

Accepted:

```text
closure:
  final components        1829
  nontrivial              39
  max component           31
  gate                    PASS
  historical quarantine   18 IDs, no propagation expansion

split candidate:
  TRAIN                   1504
  DEV                      198
  FINAL HOLDOUT             298
  hard gate               PASS
  sample overlap          none
  component split         none
  union                   2000
  all class-min gates     PASS
```

Formal E3 result:

```text
E3 = PASS / CLOSED
```

Claim boundary remains:

> no exact/review-near-duplicate leakage under the frozen visual leakage-component rule

not true scene-independent generalization.

## E4 freeze semantics

E4 freezes exactly the reviewed E3 candidate. It is not allowed to seal another
`hard_gate_passed=true` candidate.

Private artifacts are separated:

```text
TRAIN/DEV ACCESS:
  contains TRAIN IDs + DEV IDs
  contains FINAL HOLDOUT count + commitment only
  contains no FINAL HOLDOUT IDs

FINAL HOLDOUT SEALED:
  contains FINAL HOLDOUT IDs
  open forbidden until T1-GR final adjudication
```

The public freeze contains counts and commitments only.

E4 does NOT itself authorize Step1 training. After E4 seal verification:

```text
E5 entry = authorized
Step1 training = still HOLD until E5 recipe/view gates pass
```
