# E2 Formal Adjudication After ZIP Forensics

日期：2026-08-19

## 1. Depth dual encoding

### Class A — 1851 PNG

Observed:

```text
1920×1080
PNG
16-bit
1 encoded channel
cv2 IMREAD_UNCHANGED -> uint16 H×W
sample max = 19999
```

Frozen semantic class:

```text
DEPTH_METRIC_U16_MM
```

### Class B — 149 JPG

Observed:

```text
640×360
JPEG
8-bit
3 encoded channels
cv2 IMREAD_UNCHANGED -> uint8 H×W×3
sample channel differences = 0
range 0..255
```

Frozen semantic class:

```text
DEPTH_U8_GRAY_DERIVED_UNKNOWN_SCALE
```

禁止：

```text
×255
linear mm reconstruction
claiming metric millimeters
```

T1-GR does not use Depth:

```text
depth_training_use = QUARANTINED
depth_go = false
```

D-series requires a separate calibration/semantic qualification stage.

## 2. IR is also a mixed representation domain

Observed:

```text
149 JPG:
  640×360, uint8, 3ch
  sampled channel differences = 0

1851 PNG:
  1920×1080, uint8, 3ch
  sampled channel differences non-zero, sometimes large
```

Therefore do not write:

```text
IR = replicated grayscale 3ch
```

for the full 2000-image dataset.

For E2 contract, freeze observed encoding facts only.
Final scalar/3ch IR preprocessing belongs to T1-GR design after TRAIN/DEV split freeze;
FINAL HOLDOUT must not be used to choose it.

Old-17 median-channel preprocessing is a historical baseline, not an automatic formal-data truth.

## 3. Labels

The forensic script reports 332 samples under its strict checker, but that checker includes a
derived-corner condition (`cx±w/2`, `cy±h/2` must remain in [0,1]).

Ultralytics 8.4.56 verification does not use that derived-corner condition. It checks:
- effective 5-column detect labels;
- normalized coordinate scalar maximum <= 1.01;
- overall minimum >= -0.01;
- class index < num_classes.

Therefore:

```text
332 != 332 proven corrupt labels
```

Run `t1gr_label_error_taxonomy.py`.

Formal policy after taxonomy:

```text
HARD_SCHEMA_OR_CLASS
  => E2 FAIL

ULTRALYTICS_8_4_56_REJECT
  => Step1 FAIL / E2 HOLD

STRICT_[0,1]_ONLY
  => explicit tolerance adjudication required

DERIVED_CORNER_OVERFLOW only
  => diagnostic; does not by itself invalidate official labels for the frozen trainer

DUPLICATE_ROWS
  => report; raw labels stay unchanged; effective loader behavior must be logged

EMPTY_BACKGROUND
  => valid if no objects
```

Do not delete or clip official labels before taxonomy.

## 4. Group / leakage rule

ID grammar is not a defensible scene rule:

```text
single-token IDs: 1052
unique first token: 1072/2000
unique first-two-token: 733
```

Formal rule remains:

```text
group_rule_status = UNRESOLVED
```

Next E3 preparation must build a conservative leakage graph:

```text
A. historical-contamination audit
   formal2000 vs old17
   exact hash + perceptual/re-encoding similarity

B. exact/near-duplicate graph inside formal2000

C. connected components become minimum indivisible leakage groups

D. JPG/PNG source class is a stratification variable, not automatically a scene group

E. class image-count + box-count balance applied at group level
```

If no stronger official/metadata scene grouping appears, the eventual claim must be:

> no exact/near-duplicate leakage under the frozen visual leakage-component rule

and NOT:

> true scene-independent generalization.

## 5. Historical contamination is mandatory

The old 17 images have already influenced A2→T1-TR.

Any formal-2000 image that is an exact or perceptual match to those historical images:

```text
NEVER FINAL HOLDOUT
```

It should be quarantined to TRAIN (or excluded from T1-GR adjudication), with the choice frozen
before split generation.

## 6. Current route

```text
E1                              PASS
E2 structure/pairing             PASS
E2 class map                     PASS
E2 Depth encoding facts          PASS
E2 IR encoding facts             PASS
E2 label validity                HOLD: taxonomy
E2/E3 leakage grouping           HOLD

SPLIT                            HOLD
STEP1                            HOLD
T1-GR TRAINING                   HOLD
Depth                            HOLD
Production                       HOLD
```
