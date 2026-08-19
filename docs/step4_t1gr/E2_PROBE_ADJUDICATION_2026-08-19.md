# E2 Probe Adjudication — 2026-08-19

## Current status

```text
E1 data available                  PASS
E2a ZIP member pairing             provisional positive evidence
E2b class map / label schema       freeze from official rule
E2c Depth physical semantics       HOLD
E2d scene/sequence/source group    HOLD
split                              HOLD
Step1                              HOLD
```

## Official facts frozen for E2

```text
n_classes = 12

0 person
1 boat
2 animal
3 seat
4 sign
5 bicycle
6 car
7 ball
8 light
9 garbage can
10 uav
11 tricycle

label = [class_id, norm_center_x, norm_center_y, norm_w, norm_h]
```

The official rule currently states that RGB / Infrared / Depth use a PNG+JPG mixture
and the three modalities of one sample keep the same extension.

It also states Depth is single-channel uint16 in millimeters.

These claims are not sufficient to define the actual JPG Depth numerical semantics.
The ZIP byte audit is authoritative for observed encoding.

## No group rule yet

Do NOT use the old 17-image:

```text
first_id_field_proxy_v1
```

as formal scene independence.

Before E3, resolve grouping from one of:
1. official metadata;
2. filename grammar with independently verified sequence meaning;
3. scene/sequence reconstruction audit.

If none is defensible:

```text
T1GR_SPLIT_HOLD
```

Do not fall back to random image split.

## Public/private evidence

Raw sample IDs / ID-run boundaries belong in private E2 evidence before holdout freeze.
Repo-visible reports should retain only aggregates and cryptographic commitments.

## Next gate

Run `t1gr_probe_zip_forensics.py`; review:
- all header summaries,
- all label validity,
- Depth PNG/JPG encoded precision/channels,
- runtime decoded dtype/shape/range,
- ID grammar aggregates.

No training until this closes.
