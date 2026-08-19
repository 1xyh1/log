# E5 Step1 Recipe Adjudication

## Frozen mother baseline

```text
model       YOLO26s
nc          12
end2end     true
epochs      80
batch       4
imgsz       640
seed        20260812
```

The remaining training/evaluation settings are explicitly frozen rather than inherited implicitly.

## Optimizer

`optimizer=auto` is forbidden in formal E5.

For the formal split:

```text
TRAIN images = 1504
batch        = 4
epochs       = 80

estimated batches/epoch = ceil(1504/4) = 376
estimated iterations    = 376 * 80 = 30080
```

Ultralytics 8.4.56 selects the MuSGD auto branch above 10000 iterations.

Therefore the source-equivalent resolved values are frozen before execution:

```text
optimizer       MuSGD
lr0             0.01
momentum        0.9
warmup_bias_lr  0.0
```

This is a deterministic resolution of the version-pinned auto branch, not post-result tuning.

## AMP

`amp=true` is frozen.

The framework's default AMP helper may try to instantiate/download another YOLO checkpoint.
E5 blocks that network-capable probe. Instead:

```text
mandatory 1-epoch smoke with actual amp=true
→ runtime AMP/batch/workers/optimizer checks
→ only then formal 80-epoch authorization
```

## No early-stop shortcut

`patience=100` with `epochs=80`, and the formal gate additionally requires exactly 80 rows in
`results.csv`. A shortened formal run cannot PASS.

## Match-forward rule

Once Step1 is accepted, G0/G1/G2 must inherit all non-treatment training/evaluation settings from
this frozen mother baseline unless an additive pre-registered adjudication explicitly changes them.
