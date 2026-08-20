# T1-GR G Implementation and Smoke Freeze

Upstream design commit: `3b0a304`  
Status: `FROZEN_BEFORE_IMPLEMENTATION_SMOKE`

This freeze implements the already-approved G0-N/G1-P/G2-S design.  It does
not reopen the arm definitions, seeds, E5 v2 recipe, optimizer, endpoint, or
FINAL HOLDOUT boundary.

## Loader construction

For every recipient, the loader constructs one HWC array:

```text
[visible B, visible G, visible R, selected IR grayscale]
```

The selected IR source is:

- G0-N: recipient-paired IR, used only for matched auxiliary-buffer exposure;
- G1-P: recipient-paired IR;
- G2-S: the frozen seed/epoch cyclic derangement donor;
- DEV primary evaluation: an all-zero IR plane for every arm.

At the model boundary, the auxiliary encoder receives `[selected IR,
all-zero depth]`.  IR is not duplicated into the depth channel and depth stays
out of scope.

If a wrong-source IR image has a different native size, it is first resized to
the recipient native grid.  Only then is it concatenated with recipient RGB.
All subsequent mosaic, affine, resize, pad, and flip operations process that
single four-channel array, so RGB/IR geometry cannot draw separate randomness.

Ultralytics color transforms are made modality-safe without changing the E5
visible path:

- default non-spatial Albumentations operate on visible BGR only;
- HSV operates on visible BGR only;
- the final formatter performs the E5 BGR-to-RGB reversal on the visible three
  channels and leaves IR unchanged.

Visible geometric border fill remains E5's value 114. IR border fill is zero,
including mosaic, affine, and letterbox padding. Consequently ZERO-IR DEV stays
bitwise zero after the complete preprocessing graph rather than acquiring a
114-valued frame.

Preflight compares the visible tensor and labels against a stock three-channel
Ultralytics 8.4.56 dataset under restored identical RNG state.  It also compares
the visible tensor across all three arms under the same seed/draw.

## Epoch boundary

Ultralytics' infinite loader can prefetch the next epoch while the current epoch
is finishing.  That is unsafe for an epoch-dependent G2 donor map.  The T1-GR
loader therefore shuts down and reconstructs its worker iterator after setting
the new epoch.  Worker count remains exactly 8 for training; sampling and all
arms remain matched.  Prefetched rows from an old epoch cannot enter training.

## Model correction for the formal mother baseline

The old T-series model targeted a non-end-to-end small-data run.  T1-GR uses a
`T1GRP5Model` adapter that:

1. accepts the four-channel tensor;
2. leaves the RGB backbone trainable;
3. exposes the YOLO26 end-to-end head contract;
4. uses Ultralytics `E2ELoss`, not the old hard-coded `v8DetectionLoss`;
5. keeps identical parameter keys and trainability across arms.

The evidence wording remains **auditable initialization**.  Same-seed arm
construction hashes are compared, but no claim of repeated-run numerical
reproducibility is made.

## Primary inference

The primary T1-GR endpoint uses ZERO IR on DEV.  This preserves the T1-TR
training-source replication question: whether paired IR during training changes
held-out generalization, without allowing inference-time source content to
become a second treatment.  Native paired-IR inference is diagnostic only.

## Authorization chain

```text
design audit PASS
  -> multimodal view
  -> implementation preflight PASS
  -> nine one-epoch smoke runs
  -> smoke audit PASS
  -> formal nine-run suite
  -> ZERO-IR DEV full + five LOFO evaluations
  -> cross-seed summary
```

Only the smoke audit may set `multiseed_training_authorized=true`.  Neither
preflight, smoke, formal completion, DEV evaluation, nor a positive DEV branch
may authorize opening FINAL HOLDOUT.
