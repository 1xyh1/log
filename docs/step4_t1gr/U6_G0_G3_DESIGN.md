# T1-U6 G0--G3 Design Freeze

Status: **new isolated DEV-only suite, frozen before server preflight**.

## Historical adjudication

The repository contains a genuine six-channel early-fusion implementation in
the Step3 history.  It also contains a later, deliberate Step4 mechanism
experiment that changed topology to an auxiliary encoder with P5-only
injection.  These facts do not justify describing the current package as a
no-code-change continuation of either line.  T1-U6 is a newly versioned
E5-v2 end-to-end descendant that reuses the already frozen TRAIN/DEV split.

The old experimental `G3-FE` cannot be honestly folded into pure early fusion.
Its question was whether a separately identifiable auxiliary encoder could be
frozen while a zero-initialized fusion projection remained trainable.  A single
six-channel first convolution has no such separate encoder.  Freezing its
zero-initialized IR slice would make IR permanently irrelevant and collapse to
the RGB control; initializing that slice nonzero would break RGB-equivalent
initialization.  The old `G3-FE` therefore remains a legacy P5 experiment and
is not relabeled here.

## Four physical-six-channel arms

| Arm | Train input | Native DEV input | Role |
|---|---|---|---|
| G0-N | RGB + zero IR + zero Depth/mask | same | RGB control |
| G1-P | RGB + correctly paired IR + zero Depth/mask | same | paired-IR candidate |
| G2-S | RGB + balanced fully wrong IR + zero Depth/mask | RGB + correctly paired IR | source-identity control |
| G3-D | RGB + correctly paired IR + qualified Depth/mask | same | tri-modal candidate |

All four arms instantiate the same physical six-channel YOLO26s E5-v2 model.
The first convolution starts as `[W_R,W_G,W_B,0,0,0]`; all other state is the
same seeded E5-v2 state.  Every parameter remains trainable.  No gate,
attention block, auxiliary encoder, optimizer search, or new visible
augmentation is added.

G0 is the matched control for this new suite, not a claim of bitwise training
reproduction of the three-channel E5-v2 run. Zero-aux initialization preserves
the initial function; a different convolution shape may still choose a
different numerical backend path during training.

G2 uses a seed-keyed cyclic derangement for TRAIN only.  Every epoch is a
bijection, has zero self-pairs, and uses each donor once.  Across 80 epochs a
recipient sees 80 distinct wrong donors; the design does not claim to exhaust
all 1503 alternatives.

## Depth audit and policy

The formal archive has two storage domains:

- 1851 PNG files are `uint16 HxW`. Values in `[300,19999]` millimetres are
  log-normalized and accompanied by a binary validity mask.
- 149 JPG files are `uint8 HxWx3` gray-derived images with unknown scale. They
  remain missing (`Depth=0`, `mask=0`). No multiplication by 255 or fictitious
  millimetre reconstruction is allowed.

Depth resize and geometric warp use a validity-weighted numerator and
denominator. The validity mask uses nearest interpolation and is re-binarized;
`mask==0` always forces Depth to zero. RGB, IR, Depth, mask, and boxes share one
recipient-keyed geometric draw. Only visible RGB receives photometric changes.

After the sidecar is fully hashed and format-verified, G0/G1/G2 do not decode
Depth during training, and G0 does not read IR. They emit strict zero planes.
This removes unnecessary I/O without changing a tensor or treatment.

## Training and launch design

The inherited recipe remains MuSGD, 80 epochs, batch 4, image size 640,
`lr0=0.01`, momentum 0.9, `nbs=64`, and `last.pt` primary. Frozen seeds are
20260812, 20260813, and 20260814.

Four arms and three seeds cannot produce a complete four-position Latin
square. The declared schedule is a cyclic incomplete 3x4 Latin rectangle:

- seed 20260812: G0-N, G1-P, G2-S, G3-D
- seed 20260813: G1-P, G2-S, G3-D, G0-N
- seed 20260814: G2-S, G3-D, G0-N, G1-P

Each seed is one server lane and its four arms run sequentially on the same
visible GPU. Three identical-GPU lanes may run concurrently. Each arm occurs
once per seed and misses exactly one lane position; no complete position
balance is claimed.

## Evaluation separates questions

Native DEV evaluates deployable inputs. Common-input evaluations prevent
training-treatment effects from being confused with inference-time modality
use:

- every checkpoint receives RGB-only input;
- every checkpoint receives paired IR with zero Depth/mask;
- G1 and G2 also receive a fixed wrong-IR DEV diagnostic;
- G0, G1, and G3 receive five leakage-component LOFO evaluations;
- G1 and G3 receive metric-PNG and quarantined-JPG domain diagnostics.

The original source-identity question is read from the shared RGB-only view,
especially `G1.rgb_only - G2.rgb_only`. The operational IR comparison is
`G1.native - G0.native`. The Depth comparison is `G3.native - G1.native`, and
inference dependence is `G3.native - G3.paired_ir_zero_depth`.

G1 is eligible over G0, and G3 is eligible over G1, only when at least two seed
deltas, mean, median, modality-dependency majority, worst-seed guard, and 9/15
LOFO directions pass the frozen rule. Selection order is G3, then G1, then G0.
G2 is never auto-selected; if its mean exceeds the recommendation, the summary
raises a manual-review warning because the scientific story needs inspection.

All results are DEV-only. FINAL HOLDOUT 298 remains sealed.
