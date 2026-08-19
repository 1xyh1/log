# T1-GR E2–E5 v1 → v2 Reviewer Fixes

## Verdict transition

```text
v1:
  IMPLEMENTATION CONDITIONAL PASS
  FORMAL E2–E5 HOLD

v2 target:
  package/static/synthetic gates PASS
  then allow formal E2 probe/contract work
  training remains HOLD until E2–E4 freeze + Step1 recipe/view are closed
```

## P0-1 — FINAL HOLDOUT seal

Fixed:

- split proposal no longer defaults into repo; it is private and required outside repo.
- full contract is private/outside repo; repo receives sanitized public contract only.
- public split freeze contains no train/dev/holdout IDs.
- holdout gets a dedicated sealed file outside repo.
- documentation no longer claims human secrecy; it claims repo nondisclosure + runner access sealing.

## P0-2 — arbitrary dataset YAML

Fixed:

- runner/evaluator removed `--data`.
- only hash-pinned `view_manifest.json` accepted.
- view manifest records recipe/contract/split pins, dataset YAML SHA, train/dev IDs SHA and every copied RGB/label SHA.
- runner/evaluator rescan actual files and fail on extra/missing samples.

## P0-3 — format/full-hash not gated

Fixed:

- full hash is unconditional in formal contract builder.
- format expectation must be explicitly filled from observed formal data.
- dtype/ndim/channels/HW/readability/cross-modal spatial compatibility are hard gates.
- expected formal sample count is a hard gate.
- class names/count and strict label geometry are hard gates.

## P0-4 — Step1 recipe not frozen

Fixed:

- explicit training spec covers optimizer/LR/nbs/warmup/augmentation/eval semantics.
- runtime base-checkpoint SHA and Ultralytics version rechecked.
- trainer effective args checked before training and from post-run `args.yaml`.
- physical `nc/end2end` checked.

## P1

Fixed:

- group-aware class-stratified split.
- per-class image + box support.
- nonempty split gate.
- explicit class coverage policy/exemptions.
- exact duplicate leakage checks for RGB/IR/Depth/triplet.
- group rule executed during contract build.
- proposal schema/gates required before freeze.
- freeze timestamp replaces self-asserted historical fact.
- label rows require exact 5 fields and valid box edges.

## Synthetic required cases

```text
bad depth dtype/shape                  MUST FAIL contract
full hash                              MUST be unconditional
rare class impossible coverage         MUST FAIL split
forged holdout file in Step1 view      MUST FAIL runner preflight
arbitrary --data                       MUST be rejected
same checkpoint path with changed data MUST FAIL SHA pin
```
