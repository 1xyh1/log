# E5 Security / Reliability Review

## Scope

E5 Step1 RGB baseline preparation and execution.

Formal data/model science logic is separated from operational hardening. The hardening below
does not change TRAIN/DEV membership, model architecture, frozen hyperparameters or evaluation
endpoint.

## Null / schema / type safety

All formal JSON inputs are schema checked. Required fields may not be absent/null.
Scientific numeric values reject bool-as-int, NaN/Inf, invalid ranges and multi-GPU device syntax.

## Duplicate requests / idempotency

Formal outputs carry:
- `request_fingerprint`
- `payload_sha256`

Same request:
- existing output integrity is rechecked;
- view files are rehashed;
- completed run rechecks `last.pt`, `args.yaml`, `results.csv` and epoch count;
- only then is idempotent reuse returned.

Different request targeting an existing formal output fails.

Private view/run-root location is cryptographically bound into the request fingerprint so changing
the private destination is not misclassified as the same request.

## Concurrency

Public formal outputs and private view construction use exclusive file locks. Atomic JSON writing uses:
temporary sibling → flush → fsync → `os.replace`.

The view is constructed in a private temporary directory and committed with one directory rename.

## Permission boundary

TRAIN/DEV access, private view, run artifacts and model weights must be outside repo.

POSIX:
- private files use owner-only mode where supported;
- run creation is protected by `umask 0077`.

Windows:
- Python file mode is not treated as proof of NTFS ACL confidentiality;
- the private root must be user-controlled.

## Timeout

Training and DEV eval use both:
- batch/epoch callback deadline checks;
- a wall-clock watchdog that also covers stalls before the first callback.

Recipe hashing/view construction use bounded deadlines.

## Exceptions

Expected gate failures emit safe error codes.
Unexpected exceptions are reported as `UNHANDLED_INTERNAL_ERROR`.
User interrupts are converted to `USER_INTERRUPT`.
Formal PASS is never published after an incomplete training run.

A private `E5_INCOMPLETE.txt` marker is left for diagnosis if training fails after run directory creation.

## Sensitive information

No E5 operational tool accepts the sealed FINAL HOLDOUT artifact.

Public reports:
- contain counts/commitments/metrics;
- contain no raw sample IDs;
- contain no local absolute paths;
- are passed through the sensitive-output scanner.

Ultralytics `args.yaml/results.csv/weights` are repo-external.

Ultralytics 8.4.56 external integration callbacks and usage analytics are disabled during E5
training/eval. Its network-capable AMP download probe is bypassed; mandatory 1-epoch smoke is the
actual AMP qualification gate before formal Step1.

## Formal boundary

Local/container tests prove implementation behavior only.

They do NOT establish:
- E4 formal seal PASS;
- E5 formal recipe/view/preflight/smoke/train/eval PASS;
- T1-GR training authorization.
