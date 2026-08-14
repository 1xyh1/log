#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a deterministic 4-16 sample train=val overfit split")
    parser.add_argument("--source", required=True, help="Source split containing one sample id per line")
    parser.add_argument("--out", default="splits/overfit.txt")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not 4 <= args.count <= 16:
        raise ValueError(f"--count must be in [4, 16], got {args.count}")
    source = Path(args.source)
    ids = list(dict.fromkeys(line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()))
    if len(ids) < args.count:
        raise ValueError(f"{source} contains {len(ids)} unique ids, fewer than requested {args.count}")
    rng = random.Random(args.seed)
    selected = sorted(rng.sample(ids, args.count))
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(selected) + "\n", encoding="utf-8")
    print(f"wrote {len(selected)} ids to {output}; use the same file for train and val")


if __name__ == "__main__":
    main()
