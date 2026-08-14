#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mmod_qaf.local_stub import load_local_yolo26


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(); p.add_argument('weights',nargs='+'); a=p.parse_args()
    reports=[]
    for name in a.weights:
        path=Path(name); model=load_local_yolo26(path)
        reports.append({
            'path':str(path), 'sha256':sha256(path), 'parameters':sum(p.numel() for p in model.parameters()),
            'layers':len(model.model), 'yaml':getattr(model,'yaml',None), 'stride':getattr(model,'stride',None).tolist(),
            'head_nc':int(model.model[-1].nc), 'reg_max':int(model.model[-1].reg_max),
        })
    print(json.dumps(reports,indent=2))
if __name__=='__main__': main()
