#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mmod_qaf.audit import save_audit

def main():
 p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--json',default='audit.json'); p.add_argument('--md',default='audit.md'); a=p.parse_args()
 r=save_audit(a.data,a.json,a.md); print(f"audited {r['count']} sample groups -> {a.json}, {a.md}")
if __name__=='__main__': main()
