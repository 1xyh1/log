#!/usr/bin/env python3
from __future__ import annotations
import argparse, random, sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mmod_qaf.data import discover_samples

def main():
 p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--out',required=True); p.add_argument('--val-ratio',type=float,default=.2); p.add_argument('--seed',type=int,default=0); p.add_argument('--group-parts',type=int,default=1); a=p.parse_args()
 records=discover_samples(a.data,invalid_depth_policy='skip'); groups=defaultdict(list)
 for r in records: groups['_'.join(r.sample_id.split('_')[:a.group_parts])].append(r.sample_id)
 keys=sorted(groups); random.Random(a.seed).shuffle(keys); target=max(1,round(len(records)*a.val_ratio)); val=[]
 while keys and len(val)<target: val.extend(groups[keys.pop()])
 train=[sid for k in keys for sid in groups[k]]
 out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
 (out/'train.txt').write_text('\n'.join(sorted(train))+'\n',encoding='utf-8'); (out/'val.txt').write_text('\n'.join(sorted(val))+'\n',encoding='utf-8')
 print({'train':len(train),'val':len(val),'val_ids':sorted(val)})
if __name__=='__main__': main()
