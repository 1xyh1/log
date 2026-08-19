#!/usr/bin/env python3
"""Read-only recursive inventory of the formal dataset; makes no layout assumptions."""
from __future__ import annotations
import argparse,json,os,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
 p=argparse.ArgumentParser(); p.add_argument('--dataset-root',required=True); p.add_argument('--out',default='reports/step4_t1gr/dataset_probe.json'); a=p.parse_args()
 root=Path(a.dataset_root)
 if not root.is_dir(): raise SystemExit(f'DATASET_ROOT_NOT_FOUND:{root}')
 files=[x for x in root.rglob('*') if x.is_file()]
 ext=Counter(x.suffix.lower() or '<none>' for x in files); dirs=Counter(str(x.parent.relative_to(root)) for x in files)
 top=[]
 for d,n in dirs.most_common(100): top.append({'relative_dir':d,'file_count':n})
 sample=[]
 for x in files[:200]:
  st=x.stat(); sample.append({'relative_path':str(x.relative_to(root)).replace('\\','/'),'suffix':x.suffix.lower(),'bytes':st.st_size})
 report={'schema':'t1gr-dataset-probe-v1','read_only':True,'dataset_root':str(root.resolve()),'total_files':len(files),'extension_counts':dict(sorted(ext.items())),'directories_by_file_count':top,'first_200_files':sample,'next_action':'fill config/t1gr_layout_spec.json from observed structure; do not split/train yet'}
 out=ROOT/a.out; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps({'total_files':len(files),'extensions':dict(ext),'out':str(out)},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
