#!/usr/bin/env python3
from __future__ import annotations
import argparse, collections, hashlib, json, math, zipfile
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np

MODS=("visible","infrared")

def sha256_bytes(x:bytes)->str: return hashlib.sha256(x).hexdigest()
def sha256_file(p:Path)->str:
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for c in iter(lambda:f.read(1<<20),b""): h.update(c)
    return h.hexdigest()
def sha256_json(x)->str:
    return sha256_bytes(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode())

def modality_of(name:str):
    parts=[x for x in name.replace("\\","/").split("/") if x]
    if len(parts)<2:return None
    m=parts[-2].lower()
    return m if m in MODS else None

def sid_of(name:str)->str:return Path(name).stem

def to_gray(arr:np.ndarray,modality:str)->np.ndarray:
    if arr.ndim==2:g=arr.astype(np.float32)
    elif arr.ndim==3:
        if modality=="infrared": g=np.median(arr.astype(np.float32),axis=2)
        else: g=cv2.cvtColor(arr,cv2.COLOR_BGR2GRAY).astype(np.float32)
    else: raise ValueError(f"unsupported ndim:{arr.ndim}")
    lo=float(np.percentile(g,1)); hi=float(np.percentile(g,99))
    if hi>lo:g=np.clip((g-lo)/(hi-lo),0,1)
    elif float(g.max())>float(g.min()):g=(g-float(g.min()))/(float(g.max())-float(g.min()))
    else:g=np.zeros_like(g,dtype=np.float32)
    return g

def norm_vec(x:np.ndarray,size:int)->np.ndarray:
    y=cv2.resize(x,(size,size),interpolation=cv2.INTER_AREA).astype(np.float32).reshape(-1)
    y-=float(y.mean()); n=float(np.linalg.norm(y))
    if n>1e-12:y/=n
    return y

def edge_vec(x:np.ndarray,size:int)->np.ndarray:
    y=cv2.resize(x,(size,size),interpolation=cv2.INTER_AREA).astype(np.float32)
    gx=cv2.Sobel(y,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(y,cv2.CV_32F,0,1,ksize=3)
    return norm_vec(cv2.magnitude(gx,gy),size)

def phash(x:np.ndarray,dct_size:int=32,hash_size:int=8)->int:
    y=cv2.resize(x,(dct_size,dct_size),interpolation=cv2.INTER_AREA).astype(np.float32)
    d=cv2.dct(y); vals=d[:hash_size,:hash_size].reshape(-1)[1:]; med=float(np.median(vals))
    out=0
    for b in vals>med: out=(out<<1)|int(bool(b))
    return out

@dataclass
class Feature:
    sid:str; ext:str; raw_sha256:str; phash:int; vec:np.ndarray; edge:np.ndarray

def feature_from_bytes(sid:str,ext:str,raw:bytes,modality:str,pol:dict)->Feature:
    arr=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_UNCHANGED)
    if arr is None:raise RuntimeError(f"IMDECODE_FAILED:{modality}:{sid}")
    g=to_gray(arr,modality); fp=pol["fingerprint"]
    return Feature(sid,ext,sha256_bytes(raw),phash(g,int(fp["phash_dct_size"]),int(fp["phash_size"])),norm_vec(g,int(fp["corr_thumb_size"])),edge_vec(g,int(fp["edge_thumb_size"])))

def feature_from_file(sid:str,p:Path,modality:str,pol:dict)->Feature:
    return feature_from_bytes(sid,p.suffix.lower(),p.read_bytes(),modality,pol)

def ham(a:int,b:int)->int:return (a^b).bit_count()
def corr(a:np.ndarray,b:np.ndarray)->float:return float(np.dot(a,b))

def pair_metrics(a:dict[str,Feature],b:dict[str,Feature])->dict:
    return {
      "visible_phash":ham(a["visible"].phash,b["visible"].phash),
      "infrared_phash":ham(a["infrared"].phash,b["infrared"].phash),
      "visible_corr":corr(a["visible"].vec,b["visible"].vec),
      "infrared_corr":corr(a["infrared"].vec,b["infrared"].vec),
      "visible_edge_corr":corr(a["visible"].edge,b["visible"].edge),
      "infrared_edge_corr":corr(a["infrared"].edge,b["infrared"].edge),
      "visible_raw_exact":a["visible"].raw_sha256==b["visible"].raw_sha256,
      "infrared_raw_exact":a["infrared"].raw_sha256==b["infrared"].raw_sha256,
    }

def classify(m:dict,pol:dict)->str:
    if m["visible_raw_exact"] and m["infrared_raw_exact"]:return "EXACT_BOTH_MODALITIES"
    s=pol["strong_edge"]
    if (m["visible_phash"]<=s["visible_phash_hamming_max"] and m["infrared_phash"]<=s["infrared_phash_hamming_max"] and m["visible_corr"]>=s["visible_corr_min"] and m["infrared_corr"]>=s["infrared_corr_min"] and m["visible_edge_corr"]>=s["visible_edge_corr_min"] and m["infrared_edge_corr"]>=s["infrared_edge_corr_min"]):return "STRONG_NEAR_DUPLICATE"
    r=pol["review_edge"]
    if (m["visible_phash"]<=r["visible_phash_hamming_max"] and m["infrared_phash"]<=r["infrared_phash_hamming_max"] and m["visible_corr"]>=r["visible_corr_min"] and m["infrared_corr"]>=r["infrared_corr_min"] and m["visible_edge_corr"]>=r["visible_edge_corr_min"] and m["infrared_edge_corr"]>=r["infrared_edge_corr_min"]):return "REVIEW_NEAR_DUPLICATE"
    return "NONE"

def candidate_by_phash(a:dict[str,Feature],b:dict[str,Feature],pol:dict)->bool:
    g=pol["candidate_gate"]
    return ham(a["visible"].phash,b["visible"].phash)<=g["visible_phash_hamming_max"] and ham(a["infrared"].phash,b["infrared"].phash)<=g["infrared_phash_hamming_max"]

class DSU:
    def __init__(self,ids):self.p={x:x for x in ids};self.sz={x:1 for x in ids}
    def find(self,x):
        while self.p[x]!=x:self.p[x]=self.p[self.p[x]];x=self.p[x]
        return x
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a==b:return
        if self.sz[a]<self.sz[b]:a,b=b,a
        self.p[b]=a;self.sz[a]+=self.sz[b]
    def comps(self):
        d=collections.defaultdict(list)
        for x in self.p:d[self.find(x)].append(x)
        return [sorted(v) for v in d.values()]

def locate_old_file(raw:Path,sub:str,sid:str)->Path:
    c=list((raw/sub).glob(f"{sid}.*"))
    if len(c)!=1:raise RuntimeError(f"HISTORICAL_FILE_RESOLUTION:{sub}:{sid}:{len(c)}")
    return c[0]

def load_historical(contract_path:Path,root_override:str|None,pol:dict):
    c=json.loads(contract_path.read_text(encoding="utf-8"));ids=c.get("usable_ids") or c.get("all17_ids")
    if not ids or len(ids)!=17:raise RuntimeError(f"HISTORICAL_17_IDS_EXPECTED:{0 if not ids else len(ids)}")
    raw=Path(root_override) if root_override else Path(c["_raw_dir"])
    if not raw.is_dir():raise RuntimeError(f"HISTORICAL_RAW_NOT_FOUND:{raw}")
    feats={};verified=0
    for s in ids:
        feats[s]={}
        for m in MODS:
            p=locate_old_file(raw,m,s);actual=sha256_file(p);expected=c.get("file_hashes",{}).get(s,{}).get(m,{}).get("sha256")
            if expected:
                if actual!=expected:raise RuntimeError(f"HISTORICAL_HASH_DRIFT:{s}:{m}")
                verified+=1
            feats[s][m]=feature_from_file(s,p,m,pol)
    return c,feats,verified,str(raw.resolve())

def formal_zip_features(zp:Path,pol:dict):
    feats=collections.defaultdict(dict);metadata=[];exts=collections.defaultdict(dict)
    with zipfile.ZipFile(zp,"r") as z:
        infos=[]
        for info in z.infolist():
            if info.is_dir():continue
            m=modality_of(info.filename)
            if m is None:continue
            s=sid_of(info.filename);infos.append((m,s,info));metadata.append((info.filename,info.CRC,info.file_size,info.compress_size))
        counts=collections.Counter(m for m,_,_ in infos)
        if counts["visible"]!=2000 or counts["infrared"]!=2000:raise RuntimeError(f"FORMAL_COUNTS_BAD:{dict(counts)}")
        for m,s,info in infos:
            if m in feats[s]:raise RuntimeError(f"FORMAL_DUPLICATE_ID:{m}:{s}")
            raw=z.read(info);feats[s][m]=feature_from_bytes(s,Path(info.filename).suffix.lower(),raw,m,pol);exts[s][m]=Path(info.filename).suffix.lower()
    bad=[s for s,x in feats.items() if set(x)!=set(MODS)]
    if bad:raise RuntimeError(f"FORMAL_PAIRING_BAD:{len(bad)}")
    return dict(feats),dict(exts),sha256_json(sorted(metadata))

def size_hist(comps):
    c=collections.Counter(len(x) for x in comps);return {str(k):v for k,v in sorted(c.items())}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--formal-zip",required=True);ap.add_argument("--historical-contract",required=True);ap.add_argument("--historical-root",default=None);ap.add_argument("--policy",default="config/t1gr_visual_leakage_policy.json");ap.add_argument("--private-out",required=True);ap.add_argument("--public-out",default="reports/step4_t1gr/visual_leakage_public.json");a=ap.parse_args()
    repo=Path(__file__).resolve().parents[1];pol=json.loads((repo/a.policy).read_text(encoding="utf-8"));pol_sha=sha256_file(repo/a.policy)
    priv=Path(a.private_out).resolve()
    try:priv.relative_to(repo.resolve());raise SystemExit("PRIVATE_OUT_MUST_BE_OUTSIDE_REPO")
    except ValueError:pass
    hc_path=Path(a.historical_contract);hc_sha=sha256_file(hc_path);_,hist,verified,hist_raw=load_historical(hc_path,a.historical_root,pol);formal,exts,zip_commit=formal_zip_features(Path(a.formal_zip),pol);fids=sorted(formal)
    hist_edges=[];force=set();hist_counts=collections.Counter()
    for hs,hf in hist.items():
        for fs in fids:
            ff=formal[fs]
            if not candidate_by_phash(hf,ff,pol):continue
            met=pair_metrics(hf,ff);cl=classify(met,pol)
            if cl=="NONE":continue
            hist_edges.append({"historical_id":hs,"formal_id":fs,"class":cl,"metrics":met});hist_counts[cl]+=1;force.add(fs)
    dsu=DSU(fids);strong=[];review=[];cand=0
    for i,ai in enumerate(fids):
        af=formal[ai]
        for bi in fids[i+1:]:
            bf=formal[bi]
            if not candidate_by_phash(af,bf,pol):continue
            cand+=1;met=pair_metrics(af,bf);cl=classify(met,pol)
            if cl in {"EXACT_BOTH_MODALITIES","STRONG_NEAR_DUPLICATE"}:strong.append({"a":ai,"b":bi,"class":cl,"metrics":met});dsu.union(ai,bi)
            elif cl=="REVIEW_NEAR_DUPLICATE":review.append({"a":ai,"b":bi,"class":cl,"metrics":met})
    comps=dsu.comps();nontrivial=[x for x in comps if len(x)>1];max_comp=max((len(x) for x in comps),default=0);limit=max(int(pol["internal_graph_policy"]["max_component_abs_before_manual_review"]),math.ceil(len(fids)*float(pol["internal_graph_policy"]["max_component_fraction_before_manual_review"])));graph_gate="PASS" if max_comp<=limit else "HOLD_OVERSIZED_COMPONENT"
    force_domain=collections.Counter(exts[s]["visible"] for s in force)
    priv_data={"schema":"t1gr-visual-leakage-private-v1","policy_sha256":pol_sha,"historical_contract_sha256":hc_sha,"historical_raw_root":hist_raw,"historical_hash_entries_verified":verified,"formal_zip_path":str(Path(a.formal_zip).resolve()),"formal_zip_member_metadata_commitment":zip_commit,"historical_matches":hist_edges,"force_train_ids_due_historical_contamination":sorted(force),"internal_strong_edges":strong,"internal_review_edges":review,"internal_components_all":sorted(comps,key=lambda x:(-len(x),x)),"internal_nontrivial_components":sorted(nontrivial,key=lambda x:(-len(x),x)),"note":"No split assignment is made here."}
    pub={"schema":"t1gr-visual-leakage-public-v1","read_only":True,"formal_n":len(fids),"historical_n":len(hist),"policy_sha256":pol_sha,"historical_contract_sha256":hc_sha,"historical_hash_entries_verified":verified,"formal_zip_member_metadata_commitment":zip_commit,"historical_match_class_counts":dict(hist_counts),"historical_contaminated_formal_count":len(force),"historical_contaminated_domain_counts":dict(sorted(force_domain.items())),"force_train_ids_commitment":sha256_json(sorted(force)),"internal_candidate_pairs_after_phash":cand,"internal_strong_edge_count":len(strong),"internal_review_edge_count":len(review),"internal_component_count":len(comps),"internal_nontrivial_component_count":len(nontrivial),"internal_component_size_distribution":size_hist(comps),"internal_max_component_size":max_comp,"internal_component_review_limit":limit,"internal_graph_gate":graph_gate,"private_ids_in_public_report":False,"adjudication":{"historical_matches":"all STRONG/REVIEW matches conservatively ineligible for FINAL HOLDOUT","internal_strong_edges":"minimum indivisible leakage components","internal_review_edges":"review before final split freeze; not auto-connected","jpg_png":"stratification factor only, not a scene group","scene_independence_claim":"FORBIDDEN_WITHOUT_INDEPENDENT_METADATA"},"e3_status":"HOLD_PENDING_REVIEW_EDGES_AND_COMPONENT_ADJUDICATION","step1_status":"HOLD"}
    priv.parent.mkdir(parents=True,exist_ok=True);priv.write_text(json.dumps(priv_data,ensure_ascii=False,indent=2),encoding="utf-8");out=repo/a.public_out;out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise SystemExit(f"REFUSE_OVERWRITE:{out}")
    out.write_text(json.dumps(pub,ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps(pub,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
