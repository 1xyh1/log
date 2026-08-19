from __future__ import annotations
import importlib.util,json,sys
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('m',ROOT/'scripts/t1gr_visual_leakage_audit.py')
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
POL=json.loads((ROOT/'config/t1gr_visual_leakage_policy.json').read_text())

def enc(img,ext):
    ok,b=cv2.imencode(ext,img);assert ok;return b.tobytes()
def make_scene(seed=0):
    rng=np.random.default_rng(seed);x=np.zeros((180,320,3),np.uint8)
    cv2.rectangle(x,(30,40),(150,130),(180,120,60),-1);cv2.circle(x,(240,90),35,(30,220,180),-1)
    return cv2.add(x,rng.integers(0,10,x.shape,dtype=np.uint8))
def feat_pair(img,ext='.png'):
    ir=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);ir3=np.repeat(ir[...,None],3,axis=2)
    return {'visible':m.feature_from_bytes('x',ext,enc(img,ext),'visible',POL),'infrared':m.feature_from_bytes('x',ext,enc(ir3,ext),'infrared',POL)}

def test_exact_class():
    a=feat_pair(make_scene(1));b=feat_pair(make_scene(1));assert m.classify(m.pair_metrics(a,b),POL)=='EXACT_BOTH_MODALITIES'
def test_reencode_resize_candidate():
    a=feat_pair(make_scene(2),'.png');small=cv2.resize(make_scene(2),(160,90),interpolation=cv2.INTER_AREA);b=feat_pair(small,'.jpg');assert m.candidate_by_phash(a,b,POL);assert m.classify(m.pair_metrics(a,b),POL) in {'STRONG_NEAR_DUPLICATE','REVIEW_NEAR_DUPLICATE'}
def test_unrelated_none():
    a=feat_pair(make_scene(3));x=np.zeros((180,320,3),np.uint8)
    for i in range(0,320,20):cv2.line(x,(i,0),(319-i,179),(255,255,255),3)
    b=feat_pair(x);assert m.classify(m.pair_metrics(a,b),POL)=='NONE'
def test_dsu():
    d=m.DSU(['a','b','c','d']);d.union('a','b');d.union('b','c');assert sorted(len(x) for x in d.comps())==[1,3]
def test_policy_review_force_train():assert POL['historical_policy']['review_match']=='FORCE_TRAIN_CONSERVATIVE'
def test_only_strong_connected():assert POL['internal_graph_policy']['connected_component_edges']=='STRONG_ONLY'
def test_claim_boundary():assert 'not true scene-independent' in POL['claim_boundary']
def test_depth_not_in_modalities():assert POL['modalities']==['visible','infrared']
