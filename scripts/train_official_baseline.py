#!/usr/bin/env python3
from __future__ import annotations

import argparse
from ultralytics import YOLO


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--weights',required=True); p.add_argument('--data',required=True)
    p.add_argument('--epochs',type=int,default=80); p.add_argument('--imgsz',type=int,default=768)
    p.add_argument('--batch',type=int,default=16); p.add_argument('--device',default='0')
    p.add_argument('--project',default='runs/baselines'); p.add_argument('--name',default='rgb_s')
    a=p.parse_args()
    model=YOLO(a.weights)
    model.train(data=a.data,epochs=a.epochs,imgsz=a.imgsz,batch=a.batch,device=a.device,project=a.project,name=a.name,max_det=100)
if __name__=='__main__': main()
