#!/usr/bin/env python3
"""Classify formal YOLO label issues without modifying data.

This audit separates:
A) hard schema/semantic errors,
B) official-[0,1] scalar-bound issues,
C) Ultralytics-8.4.56 tolerance-only issues,
D) derived xyxy corner overflow only,
E) duplicate rows,
F) empty/background labels.

It reads label TXT members directly from the training ZIP and uses the visible
member extension only to report JPG/PNG domain breakdown. No dataset extraction.
"""
from __future__ import annotations
import argparse, collections, json, math, zipfile
from pathlib import Path

CLASSES = [
    "person","boat","animal","seat","sign","bicycle",
    "car","ball","light","garbage can","uav","tricycle"
]

def modality_of(name: str):
    p=[x for x in name.replace("\\","/").split("/") if x]
    return p[-2].lower() if len(p)>=2 else None

def sid(name: str): return Path(name).stem

def parse_label(raw: bytes, nc: int = 12):
    out = {
        "hard_errors": [],
        "strict_scalar_errors": [],
        "ultralytics_tolerance_errors": [],
        "corner_overflows": [],
        "duplicate_rows": 0,
        "n_boxes": 0,
        "empty": False,
    }
    try:
        text=raw.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        out["hard_errors"].append(f"decode:{e}")
        return out

    rows=[]
    nonblank=[x for x in text.splitlines() if x.strip()]
    if not nonblank:
        out["empty"]=True
        return out

    for ln,line in enumerate(nonblank,1):
        parts=line.split()
        if len(parts)!=5:
            out["hard_errors"].append(f"line{ln}:field_count:{len(parts)}")
            continue
        try:
            vals=[float(x) for x in parts]
        except ValueError:
            out["hard_errors"].append(f"line{ln}:non_numeric")
            continue
        c,cx,cy,w,h=vals
        if not all(math.isfinite(x) for x in vals):
            out["hard_errors"].append(f"line{ln}:nonfinite")
            continue
        if not c.is_integer():
            out["hard_errors"].append(f"line{ln}:class_not_integer:{c}")
            continue
        ci=int(c)
        if not (0<=ci<nc):
            out["hard_errors"].append(f"line{ln}:class_out_of_range:{ci}")
        if w<=0 or h<=0:
            out["hard_errors"].append(f"line{ln}:nonpositive_wh:{w},{h}")

        coords=(cx,cy,w,h)
        # Official normalized representation: [0,1] for scalar xywh.
        bad_strict=[v for v in coords if v<0 or v>1]
        if bad_strict:
            out["strict_scalar_errors"].append(
                {"line":ln,"values":coords,"offenders":bad_strict}
            )
        # Ultralytics 8.4.56 verifier checks points.max() <=1.01 and lb.min() >= -0.01.
        bad_u=[v for v in coords if v < -0.01 or v > 1.01]
        if bad_u:
            out["ultralytics_tolerance_errors"].append(
                {"line":ln,"values":coords,"offenders":bad_u}
            )

        x1,x2=cx-w/2,cx+w/2
        y1,y2=cy-h/2,cy+h/2
        over=max(0.0,-x1,x2-1.0,-y1,y2-1.0)
        if over>0:
            out["corner_overflows"].append(
                {"line":ln,"overflow":over,"xywh":coords}
            )
        rows.append(tuple(vals))
        out["n_boxes"] += 1

    out["duplicate_rows"] = len(rows)-len(set(rows))
    return out

def bucket_overflow(v: float):
    if v <= 1e-6: return "<=1e-6"
    if v <= 1e-4: return "(1e-6,1e-4]"
    if v <= 1e-3: return "(1e-4,1e-3]"
    if v <= 1e-2: return "(1e-3,1e-2]"
    if v <= 5e-2: return "(1e-2,5e-2]"
    return ">5e-2"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--zip",required=True)
    ap.add_argument("--public-out",default="reports/step4_t1gr/label_error_taxonomy_public.json")
    a=ap.parse_args()

    zp=Path(a.zip)
    root=Path(__file__).resolve().parents[1]
    by={"labels":{},"visible":{}}
    with zipfile.ZipFile(zp) as z:
        for info in z.infolist():
            if info.is_dir(): continue
            m=modality_of(info.filename)
            if m in by:
                by[m][sid(info.filename)] = info

        ids=set(by["labels"]) & set(by["visible"])
        tax=collections.Counter()
        tax_by_ext=collections.defaultdict(collections.Counter)
        overflow_bins=collections.Counter()
        duplicate_samples=0
        empty_samples=0
        samples={}
        for s in sorted(ids):
            r=parse_label(z.read(by["labels"][s]))
            ext=Path(by["visible"][s].filename).suffix.lower()
            flags=[]
            if r["hard_errors"]: flags.append("HARD_SCHEMA_OR_CLASS")
            if r["ultralytics_tolerance_errors"]: flags.append("ULTRALYTICS_8_4_56_REJECT")
            elif r["strict_scalar_errors"]: flags.append("STRICT_[0,1]_ONLY")
            if r["corner_overflows"]:
                flags.append("DERIVED_CORNER_OVERFLOW")
                for x in r["corner_overflows"]:
                    overflow_bins[bucket_overflow(float(x["overflow"]))]+=1
            if r["duplicate_rows"]:
                flags.append("DUPLICATE_ROWS")
                duplicate_samples += 1
            if r["empty"]:
                flags.append("EMPTY_BACKGROUND")
                empty_samples += 1
            if not flags: flags=["CLEAN"]
            for f in set(flags):
                tax[f]+=1; tax_by_ext[ext][f]+=1
            samples[s]=r

    # Mutually useful adjudication counts.
    only_corner=0
    corner_and_no_hard=0
    hard=0
    strict_only=0
    u_reject=0
    for r in samples.values():
        if r["hard_errors"]: hard+=1
        if r["ultralytics_tolerance_errors"]: u_reject+=1
        if r["strict_scalar_errors"] and not r["ultralytics_tolerance_errors"] and not r["hard_errors"]:
            strict_only += 1
        if r["corner_overflows"] and not r["hard_errors"] and not r["ultralytics_tolerance_errors"]:
            corner_and_no_hard += 1
            if not r["strict_scalar_errors"] and not r["duplicate_rows"]:
                only_corner += 1

    report={
        "schema":"t1gr-label-error-taxonomy-public-v1",
        "read_only":True,
        "n_label_visible_pairs":len(samples),
        "class_names":CLASSES,
        "sample_category_counts":dict(sorted(tax.items())),
        "sample_category_counts_by_visible_extension":{
            e:dict(sorted(c.items())) for e,c in sorted(tax_by_ext.items())
        },
        "derived_corner_overflow_box_bins":dict(overflow_bins),
        "adjudication_counts":{
            "hard_schema_or_class_samples":hard,
            "ultralytics_8_4_56_reject_samples":u_reject,
            "strict_[0,1]_but_ultralytics_tolerates_samples":strict_only,
            "corner_overflow_without_hard_or_ultralytics_reject":corner_and_no_hard,
            "corner_overflow_only_samples":only_corner,
            "duplicate_row_samples":duplicate_samples,
            "empty_background_samples":empty_samples,
        },
        "policy_notes":{
            "hard_schema_or_class":"formal blocker",
            "ultralytics_8_4_56_reject":"formal Step1 blocker",
            "strict_[0,1]_only":"review/record; trainer tolerance differs from strict documentation",
            "derived_corner_overflow":"diagnostic only unless another hard category is present",
            "duplicate_rows":"record effective loader behavior; do not silently edit raw labels",
            "empty_background":"valid if intentional",
        },
        "raw_sample_ids_in_public_report":False,
    }
    out=root/a.public_out
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists(): raise SystemExit(f"REFUSE_OVERWRITE:{out}")
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
