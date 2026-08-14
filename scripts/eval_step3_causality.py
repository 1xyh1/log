#!/usr/bin/env python3
"""Step 3-A causality evaluation: NORMAL / ZERO / SHUFFLE x best.pt+last.pt x
train11/val6/all17 (+ per-class, late10, val6 leave-one-out sensitivity).

Manual val loop (DetMetrics + stock match_predictions semantics copied with
provenance) because the stock validator's AutoBackend path assumes 3-channel data.
Primary axis = last.pt NORMAL/ZERO/SHUFFLE (protocol).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal.raw_sample_index import build_contract, group_of, OUT_DEFAULT  # noqa: E402
from multimodal.trimodal_dataset import GROUPS, TriModalDataset  # noqa: E402
from ultralytics.utils.metrics import DetMetrics, box_iou  # noqa: E402
from ultralytics.utils.nms import non_max_suppression  # noqa: E402

IOUV = torch.linspace(0.5, 0.95, 10)  # stock iouv


def match_predictions(pred_classes, true_classes, iou, iouv):
    """Copy of BaseValidator.match_predictions semantics (ultralytics 8.4.56,
    engine/validator.py) — 10-threshold greedy per-image matching."""
    correct = np.zeros((pred_classes.shape[0], iouv.shape[0])).astype(bool)
    correct_class = (true_classes[:, None] == pred_classes).cpu().numpy()
    for i in range(len(iouv)):
        x = torch.where(iou >= iouv[i])
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = correct_class[
                matches[:, 0].astype(int), matches[:, 1].astype(int)]
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


def group_aware_derangement(ids: list[str], shuffle_map: dict, rand=None) -> dict:
    """No-self-match derangement preferring donor_group != rgb_group (deterministic greedy).
    Returns {sample_id: donor_id}. D/M pair travels together (per-sample aux planes)."""
    groups = {}
    for sid in ids:
        groups.setdefault(group_of(sid), []).append(sid)
    donors = {sid: [d for g, ds in groups.items() if g != group_of(sid) for d in ds]
              for sid in ids}
    fallback = [d for d in ids]
    result = {}
    used = set()
    order = ids if rand is None else [ids[i] for i in rand.permutation(len(ids)).tolist()]
    for sid in order:
        pool = [d for d in donors[sid] if d != sid and d not in used] or \
               [d for d in fallback if d != sid and d not in used]
        if not pool:
            # release used cross-group donors and retry (guaranteed for our sets)
            pool = [d for d in donors[sid] if d != sid] or [d for d in fallback if d != sid]
        d = pool[0]
        result[sid] = d
        used.add(d)
    assert all(result[s] != s for s in result), "derangement violated"
    return result


def eval_set(model, dataset, device, names) -> dict:
    model.eval()
    stats = []
    n_gt_boxes = 0
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            sid = sample["sample_id"]
            img = torch.as_tensor(sample["img"], dtype=torch.float32)[None].to(device)
            out = model._predict_once(img)
            if isinstance(out, dict):
                out = out["one2many"]
            preds = non_max_suppression(out, conf_thres=0.001, iou_thres=0.7,
                                        nc=int(model.model[-1].nc), max_det=100)[0]
            gt_cls = torch.as_tensor(sample["cls"], dtype=torch.float32).squeeze(-1)
            gt_xywh = torch.as_tensor(sample["bboxes"], dtype=torch.float32)
            gt_xyxy = gt_xywh.clone()
            gt_xyxy[:, [0, 1]] -= gt_xyxy[:, [2, 3]] / 2
            gt_xyxy[:, [2, 3]] += gt_xyxy[:, [0, 1]] - gt_xyxy[:, [2, 3]] / 2
            gt_xyxy = gt_xyxy * 640.0  # normalized xywh -> pixel xyxy in letterbox space
            n_gt_boxes += len(gt_cls)
            if preds.shape[0] == 0 or len(gt_cls) == 0:
                tp = np.zeros((preds.shape[0], 10), dtype=bool)
            else:
                iou = box_iou(gt_xyxy.to(device), preds[:, :4])
                tp = match_predictions(preds[:, 5].long(), gt_cls.long().to(device),
                                       iou, IOUV).cpu().numpy()
            stats.append({"tp": tp, "conf": preds[:, 4].cpu().numpy(),
                          "pred_cls": preds[:, 5].cpu().numpy(),
                          "target_cls": gt_cls.numpy(),
                          "target_img": gt_cls.numpy(), "im_name": sid})
    metrics = DetMetrics(names={int(k): v for k, v in names.items()})
    for s in stats:
        metrics.update_stats(s)
    metrics.process()
    rd = metrics.results_dict
    out = {"map50": float(rd["metrics/mAP50(B)"]), "map50_95": float(rd["metrics/mAP50-95(B)"]),
           "n_images": len(dataset), "n_gt_boxes": n_gt_boxes}
    per = {}
    for i, ci in enumerate(metrics.box.ap_class_index.tolist()):
        per[str(ci)] = {"ap50": round(float(metrics.box.ap50[i]), 4),
                        "ap50_95": round(float(metrics.box.ap[i]), 4)}
    out["per_class"] = per
    return out


def late10(run_dir: Path) -> dict:
    import csv
    rows = list(csv.DictReader((run_dir / "results.csv").open()))
    vals = [float(r["metrics/mAP50-95(B)"]) for r in rows if r.get("metrics/mAP50-95(B)")]
    last10 = vals[-10:]
    return {"mean": round(statistics.mean(last10), 4),
            "median": round(statistics.median(last10), 4),
            "std": round(statistics.stdev(last10), 4) if len(last10) > 1 else None,
            "min": round(min(last10), 4), "max": round(max(last10), 4)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group", choices=["C0-N", "C1-I", "C2-D"], required=True)
    p.add_argument("--project", default="runs/step3_earlyfusion")
    p.add_argument("--contract", default=OUT_DEFAULT)
    p.add_argument("--device", default="0")
    a = p.parse_args()

    contract = build_contract(out_path=a.contract)
    a.device = ("cuda:0" if a.device in {"0", "cuda"} and torch.cuda.is_available()
                else ("cpu" if a.device in {"0", "cuda"} else a.device))
    import yaml as _yaml
    names = _yaml.safe_load(Path(
        "D:/pycharm/Python Develop/YOLO_1/v031_step1_rgb_sample/dataset.yaml")
        .read_text(encoding="utf-8"))["names"]
    names = {int(k): v for k, v in names.items()}

    # deterministic shuffle maps (fixed across checkpoints; saved to run dir)
    run_dir = Path(a.project) / a.group
    shuffle_maps = {}
    for split, key in (("train", "shuffle_map_train.json"), ("val", "shuffle_map_val.json"),
                       ("all17", "shuffle_map_all17.json")):
        fp = run_dir / key
        if fp.exists():
            shuffle_maps[split] = json.loads(fp.read_text(encoding="utf-8"))
        else:
            ids = contract[f"{split}_ids"]
            shuffle_maps[split] = group_aware_derangement(ids, {})
            fp.write_text(json.dumps(shuffle_maps[split], indent=2), encoding="utf-8")

    results = {"group": a.group, "contract_sha": contract.get("format_gate_passed"),
               "late10": late10(run_dir)}
    for ck_name in ("last.pt", "best.pt"):
        ck = torch.load(run_dir / "weights" / ck_name, map_location="cpu", weights_only=False)
        # 8.4.56 saves the nn.Module-passed model under "ema" (ckpt["model"] is None);
        # EMA weights are also the standard evaluation choice (validator uses ema).
        model = (ck.get("ema") or ck.get("model")).float()  # trainer saves ema in half
        model.eval()
        model = model.to(a.device)
        results[ck_name] = {}
        for variant in ("NORMAL", "ZERO", "SHUFFLE"):
            results[ck_name][variant] = {}
            for split in ("train", "val", "all17"):
                ds = TriModalDataset(
                    contract, split=split, group=a.group, augment=False,
                    aux_zero=(variant == "ZERO"),
                    aux_id_map=(shuffle_maps[split] if variant == "SHUFFLE" else None))
                results[ck_name][variant][split] = eval_set(model, ds, a.device, names)
        print(f"[{a.group}] {ck_name} NORMAL val mAP50-95="
              f"{results[ck_name]['NORMAL']['val']['map50_95']:.4f} "
              f"ZERO={results[ck_name]['ZERO']['val']['map50_95']:.4f} "
              f"SHUFFLE={results[ck_name]['SHUFFLE']['val']['map50_95']:.4f}")

    # val6 leave-one-out sensitivity (last.pt, NORMAL; diagnostic only)
    if a.group != "C0-N":
        _bc = torch.load(run_dir / "weights" / "last.pt", map_location="cpu", weights_only=False)
        base_ck = (_bc.get("ema") or _bc.get("model")).float().eval().to(a.device)
        c0_dir = Path(a.project) / "C0-N"
        _c0 = torch.load(c0_dir / "weights" / "last.pt", map_location="cpu", weights_only=False)
        c0_ck = (_c0.get("ema") or _c0.get("model")).float().eval().to(a.device)
        val_ids = contract["val_ids"]
        deltas = []
        fold_support = []
        for j in range(len(val_ids)):
            sub = [s for i, s in enumerate(val_ids) if i != j]
            sub_contract = {**contract, "val_ids": sub}
            ds_c = TriModalDataset(sub_contract, split="val", group=a.group, augment=False)
            ds_0 = TriModalDataset(sub_contract, split="val", group="C0-N", augment=False)
            m_c = eval_set(base_ck, ds_c, a.device, names)["map50_95"]
            m_0 = eval_set(c0_ck, ds_0, a.device, names)["map50_95"]
            deltas.append(round(m_c - m_0, 4))
            fold_support.append({"removed": val_ids[j],
                                 "classes_left": sorted(set(int(x.split()[0]) for s in sub
                                        for x in (Path(contract["_labels_dir"]) /
                                                  f"{s}.txt").read_text().splitlines()
                                        if x.strip()))})
        results["val6_loo"] = {
            "note": "diagnostic only; NOT a significance test",
            "deltas": deltas, "positive_folds": sum(d > 0 for d in deltas),
            "median_delta": round(float(np.median(deltas)), 4),
            "min_delta": min(deltas), "max_delta": max(deltas),
            "fold_class_support": fold_support}

    out = run_dir / "eval_step3_causality.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
