#!/usr/bin/env python3
"""Read-only forensic audit INSIDE the formal AIC2026 training ZIP.

Does not extract the dataset. Produces:
- private report: may contain raw sample IDs / filename grammar evidence; keep OUTSIDE repo.
- public report: aggregate facts only; safe for repo before FINAL HOLDOUT is frozen.

Authority:
- ZIP central directory for member inventory/pairing.
- PNG/JPEG headers for encoded precision/channels/dimensions.
- cv2.imdecode(IMREAD_UNCHANGED) on a deterministic small sample for runtime representation.
- all label TXT members for exact-format/class/geometry audit.

This tool does NOT invent a scene/sequence grouping rule.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import math
import re
import struct
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

MODS = ("visible", "infrared", "depth", "labels")
IMG_MODS = ("visible", "infrared", "depth")
IMG_EXTS = {".png", ".jpg", ".jpeg"}
LABEL_EXTS = {".txt"}
OFFICIAL_CLASS_NAMES = [
    "person", "boat", "animal", "seat", "sign", "bicycle",
    "car", "ball", "light", "garbage can", "uav", "tricycle",
]
PNG_SIG = b"\x89PNG\r\n\x1a\n"
JPEG_SOF = {
    0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
    0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
}

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sha256_json(obj) -> str:
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(raw)

def modality_of(name: str):
    parts = [p for p in name.replace("\\", "/").split("/") if p]
    if len(parts) < 2:
        return None
    parent = parts[-2].lower()
    return parent if parent in MODS else None

def sample_id_of(name: str) -> str:
    return Path(name).stem

def png_header(prefix: bytes) -> dict:
    if len(prefix) < 29 or prefix[:8] != PNG_SIG:
        return {"ok": False, "error": "bad_png_signature_or_short"}
    if prefix[12:16] != b"IHDR":
        return {"ok": False, "error": "missing_IHDR"}
    w, h = struct.unpack(">II", prefix[16:24])
    bit_depth = int(prefix[24])
    color_type = int(prefix[25])
    channels = {0:1, 2:3, 3:1, 4:2, 6:4}.get(color_type)
    return {
        "ok": True, "format": "PNG", "width": w, "height": h,
        "bit_depth": bit_depth, "color_type": color_type, "encoded_channels": channels,
    }

def jpeg_header(prefix: bytes) -> dict:
    if len(prefix) < 4 or prefix[:2] != b"\xff\xd8":
        return {"ok": False, "error": "bad_jpeg_signature_or_short"}
    i = 2
    n = len(prefix)
    while i + 3 < n:
        while i < n and prefix[i] != 0xFF:
            i += 1
        while i < n and prefix[i] == 0xFF:
            i += 1
        if i >= n:
            break
        marker = prefix[i]
        i += 1
        if marker in {0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
            continue
        if marker == 0xDA:  # SOS reached before SOF
            break
        if i + 1 >= n:
            break
        seglen = (prefix[i] << 8) | prefix[i+1]
        if seglen < 2 or i + seglen > n:
            break
        if marker in JPEG_SOF:
            if seglen < 8:
                return {"ok": False, "error": "short_SOF"}
            precision = int(prefix[i+2])
            h = (prefix[i+3] << 8) | prefix[i+4]
            w = (prefix[i+5] << 8) | prefix[i+6]
            comps = int(prefix[i+7])
            return {
                "ok": True, "format": "JPEG", "sof_marker": f"0x{marker:02X}",
                "precision_bits": precision, "width": w, "height": h,
                "encoded_channels": comps,
            }
        i += seglen
    return {"ok": False, "error": "SOF_not_found_in_prefix"}

def header_from_zip(zf: zipfile.ZipFile, info: zipfile.ZipInfo) -> dict:
    ext = Path(info.filename).suffix.lower()
    with zf.open(info, "r") as f:
        if ext == ".png":
            prefix = f.read(64)
            return png_header(prefix)
        if ext in {".jpg", ".jpeg"}:
            prefix = f.read(262144)
            return jpeg_header(prefix)
    return {"ok": False, "error": f"unsupported_extension:{ext}"}

def decode_member(zf: zipfile.ZipFile, info: zipfile.ZipInfo) -> dict:
    raw = zf.read(info)
    arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        return {"ok": False, "error": "cv2_imdecode_failed"}
    out = {
        "ok": True,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": int(arr.min()) if arr.size else None,
        "max": int(arr.max()) if arr.size else None,
    }
    if arr.ndim == 3:
        out["channels"] = arr.shape[2]
        if arr.shape[2] == 3:
            x = arr.astype(np.int32)
            out["channel_max_abs_diff_01"] = int(np.abs(x[...,0]-x[...,1]).max())
            out["channel_max_abs_diff_12"] = int(np.abs(x[...,1]-x[...,2]).max())
    else:
        out["channels"] = 1
    return out

def label_audit(raw: bytes, nc: int) -> dict:
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        return {"ok": False, "errors": [f"decode:{e}"], "classes": [], "n_boxes": 0}
    errors, classes = [], []
    n_boxes = 0
    for ln, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"line{ln}:field_count={len(parts)}")
            continue
        try:
            cf = float(parts[0])
            vals = [float(x) for x in parts[1:]]
        except ValueError:
            errors.append(f"line{ln}:non_numeric")
            continue
        if not cf.is_integer():
            errors.append(f"line{ln}:class_not_integer:{cf}")
            continue
        c = int(cf)
        if not (0 <= c < nc):
            errors.append(f"line{ln}:class_out_of_range:{c}")
        if not all(math.isfinite(v) for v in vals):
            errors.append(f"line{ln}:nonfinite")
            continue
        cx, cy, w, h = vals
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
            errors.append(f"line{ln}:normalized_bounds:{vals}")
        if cx-w/2 < -1e-9 or cx+w/2 > 1+1e-9 or cy-h/2 < -1e-9 or cy+h/2 > 1+1e-9:
            errors.append(f"line{ln}:box_exceeds_image:{vals}")
        classes.append(c); n_boxes += 1
    return {"ok": not errors, "errors": errors[:50], "classes": classes, "n_boxes": n_boxes}

def summarize_counter(c: collections.Counter) -> dict:
    return {str(k): int(v) for k, v in sorted(c.items(), key=lambda kv: str(kv[0]))}

def id_grammar(ids: list[str]) -> dict:
    token_count = collections.Counter(len(x.split("_")) for x in ids)
    first_token = collections.Counter(x.split("_")[0] for x in ids)
    prefix2 = collections.Counter("_".join(x.split("_")[:2]) for x in ids if len(x.split("_")) >= 2)
    # Do not place raw IDs/prefix values in PUBLIC output; private only.
    return {
        "token_count_distribution": summarize_counter(token_count),
        "n_unique_first_token": len(first_token),
        "first_token_size_distribution": summarize_counter(collections.Counter(first_token.values())),
        "n_unique_first_two_tokens": len(prefix2),
        "first_two_token_size_distribution": summarize_counter(collections.Counter(prefix2.values())),
        "private_first_token_counts": dict(first_token.most_common()),
        "private_first_two_token_counts": dict(prefix2.most_common()),
    }

def extension_runs(ids: list[str], ext_by_id: dict[str,str]) -> list[dict]:
    runs = []
    if not ids:
        return runs
    start = prev = ids[0]
    ext = ext_by_id[start]
    count = 1
    for sid in ids[1:]:
        e = ext_by_id[sid]
        if e == ext:
            prev = sid; count += 1
        else:
            runs.append({"start_id": start, "end_id": prev, "extension": ext, "count": count})
            start = prev = sid; ext = e; count = 1
    runs.append({"start_id": start, "end_id": prev, "extension": ext, "count": count})
    return runs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--private-out", required=True,
                    help="MUST be outside repo; contains raw ID evidence")
    ap.add_argument("--public-out", default="reports/step4_t1gr/zip_forensic_public.json")
    ap.add_argument("--expected-samples", type=int, default=2000)
    ap.add_argument("--decode-per-format", type=int, default=8)
    a = ap.parse_args()

    zp = Path(a.zip)
    if not zp.is_file():
        raise SystemExit(f"ZIP_NOT_FOUND:{zp}")
    private_out = Path(a.private_out).resolve()
    repo_root = Path(__file__).resolve().parents[1]
    try:
        private_out.relative_to(repo_root.resolve())
        raise SystemExit("PRIVATE_OUT_MUST_BE_OUTSIDE_REPO")
    except ValueError:
        pass

    by_mod = {m: {} for m in MODS}
    duplicate_ids = {m: [] for m in MODS}
    ext_counts = {m: collections.Counter() for m in MODS}
    header_counts = {m: collections.Counter() for m in IMG_MODS}
    header_errors = []
    infos_by_mod = {m: {} for m in MODS}

    class_boxes = collections.Counter()
    class_images = collections.Counter()
    label_errors = {}
    decoded_samples = []

    with zipfile.ZipFile(zp, "r") as zf:
        infos = [x for x in zf.infolist() if not x.is_dir()]
        for info in infos:
            mod = modality_of(info.filename)
            if mod is None:
                continue
            sid = sample_id_of(info.filename)
            if sid in by_mod[mod]:
                duplicate_ids[mod].append(sid)
            by_mod[mod][sid] = info.filename
            infos_by_mod[mod][sid] = info
            ext_counts[mod][Path(info.filename).suffix.lower()] += 1

        id_sets = {m: set(by_mod[m]) for m in MODS}
        common = set.intersection(*(id_sets[m] for m in MODS))
        union = set.union(*(id_sets[m] for m in MODS))
        same_sets = all(id_sets[m] == id_sets[MODS[0]] for m in MODS)

        # All labels are small enough to validate exactly.
        for sid, info in infos_by_mod["labels"].items():
            la = label_audit(zf.read(info), len(OFFICIAL_CLASS_NAMES))
            if not la["ok"]:
                label_errors[sid] = la["errors"]
            present = set(la["classes"])
            for c in la["classes"]:
                class_boxes[c] += 1
            for c in present:
                class_images[c] += 1

        # Header audit of ALL image members. Cheap: PNG 64 bytes; JPEG header prefix only.
        headers_private = {m: {} for m in IMG_MODS}
        for mod in IMG_MODS:
            for sid, info in infos_by_mod[mod].items():
                h = header_from_zip(zf, info)
                headers_private[mod][sid] = h
                if not h.get("ok"):
                    header_errors.append({"modality": mod, "sample_id": sid, "header": h})
                key = (
                    Path(info.filename).suffix.lower(),
                    h.get("format"), h.get("bit_depth", h.get("precision_bits")),
                    h.get("encoded_channels"), h.get("width"), h.get("height"),
                    h.get("ok"),
                )
                header_counts[mod][key] += 1

        # Deterministic runtime decode samples per modality/extension.
        for mod in IMG_MODS:
            ext_to_ids = collections.defaultdict(list)
            for sid, info in infos_by_mod[mod].items():
                ext_to_ids[Path(info.filename).suffix.lower()].append(sid)
            for ext, ids in sorted(ext_to_ids.items()):
                for sid in sorted(ids)[:a.decode_per_format]:
                    dec = decode_member(zf, infos_by_mod[mod][sid])
                    decoded_samples.append({
                        "modality": mod, "extension": ext, "sample_id": sid,
                        "decode": dec,
                    })

        # Per-sample triplet extension identity.
        triplet_ext_mismatch = []
        triplet_shape_header_mismatch = []
        ext_by_id = {}
        for sid in sorted(common):
            exts = {
                m: Path(infos_by_mod[m][sid].filename).suffix.lower()
                for m in IMG_MODS
            }
            if len(set(exts.values())) != 1:
                triplet_ext_mismatch.append({"sample_id": sid, "extensions": exts})
            else:
                ext_by_id[sid] = exts["visible"]
            hs = {m: headers_private[m][sid] for m in IMG_MODS}
            dims = {(h.get("width"), h.get("height")) for h in hs.values() if h.get("ok")}
            if len(dims) > 1:
                triplet_shape_header_mismatch.append({"sample_id": sid, "headers": hs})

    ids_sorted = sorted(common)
    grammar = id_grammar(ids_sorted)
    runs = extension_runs(ids_sorted, ext_by_id)

    private = {
        "schema": "t1gr-zip-forensic-private-v1",
        "read_only": True,
        "zip_path": str(zp.resolve()),
        "zip_bytes": zp.stat().st_size,
        "expected_samples": a.expected_samples,
        "member_counts": {m: len(by_mod[m]) for m in MODS},
        "extension_counts": {m: summarize_counter(ext_counts[m]) for m in MODS},
        "id_sets_equal": same_sets,
        "common_id_count": len(common),
        "union_id_count": len(union),
        "duplicate_ids": duplicate_ids,
        "triplet_extension_mismatch": triplet_ext_mismatch,
        "triplet_header_dimension_mismatch": triplet_shape_header_mismatch,
        "header_errors": header_errors,
        "header_counts": {
            m: [
                {
                    "extension": k[0], "format": k[1], "precision_or_bit_depth": k[2],
                    "encoded_channels": k[3], "width": k[4], "height": k[5],
                    "header_ok": k[6], "count": v,
                }
                for k, v in header_counts[m].items()
            ] for m in IMG_MODS
        },
        "decode_samples": decoded_samples,
        "labels": {
            "errors": label_errors,
            "box_count_by_class": {str(c): int(class_boxes[c]) for c in range(12)},
            "image_count_by_class": {str(c): int(class_images[c]) for c in range(12)},
            "class_names": OFFICIAL_CLASS_NAMES,
        },
        "id_grammar": grammar,
        "extension_runs_private": runs,
        "raw_member_name_sha256_commitment": sha256_json(
            sorted(x for m in MODS for x in by_mod[m].values())
        ),
    }

    depth_header_summary = private["header_counts"]["depth"]
    public = {
        "schema": "t1gr-zip-forensic-public-v1",
        "read_only": True,
        "expected_samples": a.expected_samples,
        "member_counts": private["member_counts"],
        "extension_counts": private["extension_counts"],
        "id_sets_equal": same_sets,
        "common_id_count": len(common),
        "duplicate_id_counts": {m: len(v) for m, v in duplicate_ids.items()},
        "triplet_extension_mismatch_count": len(triplet_ext_mismatch),
        "triplet_header_dimension_mismatch_count": len(triplet_shape_header_mismatch),
        "header_error_count": len(header_errors),
        "header_counts": private["header_counts"],
        "decode_samples_redacted": [
            {
                "modality": x["modality"], "extension": x["extension"],
                "decode": x["decode"],
            } for x in decoded_samples
        ],
        "labels": {
            "error_sample_count": len(label_errors),
            "box_count_by_class": private["labels"]["box_count_by_class"],
            "image_count_by_class": private["labels"]["image_count_by_class"],
            "class_names": OFFICIAL_CLASS_NAMES,
        },
        "id_grammar_aggregate": {
            k: v for k, v in grammar.items() if not k.startswith("private_")
        },
        "extension_run_count": len(runs),
        "zip_member_names_sha256_commitment": private["raw_member_name_sha256_commitment"],
        "formal_gate": {
            "counts_2000_each": all(len(by_mod[m]) == a.expected_samples for m in MODS),
            "id_sets_equal": same_sets,
            "no_duplicate_ids": all(not v for v in duplicate_ids.values()),
            "triplet_extensions_equal": not triplet_ext_mismatch,
            "triplet_dimensions_equal": not triplet_shape_header_mismatch,
            "all_image_headers_parse": not header_errors,
            "labels_exact_valid": not label_errors,
        },
        "depth_semantics_status": "REQUIRES_REVIEW_OF_HEADER_AND_DECODE_EVIDENCE",
        "group_rule_status": "UNRESOLVED_DO_NOT_SPLIT",
        "step1_status": "HOLD",
    }
    public["formal_gate"]["all_structural_checks_pass"] = all(public["formal_gate"].values())

    private_out.parent.mkdir(parents=True, exist_ok=True)
    private_out.write_text(json.dumps(private, ensure_ascii=False, indent=2), encoding="utf-8")
    pub = repo_root / a.public_out
    pub.parent.mkdir(parents=True, exist_ok=True)
    if pub.exists():
        raise SystemExit(f"REFUSE_OVERWRITE:{pub}")
    pub.write_text(json.dumps(public, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "schema": public["schema"],
        "public_out": str(pub),
        "private_out": str(private_out),
        "structural_gate": public["formal_gate"],
        "depth_header_counts": depth_header_summary,
        "label_error_samples": len(label_errors),
        "group_rule_status": public["group_rule_status"],
        "step1_status": "HOLD",
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
