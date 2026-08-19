from __future__ import annotations
import importlib.util
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "probe", ROOT/"scripts/t1gr_probe_zip_forensics.py"
)
m = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(m)

def png_bytes(w=640,h=480,bit=16,color=0):
    return (
        m.PNG_SIG
        + struct.pack(">I", 13) + b"IHDR"
        + struct.pack(">II", w,h)
        + bytes([bit,color,0,0,0])
        + b"\x00\x00\x00\x00"
    )

def jpeg_sof_bytes(w=640,h=480,precision=8,components=1):
    # SOI + SOF0 segment; sufficient for header parser unit test.
    seglen = 8 + 3*components
    comp = b"".join(bytes([i+1,0x11,0]) for i in range(components))
    return (
        b"\xff\xd8\xff\xc0"
        + struct.pack(">H", seglen)
        + bytes([precision])
        + struct.pack(">HH", h,w)
        + bytes([components])
        + comp
        + b"\xff\xd9"
    )

def test_png_16bit_gray_header():
    h=m.png_header(png_bytes())
    assert h["ok"] and h["bit_depth"]==16 and h["encoded_channels"]==1
    assert (h["width"],h["height"])==(640,480)

def test_jpeg_8bit_gray_header():
    h=m.jpeg_header(jpeg_sof_bytes())
    assert h["ok"] and h["precision_bits"]==8 and h["encoded_channels"]==1
    assert (h["width"],h["height"])==(640,480)

def test_jpeg_8bit_three_channel_header():
    h=m.jpeg_header(jpeg_sof_bytes(1920,1080,8,3))
    assert h["ok"] and h["encoded_channels"]==3

def test_strict_label_valid():
    x=m.label_audit(b"0 0.5 0.5 0.2 0.2\n11 0.5 0.5 1 1\n",12)
    assert x["ok"] and x["n_boxes"]==2

def test_label_six_columns_rejected():
    x=m.label_audit(b"0 0.5 0.5 0.2 0.2 0.9\n",12)
    assert not x["ok"]

def test_label_box_outside_rejected():
    x=m.label_audit(b"0 0.05 0.5 0.2 0.2\n",12)
    assert not x["ok"]

def test_id_grammar_no_public_raw_ids_required():
    g=m.id_grammar(["001_a_01","001_a_02","002_b_01"])
    assert g["n_unique_first_token"]==2
    assert "private_first_token_counts" in g

def test_extension_runs():
    ids=["a","b","c","d"]
    r=m.extension_runs(ids,{"a":".jpg","b":".jpg","c":".png","d":".png"})
    assert [x["count"] for x in r]==[2,2]
