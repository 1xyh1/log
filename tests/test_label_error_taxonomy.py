from pathlib import Path
import importlib.util, json
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("m",ROOT/"scripts/t1gr_label_error_taxonomy.py")
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def test_clean():
    r=m.parse_label(b"0 0.5 0.5 0.2 0.2\n",12)
    assert not r["hard_errors"] and not r["strict_scalar_errors"] and not r["corner_overflows"]

def test_corner_overflow_not_hard():
    r=m.parse_label(b"0 0.05 0.5 0.2 0.2\n",12)
    assert not r["hard_errors"]
    assert not r["ultralytics_tolerance_errors"]
    assert r["corner_overflows"]

def test_six_columns_hard():
    r=m.parse_label(b"0 0.5 0.5 0.2 0.2 1\n",12)
    assert r["hard_errors"]

def test_class_noninteger_hard():
    r=m.parse_label(b"0.5 0.5 0.5 0.2 0.2\n",12)
    assert r["hard_errors"]

def test_strict_only_ultralytics_tolerance():
    r=m.parse_label(b"0 1.005 0.5 0.01 0.01\n",12)
    assert r["strict_scalar_errors"]
    assert not r["ultralytics_tolerance_errors"]

def test_ultralytics_reject():
    r=m.parse_label(b"0 1.02 0.5 0.01 0.01\n",12)
    assert r["ultralytics_tolerance_errors"]

def test_duplicate():
    x=b"0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n"
    assert m.parse_label(x,12)["duplicate_rows"]==1

def test_empty_valid_background():
    r=m.parse_label(b"\n",12)
    assert r["empty"] and not r["hard_errors"]
