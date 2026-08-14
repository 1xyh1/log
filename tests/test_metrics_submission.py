from pathlib import Path

import numpy as np

from mmod_qaf.metrics import Detection, GroundTruth, evaluate_map
from mmod_qaf.submission import make_submission_zip, validate_submission_dir


def test_perfect_map():
    gt=[GroundTruth('a',0,np.array([0,0,10,10],np.float32))]
    pred=[Detection('a',0,.9,np.array([0,0,10,10],np.float32))]
    result=evaluate_map(pred,gt)
    assert abs(result['mAP50-95']-1.0)<1e-6


def test_submission_requires_empty_files(tmp_path: Path):
    (tmp_path/'a.txt').write_text('',encoding='utf-8')
    errors=validate_submission_dir(tmp_path,['a','b'])
    assert any('b.txt' in e for e in errors)
    (tmp_path/'b.txt').write_text('0 0.5 0.5 0.1 0.1 0.9\n',encoding='utf-8')
    assert not validate_submission_dir(tmp_path,['a','b'])


def test_submission_zip_contains_only_flat_txt_files(tmp_path: Path):
    import zipfile

    pred = tmp_path / "pred"
    pred.mkdir()
    (pred / "a.txt").write_text("", encoding="utf-8")
    (pred / "b.txt").write_text("1 0.5 0.5 0.1 0.1 0.9\n", encoding="utf-8")
    (pred / "ignored.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "submission.zip"
    make_submission_zip(pred, output)
    with zipfile.ZipFile(output) as archive:
        assert archive.namelist() == ["a.txt", "b.txt"]
