from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_make_overfit_split_is_deterministic_and_bounded(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("\n".join(f"sample-{i:02d}" for i in range(20)) + "\n", encoding="utf-8")
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "make_overfit_split.py"),
        "--source",
        str(source),
        "--count",
        "8",
        "--seed",
        "17",
    ]
    subprocess.run([*command, "--out", str(first)], check=True, capture_output=True, text=True)
    subprocess.run([*command, "--out", str(second)], check=True, capture_output=True, text=True)
    assert first.read_bytes() == second.read_bytes()
    selected = first.read_text(encoding="utf-8").splitlines()
    assert len(selected) == len(set(selected)) == 8
