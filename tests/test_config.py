from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))


def test_same_loop_rgb_baseline_changes_only_the_model_branch():
    baseline = _load("b0_s_rgb_same_loop.yaml")
    for candidate_name in ("c1_s_p5_concat.yaml", "c2_s_p45_concat.yaml", "c4_s_p45_qaf.yaml"):
        candidate = _load(candidate_name)
        assert baseline["weights"] == candidate["weights"]
        assert baseline["device"] == candidate["device"]
        assert baseline["seed"] == candidate["seed"]
        assert baseline["deterministic"] == candidate["deterministic"]
        assert baseline["max_det"] == candidate["max_det"]
        assert baseline["val_conf"] == candidate["val_conf"]
        assert baseline["data"] == candidate["data"]
        assert baseline["train"] == candidate["train"]

    assert baseline["model"]["rgb_only"] is True
    assert not any(baseline["model"][name] for name in ("fuse_p3", "fuse_p4", "fuse_p5"))


def test_shipped_config_paths_exist_and_preprocessing_is_explicit():
    for path in sorted((ROOT / "configs").glob("*.yaml")):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        for split in ("train", "val"):
            data = config["data"][split]
            assert (ROOT / data["ids_file"]).resolve().is_file()
            for field in ("imgsz", "ir_mode", "depth_min_mm", "depth_max_mm", "depth_resize"):
                assert field in data, f"{path.name}: data.{split}.{field} is implicit"
