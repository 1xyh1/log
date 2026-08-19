from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

# ============================================================
# Step 3 源码镜像同步：docs 日志 + src/multimodal + scripts + tests + 小型 reports
# ============================================================

SOURCE_ROOT = Path(
    r"C:\Users\xyh23\Documents\ChatGPT\多模态模型"
    r"\multimodal_yolo26_qaf_v0_3"
)

MIRROR_REPO = Path(
    r"D:\pycharm\Python Develop\YOLO_1\step3_log_sync"
)

REMOTE = "origin"

POLL_SECONDS = 2
DEBOUNCE_SECONDS = 20

# 允许同步的相对路径规则（前缀）
SYNC_PREFIXES = (
    "src/multimodal/",
    "scripts/",
    "tests/",
    "docs/",
    "reports/step",
    "T_SERIES_README.md",
    "T_SERIES_IMPLEMENTATION_VALIDATION.json",
    "T1S_README.md",
    "T1S_IMPLEMENTATION_VALIDATION.json",
    "T1TR_README.md",
    "T1TR_IMPLEMENTATION_VALIDATION.json",
)

# 结果目录：只同步文本与必要图（results.png = val 曲线）。
# 收窄到当前冻结/在研实验（避免历史 step1/step2 结果淹没 log 仓库）。
# F1 仍沿用下方文本/必要曲线白名单；checkpoint 和原始数据继续明确屏蔽。
RESULTS_PREFIXES = (
    "runs/step3_earlyfusion/",
    "runs/step4_f0/",
    "runs/step4_f1_ir_gate/",
    "runs/step4_f1_b_corruption/",
    "runs/step4_f1_c/",
    "runs/step4_tseries/",
    "runs/step4_tseries_smoke/",
    "runs/step4_t1tr/",
    "runs/step4_t1tr_smoke/",
)
RESULTS_TEXT_SUFFIXES = {".json", ".jsonl", ".csv", ".yaml", ".txt"}
RESULTS_FIGURE_NAMES = {"results.png"}

# 明确屏蔽
BLOCK_SUFFIXES = {
    ".pt", ".pth", ".onnx", ".engine", ".zip", ".png", ".jpg", ".jpeg",
    ".pyc", ".log", ".tmp",
}
BLOCK_NAMES = {".env", "token", "key", "secret"}
BLOCK_PATH_PARTS = {".venv", "__pycache__", ".git", "sample_multimodal",
                    "datasets", "data", "images", "labels"}

MANIFEST_NAME = "SOURCE_MIRROR_MANIFEST.json"


def git(*args: str, check: bool = True):
    p = subprocess.run(
        ["git", *args],
        cwd=MIRROR_REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and p.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed\nstdout:\n{p.stdout}\nstderr:\n{p.stderr}"
        )
    return p


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_allowed(rel: Path) -> bool:
    s = rel.as_posix()
    parts = set(rel.parts)
    if parts & BLOCK_PATH_PARTS:
        return False
    low = rel.name.lower()
    if any(low.startswith(b) for b in BLOCK_NAMES):
        return False
    if rel.name.startswith("."):
        return False
    if any(s.startswith(p) for p in RESULTS_PREFIXES):
        if rel.suffix.lower() in RESULTS_TEXT_SUFFIXES:
            return True
        return rel.name in RESULTS_FIGURE_NAMES
    if not any(s == p.rstrip("/") or s.startswith(p) for p in SYNC_PREFIXES):
        return False
    if rel.suffix.lower() in BLOCK_SUFFIXES:
        return False
    return True


def iter_source_files():
    for p in sorted(SOURCE_ROOT.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(SOURCE_ROOT)
        if is_allowed(rel):
            yield rel


def source_state():
    state = {}
    for rel in iter_source_files():
        st = (SOURCE_ROOT / rel).stat()
        state[rel.as_posix()] = (st.st_mtime_ns, st.st_size)
    return state


def mirror_path(rel: Path) -> Path:
    return MIRROR_REPO / rel


def write_manifest(changed: list[str]):
    entries = {}
    for rel_str, _ in sorted(source_state().items()):
        rel = Path(rel_str)
        src = SOURCE_ROOT / rel
        raw = src.read_bytes()
        eol = "crlf" if b"\r\n" in raw else "lf"
        canonical_lf = raw.replace(b"\r\n", b"\n")
        entries[rel_str] = {
            "sha256": sha256(src),          # source disk bytes (append-only key)
            "size_bytes": src.stat().st_size,
            "mtime_ns": src.stat().st_mtime_ns,
            # EOL provenance (reviewer 2026-08-16): GitHub stores LF bytes, the
            # local disk stores CRLF, so a disk-byte SHA never matches the
            # GitHub blob.  canonical_lf_sha256 matches the remote blob for
            # text files with no other diff; source_eol records the local EOL.
            "canonical_lf_sha256": hashlib.sha256(canonical_lf).hexdigest(),
            "source_eol": eol,
        }
    manifest = {
        "generated_at_local": datetime.now().astimezone().isoformat(),
        "source_root": str(SOURCE_ROOT),
        "files": entries,
        "changed_in_this_sync": changed,
    }
    (MIRROR_REPO / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def stage_whitelist():
    """只 add 白名单，不 git add .；同时跟踪镜像内删除。

    注意：git add 多路径中任一不存在会整体中止，必须过滤为实际存在的路径。
    """
    specs = ["src/multimodal", "scripts", "tests", "docs", "reports", "runs",
             MANIFEST_NAME, "watch_step3_source_mirror.py",
             "start_step3_source_mirror.bat", ".gitignore", "README.md",
             "T_SERIES_README.md", "T_SERIES_IMPLEMENTATION_VALIDATION.json",
             "T1S_README.md", "T1S_IMPLEMENTATION_VALIDATION.json",
             "T1TR_README.md", "T1TR_IMPLEMENTATION_VALIDATION.json"]
    existing = [s for s in specs if (MIRROR_REPO / s).exists()]
    if existing:
        git("add", "--", *existing, check=False)
    tracked_dirs = [s for s in ("src/multimodal", "scripts", "tests", "docs", "reports", "runs")
                    if (MIRROR_REPO / s).exists()]
    if tracked_dirs:
        git("add", "-u", "--", *tracked_dirs, check=False)


def staged_changes_exist() -> bool:
    p = git("diff", "--cached", "--quiet", check=False)
    return p.returncode == 1


def sync_once():
    state = source_state()
    # 删除镜像里多余的旧文件
    mirror_files = set()
    for p in MIRROR_REPO.rglob("*"):
        if not p.is_file() or ".git" in p.parts:
            continue
        rel = p.relative_to(MIRROR_REPO)
        if rel.name == MANIFEST_NAME or rel.name in {"watch_step3_source_mirror.py",
                                                      "start_step3_source_mirror.bat",
                                                      ".gitignore", "README.md",
                                                      "watch_step3_log.py", "start_watch.bat"}:
            continue
        if is_allowed(rel) and rel.as_posix() not in state:
            mirror_files.add(rel)
    changed = []
    for rel_str in sorted(state):
        rel = Path(rel_str)
        src = SOURCE_ROOT / rel
        dst = mirror_path(rel)
        if not dst.exists() or sha256(dst) != sha256(src):
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            changed.append(rel_str)
    for rel in mirror_files:
        (MIRROR_REPO / rel).unlink(missing_ok=True)
        changed.append(rel.as_posix() + " (deleted)")

    if changed:
        write_manifest(changed)
        stage_whitelist()
        if not staged_changes_exist():
            print("[sync] nothing staged")
            return
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        summary = ", ".join(changed[:5]) + (f" ...(+{len(changed)-5})" if len(changed) > 5 else "")
        git("commit", "-m", f"auto: mirror Step3 sources {now}\n\n{summary}")
        sha = git("rev-parse", "--short", "HEAD").stdout.strip()
        branch = git("branch", "--show-current").stdout.strip()
        print(f"[sync] committed {sha}: {len(changed)} files")
        p = git("push", REMOTE, branch, check=False)
        if p.returncode != 0:
            print("[sync] PUSH FAILED")
            print(p.stderr)
            print("[sync] 本地 commit 已保留")
        else:
            print("[sync] PUSH OK")
    else:
        print("[sync] content unchanged")


def main():
    if not (MIRROR_REPO / ".git").exists():
        raise RuntimeError(f"{MIRROR_REPO} 不是 Git 仓库")
    print("=" * 70)
    print("Step 3 source mirror watcher")
    print("=" * 70)
    print(f"source : {SOURCE_ROOT}")
    print(f"mirror : {MIRROR_REPO}")
    print(f"branch : {git('branch', '--show-current').stdout.strip()}")
    print(f"poll={POLL_SECONDS}s debounce={DEBOUNCE_SECONDS}s")
    print("first full sync ...")
    sync_once()
    print("monitoring ... Ctrl+C 停止")
    old_state = source_state()
    dirty_at = None
    while True:
        try:
            time.sleep(POLL_SECONDS)
            new_state = source_state()
            if new_state != old_state:
                old_state = new_state
                dirty_at = time.monotonic()
                print(f"[watch] changed {datetime.now():%H:%M:%S}")
                continue
            if dirty_at is not None and time.monotonic() - dirty_at >= DEBOUNCE_SECONDS:
                try:
                    sync_once()
                except Exception as e:
                    print(f"[sync] ERROR: {e}")
                dirty_at = None
        except KeyboardInterrupt:
            print("\n[watch] stopped")
            break


if __name__ == "__main__":
    main()
