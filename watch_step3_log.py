from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path


# ============================================================
# 改这里：你的本地 Step 3 日志
# ============================================================

SOURCE_FILE = Path(
    r"C:\Users\xyh23\Documents\ChatGPT\多模态模型"
    r"\multimodal_yolo26_qaf_v0_3"
    r"\docs\STEP3_IMPLEMENTATION_LOG.md"
)

# 当前脚本所在目录 = 1xyh1/log 的本地 clone
REPO_DIR = Path(__file__).resolve().parent

DEST_FILE = REPO_DIR / "STEP3_IMPLEMENTATION_LOG.md"
META_FILE = REPO_DIR / "sync_meta.json"

REMOTE = "origin"

POLL_SECONDS = 2
DEBOUNCE_SECONDS = 20


def git(*args: str, check: bool = True):
    p = subprocess.run(
        ["git", *args],
        cwd=REPO_DIR,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if check and p.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed\n"
            f"stdout:\n{p.stdout}\n"
            f"stderr:\n{p.stderr}"
        )

    return p


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def source_signature():
    if not SOURCE_FILE.exists():
        return None

    st = SOURCE_FILE.stat()

    return (
        st.st_mtime_ns,
        st.st_size,
    )


def current_branch() -> str:
    return git(
        "branch",
        "--show-current",
    ).stdout.strip()


def staged_changes_exist() -> bool:
    # 0 = 无差异
    # 1 = 有差异
    p = git(
        "diff",
        "--cached",
        "--quiet",
        check=False,
    )

    return p.returncode == 1


def sync_once():
    if not SOURCE_FILE.exists():
        print(f"[sync] source missing: {SOURCE_FILE}")
        return

    source_hash = sha256(SOURCE_FILE)

    if DEST_FILE.exists():
        dest_hash = sha256(DEST_FILE)
    else:
        dest_hash = None

    if source_hash == dest_hash:
        print("[sync] content unchanged")
        return

    # 复制日志
    shutil.copy2(SOURCE_FILE, DEST_FILE)

    meta = {
        "source_file": str(SOURCE_FILE),
        "synced_at_local": datetime.now().astimezone().isoformat(),
        "sha256": source_hash,
        "size_bytes": SOURCE_FILE.stat().st_size,
    }

    META_FILE.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # 只提交这两个文件，不 git add .
    git(
        "add",
        "--",
        DEST_FILE.name,
        META_FILE.name,
    )

    if not staged_changes_exist():
        print("[sync] nothing staged")
        return

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    git(
        "commit",
        "-m",
        f"auto: sync Step3 implementation log {now}",
    )

    sha = git(
        "rev-parse",
        "--short",
        "HEAD",
    ).stdout.strip()

    branch = current_branch()

    print(f"[sync] committed {sha}")
    print(f"[sync] pushing origin/{branch}")

    p = git(
        "push",
        REMOTE,
        branch,
        check=False,
    )

    if p.returncode != 0:
        print("[sync] PUSH FAILED")
        print(p.stderr)
        print(
            "[sync] 本地 commit 已保留；"
            "解决网络/认证后执行 git push 即可。"
        )
    else:
        print("[sync] PUSH OK")


def main():
    if not (REPO_DIR / ".git").exists():
        raise RuntimeError(
            f"{REPO_DIR} 不是 Git 仓库"
        )

    print("=" * 70)
    print("Step 3 implementation log watcher")
    print("=" * 70)
    print(f"source : {SOURCE_FILE}")
    print(f"repo   : {REPO_DIR}")
    print(f"target : {DEST_FILE}")
    print(f"branch : {current_branch()}")
    print(
        f"poll={POLL_SECONDS}s, "
        f"debounce={DEBOUNCE_SECONDS}s"
    )
    print("Ctrl+C 停止")
    print()

    old_signature = source_signature()
    dirty_at = None

    while True:
        try:
            time.sleep(POLL_SECONDS)

            new_signature = source_signature()

            if new_signature != old_signature:
                old_signature = new_signature
                dirty_at = time.monotonic()

                print(
                    f"[watch] changed "
                    f"{datetime.now():%H:%M:%S}"
                )

                continue

            if (
                dirty_at is not None
                and time.monotonic() - dirty_at
                >= DEBOUNCE_SECONDS
            ):
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
