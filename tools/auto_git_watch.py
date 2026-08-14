from __future__ import annotations

import hashlib
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

# 只监控这些目录/文件
WATCH_TARGETS = [
    ROOT / "src" / "multimodal",
    ROOT / "scripts",
    ROOT / "tests",
    ROOT / "docs" / "STEP3_IMPLEMENTATION_LOG.md",
]

# 只自动提交文本/源码
ALLOWED_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
}

POLL_SECONDS = 2.0

# 最后一次修改后多少秒没有继续变化，才 commit
DEBOUNCE_SECONDS = 20.0

# 是否自动 push
AUTO_PUSH = True

# push 失败后不会删除 commit，下一轮再重试
REMOTE = "origin"


# ============================================================
# Helpers
# ============================================================

def run_git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    cmd = ["git", *args]

    result = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if check and result.returncode != 0:
        raise RuntimeError(
            f"git command failed:\n"
            f"{' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    return result


def repo_is_git() -> bool:
    result = run_git("rev-parse", "--is-inside-work-tree", check=False)
    return result.returncode == 0 and result.stdout.strip() == "true"


def current_branch() -> str:
    return run_git(
        "rev-parse",
        "--abbrev-ref",
        "HEAD",
    ).stdout.strip()


def is_allowed_file(path: Path) -> bool:
    if not path.is_file():
        return False

    return path.suffix.lower() in ALLOWED_SUFFIXES


def iter_watched_files():
    seen = set()

    for target in WATCH_TARGETS:
        if target.is_file():
            if is_allowed_file(target):
                p = target.resolve()
                if p not in seen:
                    seen.add(p)
                    yield p

        elif target.is_dir():
            for p in target.rglob("*"):
                if not is_allowed_file(p):
                    continue

                # 忽略隐藏目录
                if any(part.startswith(".") for part in p.relative_to(ROOT).parts):
                    continue

                p = p.resolve()

                if p not in seen:
                    seen.add(p)
                    yield p


def file_signature(path: Path):
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None

    return (
        stat.st_mtime_ns,
        stat.st_size,
    )


def snapshot():
    state = {}

    for p in iter_watched_files():
        state[str(p)] = file_signature(p)

    return state


def changed_paths(old, new):
    paths = set(old) | set(new)

    changed = []

    for path in paths:
        if old.get(path) != new.get(path):
            changed.append(Path(path))

    return changed


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def git_has_changes() -> bool:
    result = run_git("status", "--porcelain")

    return bool(result.stdout.strip())


def staged_has_changes() -> bool:
    result = run_git(
        "diff",
        "--cached",
        "--quiet",
        check=False,
    )

    return result.returncode == 1


def stage_watch_targets():
    """
    只 add 白名单中的文件，不执行 git add .
    """
    current_files = list(iter_watched_files())

    if current_files:
        run_git(
            "add",
            "--",
            *[relative(p) for p in current_files],
        )

    # 已删除的受监控文件也要进入 staged
    #
    # git add -u 仅对这些目录执行
    for target in WATCH_TARGETS:
        try:
            rel = target.resolve().relative_to(ROOT.resolve()).as_posix()
        except ValueError:
            continue

        run_git(
            "add",
            "-u",
            "--",
            rel,
            check=False,
        )


def commit_and_push(changed):
    stage_watch_targets()

    if not staged_has_changes():
        print("[watch] no staged changes")
        return

    names = [relative(p) for p in changed if p.exists()]

    if len(names) <= 3:
        summary = ", ".join(names)
    else:
        summary = f"{len(names)} watched files"

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    message = f"auto: sync Step3 changes ({ts})"

    print()
    print("=" * 70)
    print(f"[watch] committing: {summary}")
    print(f"[watch] message: {message}")

    run_git(
        "commit",
        "-m",
        message,
    )

    commit_sha = run_git(
        "rev-parse",
        "--short",
        "HEAD",
    ).stdout.strip()

    print(f"[watch] committed {commit_sha}")

    if AUTO_PUSH:
        branch = current_branch()

        print(f"[watch] pushing {REMOTE}/{branch} ...")

        result = run_git(
            "push",
            REMOTE,
            branch,
            check=False,
        )

        if result.returncode == 0:
            print(f"[watch] push OK: {REMOTE}/{branch}")
        else:
            print("[watch] PUSH FAILED")
            print(result.stderr)
            print(
                "[watch] commit remains locally; "
                "the watcher will not delete it."
            )

    print("=" * 70)
    print()


def main():
    print("=" * 70)
    print("Step 3 Git Auto Watch")
    print("=" * 70)
    print(f"repo:   {ROOT}")
    print(f"branch: {current_branch() if repo_is_git() else 'N/A'}")
    print(f"poll:   {POLL_SECONDS}s")
    print(f"debounce: {DEBOUNCE_SECONDS}s")
    print(f"auto push: {AUTO_PUSH}")
    print()

    if not repo_is_git():
        raise RuntimeError(
            f"{ROOT} is not a Git repository"
        )

    old_state = snapshot()

    dirty_since = None
    accumulated_changes = set()

    print("[watch] monitoring... Ctrl+C to stop.")

    try:
        while True:
            time.sleep(POLL_SECONDS)

            new_state = snapshot()
            diff = changed_paths(old_state, new_state)

            if diff:
                dirty_since = time.monotonic()

                for p in diff:
                    accumulated_changes.add(p)

                print(
                    "[watch] changed:",
                    ", ".join(
                        relative(p)
                        if p.exists()
                        else f"{p.name} (deleted)"
                        for p in diff
                    )
                )

                old_state = new_state
                continue

            if (
                dirty_since is not None
                and time.monotonic() - dirty_since
                >= DEBOUNCE_SECONDS
            ):
                try:
                    commit_and_push(
                        sorted(
                            accumulated_changes,
                            key=lambda p: str(p),
                        )
                    )
                except Exception as exc:
                    print(f"[watch] ERROR: {exc}")

                dirty_since = None
                accumulated_changes.clear()

    except KeyboardInterrupt:
        print("\n[watch] stopped.")


if __name__ == "__main__":
    main()
