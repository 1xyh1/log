"""Fail-closed local I/O helpers for T1-GR E3 formal tooling.

No network access. Designed for Windows/Python 3.11 and POSIX test runners.
The module prevents accidental repo leakage of private IDs, provides atomic/idempotent
writes, cooperative concurrency locking, bounded JSON/ZIP reads, and wall-clock deadlines.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import stat
import time
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

class GateError(RuntimeError):
    def __init__(self, code: str, safe_detail: str = ""):
        self.code = code
        self.safe_detail = safe_detail
        super().__init__(code + (f":{safe_detail}" if safe_detail else ""))


def fail(code: str, safe_detail: str = "") -> None:
    raise GateError(code, safe_detail)


def sha256_file(path: Path, deadline: "Deadline|None" = None) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while True:
                if deadline: deadline.check("HASH_TIMEOUT")
                b = f.read(1 << 20)
                if not b: break
                h.update(b)
    except PermissionError:
        fail("READ_PERMISSION_DENIED")
    except OSError:
        fail("READ_IO_ERROR")
    return h.hexdigest()


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(obj)).hexdigest()


def require_nonempty_string(x: Any, code: str) -> str:
    if not isinstance(x, str) or not x.strip() or any(ord(c) < 32 for c in x):
        fail(code)
    if len(x) > 512:
        fail(code)
    return x


def require_dict(obj: Any, code: str) -> dict:
    if not isinstance(obj, dict): fail(code)
    return obj


def require_list(obj: Any, code: str) -> list:
    if not isinstance(obj, list): fail(code)
    return obj


def require_keys(d: dict, keys: tuple[str, ...], code: str) -> None:
    missing = [k for k in keys if k not in d or d[k] is None]
    if missing: fail(code, f"missing_count={len(missing)}")


def read_json_bounded(path: Path, max_bytes: int, expected_schema: str|None = None) -> dict:
    try:
        st = path.stat()
    except FileNotFoundError:
        fail("INPUT_NOT_FOUND")
    except PermissionError:
        fail("READ_PERMISSION_DENIED")
    except OSError:
        fail("INPUT_STAT_ERROR")
    if not path.is_file(): fail("INPUT_NOT_FILE")
    if st.st_size <= 0 or st.st_size > int(max_bytes): fail("INPUT_SIZE_OUT_OF_RANGE")
    try:
        raw = path.read_bytes()
        obj = json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError:
        fail("JSON_UTF8_ERROR")
    except json.JSONDecodeError:
        fail("JSON_PARSE_ERROR")
    except PermissionError:
        fail("READ_PERMISSION_DENIED")
    except OSError:
        fail("READ_IO_ERROR")
    if not isinstance(obj, dict): fail("JSON_TOPLEVEL_NOT_OBJECT")
    if expected_schema is not None and obj.get("schema") != expected_schema:
        fail("BAD_SCHEMA")
    return obj



def stat_token(path: Path) -> tuple[int, int]:
    try:
        st = path.stat()
    except FileNotFoundError:
        fail("INPUT_NOT_FOUND")
    except PermissionError:
        fail("READ_PERMISSION_DENIED")
    except OSError:
        fail("INPUT_STAT_ERROR")
    return int(st.st_size), int(st.st_mtime_ns)


def require_unchanged(path: Path, before: tuple[int, int], code: str = "INPUT_CHANGED_DURING_RUN") -> None:
    if stat_token(path) != before:
        fail(code)

def resolved_repo(path: Path) -> Path:
    try: return path.resolve(strict=True)
    except OSError: fail("REPO_RESOLVE_ERROR")


def is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve(strict=False).relative_to(parent.resolve(strict=False)); return True
    except ValueError:
        return False


def ensure_private_output(path: Path, repo: Path, must_parent_exist: bool = True) -> Path:
    p = path.expanduser().resolve(strict=False)
    r = repo.resolve(strict=True)
    if is_within(p, r): fail("PRIVATE_OUTPUT_INSIDE_REPO")
    parent = p.parent
    if must_parent_exist and not parent.is_dir(): fail("PRIVATE_PARENT_NOT_FOUND")
    if not must_parent_exist:
        try: parent.mkdir(parents=True, exist_ok=True)
        except OSError: fail("PRIVATE_PARENT_CREATE_FAILED")
    if not os.access(parent, os.W_OK): fail("PRIVATE_PARENT_NOT_WRITABLE")
    if p.exists() and p.is_dir(): fail("OUTPUT_IS_DIRECTORY")
    return p


def ensure_public_output(repo: Path, rel: str, allowed_prefix: str) -> Path:
    rel = require_nonempty_string(rel, "PUBLIC_OUTPUT_EMPTY")
    rp = PurePosixPath(rel.replace("\\", "/"))
    if rp.is_absolute() or ".." in rp.parts: fail("PUBLIC_OUTPUT_PATH_ESCAPE")
    prefix = PurePosixPath(allowed_prefix)
    try: rp.relative_to(prefix)
    except ValueError: fail("PUBLIC_OUTPUT_PREFIX_VIOLATION")
    target = (repo / Path(*rp.parts)).resolve(strict=False)
    if not is_within(target, repo): fail("PUBLIC_OUTPUT_PATH_ESCAPE")
    try: target.parent.mkdir(parents=True, exist_ok=True)
    except OSError: fail("PUBLIC_PARENT_CREATE_FAILED")
    if not os.access(target.parent, os.W_OK): fail("PUBLIC_PARENT_NOT_WRITABLE")
    if target.exists() and target.is_dir(): fail("OUTPUT_IS_DIRECTORY")
    return target


def ensure_private_input(path: Path, repo: Path) -> Path:
    p = path.expanduser().resolve(strict=False)
    r = repo.resolve(strict=True)
    if is_within(p, r): fail("PRIVATE_INPUT_INSIDE_REPO")
    try:
        if not p.is_file(): fail("PRIVATE_INPUT_NOT_FILE")
    except OSError:
        fail("PRIVATE_INPUT_STAT_ERROR")
    if not os.access(p, os.R_OK): fail("PRIVATE_INPUT_NOT_READABLE")
    return p


def ensure_repo_input(repo: Path, rel: str, allowed_prefix: str) -> Path:
    rel = require_nonempty_string(rel, "REPO_INPUT_EMPTY")
    rp = PurePosixPath(rel.replace("\\", "/"))
    if rp.is_absolute() or ".." in rp.parts: fail("REPO_INPUT_PATH_ESCAPE")
    prefix = PurePosixPath(allowed_prefix)
    try: rp.relative_to(prefix)
    except ValueError: fail("REPO_INPUT_PREFIX_VIOLATION")
    target = (repo / Path(*rp.parts)).resolve(strict=False)
    if not is_within(target, repo): fail("REPO_INPUT_PATH_ESCAPE")
    if not target.is_file(): fail("REPO_INPUT_NOT_FILE")
    if not os.access(target, os.R_OK): fail("REPO_INPUT_NOT_READABLE")
    return target


SENSITIVE_KEYS = {
    "ids", "all_components", "components", "edges", "internal_strong_edges", "internal_review_edges",
    "historical_matches", "sample_id", "formal_id", "historical_id", "force_train_ids",
    "force_train_seed_ids", "force_train_components", "force_train_ids_after_component_propagation",
    "path", "root", "private_path", "private_out", "zip_path", "historical_raw_root",
}
PATH_RE = re.compile(r"(?:^[A-Za-z]:[\\/])|(?:^/)|(?:\\\\)|(?:[\\/].+[\\/])")


def assert_public_safe(obj: Any) -> None:
    def walk(x: Any, key: str = ""):
        if isinstance(x, dict):
            for k, v in x.items():
                lk = str(k).lower()
                if lk in SENSITIVE_KEYS: fail("PUBLIC_SENSITIVE_KEY", lk)
                if lk.endswith("_ids") and "commitment" not in lk and "count" not in lk:
                    fail("PUBLIC_SENSITIVE_KEY", lk)
                walk(v, lk)
        elif isinstance(x, list):
            if any(isinstance(v, str) for v in x): fail("PUBLIC_STRING_LIST_FORBIDDEN")
            for v in x: walk(v, key)
        elif isinstance(x, str):
            if PATH_RE.search(x): fail("PUBLIC_PATH_STRING_FORBIDDEN")
    walk(obj)


def _chmod_private(path: Path) -> str:
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        if os.name != "nt": fail("PRIVATE_CHMOD_FAILED")
        return "WINDOWS_ACL_NOT_PROVEN"
    if os.name != "nt":
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode & 0o077: fail("PRIVATE_MODE_TOO_OPEN")
        return oct(mode)
    return "WINDOWS_ACL_NOT_PROVEN"


def _with_integrity_fields(obj: dict, request_fingerprint: str) -> dict:
    base = dict(obj)
    base.pop("request_fingerprint", None)
    base.pop("payload_sha256", None)
    payload_sha = sha256_json(base)
    base["payload_sha256"] = payload_sha
    base["request_fingerprint"] = request_fingerprint
    return base


def check_existing_output(path: Path, request_fingerprint: str) -> tuple[dict, str] | None:
    if not path.exists(): return None
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        fail("EXISTING_OUTPUT_CORRUPT")
    if not isinstance(existing, dict): fail("EXISTING_OUTPUT_CORRUPT")
    if existing.get("request_fingerprint") != request_fingerprint:
        fail("OUTPUT_CONFLICT_DIFFERENT_REQUEST")
    claimed = existing.get("payload_sha256")
    if not isinstance(claimed, str) or len(claimed) != 64:
        fail("EXISTING_OUTPUT_INTEGRITY_MISSING")
    base = dict(existing); base.pop("request_fingerprint", None); base.pop("payload_sha256", None)
    if sha256_json(base) != claimed:
        fail("EXISTING_OUTPUT_INTEGRITY_FAIL")
    return existing, sha256_file(path)


def atomic_json_write(path: Path, obj: dict, *, private: bool, request_fingerprint: str) -> tuple[str, bool]:
    """Write deterministically; same request is idempotent, different request conflicts."""
    final_obj = _with_integrity_fields(obj, request_fingerprint)
    raw = json.dumps(final_obj, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    expected_sha = hashlib.sha256(raw).hexdigest()
    existing = check_existing_output(path, request_fingerprint)
    if existing is not None:
        _, current_sha = existing
        if current_sha != expected_sha:
            fail("OUTPUT_CONFLICT_SAME_REQUEST_DIFFERENT_CONTENT")
        return current_sha, True
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    mode = 0o600 if private else 0o644
    try:
        fd = os.open(tmp, flags, mode)
        try:
            with os.fdopen(fd, "wb", closefd=True) as f:
                f.write(raw); f.flush(); os.fsync(f.fileno())
            if private: _chmod_private(tmp)
            os.replace(tmp, path)
            if os.name != "nt":
                try:
                    dfd = os.open(str(path.parent), os.O_RDONLY)
                    try: os.fsync(dfd)
                    finally: os.close(dfd)
                except OSError: pass
        except Exception:
            try: tmp.unlink(missing_ok=True)
            except OSError: pass
            raise
    except FileExistsError:
        fail("TEMPFILE_COLLISION")
    except PermissionError:
        fail("WRITE_PERMISSION_DENIED")
    except OSError:
        fail("WRITE_IO_ERROR")
    return sha256_file(path), False


@dataclass
class Deadline:
    seconds: float
    start: float = 0.0
    def __post_init__(self):
        if not isinstance(self.seconds, (int, float)) or not math_isfinite_positive(float(self.seconds)):
            fail("BAD_TIMEOUT")
        self.start = time.monotonic()
    def check(self, code: str = "OPERATION_TIMEOUT"):
        if time.monotonic() - self.start > float(self.seconds): fail(code)
    @property
    def remaining(self) -> float:
        return max(0.0, float(self.seconds) - (time.monotonic() - self.start))


def math_isfinite_positive(x: float) -> bool:
    return x > 0 and x < float("inf")


def _pid_alive(pid: int) -> bool:
    if pid <= 0: return False
    try:
        os.kill(pid, 0); return True
    except ProcessLookupError: return False
    except PermissionError: return True
    except OSError: return False


@contextlib.contextmanager
def file_lock(lock_path: Path, wait_seconds: float, stale_seconds: float):
    deadline = time.monotonic() + max(0.0, float(wait_seconds))
    payload = json.dumps({"pid": os.getpid(), "created_unix": time.time()}).encode()
    while True:
        try:
            fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(payload); f.flush(); os.fsync(f.fileno())
            break
        except FileExistsError:
            stale = False
            try:
                age = time.time() - lock_path.stat().st_mtime
                info = json.loads(lock_path.read_text(encoding="utf-8"))
                stale = age > float(stale_seconds) and not _pid_alive(int(info.get("pid", -1)))
            except Exception:
                # Unknown/corrupt lock is not auto-deleted until well past stale threshold.
                try: stale = time.time() - lock_path.stat().st_mtime > float(stale_seconds) * 2
                except OSError: stale = False
            if stale:
                try: lock_path.unlink(); continue
                except OSError: pass
            if time.monotonic() >= deadline: fail("CONCURRENT_RUN_LOCKED")
            time.sleep(0.1)
        except PermissionError:
            fail("LOCK_PERMISSION_DENIED")
        except OSError:
            fail("LOCK_IO_ERROR")
    try:
        yield
    finally:
        try: lock_path.unlink(missing_ok=True)
        except OSError: pass


def safe_error_message(exc: Exception) -> str:
    if isinstance(exc, GateError):
        return exc.code + (f":{exc.safe_detail}" if exc.safe_detail else "")
    return "UNHANDLED_INTERNAL_ERROR"


def validate_zip_name(name: str) -> None:
    if not isinstance(name, str) or not name or "\x00" in name: fail("ZIP_MEMBER_NAME_INVALID")
    pp = PurePosixPath(name.replace("\\", "/"))
    if pp.is_absolute() or ".." in pp.parts: fail("ZIP_MEMBER_PATH_UNSAFE")


def validate_identifier(x: Any) -> str:
    return require_nonempty_string(x, "INVALID_SAMPLE_IDENTIFIER")
