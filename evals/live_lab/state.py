"""Byte fingerprints, immutable snapshots, and safe atomic restoration."""

from __future__ import annotations

import fcntl
import ctypes
import hashlib
import json
import os
import shutil
import stat
import tempfile
import threading
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .contracts import EquivalenceReport, FixtureManifest, StateFingerprint, StateSnapshot


_THREAD_LOCKS_GUARD = threading.Lock()
_THREAD_LOCKS: dict[Path, threading.RLock] = {}
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_SENTINELS = {
    "depth_start": b"DEPTH-START-SENTINEL",
    "depth_end": b"DEPTH-END-SENTINEL",
    "cross_agent": b"CROSS-AGENT-SENTINEL",
    "slug_space": b"SPACE-COLLISION-SENTINEL",
    "slug_hyphen": b"HYPHEN-COLLISION-SENTINEL",
}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _regular_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise ValueError(f"state root is not a directory: {root}")
    paths: list[Path] = []
    for path in sorted(root.rglob("*")):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ValueError(f"state trees may not contain symlinks: {path}")
        if stat.S_ISREG(mode):
            paths.append(path)
        elif not stat.S_ISDIR(mode):
            raise ValueError(f"state trees may contain only directories and files: {path}")
    return paths


def _count_entries(payload: bytes) -> int:
    if not payload:
        return 0
    return payload.count(b"\n") + int(not payload.endswith(b"\n"))


def _logical_identity_digest(data_dir: Path) -> str:
    manifest_path = data_dir / "live_lab" / "fixture_manifest.json"
    manifest = FixtureManifest.model_validate_json(manifest_path.read_bytes())
    logical = sorted(
        (
            agent.logical_id,
            agent.name,
            agent.purpose,
            tuple(agent.aliases),
            agent.status,
        )
        for agent in manifest.agents
    )
    rendered = json.dumps(logical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256(rendered.encode("utf-8"))


def fingerprint_state(data_dir: Path) -> StateFingerprint:
    """Hash a state tree without decoding or rewriting journal bytes."""

    root = data_dir.resolve(strict=True)
    paths = _regular_files(root)
    file_sha256: dict[str, str] = {}
    journals: list[tuple[str, bytes]] = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        payload = path.read_bytes()
        file_sha256[relative] = _sha256(payload)
        if path.parent.name == "execution_agents" and path.suffix == ".log":
            journals.append((relative, payload))

    roster_path = root / "execution_agents" / "roster.json"
    roster_bytes = roster_path.read_bytes()
    roster_payload = json.loads(roster_bytes)
    if isinstance(roster_payload, list):
        roster_count = len(roster_payload)
    elif isinstance(roster_payload, dict) and isinstance(roster_payload.get("agents"), list):
        roster_count = len(roster_payload["agents"])
    else:
        raise ValueError("roster.json is neither a historical list nor an enhanced directory")

    aggregate = hashlib.sha256()
    journal_bytes = 0
    journal_entries = 0
    combined_journal_bytes: list[bytes] = []
    for relative, payload in journals:
        encoded_path = relative.encode("utf-8")
        aggregate.update(len(encoded_path).to_bytes(8, "big"))
        aggregate.update(encoded_path)
        aggregate.update(len(payload).to_bytes(8, "big"))
        aggregate.update(payload)
        journal_bytes += len(payload)
        journal_entries += _count_entries(payload)
        combined_journal_bytes.append(payload)
    combined = b"".join(combined_journal_bytes)

    return StateFingerprint(
        file_sha256=file_sha256,
        roster_sha256=_sha256(roster_bytes),
        roster_count=roster_count,
        logical_identity_digest=_logical_identity_digest(root),
        raw_journal_digest=aggregate.hexdigest(),
        journal_bytes=journal_bytes,
        journal_entries=journal_entries,
        sentinel_checks={name: marker in combined for name, marker in _SENTINELS.items()},
    )


def compare_logical_state(
    baseline: StateFingerprint,
    enhanced: StateFingerprint,
) -> EquivalenceReport:
    """Compare semantic identities and raw journals, never roster JSON bytes."""

    checks = {
        "roster_count": baseline.roster_count == enhanced.roster_count,
        "logical_identity_digest": (
            baseline.logical_identity_digest == enhanced.logical_identity_digest
        ),
        "raw_journal_digest": baseline.raw_journal_digest == enhanced.raw_journal_digest,
        "journal_bytes": baseline.journal_bytes == enhanced.journal_bytes,
        "journal_entries": baseline.journal_entries == enhanced.journal_entries,
        "sentinel_checks": baseline.sentinel_checks == enhanced.sentinel_checks,
    }
    differences = tuple(name for name, matches in checks.items() if not matches)
    return EquivalenceReport(equivalent=not differences, checks=checks, differences=differences)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    for path in _regular_files(root):
        _fsync_file(path)
    for path in sorted((item for item in root.rglob("*") if item.is_dir()), reverse=True):
        _fsync_directory(path)
    _fsync_directory(root)


def _set_snapshot_read_only(root: Path) -> None:
    for path in _regular_files(root):
        path.chmod(0o444)
    for path in sorted((item for item in root.rglob("*") if item.is_dir()), reverse=True):
        path.chmod(0o555)
    root.chmod(0o555)


def create_snapshot(data_dir: Path, snapshot_dir: Path) -> StateSnapshot:
    """Create a new immutable byte-for-byte snapshot."""

    source = data_dir.resolve(strict=True)
    _regular_files(source)
    destination = snapshot_dir.resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"snapshot already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, symlinks=False)
    _fsync_tree(destination)
    fingerprint = fingerprint_state(destination)
    _set_snapshot_read_only(destination)
    _fsync_directory(destination.parent)
    return StateSnapshot(snapshot_dir=str(destination), fingerprint=fingerprint)


def _validated_target(data_dir: Path, allowed_root: Path) -> tuple[Path, Path]:
    allowed = allowed_root.resolve(strict=True)
    target = data_dir.resolve(strict=False)
    unsafe = {Path("/"), Path.home().resolve(), _REPOSITORY_ROOT}
    if target in unsafe:
        raise ValueError(f"unsafe restore target: {target}")
    if target == allowed or allowed not in target.parents:
        raise ValueError(f"restore target is outside allowed_root: {target}")
    return target, allowed


def _thread_lock(path: Path) -> threading.RLock:
    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(path, threading.RLock())


@contextmanager
def runtime_lock(data_dir: Path, *, allowed_root: Path) -> Iterator[None]:
    """Serialize a run/reset pair for one runtime tree."""

    target, _ = _validated_target(data_dir, allowed_root)
    lock_path = target.parent / f".{target.name}.live-lab.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _thread_lock(lock_path):
        with lock_path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _copy_snapshot_bytes(source: Path, destination: Path) -> None:
    destination.mkdir(mode=0o755)
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        target = destination / relative
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ValueError(f"snapshot may not contain symlinks: {path}")
        if stat.S_ISDIR(mode):
            target.mkdir(mode=0o755)
        elif stat.S_ISREG(mode):
            target.parent.mkdir(parents=True, exist_ok=True)
            with path.open("rb") as reader, target.open("wb") as writer:
                shutil.copyfileobj(reader, writer)
                writer.flush()
                os.fsync(writer.fileno())
            target.chmod(0o644)
        else:
            raise ValueError(f"snapshot contains unsupported entry: {path}")
    _fsync_tree(destination)


def _atomic_exchange(left: Path, right: Path) -> None:
    """Atomically swap two existing paths on supported production platforms."""

    libc = ctypes.CDLL(None, use_errno=True)
    encoded_left = os.fsencode(left)
    encoded_right = os.fsencode(right)
    if sys.platform == "darwin":
        rename = libc.renamex_np
        rename.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        rename.restype = ctypes.c_int
        result = rename(encoded_left, encoded_right, 0x00000002)
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        result = rename(-100, encoded_left, -100, encoded_right, 0x00000002)
    else:
        raise RuntimeError("atomic directory exchange is unsupported on this platform")
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), str(right))


def restore_snapshot(
    snapshot: StateSnapshot,
    data_dir: Path,
    *,
    allowed_root: Path,
) -> StateFingerprint:
    """Restore an immutable snapshot through a sibling temporary directory."""

    target, _ = _validated_target(data_dir, allowed_root)
    snapshot_root = Path(snapshot.snapshot_dir).resolve(strict=True)
    if fingerprint_state(snapshot_root) != snapshot.fingerprint:
        raise ValueError("snapshot fingerprint does not match immutable snapshot metadata")

    with runtime_lock(target, allowed_root=allowed_root):
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{target.name}.restore-", dir=target.parent)
        ).resolve(strict=True)
        temporary.rmdir()
        try:
            _copy_snapshot_bytes(snapshot_root, temporary)
            if target.exists():
                _atomic_exchange(temporary, target)
            else:
                os.replace(temporary, target)
            _fsync_directory(target.parent)
            if temporary.exists():
                shutil.rmtree(temporary)
                _fsync_directory(target.parent)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    restored = fingerprint_state(target)
    if restored != snapshot.fingerprint:
        raise RuntimeError("restored state does not match snapshot fingerprint")
    return restored
