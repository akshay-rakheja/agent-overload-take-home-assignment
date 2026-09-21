"""Byte fingerprints, immutable snapshots, and safe atomic restoration."""

from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import json
import os
import secrets
import shutil
import stat
import sys
import threading
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from uuid import NAMESPACE_URL, uuid5

from server.services.execution.log_store import execution_log_slug
from server.services.execution.models import AgentRecord, normalize_agent_text

from .contracts import EquivalenceReport, FixtureManifest, StateFingerprint, StateSnapshot


_THREAD_LOCKS_GUARD = threading.Lock()
_THREAD_LOCKS: dict[Path, threading.RLock] = {}
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


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


def _legacy_identity_and_ownership(names: list[str]) -> list[tuple[str, str | None]]:
    normalized_occurrences: dict[str, int] = {}
    slug_counts = Counter(execution_log_slug(name) for name in names)
    values: list[tuple[str, str | None]] = []
    for name in names:
        normalized = normalize_agent_text(name)
        occurrence = normalized_occurrences.get(normalized, 0)
        normalized_occurrences[normalized] = occurrence + 1
        stable_id = uuid5(NAMESPACE_URL, f"openpoke-legacy:{normalized}:{occurrence}")
        slug = execution_log_slug(name)
        owner = slug if slug_counts[slug] == 1 else None
        values.append((str(stable_id), owner))
    return values


def _logical_identity_digest(
    roster_payload: object,
    manifest: FixtureManifest,
) -> str:
    canonical = list(manifest.agents)
    projection: list[dict[str, object]] = []
    if isinstance(roster_payload, list):
        names = [str(raw).strip() or "agent" for raw in roster_payload]
        identities = _legacy_identity_and_ownership(names)
        for index, (name, (agent_id, owner)) in enumerate(zip(names, identities)):
            logical = canonical[index] if index < len(canonical) else None
            projection.append(
                {
                    "logical_id": logical.logical_id if logical else f"unmapped:{index}",
                    "agent_id": agent_id,
                    "name": name,
                    "normalized_name": normalize_agent_text(name),
                    "purpose": logical.purpose if logical else "",
                    "aliases": sorted(logical.aliases) if logical else [],
                    "status": logical.status if logical else "unmapped",
                    "memory_summary": logical.memory_summary if logical else "",
                    "use_count": 0,
                    "schema_version": 1,
                    "legacy_storage_slug": owner,
                    "legacy_quarantined": owner is None,
                }
            )
    elif isinstance(roster_payload, dict) and isinstance(roster_payload.get("agents"), list):
        records = [AgentRecord.model_validate(item) for item in roster_payload["agents"]]
        for index, record in enumerate(records):
            logical = canonical[index] if index < len(canonical) else None
            owner = (
                execution_log_slug(record.legacy_storage_key)
                if record.legacy_storage_key is not None
                else None
            )
            projection.append(
                {
                    "logical_id": logical.logical_id if logical else f"unmapped:{index}",
                    "agent_id": str(record.agent_id),
                    "name": record.name,
                    "normalized_name": record.normalized_name,
                    "purpose": record.purpose,
                    "aliases": sorted(record.aliases),
                    "status": record.status.value,
                    "memory_summary": record.memory_summary,
                    "use_count": record.use_count,
                    "schema_version": record.schema_version,
                    "legacy_storage_slug": owner,
                    "legacy_quarantined": owner is None,
                }
            )
    else:
        raise ValueError("roster.json is neither a historical list nor an enhanced directory")

    rendered = json.dumps(
        sorted(projection, key=lambda item: str(item["logical_id"])),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return _sha256(rendered.encode("utf-8"))


def _sentinel_checks(
    manifest: FixtureManifest,
    journals: dict[str, bytes],
) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for sentinel in manifest.sentinels:
        marker = sentinel.marker.encode("utf-8")
        owner_payload = journals.get(sentinel.expected_relative_path, b"")
        absent_elsewhere = all(
            marker not in payload
            for relative, payload in journals.items()
            if relative != sentinel.expected_relative_path
        )
        checks[sentinel.sentinel_id] = marker in owner_payload and absent_elsewhere
    return checks


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
    manifest = FixtureManifest.model_validate_json(
        (root / "live_lab" / "fixture_manifest.json").read_bytes()
    )
    if isinstance(roster_payload, list):
        roster_count = len(roster_payload)
    elif isinstance(roster_payload, dict) and isinstance(roster_payload.get("agents"), list):
        roster_count = len(roster_payload["agents"])
    else:
        raise ValueError("roster.json is neither a historical list nor an enhanced directory")

    aggregate = hashlib.sha256()
    journal_bytes = 0
    journal_entries = 0
    journal_payloads: dict[str, bytes] = {}
    for relative, payload in journals:
        encoded_path = relative.encode("utf-8")
        aggregate.update(len(encoded_path).to_bytes(8, "big"))
        aggregate.update(encoded_path)
        aggregate.update(len(payload).to_bytes(8, "big"))
        aggregate.update(payload)
        journal_bytes += len(payload)
        journal_entries += _count_entries(payload)
        journal_payloads[relative] = payload

    return StateFingerprint(
        file_sha256=file_sha256,
        roster_sha256=_sha256(roster_bytes),
        roster_count=roster_count,
        logical_identity_digest=_logical_identity_digest(roster_payload, manifest),
        raw_journal_digest=aggregate.hexdigest(),
        journal_bytes=journal_bytes,
        journal_entries=journal_entries,
        sentinel_checks=_sentinel_checks(manifest, journal_payloads),
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
        "sentinel_checks": (
            baseline.sentinel_checks == enhanced.sentinel_checks
            and all(baseline.sentinel_checks.values())
            and all(enhanced.sentinel_checks.values())
        ),
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


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _reject_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise ValueError(f"symlink component is not allowed in restore path: {current}")


def _validated_target(data_dir: Path, allowed_root: Path) -> tuple[Path, Path]:
    allowed = _absolute_path(allowed_root)
    target = _absolute_path(data_dir)
    _reject_symlink_components(allowed)
    _reject_symlink_components(target)
    allowed_resolved = allowed.resolve(strict=True)
    target_resolved = target.resolve(strict=False)
    unsafe = {Path("/"), Path.home().resolve(), _REPOSITORY_ROOT}
    if target_resolved in unsafe:
        raise ValueError(f"unsafe restore target: {target}")
    if target == allowed or allowed not in target.parents:
        raise ValueError(f"restore target is outside allowed_root: {target}")
    if target_resolved == allowed_resolved or allowed_resolved not in target_resolved.parents:
        raise ValueError(f"restore target is outside allowed_root: {target}")
    return target, allowed


def _thread_lock(path: Path) -> threading.RLock:
    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(path, threading.RLock())


@contextmanager
def runtime_lock(data_dir: Path, *, allowed_root: Path) -> Iterator[None]:
    """Serialize a run/reset pair for one runtime tree."""

    target, allowed = _validated_target(data_dir, allowed_root)
    relative = target.relative_to(allowed).as_posix().encode("utf-8")
    lock_name = f"{_sha256(relative)}.lock"
    lock_path = allowed / ".live-lab-locks" / lock_name
    directory_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    with _thread_lock(lock_path):
        allowed_before = allowed.stat(follow_symlinks=False)
        allowed_fd = os.open(allowed, directory_flags)
        lock_dir_fd: int | None = None
        try:
            if not os.path.samestat(allowed_before, os.fstat(allowed_fd)):
                raise ValueError("allowed_root changed while acquiring runtime lock")
            try:
                os.mkdir(".live-lab-locks", mode=0o700, dir_fd=allowed_fd)
            except FileExistsError:
                pass
            lock_dir_fd = os.open(".live-lab-locks", directory_flags, dir_fd=allowed_fd)
            lock_flags = os.O_CREAT | os.O_RDWR
            if hasattr(os, "O_NOFOLLOW"):
                lock_flags |= os.O_NOFOLLOW
            lock_fd = os.open(lock_name, lock_flags, 0o600, dir_fd=lock_dir_fd)
            handle = os.fdopen(lock_fd, "a+b")
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                confirmed_target, confirmed_allowed = _validated_target(target, allowed)
                if confirmed_target != target or confirmed_allowed != allowed:
                    raise ValueError("restore target changed while acquiring lock")
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                handle.close()
        finally:
            if lock_dir_fd is not None:
                os.close(lock_dir_fd)
            os.close(allowed_fd)


def _directory_open_flags() -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def _copy_snapshot_directory_at(source: Path, destination_fd: int) -> None:
    for path in sorted(source.iterdir(), key=lambda item: item.name):
        name = path.name
        source_stat = path.lstat()
        mode = source_stat.st_mode
        if stat.S_ISLNK(mode):
            raise ValueError(f"snapshot may not contain symlinks: {path}")
        if stat.S_ISDIR(mode):
            os.mkdir(name, mode=0o755, dir_fd=destination_fd)
            child_fd = os.open(name, _directory_open_flags(), dir_fd=destination_fd)
            try:
                _copy_snapshot_directory_at(path, child_fd)
                os.fsync(child_fd)
            finally:
                os.close(child_fd)
        elif stat.S_ISREG(mode):
            source_flags = os.O_RDONLY
            if hasattr(os, "O_NOFOLLOW"):
                source_flags |= os.O_NOFOLLOW
            source_fd = os.open(path, source_flags)
            if not os.path.samestat(source_stat, os.fstat(source_fd)):
                os.close(source_fd)
                raise ValueError(f"snapshot entry changed while copying: {path}")
            destination_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                destination_flags |= os.O_NOFOLLOW
            try:
                destination_file_fd = os.open(
                    name,
                    destination_flags,
                    0o600,
                    dir_fd=destination_fd,
                )
            except BaseException:
                os.close(source_fd)
                raise
            with os.fdopen(source_fd, "rb") as reader, os.fdopen(
                destination_file_fd,
                "wb",
            ) as writer:
                shutil.copyfileobj(reader, writer)
                writer.flush()
                os.fchmod(writer.fileno(), 0o644)
                os.fsync(writer.fileno())
        else:
            raise ValueError(f"snapshot contains unsupported entry: {path}")


def _copy_snapshot_bytes(source: Path, parent_fd: int, destination_name: str) -> None:
    destination_fd = os.open(
        destination_name,
        _directory_open_flags(),
        dir_fd=parent_fd,
    )
    try:
        _copy_snapshot_directory_at(source, destination_fd)
        os.fsync(destination_fd)
    finally:
        os.close(destination_fd)


def _create_temporary_directory_at(parent_fd: int, target_name: str) -> tuple[str, os.stat_result]:
    prefix = f".{target_name[:48]}.restore-"
    for _ in range(100):
        name = f"{prefix}{secrets.token_hex(8)}"
        try:
            os.mkdir(name, mode=0o755, dir_fd=parent_fd)
        except FileExistsError:
            continue
        return name, os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    raise FileExistsError("could not allocate a unique restore staging directory")


def _stat_at(parent_fd: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _open_pinned_target_at(parent_fd: int, name: str) -> tuple[int | None, os.stat_result | None]:
    try:
        descriptor = os.open(name, _directory_open_flags(), dir_fd=parent_fd)
    except FileNotFoundError:
        return None, None
    except OSError as error:
        raise ValueError(f"restore target is not a safe directory: {name}") from error
    return descriptor, os.fstat(descriptor)


def _atomic_exchange_at(parent_fd: int, left_name: str, right_name: str) -> None:
    """Atomically swap two existing paths on supported production platforms."""

    libc = ctypes.CDLL(None, use_errno=True)
    encoded_left = os.fsencode(left_name)
    encoded_right = os.fsencode(right_name)
    if sys.platform == "darwin":
        rename = libc.renameatx_np
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        result = rename(parent_fd, encoded_left, parent_fd, encoded_right, 0x00000002)
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
        result = rename(parent_fd, encoded_left, parent_fd, encoded_right, 0x00000002)
    else:
        raise RuntimeError("atomic directory exchange is unsupported on this platform")
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), right_name)


def _atomic_install_at(parent_fd: int, source_name: str, target_name: str) -> None:
    """Atomically install a new target without replacing a concurrently created object."""

    libc = ctypes.CDLL(None, use_errno=True)
    encoded_source = os.fsencode(source_name)
    encoded_target = os.fsencode(target_name)
    if sys.platform == "darwin":
        rename = libc.renameatx_np
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        result = rename(parent_fd, encoded_source, parent_fd, encoded_target, 0x00000004)
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
        result = rename(parent_fd, encoded_source, parent_fd, encoded_target, 0x00000001)
    else:
        raise RuntimeError("atomic no-replace directory install is unsupported on this platform")
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), target_name)


def _write_cleanup_pending_metadata_at(
    parent_fd: int,
    pending_name: str,
    *,
    role: str,
    target_name: str,
    expected: os.stat_result,
) -> None:
    metadata = json.dumps(
        {
            "device": expected.st_dev,
            "inode": expected.st_ino,
            "role": role,
            "schema_version": 1,
            "state": "cleanup_pending",
            "target_name": target_name,
            "tree_name": pending_name,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(f"{pending_name}.json", flags, 0o600, dir_fd=parent_fd)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(metadata)
        handle.flush()
        os.fsync(handle.fileno())


def _quarantine_cleanup_if_same_at(
    parent_fd: int,
    name: str,
    expected: os.stat_result,
    *,
    role: str,
    target_name: str,
) -> bool:
    """Move proven cleanup state aside without ever recursively deleting by name."""

    current = _stat_at(parent_fd, name)
    if current is None or not os.path.samestat(current, expected):
        return False
    prefix = f".{target_name[:48]}.cleanup-pending-{role}-"
    for _ in range(100):
        pending_name = f"{prefix}{secrets.token_hex(8)}"
        try:
            _atomic_install_at(parent_fd, name, pending_name)
        except OSError as error:
            if error.errno == errno.EEXIST:
                continue
            return False
        except RuntimeError:
            return False

        moved = _stat_at(parent_fd, pending_name)
        original_name = _stat_at(parent_fd, name)
        if (
            moved is not None
            and os.path.samestat(moved, expected)
            and original_name is None
        ):
            try:
                _write_cleanup_pending_metadata_at(
                    parent_fd,
                    pending_name,
                    role=role,
                    target_name=target_name,
                    expected=expected,
                )
            except OSError:
                return False
            finally:
                os.fsync(parent_fd)
            return True

        if moved is not None and original_name is None:
            try:
                _atomic_install_at(parent_fd, pending_name, name)
            except (OSError, RuntimeError):
                return False
            rolled_back = _stat_at(parent_fd, name)
            pending_after_rollback = _stat_at(parent_fd, pending_name)
            if (
                rolled_back is None
                or not os.path.samestat(rolled_back, moved)
                or pending_after_rollback is not None
            ):
                return False
        return False
    return False


def restore_snapshot(
    snapshot: StateSnapshot,
    data_dir: Path,
    *,
    allowed_root: Path,
) -> StateFingerprint:
    """Restore an immutable snapshot through a sibling temporary directory."""

    target, allowed = _validated_target(data_dir, allowed_root)
    snapshot_root = Path(snapshot.snapshot_dir).resolve(strict=True)
    if fingerprint_state(snapshot_root) != snapshot.fingerprint:
        raise ValueError("snapshot fingerprint does not match immutable snapshot metadata")

    with runtime_lock(target, allowed_root=allowed):
        target, allowed = _validated_target(target, allowed)
        target.parent.mkdir(parents=True, exist_ok=True)
        target, allowed = _validated_target(target, allowed)
        parent_before = target.parent.stat(follow_symlinks=False)
        parent_fd = os.open(target.parent, _directory_open_flags())
        if not os.path.samestat(parent_before, os.fstat(parent_fd)):
            os.close(parent_fd)
            raise ValueError("restore target parent changed before temporary preparation")
        target_fd: int | None = None
        temporary_name: str | None = None
        cleanup_identity: os.stat_result | None = None
        cleanup_role = "staging"
        try:
            target_fd, pinned_target = _open_pinned_target_at(parent_fd, target.name)
            temporary_name, staged_identity = _create_temporary_directory_at(
                parent_fd,
                target.name,
            )
            cleanup_identity = staged_identity
            _copy_snapshot_bytes(snapshot_root, parent_fd, temporary_name)
            confirmed_target, _ = _validated_target(target, allowed)
            parent_after = confirmed_target.parent.stat(follow_symlinks=False)
            if confirmed_target != target or not os.path.samestat(parent_before, parent_after):
                raise ValueError("restore target path changed during snapshot preparation")
            if not os.path.samestat(parent_before, os.fstat(parent_fd)):
                raise ValueError("restore target parent changed during snapshot preparation")

            current_target = _stat_at(parent_fd, target.name)
            current_staged = _stat_at(parent_fd, temporary_name)
            if current_staged is None or not os.path.samestat(current_staged, staged_identity):
                raise RuntimeError("restore staging directory changed before atomic exchange")
            if pinned_target is not None:
                if current_target is None or not os.path.samestat(current_target, pinned_target):
                    raise RuntimeError("restore target changed before atomic exchange")
                _atomic_exchange_at(parent_fd, temporary_name, target.name)
                installed_target = _stat_at(parent_fd, target.name)
                exchanged_target = _stat_at(parent_fd, temporary_name)
                valid_exchange = (
                    installed_target is not None
                    and os.path.samestat(installed_target, staged_identity)
                    and exchanged_target is not None
                    and os.path.samestat(exchanged_target, pinned_target)
                )
                if not valid_exchange:
                    if (
                        installed_target is not None
                        and os.path.samestat(installed_target, staged_identity)
                        and exchanged_target is not None
                    ):
                        unexpected_target = exchanged_target
                        _atomic_exchange_at(parent_fd, temporary_name, target.name)
                        rolled_back_target = _stat_at(parent_fd, target.name)
                        rolled_back_staging = _stat_at(parent_fd, temporary_name)
                        if (
                            rolled_back_target is None
                            or not os.path.samestat(rolled_back_target, unexpected_target)
                            or rolled_back_staging is None
                            or not os.path.samestat(rolled_back_staging, staged_identity)
                        ):
                            cleanup_identity = None
                            raise RuntimeError(
                                "restore target changed during atomic exchange and rollback failed"
                            )
                    else:
                        cleanup_identity = None
                    raise RuntimeError("restore target changed during atomic exchange")
                cleanup_identity = pinned_target
                cleanup_role = "old-target"
            else:
                if current_target is not None:
                    raise RuntimeError("restore target appeared before atomic install")
                _atomic_install_at(parent_fd, temporary_name, target.name)
                installed_target = _stat_at(parent_fd, target.name)
                if installed_target is None or not os.path.samestat(
                    installed_target,
                    staged_identity,
                ):
                    cleanup_identity = None
                    raise RuntimeError("restore target changed during atomic install")
                temporary_name = None
                cleanup_identity = None
            os.fsync(parent_fd)
            if temporary_name is not None and cleanup_identity is not None:
                _quarantine_cleanup_if_same_at(
                    parent_fd,
                    temporary_name,
                    cleanup_identity,
                    role=cleanup_role,
                    target_name=target.name,
                )
                temporary_name = None
                cleanup_identity = None
            os.fsync(parent_fd)
        finally:
            if temporary_name is not None and cleanup_identity is not None:
                _quarantine_cleanup_if_same_at(
                    parent_fd,
                    temporary_name,
                    cleanup_identity,
                    role=cleanup_role,
                    target_name=target.name,
                )
            if target_fd is not None:
                os.close(target_fd)
            os.close(parent_fd)

    target, _ = _validated_target(target, allowed)
    restored = fingerprint_state(target)
    if restored != snapshot.fingerprint:
        raise RuntimeError("restored state does not match snapshot fingerprint")
    return restored
