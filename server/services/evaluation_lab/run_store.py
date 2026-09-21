"""Atomic descriptor-relative persistence for paired Evaluation Lab runs."""

from __future__ import annotations

import errno
import fcntl
import json
import os
import secrets
import stat
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Iterator
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from .redaction import redact_value

if TYPE_CHECKING:
    from .orchestrator import PairedRunResult


_EXECUTION_LEASE_NAME = ".execution-owner.json"


class ExecutionLease(BaseModel):
    """Persisted conservative proof that one process still owns execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    run_id: UUID
    owner_id: UUID
    pid: int
    acquired_at: datetime


class RunStore:
    """Store one canonical JSON document per run below an ignored ``.lab`` root."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        parts = self.root.parts
        if ".." in parts:
            raise ValueError("run-store root must not contain path traversal")
        indexes = [index for index, part in enumerate(parts) if part == ".lab"]
        if not indexes or indexes[-1] == len(parts) - 1:
            raise ValueError("run-store root must be below the ignored .lab directory")
        index = indexes[-1]
        self._trusted_parent = Path(*parts[:index]) if parts[:index] else Path(".")
        self._root_components = parts[index:]
        self._lock = threading.RLock()
        self._bound_identity: tuple[int, int] | None = None

    @staticmethod
    def _directory_flags() -> int:
        return os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)

    def _open_root(self, *, create: bool) -> int | None:
        flags = self._directory_flags()
        try:
            descriptor = os.open(self._trusted_parent, flags)
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise ValueError("run-store root must not contain a symlink") from exc
            raise
        try:
            for component in self._root_components:
                if create:
                    try:
                        os.mkdir(component, 0o700, dir_fd=descriptor)
                    except FileExistsError:
                        pass
                try:
                    child = os.open(component, flags, dir_fd=descriptor)
                except FileNotFoundError:
                    if not create:
                        os.close(descriptor)
                        return None
                    raise
                except OSError as exc:
                    if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                        raise ValueError("run-store root must not contain a symlink") from exc
                    raise
                os.close(descriptor)
                descriptor = child
            identity = os.fstat(descriptor)
            observed = (identity.st_dev, identity.st_ino)
            if self._bound_identity is None:
                self._bound_identity = observed
            elif self._bound_identity != observed:
                raise ValueError("run-store root identity changed")
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    @staticmethod
    def _read_file(root_fd: int, name: str) -> bytes | None:
        try:
            metadata = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        except FileNotFoundError:
            return None
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("run-store entry must be a regular file")
        try:
            descriptor = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=root_fd,
            )
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.EMLINK}:
                raise ValueError("run-store entry must not be a symlink") from exc
            raise
        try:
            chunks: list[bytes] = []
            while chunk := os.read(descriptor, 64 * 1024):
                chunks.append(chunk)
            return b"".join(chunks)
        finally:
            os.close(descriptor)

    @staticmethod
    def _atomic_write(root_fd: int, name: str, payload: bytes) -> None:
        temporary = f".{name}.{secrets.token_hex(8)}.tmp"
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=root_fd,
            )
            try:
                view = memoryview(payload)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise OSError("short run-store write")
                    view = view[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.replace(temporary, name, src_dir_fd=root_fd, dst_dir_fd=root_fd)
            os.fsync(root_fd)
        except BaseException:
            try:
                os.unlink(temporary, dir_fd=root_fd)
            except FileNotFoundError:
                pass
            raise

    @staticmethod
    def _serialize(record: "PairedRunResult") -> bytes:
        safe = redact_value(
            record.model_dump(mode="json", exclude_computed_fields=True)
        )
        return (
            json.dumps(
                safe,
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )

    @staticmethod
    def _decode(payload: bytes, *, requested: UUID) -> "PairedRunResult":
        from .orchestrator import PairedRunResult

        if not payload.endswith(b"\n"):
            raise ValueError("run-store document is truncated")
        try:
            record = PairedRunResult.model_validate_json(payload)
        except Exception as exc:
            raise ValueError("run-store document is malformed") from exc
        if record.run_id != requested:
            raise ValueError("run-store document does not match requested run")
        return record

    @staticmethod
    def _name(run_id: UUID) -> str:
        return f"{run_id}.json"

    def _locked_root(self, *, create: bool, exclusive: bool) -> Iterator[int]:
        class _RootContext:
            def __init__(inner, store: "RunStore") -> None:
                inner.store = store
                inner.fd: int | None = None

            def __enter__(inner) -> int | None:
                inner.store._lock.acquire()
                try:
                    inner.fd = inner.store._open_root(create=create)
                    if inner.fd is not None:
                        fcntl.flock(inner.fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
                    return inner.fd
                except BaseException:
                    inner.store._lock.release()
                    raise

            def __exit__(inner, *_exc) -> None:
                try:
                    if inner.fd is not None:
                        fcntl.flock(inner.fd, fcntl.LOCK_UN)
                        os.close(inner.fd)
                finally:
                    inner.store._lock.release()

        return _RootContext(self)  # type: ignore[return-value]

    def get(self, run_id: UUID) -> "PairedRunResult | None":
        if not isinstance(run_id, UUID):
            raise ValueError("run_id must be a UUID")
        with self._locked_root(create=False, exclusive=False) as root_fd:
            if root_fd is None:
                return None
            payload = self._read_file(root_fd, self._name(run_id))
            return None if payload is None else self._decode(payload, requested=run_id)

    def list_records(self) -> tuple["PairedRunResult", ...]:
        with self._locked_root(create=False, exclusive=False) as root_fd:
            if root_fd is None:
                return ()
            records: list["PairedRunResult"] = []
            for name in sorted(os.listdir(root_fd)):
                if name.startswith(".") or not name.endswith(".json"):
                    continue
                try:
                    run_id = UUID(name[:-5])
                except ValueError as exc:
                    raise ValueError("run-store contains an unexpected entry") from exc
                payload = self._read_file(root_fd, name)
                assert payload is not None
                records.append(self._decode(payload, requested=run_id))
            return tuple(records)

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def execution_lease(self) -> ExecutionLease | None:
        with self._locked_root(create=False, exclusive=False) as root_fd:
            if root_fd is None:
                return None
            payload = self._read_file(root_fd, _EXECUTION_LEASE_NAME)
            if payload is None:
                return None
            try:
                return ExecutionLease.model_validate_json(payload)
            except Exception as exc:
                raise ValueError("execution lease is malformed") from exc

    def execution_owner_is_alive(self, run_id: UUID | None = None) -> bool:
        lease = self.execution_lease()
        if lease is None or (run_id is not None and lease.run_id != run_id):
            return False
        return self._pid_alive(lease.pid)

    def acquire_execution(self, run_id: UUID, owner_id: UUID) -> ExecutionLease:
        lease = ExecutionLease(
            run_id=run_id,
            owner_id=owner_id,
            pid=os.getpid(),
            acquired_at=datetime.now(UTC),
        )
        with self._locked_root(create=True, exclusive=True) as root_fd:
            assert root_fd is not None
            payload = self._read_file(root_fd, _EXECUTION_LEASE_NAME)
            if payload is not None:
                try:
                    current = ExecutionLease.model_validate_json(payload)
                except Exception as exc:
                    raise ValueError("execution lease is malformed") from exc
                if (current.run_id, current.owner_id, current.pid) == (
                    lease.run_id,
                    lease.owner_id,
                    lease.pid,
                ):
                    return current
                if self._pid_alive(current.pid):
                    raise RuntimeError("a live execution owner already holds the run store")
            self._atomic_write(
                root_fd,
                _EXECUTION_LEASE_NAME,
                lease.model_dump_json().encode("utf-8") + b"\n",
            )
            return lease

    def release_execution(self, run_id: UUID, owner_id: UUID) -> bool:
        with self._locked_root(create=False, exclusive=True) as root_fd:
            if root_fd is None:
                return False
            payload = self._read_file(root_fd, _EXECUTION_LEASE_NAME)
            if payload is None:
                return False
            try:
                current = ExecutionLease.model_validate_json(payload)
            except Exception as exc:
                raise ValueError("execution lease is malformed") from exc
            if (current.run_id, current.owner_id) != (run_id, owner_id):
                raise RuntimeError("execution lease belongs to another owner")
            os.unlink(_EXECUTION_LEASE_NAME, dir_fd=root_fd)
            os.fsync(root_fd)
            return True

    def clear_stale_execution(self, run_id: UUID) -> bool:
        with self._locked_root(create=False, exclusive=True) as root_fd:
            if root_fd is None:
                return False
            payload = self._read_file(root_fd, _EXECUTION_LEASE_NAME)
            if payload is None:
                return False
            try:
                current = ExecutionLease.model_validate_json(payload)
            except Exception as exc:
                raise ValueError("execution lease is malformed") from exc
            if current.run_id != run_id:
                return False
            if self._pid_alive(current.pid):
                raise RuntimeError("cannot clear a live execution owner")
            os.unlink(_EXECUTION_LEASE_NAME, dir_fd=root_fd)
            os.fsync(root_fd)
            return True

    def create(self, record: "PairedRunResult") -> None:
        from .orchestrator import PairedRunResult

        if not isinstance(record, PairedRunResult) or record.generation != 0:
            raise ValueError("new run record must begin at generation zero")
        with self._locked_root(create=True, exclusive=True) as root_fd:
            assert root_fd is not None
            name = self._name(record.run_id)
            if self._read_file(root_fd, name) is not None:
                raise FileExistsError("run record already exists")
            self._atomic_write(root_fd, name, self._serialize(record))

    def save(self, record: "PairedRunResult") -> None:
        from .orchestrator import PairedRunResult

        if not isinstance(record, PairedRunResult):
            raise TypeError("save requires a PairedRunResult")
        with self._locked_root(create=False, exclusive=True) as root_fd:
            if root_fd is None:
                raise FileNotFoundError("run record does not exist")
            name = self._name(record.run_id)
            payload = self._read_file(root_fd, name)
            if payload is None:
                raise FileNotFoundError("run record does not exist")
            current = self._decode(payload, requested=record.run_id)
            if record.generation != current.generation + 1:
                raise RuntimeError("stale or skipped run-store generation")
            self._atomic_write(root_fd, name, self._serialize(record))


__all__ = ["ExecutionLease", "RunStore"]
