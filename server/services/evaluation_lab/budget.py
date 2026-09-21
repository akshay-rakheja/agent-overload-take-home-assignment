"""Atomic cumulative cost reservations for credential-bearing lab calls."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import secrets
import stat
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4, uuid5

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .decimal_math import exact_add, exact_subtract, exact_sum, validate_exact_decimal
from .pricing import (
    CostSource,
    PricingSnapshot,
    calculate_cost,
    conservative_reservation,
)
from .usage import normalize_usage


DEFAULT_COST_CAP_USD = Decimal("10.00")
_STATE_NAME = "cost-ledger.json"
_STATE_VERSION = 1
_OWNER_DIRECTORY = ".cost-ledger-owners"


def _amount(value: object, *, name: str, positive: bool = False) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a Decimal-compatible number") from exc
    if not result.is_finite() or result < 0 or (positive and result == 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return validate_exact_decimal(result, field=name)


class BudgetExceeded(RuntimeError):
    """Raised before transport when the next reservation would cross the cap."""

    def __init__(self, snapshot: "LedgerSnapshot", requested_usd: Decimal) -> None:
        self.snapshot = snapshot
        self.requested_usd = requested_usd
        super().__init__("evaluation cost cap would be exceeded")


class Reservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    call_id: UUID
    amount_usd: Decimal

    @field_validator("amount_usd", mode="before")
    @classmethod
    def _valid_amount(cls, value: object) -> Decimal:
        return _amount(value, name="reservation amount")


class LedgerEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    call_id: UUID
    reserved_usd: Decimal
    state: Literal["reserved", "reconciled", "abandoned"]
    settled_usd: Decimal | None = None
    source: CostSource | None = None
    outcome: str | None = None
    reason: str | None = None

    @field_validator("reserved_usd", "settled_usd", mode="before")
    @classmethod
    def _valid_amounts(cls, value: object, info):
        return None if value is None else _amount(value, name=info.field_name)

    @field_validator("outcome", "reason")
    @classmethod
    def _bounded_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized or len(normalized) > 256:
            raise ValueError("ledger labels must be bounded non-empty text")
        return normalized

    @model_validator(mode="after")
    def _coherent(self) -> "LedgerEntry":
        if self.state == "reserved" and any(
            value is not None
            for value in (self.settled_usd, self.source, self.outcome, self.reason)
        ):
            raise ValueError("active reservation cannot carry terminal fields")
        if self.state == "reconciled" and (
            self.settled_usd is None or self.source is None or self.outcome is None
        ):
            raise ValueError("reconciled reservation requires cost source and outcome")
        if self.state == "abandoned" and (
            self.settled_usd is not None or self.source is not None or self.reason is None
        ):
            raise ValueError("abandoned reservation requires only a reason")
        return self


class LedgerSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    cap_usd: Decimal
    spent_usd: Decimal
    reserved_usd: Decimal
    remaining_usd: Decimal
    entries: tuple[LedgerEntry, ...]

    @field_validator("cap_usd", "spent_usd", "reserved_usd", "remaining_usd", mode="before")
    @classmethod
    def _valid_amounts(cls, value: object, info) -> Decimal:
        return _amount(value, name=info.field_name)


class _LedgerState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    ledger_id: UUID
    generation: int = Field(ge=0)
    previous_state_sha256: str | None
    cap_usd: Decimal
    entries: tuple[LedgerEntry, ...] = ()

    @field_validator("cap_usd", mode="before")
    @classmethod
    def _valid_cap(cls, value: object) -> Decimal:
        return _amount(value, name="cap_usd", positive=True)

    @field_validator("previous_state_sha256")
    @classmethod
    def _valid_previous_hash(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.casefold()
        if len(normalized) != 64 or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError("previous state digest must be SHA-256")
        return normalized

    @model_validator(mode="after")
    def _valid_chain_position(self) -> "_LedgerState":
        if self.generation == 0 and self.previous_state_sha256 is not None:
            raise ValueError("initial ledger state cannot have a predecessor")
        if self.generation > 0 and self.previous_state_sha256 is None:
            raise ValueError("committed ledger state requires a predecessor digest")
        return self


class _LedgerOwner(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    ledger_id: UUID
    root_device: int = Field(ge=0)
    root_inode: int = Field(gt=0)
    state_generation: int = Field(ge=0)
    state_sha256: str

    @field_validator("state_sha256")
    @classmethod
    def _valid_state_hash(cls, value: str) -> str:
        normalized = value.casefold()
        if len(normalized) != 64 or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError("owner state digest must be SHA-256")
        return normalized


_THREAD_LOCKS: dict[Path, threading.RLock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


def _thread_lock(path: Path) -> threading.RLock:
    key = path.absolute()
    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(key, threading.RLock())


class CostLedger:
    """Descriptor-safe, fsynced and process-serialized cumulative ledger."""

    def __init__(
        self,
        root: Path | str,
        *,
        cap_usd: Decimal = DEFAULT_COST_CAP_USD,
    ) -> None:
        self.root = Path(root)
        self.cap_usd = _amount(cap_usd, name="cap_usd", positive=True)
        parts = self.root.parts
        if ".." in parts:
            raise ValueError("ledger root must not contain path traversal")
        indexes = [index for index, part in enumerate(parts) if part == ".lab"]
        if not indexes or indexes[-1] == len(parts) - 1:
            raise ValueError("ledger root must be below the ignored .lab directory")
        index = indexes[-1]
        self._trusted_parent = Path(*parts[:index]) if parts[:index] else Path(".")
        self._lab_component = parts[index]
        self._root_components = parts[index + 1 :]
        identity_path = "/".join(self._root_components).encode("utf-8")
        self._owner_key = hashlib.sha256(identity_path).hexdigest()
        self._lock = _thread_lock(self.root)
        self._bound_owner: _LedgerOwner | None = None
        self._active_owners_fd: int | None = None

    def _open_lab(self, *, create: bool) -> int | None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self._trusted_parent, flags)
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise ValueError("ledger root must not contain a symlink") from exc
            raise
        try:
            for component in (self._lab_component,):
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
                        raise ValueError("ledger root must not contain a symlink") from exc
                    raise
                os.close(descriptor)
                descriptor = child
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    @staticmethod
    def _open_directory(parent_fd: int, name: str, *, create: bool) -> int | None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        if create:
            try:
                os.mkdir(name, 0o700, dir_fd=parent_fd)
            except FileExistsError:
                pass
        try:
            return os.open(name, flags, dir_fd=parent_fd)
        except FileNotFoundError:
            return None
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise ValueError("ledger root must not contain a symlink") from exc
            raise

    def _open_root_from_lab(self, lab_fd: int, *, create: bool) -> int | None:
        descriptor = os.dup(lab_fd)
        try:
            for component in self._root_components:
                child = self._open_directory(descriptor, component, create=create)
                if child is None:
                    os.close(descriptor)
                    return None
                os.close(descriptor)
                descriptor = child
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    @staticmethod
    def _read_json_file(parent_fd: int, name: str) -> bytes | None:
        try:
            metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return None
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("ledger identity path must be a regular file")
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        try:
            chunks: list[bytes] = []
            while chunk := os.read(descriptor, 64 * 1024):
                chunks.append(chunk)
            return b"".join(chunks)
        finally:
            os.close(descriptor)

    @staticmethod
    def _atomic_write(parent_fd: int, name: str, payload: bytes) -> None:
        temporary = f".{name}.{secrets.token_hex(8)}.tmp"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(temporary, flags, 0o600, dir_fd=parent_fd)
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short ledger write")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.replace(temporary, name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
            os.fsync(parent_fd)
        except BaseException:
            try:
                os.unlink(temporary, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
            raise

    def _read_owner(self, owners_fd: int) -> _LedgerOwner | None:
        payload = self._read_json_file(owners_fd, f"{self._owner_key}.json")
        if payload is None:
            return None
        try:
            return _LedgerOwner.model_validate_json(payload)
        except Exception as exc:
            raise ValueError("ledger identity is malformed") from exc

    def _write_owner(self, owners_fd: int, owner: _LedgerOwner) -> None:
        self._atomic_write(
            owners_fd,
            f"{self._owner_key}.json",
            owner.model_dump_json().encode("utf-8") + b"\n",
        )

    @staticmethod
    def _state_payload(state: _LedgerState) -> bytes:
        return state.model_dump_json(indent=None).encode("utf-8") + b"\n"

    @classmethod
    def _state_digest(cls, state: _LedgerState) -> str:
        return hashlib.sha256(cls._state_payload(state)).hexdigest()

    def _assert_current_root(self, root_fd: int, owner: _LedgerOwner) -> None:
        metadata = os.fstat(root_fd)
        if (metadata.st_dev, metadata.st_ino) != (
            owner.root_device,
            owner.root_inode,
        ):
            raise ValueError("ledger root identity changed")
        lab_fd = self._open_lab(create=False)
        if lab_fd is None:
            raise ValueError("ledger root identity disappeared")
        try:
            current_fd = self._open_root_from_lab(lab_fd, create=False)
            if current_fd is None:
                raise ValueError("ledger root identity disappeared")
            try:
                current = os.fstat(current_fd)
            finally:
                os.close(current_fd)
        finally:
            os.close(lab_fd)
        if (current.st_dev, current.st_ino) != (
            owner.root_device,
            owner.root_inode,
        ):
            raise ValueError("ledger root identity changed")

    @contextmanager
    def _locked_root(self, *, exclusive: bool) -> Iterator[int]:
        with self._lock:
            lab_fd = self._open_lab(create=True)
            assert lab_fd is not None
            owners_fd: int | None = None
            lock_fd: int | None = None
            root_fd: int | None = None
            lab_locked = False
            try:
                fcntl.flock(lab_fd, fcntl.LOCK_EX)
                lab_locked = True
                owners_fd = self._open_directory(
                    lab_fd, _OWNER_DIRECTORY, create=True
                )
                assert owners_fd is not None
                lock_name = f"{self._owner_key}.lock"
                try:
                    lock_fd = os.open(
                        lock_name,
                        os.O_RDWR
                        | os.O_CREAT
                        | getattr(os, "O_NOFOLLOW", 0),
                        0o600,
                        dir_fd=owners_fd,
                    )
                except OSError as exc:
                    if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                        raise ValueError(
                            "ledger identity lock must not be a symlink"
                        ) from exc
                    raise
                if not stat.S_ISREG(os.fstat(lock_fd).st_mode):
                    raise ValueError("ledger identity lock must be a regular file")
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                fcntl.flock(lab_fd, fcntl.LOCK_UN)
                lab_locked = False
                owner = self._read_owner(owners_fd)
                if owner is None:
                    if self._bound_owner is not None:
                        raise ValueError("ledger identity disappeared")
                    existing = self._open_root_from_lab(lab_fd, create=False)
                    if existing is not None:
                        os.close(existing)
                        raise ValueError("ledger root exists without an identity")
                    root_fd = self._open_root_from_lab(lab_fd, create=True)
                    assert root_fd is not None
                    root_metadata = os.fstat(root_fd)
                    state = _LedgerState(
                        ledger_id=uuid4(),
                        generation=0,
                        previous_state_sha256=None,
                        cap_usd=self.cap_usd,
                    )
                    owner = _LedgerOwner(
                        ledger_id=state.ledger_id,
                        root_device=root_metadata.st_dev,
                        root_inode=root_metadata.st_ino,
                        state_generation=state.generation,
                        state_sha256=self._state_digest(state),
                    )
                    self._bound_owner = owner
                    self._atomic_write(
                        root_fd,
                        _STATE_NAME,
                        self._state_payload(state),
                    )
                    self._write_owner(owners_fd, owner)
                else:
                    if (
                        self._bound_owner is not None
                        and self._bound_owner.ledger_id != owner.ledger_id
                    ):
                        raise ValueError("ledger identity changed")
                    root_fd = self._open_root_from_lab(lab_fd, create=False)
                    if root_fd is None:
                        raise ValueError("ledger root identity disappeared")
                    self._bound_owner = owner
                    self._assert_current_root(root_fd, owner)
                    _, owner = self._recover_or_validate_state(
                        root_fd, owners_fd, owner
                    )
                    self._bound_owner = owner
                self._active_owners_fd = owners_fd
                yield root_fd
                current_owner = self._bound_owner
                if current_owner is None:
                    raise ValueError("ledger identity disappeared")
                self._assert_current_root(root_fd, current_owner)
                self._read_state(root_fd)
            finally:
                self._active_owners_fd = None
                if root_fd is not None:
                    os.close(root_fd)
                if lock_fd is not None:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    os.close(lock_fd)
                if owners_fd is not None:
                    os.close(owners_fd)
                if lab_locked:
                    fcntl.flock(lab_fd, fcntl.LOCK_UN)
                os.close(lab_fd)

    def _load_state(self, root_fd: int) -> tuple[_LedgerState, str]:
        try:
            metadata = os.stat(_STATE_NAME, dir_fd=root_fd, follow_symlinks=False)
        except FileNotFoundError:
            raise ValueError("ledger state identity disappeared")
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("ledger state path must be a regular file")
        descriptor = os.open(
            _STATE_NAME,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=root_fd,
        )
        try:
            chunks: list[bytes] = []
            while chunk := os.read(descriptor, 64 * 1024):
                chunks.append(chunk)
        finally:
            os.close(descriptor)
        try:
            state = _LedgerState.model_validate_json(b"".join(chunks))
        except Exception as exc:
            raise ValueError("ledger state is malformed") from exc
        if state.cap_usd != self.cap_usd:
            raise ValueError("persisted ledger cap differs from configured cap")
        if len({entry.call_id for entry in state.entries}) != len(state.entries):
            raise ValueError("ledger state contains duplicate call ids")
        return state, hashlib.sha256(b"".join(chunks)).hexdigest()

    @staticmethod
    def _state_matches_owner(
        state: _LedgerState, digest: str, owner: _LedgerOwner
    ) -> bool:
        return (
            state.ledger_id == owner.ledger_id
            and state.generation == owner.state_generation
            and digest == owner.state_sha256
        )

    def _recover_or_validate_state(
        self,
        root_fd: int,
        owners_fd: int,
        owner: _LedgerOwner,
    ) -> tuple[_LedgerState, _LedgerOwner]:
        state, digest = self._load_state(root_fd)
        if self._state_matches_owner(state, digest, owner):
            return state, owner
        if (
            state.ledger_id == owner.ledger_id
            and state.generation == owner.state_generation + 1
            and state.previous_state_sha256 == owner.state_sha256
        ):
            recovered = owner.model_copy(
                update={
                    "state_generation": state.generation,
                    "state_sha256": digest,
                }
            )
            self._write_owner(owners_fd, recovered)
            return state, recovered
        raise ValueError("ledger state rollback or digest mismatch")

    def _read_state(self, root_fd: int) -> _LedgerState:
        owner = self._bound_owner
        if owner is None:
            raise ValueError("ledger identity disappeared")
        state, digest = self._load_state(root_fd)
        if not self._state_matches_owner(state, digest, owner):
            raise ValueError("ledger state rollback or digest mismatch")
        return state

    def _write_state(self, root_fd: int, state: _LedgerState) -> _LedgerState:
        owner = self._bound_owner
        owners_fd = self._active_owners_fd
        if owner is None or owners_fd is None or state.ledger_id != owner.ledger_id:
            raise ValueError("ledger state identity changed")
        self._assert_current_root(root_fd, owner)
        self._read_state(root_fd)
        committed = state.model_copy(
            update={
                "generation": owner.state_generation + 1,
                "previous_state_sha256": owner.state_sha256,
            }
        )
        payload = self._state_payload(committed)
        digest = hashlib.sha256(payload).hexdigest()
        self._atomic_write(root_fd, _STATE_NAME, payload)
        updated_owner = owner.model_copy(
            update={
                "state_generation": committed.generation,
                "state_sha256": digest,
            }
        )
        self._write_owner(owners_fd, updated_owner)
        self._bound_owner = updated_owner
        self._assert_current_root(root_fd, updated_owner)
        return committed

    @staticmethod
    def _snapshot_for(state: _LedgerState) -> LedgerSnapshot:
        spent = exact_sum(
            (
                entry.settled_usd or Decimal("0")
                for entry in state.entries
                if entry.state == "reconciled"
            ),
        )
        reserved = exact_sum(
            (
                entry.reserved_usd
                for entry in state.entries
                if entry.state == "reserved"
            ),
        )
        committed = exact_add(spent, reserved)
        return LedgerSnapshot(
            cap_usd=state.cap_usd,
            spent_usd=spent,
            reserved_usd=reserved,
            remaining_usd=max(exact_subtract(state.cap_usd, committed), Decimal("0")),
            entries=tuple(sorted(state.entries, key=lambda item: str(item.call_id))),
        )

    def snapshot(self) -> LedgerSnapshot:
        with self._locked_root(exclusive=False) as root_fd:
            return self._snapshot_for(self._read_state(root_fd))

    def reserve(self, estimate: Decimal, *, call_id: UUID) -> Reservation:
        requested = _amount(estimate, name="estimate")
        if not isinstance(call_id, UUID):
            raise TypeError("call_id must be a UUID")
        with self._locked_root(exclusive=True) as root_fd:
            state = self._read_state(root_fd)
            existing = next((entry for entry in state.entries if entry.call_id == call_id), None)
            if existing is not None:
                if existing.state == "reserved" and existing.reserved_usd == requested:
                    return Reservation(call_id=call_id, amount_usd=requested)
                raise ValueError("call_id is already finalized or reserved for a different amount")
            snapshot = self._snapshot_for(state)
            projected = exact_add(snapshot.spent_usd, snapshot.reserved_usd, requested)
            if projected > state.cap_usd:
                raise BudgetExceeded(snapshot, requested)
            entry = LedgerEntry(
                call_id=call_id,
                reserved_usd=requested,
                state="reserved",
            )
            self._write_state(
                root_fd,
                _LedgerState(
                    ledger_id=state.ledger_id,
                    generation=state.generation,
                    previous_state_sha256=state.previous_state_sha256,
                    cap_usd=state.cap_usd,
                    entries=(*state.entries, entry),
                ),
            )
            return Reservation(call_id=call_id, amount_usd=requested)

    def reconcile(
        self,
        reservation: Reservation,
        actual_or_estimated: Decimal,
        *,
        source: CostSource,
        outcome: str,
    ) -> LedgerSnapshot:
        settled = _amount(actual_or_estimated, name="actual_or_estimated")
        if source is CostSource.UNAVAILABLE:
            raise ValueError("reconciled cost source cannot be unavailable")
        with self._locked_root(exclusive=True) as root_fd:
            state = self._read_state(root_fd)
            entries = list(state.entries)
            index = next(
                (position for position, item in enumerate(entries) if item.call_id == reservation.call_id),
                None,
            )
            if index is None:
                raise ValueError("reservation does not belong to this ledger")
            current = entries[index]
            if current.reserved_usd != reservation.amount_usd:
                raise ValueError("reservation amount differs from persisted ledger")
            if current.state == "reconciled":
                if (
                    current.settled_usd == settled
                    and current.source is source
                    and current.outcome == outcome.strip()
                ):
                    return self._snapshot_for(state)
                raise ValueError("reservation was reconciled with different terminal facts")
            if current.state != "reserved":
                raise ValueError("abandoned reservation cannot be reconciled")
            entries[index] = LedgerEntry(
                call_id=current.call_id,
                reserved_usd=current.reserved_usd,
                state="reconciled",
                settled_usd=settled,
                source=source,
                outcome=outcome,
            )
            new_state = _LedgerState(
                ledger_id=state.ledger_id,
                generation=state.generation,
                previous_state_sha256=state.previous_state_sha256,
                cap_usd=state.cap_usd,
                entries=tuple(entries),
            )
            self._write_state(root_fd, new_state)
            return self._snapshot_for(new_state)

    def abandon(self, reservation: Reservation, *, reason: str) -> LedgerSnapshot:
        with self._locked_root(exclusive=True) as root_fd:
            state = self._read_state(root_fd)
            entries = list(state.entries)
            index = next(
                (position for position, item in enumerate(entries) if item.call_id == reservation.call_id),
                None,
            )
            if index is None:
                raise ValueError("reservation does not belong to this ledger")
            current = entries[index]
            if current.reserved_usd != reservation.amount_usd:
                raise ValueError("reservation amount differs from persisted ledger")
            if current.state == "abandoned":
                if current.reason == reason.strip():
                    return self._snapshot_for(state)
                raise ValueError("reservation was abandoned with a different reason")
            if current.state != "reserved":
                raise ValueError("reconciled reservation cannot be abandoned")
            entries[index] = LedgerEntry(
                call_id=current.call_id,
                reserved_usd=current.reserved_usd,
                state="abandoned",
                reason=reason,
            )
            new_state = _LedgerState(
                ledger_id=state.ledger_id,
                generation=state.generation,
                previous_state_sha256=state.previous_state_sha256,
                cap_usd=state.cap_usd,
                entries=tuple(entries),
            )
            self._write_state(root_fd, new_state)
            return self._snapshot_for(new_state)

    def recover_crashed_reservations(self, *, reason: str) -> LedgerSnapshot:
        """Charge every orphaned reservation at its conservative upper bound."""

        with self._locked_root(exclusive=True) as root_fd:
            state = self._read_state(root_fd)
            changed = False
            entries: list[LedgerEntry] = []
            for entry in state.entries:
                if entry.state != "reserved":
                    entries.append(entry)
                    continue
                changed = True
                entries.append(
                    LedgerEntry(
                        call_id=entry.call_id,
                        reserved_usd=entry.reserved_usd,
                        state="reconciled",
                        settled_usd=entry.reserved_usd,
                        source=CostSource.CRASH_RECOVERY_ESTIMATE,
                        outcome="crash_recovered",
                        reason=reason,
                    )
                )
            new_state = _LedgerState(
                ledger_id=state.ledger_id,
                generation=state.generation,
                previous_state_sha256=state.previous_state_sha256,
                cap_usd=state.cap_usd,
                entries=tuple(entries),
            )
            if changed:
                self._write_state(root_fd, new_state)
            return self._snapshot_for(new_state)


class CostController:
    """Bridge request payloads and provider responses to one atomic ledger."""

    def __init__(self, *, ledger: CostLedger, pricing: PricingSnapshot) -> None:
        self.ledger = ledger
        self.pricing = pricing

    def reserve_call(
        self,
        *,
        payload: Mapping[str, object],
        max_tokens: int,
        logical_call_id: UUID,
        attempt: int,
    ) -> Reservation:
        requested_model = payload.get("model")
        if requested_model != self.pricing.model_id:
            raise ValueError("paid call model does not match pricing snapshot")
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        estimate = conservative_reservation(
            serialized_prompt=serialized,
            max_tokens=max_tokens,
            pricing=self.pricing,
        )
        if estimate.amount_usd is None:
            raise ValueError("a paid call cannot proceed without complete reservation pricing")
        attempt_id = uuid5(logical_call_id, f"transport-attempt:{attempt}")
        return self.ledger.reserve(estimate.amount_usd, call_id=attempt_id)

    def settle_response(
        self,
        reservation: Reservation,
        response: object,
        *,
        outcome: str,
    ) -> LedgerSnapshot:
        usage = normalize_usage(response)
        cost = calculate_cost(usage, self.pricing)
        if cost.amount_usd is None:
            return self.ledger.reconcile(
                reservation,
                reservation.amount_usd,
                source=CostSource.CONSERVATIVE_ESTIMATE,
                outcome=outcome,
            )
        return self.ledger.reconcile(
            reservation,
            cost.amount_usd,
            source=cost.source,
            outcome=outcome,
        )

    def settle_unreported(self, reservation: Reservation, *, outcome: str) -> LedgerSnapshot:
        return self.ledger.reconcile(
            reservation,
            reservation.amount_usd,
            source=CostSource.CONSERVATIVE_ESTIMATE,
            outcome=outcome,
        )


_ACTIVE_CONTROLLER: ContextVar[CostController | None] = ContextVar(
    "evaluation_lab_cost_controller", default=None
)


@contextmanager
def budget_scope(controller: CostController) -> Iterator[None]:
    if not isinstance(controller, CostController):
        raise TypeError("budget_scope requires a CostController")
    token: Token[CostController | None] = _ACTIVE_CONTROLLER.set(controller)
    try:
        yield
    finally:
        _ACTIVE_CONTROLLER.reset(token)


def current_cost_controller() -> CostController | None:
    return _ACTIVE_CONTROLLER.get()


__all__ = [
    "BudgetExceeded",
    "CostController",
    "CostLedger",
    "DEFAULT_COST_CAP_USD",
    "LedgerEntry",
    "LedgerSnapshot",
    "Reservation",
    "budget_scope",
    "current_cost_controller",
]
