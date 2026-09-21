"""Atomic cumulative cost reservations for credential-bearing lab calls."""

from __future__ import annotations

import errno
import fcntl
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
from uuid import UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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


def _amount(value: object, *, name: str, positive: bool = False) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a Decimal-compatible number") from exc
    if not result.is_finite() or result < 0 or (positive and result == 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


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
    cap_usd: Decimal
    entries: tuple[LedgerEntry, ...] = ()

    @field_validator("cap_usd", mode="before")
    @classmethod
    def _valid_cap(cls, value: object) -> Decimal:
        return _amount(value, name="cap_usd", positive=True)


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
        self._components = parts[index:]
        self._lock = _thread_lock(self.root)

    def _open_root(self, *, create: bool) -> int | None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self._trusted_parent, flags)
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise ValueError("ledger root must not contain a symlink") from exc
            raise
        try:
            for component in self._components:
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

    @contextmanager
    def _locked_root(self, *, exclusive: bool) -> Iterator[int]:
        with self._lock:
            descriptor = self._open_root(create=True)
            assert descriptor is not None
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
                yield descriptor
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)

    def _read_state(self, root_fd: int) -> _LedgerState:
        try:
            metadata = os.stat(_STATE_NAME, dir_fd=root_fd, follow_symlinks=False)
        except FileNotFoundError:
            return _LedgerState(cap_usd=self.cap_usd)
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
        return state

    @staticmethod
    def _write_state(root_fd: int, state: _LedgerState) -> None:
        try:
            metadata = os.stat(_STATE_NAME, dir_fd=root_fd, follow_symlinks=False)
        except FileNotFoundError:
            metadata = None
        if metadata is not None and not stat.S_ISREG(metadata.st_mode):
            raise ValueError("ledger state path must be a regular file")
        payload = state.model_dump_json(indent=None).encode("utf-8") + b"\n"
        temporary = f".{_STATE_NAME}.{secrets.token_hex(8)}.tmp"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(temporary, flags, 0o600, dir_fd=root_fd)
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
            os.replace(
                temporary,
                _STATE_NAME,
                src_dir_fd=root_fd,
                dst_dir_fd=root_fd,
            )
            os.fsync(root_fd)
        except BaseException:
            try:
                os.unlink(temporary, dir_fd=root_fd)
            except FileNotFoundError:
                pass
            raise

    @staticmethod
    def _snapshot_for(state: _LedgerState) -> LedgerSnapshot:
        spent = sum(
            (
                entry.settled_usd or Decimal("0")
                for entry in state.entries
                if entry.state == "reconciled"
            ),
            Decimal("0"),
        )
        reserved = sum(
            (
                entry.reserved_usd
                for entry in state.entries
                if entry.state == "reserved"
            ),
            Decimal("0"),
        )
        return LedgerSnapshot(
            cap_usd=state.cap_usd,
            spent_usd=spent,
            reserved_usd=reserved,
            remaining_usd=max(state.cap_usd - spent - reserved, Decimal("0")),
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
            if snapshot.spent_usd + snapshot.reserved_usd + requested > state.cap_usd:
                raise BudgetExceeded(snapshot, requested)
            entry = LedgerEntry(
                call_id=call_id,
                reserved_usd=requested,
                state="reserved",
            )
            self._write_state(
                root_fd,
                _LedgerState(cap_usd=state.cap_usd, entries=(*state.entries, entry)),
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
            new_state = _LedgerState(cap_usd=state.cap_usd, entries=tuple(entries))
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
            new_state = _LedgerState(cap_usd=state.cap_usd, entries=tuple(entries))
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
            new_state = _LedgerState(cap_usd=state.cap_usd, entries=tuple(entries))
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
