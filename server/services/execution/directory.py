"""Atomic, lifecycle-aware persistence for execution-agent identities."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
import threading
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from pydantic import ValidationError

from .models import AGENT_SCHEMA_VERSION, AgentRecord, AgentStatus, normalize_agent_text


DIRECTORY_SCHEMA_VERSION = 1
Clock = Callable[[], datetime]
IdFactory = Callable[[], UUID]


class DirectoryError(RuntimeError):
    """Base class for directory persistence and lookup failures."""


class DirectoryCorruptError(DirectoryError):
    """Raised when persisted data cannot be safely interpreted."""


class UnknownAgentError(DirectoryError):
    """Raised when a stable identity does not exist."""


_LOCKS_GUARD = threading.Lock()
_PATH_LOCKS: dict[Path, threading.RLock] = {}


def _path_lock(path: Path) -> threading.RLock:
    resolved = path.resolve()
    with _LOCKS_GUARD:
        return _PATH_LOCKS.setdefault(resolved, threading.RLock())


class AgentDirectory:
    """Versioned source of truth for persistent execution-agent identity."""

    def __init__(
        self,
        path: Path,
        *,
        clock: Clock | None = None,
        id_factory: IdFactory | None = None,
    ) -> None:
        self._path = path
        self._lock_path = path.with_suffix(f"{path.suffix}.lock")
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._id_factory = id_factory or uuid4
        self._records: list[AgentRecord] = []
        self.load()

    @contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with _path_lock(self._path):
            with self._lock_path.open("a+", encoding="utf-8") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _read_json_locked(self) -> Any:
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as exc:
            raise DirectoryCorruptError(f"Could not parse agent directory {self._path}: {exc}") from exc

    def _records_from_payload(self, payload: Any) -> tuple[list[AgentRecord], bool]:
        if payload is None:
            return [], True
        if isinstance(payload, list):
            return self._migrate_legacy_names(payload), True
        if not isinstance(payload, dict):
            raise DirectoryCorruptError("Agent directory must contain an object or a legacy name list")
        if payload.get("schema_version") != DIRECTORY_SCHEMA_VERSION:
            raise DirectoryCorruptError(
                f"Unsupported directory schema version: {payload.get('schema_version')!r}"
            )
        agents = payload.get("agents")
        if not isinstance(agents, list):
            raise DirectoryCorruptError("Agent directory field 'agents' must be a list")
        try:
            return [AgentRecord.model_validate(item) for item in agents], False
        except ValidationError as exc:
            raise DirectoryCorruptError(f"Invalid agent directory record: {exc}") from exc

    def _migrate_legacy_names(self, payload: list[Any]) -> list[AgentRecord]:
        now = self._now()
        seen: dict[str, int] = {}
        migrated: list[AgentRecord] = []
        for raw_name in payload:
            name = str(raw_name).strip() or "agent"
            normalized = normalize_agent_text(name)
            occurrence = seen.get(normalized, 0)
            seen[normalized] = occurrence + 1
            stable_id = uuid5(NAMESPACE_URL, f"openpoke-legacy:{normalized}:{occurrence}")
            migrated.append(
                AgentRecord(
                    agent_id=stable_id,
                    name=name,
                    purpose=f"Handle tasks related to: {name}",
                    aliases=(name,),
                    status=AgentStatus.HOT,
                    created_at=now,
                    last_used_at=now,
                    legacy_storage_key=name if occurrence == 0 else None,
                )
            )
        return migrated

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("AgentDirectory clock must return a timezone-aware datetime")
        return value

    def _serialize(self, records: Iterable[AgentRecord]) -> str:
        payload = {
            "schema_version": DIRECTORY_SCHEMA_VERSION,
            "agents": [record.model_dump(mode="json") for record in records],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    def _write_locked(self, records: list[AgentRecord]) -> None:
        rendered = self._serialize(records)
        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=self._path.parent,
            prefix=f".{self._path.name}.",
            suffix=".tmp",
            text=True,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
                handle.write(rendered)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self._path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def _load_locked(self) -> list[AgentRecord]:
        records, should_write = self._records_from_payload(self._read_json_locked())
        if should_write:
            self._write_locked(records)
        return records

    def load(self) -> None:
        """Reload persisted records, migrating legacy name lists in place."""

        with self._exclusive_lock():
            self._records = self._load_locked()

    def save(self) -> None:
        """Persist the current in-memory snapshot atomically."""

        with self._exclusive_lock():
            self._write_locked(self._records)

    def list_records(self) -> list[AgentRecord]:
        """Return all records, including dormant and archived identities."""

        self.load()
        return list(self._records)

    def create(
        self,
        *,
        name: str,
        purpose: str,
        aliases: Iterable[str] = (),
        memory_summary: str = "",
    ) -> AgentRecord:
        """Create a new hot identity and persist it without clobbering peers."""

        with self._exclusive_lock():
            records = self._load_locked()
            now = self._now()
            record = AgentRecord(
                agent_id=self._id_factory(),
                name=name,
                purpose=purpose,
                aliases=tuple(aliases),
                status=AgentStatus.HOT,
                created_at=now,
                last_used_at=now,
                memory_summary=memory_summary,
                schema_version=AGENT_SCHEMA_VERSION,
            )
            records.append(record)
            self._write_locked(records)
            self._records = records
            return record

    def _require_from(self, records: Iterable[AgentRecord], agent_id: UUID) -> AgentRecord:
        for record in records:
            if record.agent_id == agent_id:
                return record
        raise UnknownAgentError(f"Unknown execution agent ID: {agent_id}")

    def require(self, agent_id: UUID | str) -> AgentRecord:
        """Return one stable identity or fail closed."""

        try:
            parsed_id = agent_id if isinstance(agent_id, UUID) else UUID(agent_id)
        except (TypeError, ValueError, AttributeError) as exc:
            raise UnknownAgentError(f"Unknown execution agent ID: {agent_id}") from exc
        self.load()
        return self._require_from(self._records, parsed_id)

    def _replace(self, agent_id: UUID | str, transform: Callable[[AgentRecord], AgentRecord]) -> AgentRecord:
        try:
            parsed_id = agent_id if isinstance(agent_id, UUID) else UUID(agent_id)
        except (TypeError, ValueError, AttributeError) as exc:
            raise UnknownAgentError(f"Unknown execution agent ID: {agent_id}") from exc

        with self._exclusive_lock():
            records = self._load_locked()
            current = self._require_from(records, parsed_id)
            updated = transform(current)
            records = [updated if record.agent_id == parsed_id else record for record in records]
            self._write_locked(records)
            self._records = records
            return updated

    def transition(self, agent_id: UUID | str, status: AgentStatus | str) -> AgentRecord:
        """Move an identity to an explicit lifecycle state."""

        parsed_status = status if isinstance(status, AgentStatus) else AgentStatus(status)
        return self._replace(agent_id, lambda record: record.model_copy(update={"status": parsed_status}))

    def mark_used(self, agent_id: UUID | str) -> AgentRecord:
        """Record a successful dispatch and reactivate the identity."""

        now = self._now()
        return self._replace(
            agent_id,
            lambda record: record.model_copy(
                update={
                    "status": AgentStatus.HOT,
                    "last_used_at": now,
                    "use_count": record.use_count + 1,
                }
            ),
        )

    def find_by_alias(self, value: str) -> list[AgentRecord]:
        """Return every exact normalized name or alias match without merging."""

        needle = normalize_agent_text(value)
        return [
            record
            for record in self.list_records()
            if needle == record.normalized_name or needle in record.normalized_aliases
        ]

    def clear(self) -> None:
        """Clear directory identities without touching execution logs."""

        with self._exclusive_lock():
            self._write_locked([])
            self._records = []
