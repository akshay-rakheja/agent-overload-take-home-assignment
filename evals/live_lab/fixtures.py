"""Deterministic fixture manifests and runtime materializers."""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone
from importlib.resources import files
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from server.services.execution.directory import DIRECTORY_SCHEMA_VERSION, AgentDirectory
from server.services.execution.log_store import _encode_payload, execution_log_slug
from server.services.execution.models import AgentStatus

from .contracts import (
    ExpectedAction,
    FixtureManifest,
    JournalEntry,
    LogicalAgent,
    ScenarioExpectation,
    StateFingerprint,
)


_ALLOWED_ROSTER_SIZES = {10, 100, 500, 1_000}
_BASE_TIME = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)


def _read_spec() -> dict[str, object]:
    resource = files("evals.live_lab").joinpath("fixture_spec.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def _logical_id(seed: int, key: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"openpoke-live-lab:{seed}:{key}"))


def _entry(tag: str, offset: int, payload: str) -> JournalEntry:
    timestamp = (_BASE_TIME + timedelta(seconds=offset)).isoformat().replace("+00:00", "Z")
    return JournalEntry(tag=tag, timestamp=timestamp, payload=payload)


def _four_entry_episode(prefix: str, offset: int) -> tuple[JournalEntry, ...]:
    return (
        _entry("agent_request", offset, f"{prefix} request"),
        _entry("agent_action", offset + 1, f"{prefix} action"),
        _entry("tool_response", offset + 2, f"local_fixture: {prefix} result"),
        _entry("agent_response", offset + 3, f"{prefix} response"),
    )


def _history(template: str, depth_episode_count: int) -> tuple[JournalEntry, ...]:
    if template == "depth":
        entries: list[JournalEntry] = []
        for episode in range(depth_episode_count):
            marker = f"DEPTH-EPISODE-{episode:04d}"
            if episode == 0:
                marker += " DEPTH-START-SENTINEL"
            if episode == depth_episode_count - 1:
                marker += " DEPTH-END-SENTINEL"
            entries.extend(_four_entry_episode(marker, episode * 4))
        return tuple(entries)

    labels = {
        "old_relevant": "OLD-RELEVANT Instagram security review",
        "recent_distractor": "RECENT-DISTRACTOR Instagram engagement review",
        "standard": "FABRICATED receipt review",
        "collision_a": "SPACE-COLLISION-SENTINEL",
        "collision_b": "HYPHEN-COLLISION-SENTINEL",
        "same_name_a": "RETENTION-CAMPAIGN-SENTINEL",
        "same_name_b": "PARTNER-CAMPAIGN-SENTINEL",
        "unicode_a": "ACCENTED-UNICODE-SENTINEL",
        "unicode_b": "PLAIN-UNICODE-SENTINEL CROSS-AGENT-SENTINEL",
    }
    return _four_entry_episode(labels[template], 20_000 + sorted(labels).index(template) * 4)


def build_fixture_manifest(*, seed: int, roster_size: int) -> FixtureManifest:
    """Build a deterministic credential-free manifest without global RNG state."""

    if roster_size not in _ALLOWED_ROSTER_SIZES:
        raise ValueError(f"roster_size must be one of {sorted(_ALLOWED_ROSTER_SIZES)}")
    rng = random.Random(seed)
    spec = _read_spec()
    depth_episode_count = int(spec["depth_episode_count"])
    agents_by_key: dict[str, LogicalAgent] = {}
    agents: list[LogicalAgent] = []

    for raw in spec["named_agents"]:  # type: ignore[index]
        item = dict(raw)
        key = str(item["key"])
        agent = LogicalAgent(
            logical_id=_logical_id(seed, key),
            name=str(item["name"]),
            purpose=str(item["purpose"]),
            aliases=tuple(str(alias) for alias in item["aliases"]),
            status=str(item["status"]),
            memory_summary=str(item["memory_summary"]),
            journal_entries=_history(str(item["history_template"]), depth_episode_count),
        )
        agents_by_key[key] = agent
        agents.append(agent)

    for index in range(roster_size - len(agents)):
        key = f"unrelated-{index:04d}"
        topic = rng.choice(("inventory", "travel", "gardening", "maintenance", "publishing"))
        nonce = rng.randrange(1_000_000)
        agents.append(
            LogicalAgent(
                logical_id=_logical_id(seed, key),
                name=f"Unrelated Workflow {index + 1:04d}",
                purpose=f"Handle fabricated {topic} distractor {nonce:06d}",
                aliases=(f"distractor {index + 1}",),
                status="hot",
                memory_summary="Unrelated synthetic workflow.",
                journal_entries=_four_entry_episode(
                    f"UNRELATED-{index:04d}-{nonce:06d}",
                    30_000 + index * 4,
                ),
            )
        )

    rng.shuffle(agents)
    scenarios = tuple(
        ScenarioExpectation(
            scenario_id=str(raw["scenario_id"]),
            expected_action=ExpectedAction(str(raw["action"])),
            expected_logical_identity=(
                agents_by_key[str(raw["agent_key"])].logical_id if raw["agent_key"] else None
            ),
            expected_gmail_fact_ids=(),
        )
        for raw in spec["scenarios"]  # type: ignore[index]
    )
    return FixtureManifest(seed=seed, roster_size=roster_size, agents=tuple(agents), scenarios=scenarios)


def serialize_manifest(manifest: FixtureManifest) -> bytes:
    """Return canonical sorted-key JSON with a required trailing newline."""

    rendered = json.dumps(
        manifest.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{rendered}\n".encode("utf-8")


def _render_journal_entry(entry: JournalEntry) -> bytes:
    payload = _encode_payload(entry.payload)
    return (
        f'<{entry.tag} timestamp="{entry.timestamp}">{payload}</{entry.tag}>\n'
    ).encode("utf-8")


def _write_legacy_state(manifest: FixtureManifest, data_dir: Path) -> None:
    if data_dir.exists() and any(data_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty data directory: {data_dir}")
    execution_dir = data_dir / "execution_agents"
    manifest_dir = data_dir / "live_lab"
    execution_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    names = [agent.name for agent in manifest.agents]
    roster_bytes = json.dumps(names, ensure_ascii=False, indent=2).encode("utf-8")
    (execution_dir / "roster.json").write_bytes(roster_bytes)
    (manifest_dir / "fixture_manifest.json").write_bytes(serialize_manifest(manifest))

    journal_chunks: dict[str, list[bytes]] = {}
    for agent in manifest.agents:
        relative_name = f"{execution_log_slug(agent.name)}.log"
        chunks = journal_chunks.setdefault(relative_name, [])
        chunks.extend(_render_journal_entry(entry) for entry in agent.journal_entries)
    for relative_name, chunks in sorted(journal_chunks.items()):
        (execution_dir / relative_name).write_bytes(b"".join(chunks))


def materialize_baseline(manifest: FixtureManifest, data_dir: Path) -> StateFingerprint:
    """Write the historical name-list roster and name-slug journals."""

    _write_legacy_state(manifest, data_dir)
    from .state import fingerprint_state

    return fingerprint_state(data_dir)


def materialize_enhanced(manifest: FixtureManifest, data_dir: Path) -> StateFingerprint:
    """Write legacy-compatible bytes and run the real production migration."""

    _write_legacy_state(manifest, data_dir)
    roster_path = data_dir / "execution_agents" / "roster.json"
    migrated = AgentDirectory(roster_path, clock=lambda: _BASE_TIME).list_records()
    if [record.name for record in migrated] != [agent.name for agent in manifest.agents]:
        raise RuntimeError("production migration changed deterministic roster ordering")
    enriched = [
        record.model_copy(
            update={
                "purpose": logical.purpose,
                "aliases": logical.aliases,
                "status": AgentStatus(logical.status),
                "memory_summary": logical.memory_summary,
            }
        )
        for record, logical in zip(migrated, manifest.agents)
    ]
    roster_payload = {
        "schema_version": DIRECTORY_SCHEMA_VERSION,
        "agents": [record.model_dump(mode="json") for record in enriched],
    }
    roster_path.write_text(
        json.dumps(roster_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    AgentDirectory(roster_path, clock=lambda: _BASE_TIME).list_records()
    from .state import fingerprint_state

    return fingerprint_state(data_dir)
