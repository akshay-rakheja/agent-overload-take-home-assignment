"""Backward-compatible name roster backed by the Agent Directory."""

from __future__ import annotations

from pathlib import Path

from .directory import AgentDirectory


class AgentRoster:
    """Compatibility adapter for callers that still consume display names."""

    def __init__(self, roster_path: Path):
        self._directory = AgentDirectory(roster_path)

    @property
    def directory(self) -> AgentDirectory:
        return self._directory

    def load(self) -> None:
        self._directory.load()

    def save(self) -> None:
        self._directory.save()

    def add_agent(self, agent_name: str) -> None:
        """Preserve legacy exact-name deduplication while creating a full record."""

        if agent_name not in self.get_agents():
            self._directory.create(
                name=agent_name,
                purpose=f"Handle tasks related to: {agent_name}",
                aliases=(agent_name,),
            )

    def get_agents(self) -> list[str]:
        return [record.name for record in self._directory.list_records()]

    def clear(self) -> None:
        self._directory.clear()


_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_ROSTER_PATH = _DATA_DIR / "execution_agents" / "roster.json"
_ROSTER_PATH_JEV = _DATA_DIR / "execution_agents" / "roster_jev.json"

_agent_roster_det = AgentRoster(_ROSTER_PATH)
_agent_roster_jev = AgentRoster(_ROSTER_PATH_JEV)


def get_agent_roster(system: str | None = None) -> AgentRoster:
    """Get the singleton compatibility roster for the requested system."""
    if system in ("enhanced_jev", "jev"):
        return _agent_roster_jev
    return _agent_roster_det


def get_agent_directory(system: str | None = None) -> AgentDirectory:
    """Get the persistent directory behind the compatibility roster."""
    return get_agent_roster(system).directory
