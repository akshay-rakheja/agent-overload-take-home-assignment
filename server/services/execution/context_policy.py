"""Bounded rendering policy for one selected execution agent's durable history."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Iterable


LogEntry = tuple[str, str, str]


def render_log_entries(entries: Iterable[LogEntry]) -> str:
    """Render entries exactly as the execution log store exposes them to prompts."""

    parts: list[str] = []
    for tag, timestamp, payload in entries:
        escaped = escape(payload, quote=False)
        if timestamp:
            parts.append(f'<{tag} timestamp="{timestamp}">{escaped}</{tag}>')
        else:
            parts.append(f"<{tag}>{escaped}</{tag}>")
    return "\n".join(parts)


@dataclass(frozen=True)
class ContextMetrics:
    raw_entry_count: int
    raw_history_characters: int
    raw_history_bytes: int
    rendered_characters: int
    rendered_bytes: int
    included_episode_count: int
    omitted_entry_count: int
    truncated_entry_count: int
    summary_used: bool


@dataclass(frozen=True)
class ContextRenderResult:
    text: str
    metrics: ContextMetrics


def _group_episodes(entries: list[LogEntry]) -> list[list[LogEntry]]:
    episodes: list[list[LogEntry]] = []
    current: list[LogEntry] = []
    for entry in entries:
        if entry[0] == "agent_request" and current:
            episodes.append(current)
            current = []
        current.append(entry)
    if current:
        episodes.append(current)
    return episodes


def _valid_summary(summary: object) -> str | None:
    if not isinstance(summary, str):
        return None
    stripped = summary.strip()
    if not stripped or "\x00" in stripped:
        return None
    return stripped


def _bounded_summary(summary: str, limit: int) -> str:
    prefix = "<memory_summary>"
    suffix = "</memory_summary>"
    escaped = escape(summary, quote=False)
    if len(prefix) + len(escaped) + len(suffix) <= limit:
        return f"{prefix}{escaped}{suffix}"

    marker = "… summary truncated …"
    available = max(0, limit - len(prefix) - len(suffix) - len(marker))
    return f"{prefix}{escaped[:available]}{marker}{suffix}"


def _truncated_entry(entry: LogEntry, limit: int) -> str:
    tag, _timestamp, payload = entry
    prefix = f'<truncated_entry tag="{escape(tag, quote=True)}" original_characters="{len(payload)}">'
    suffix = "… truncated …</truncated_entry>"
    escaped = escape(payload, quote=False)
    available = max(0, limit - len(prefix) - len(suffix))
    if available:
        head_size = available // 2
        tail_size = available - head_size
        snippet = f"{escaped[:head_size]}{escaped[-tail_size:] if tail_size else ''}"
    else:
        snippet = ""
    rendered = f"{prefix}{snippet}{suffix}"
    return rendered[:limit]


class ExecutionContextPolicy:
    """Render summary plus recent complete episodes within hard limits."""

    def __init__(self, *, max_recent_episodes: int, max_characters: int) -> None:
        if max_recent_episodes < 1:
            raise ValueError("max_recent_episodes must be positive")
        if max_characters < 200:
            raise ValueError("max_characters must be at least 200")
        self.max_recent_episodes = max_recent_episodes
        self.max_characters = max_characters

    @staticmethod
    def _omission_marker(count: int) -> str:
        if count <= 0:
            return ""
        return f'<history_omitted entries="{count}">older history omitted</history_omitted>'

    @staticmethod
    def _assemble(summary: str, episodes: list[str], omission_count: int, truncated: str = "") -> str:
        parts = [part for part in (summary, *episodes, truncated) if part]
        marker = ExecutionContextPolicy._omission_marker(omission_count)
        if marker:
            parts.append(marker)
        return "\n".join(parts)

    def render(
        self,
        entries: Iterable[LogEntry],
        *,
        memory_summary: object = None,
    ) -> ContextRenderResult:
        raw_entries = list(entries)
        raw_text = render_log_entries(raw_entries)
        valid_summary = _valid_summary(memory_summary)
        summary_text = ""
        if valid_summary is not None:
            summary_text = _bounded_summary(valid_summary, min(2_000, self.max_characters // 3))

        episodes = _group_episodes(raw_entries)
        recent_episodes = episodes[-self.max_recent_episodes :]
        rendered_recent = [render_log_entries(episode) for episode in recent_episodes]
        omitted_by_count = sum(len(episode) for episode in episodes[: -len(recent_episodes)]) if recent_episodes else 0

        complete_text = self._assemble(summary_text, rendered_recent, omitted_by_count)
        included = list(rendered_recent)
        included_entry_count = sum(len(episode) for episode in recent_episodes)
        truncated_count = 0

        if len(complete_text) > self.max_characters:
            included = []
            included_entry_count = 0
            for episode, rendered_episode in reversed(list(zip(recent_episodes, rendered_recent))):
                proposed = [rendered_episode, *included]
                proposed_count = included_entry_count + len(episode)
                omitted = len(raw_entries) - proposed_count
                if len(self._assemble(summary_text, proposed, omitted)) <= self.max_characters:
                    included = proposed
                    included_entry_count = proposed_count
                else:
                    break

            omission_count = len(raw_entries) - included_entry_count
            complete_text = self._assemble(summary_text, included, omission_count)

            if not included and raw_entries:
                marker = self._omission_marker(max(0, len(raw_entries) - 1))
                separators = int(bool(summary_text)) + int(bool(marker))
                available = self.max_characters - len(summary_text) - len(marker) - separators
                truncated = _truncated_entry(raw_entries[-1], max(0, available))
                truncated_count = 1
                complete_text = self._assemble(
                    summary_text,
                    [],
                    max(0, len(raw_entries) - 1),
                    truncated,
                )

        omitted_entry_count = max(0, len(raw_entries) - included_entry_count - truncated_count)
        if len(complete_text) > self.max_characters:
            complete_text = complete_text[: self.max_characters]

        metrics = ContextMetrics(
            raw_entry_count=len(raw_entries),
            raw_history_characters=len(raw_text),
            raw_history_bytes=len(raw_text.encode("utf-8")),
            rendered_characters=len(complete_text),
            rendered_bytes=len(complete_text.encode("utf-8")),
            included_episode_count=len(included),
            omitted_entry_count=omitted_entry_count,
            truncated_entry_count=truncated_count,
            summary_used=bool(summary_text),
        )
        return ContextRenderResult(text=complete_text, metrics=metrics)
