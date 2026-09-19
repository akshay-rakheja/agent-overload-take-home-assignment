"""Deterministic bounded hybrid retrieval over the Agent Directory."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID

from ...config import Settings
from .models import AgentRecord, AgentStatus, normalize_agent_text


_STOP_WORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "can",
    "did",
    "do",
    "for",
    "from",
    "has",
    "have",
    "him",
    "her",
    "i",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "our",
    "please",
    "the",
    "their",
    "them",
    "to",
    "us",
    "we",
    "what",
    "whether",
    "with",
    "you",
}


def _stem(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(
        stemmed
        for token in normalize_agent_text(value).split()
        if token not in _STOP_WORDS
        if (stemmed := _stem(token))
    )


def _contains_phrase(haystack: str, needle: str) -> bool:
    return bool(needle) and f" {needle} " in f" {haystack} "


@dataclass(frozen=True)
class RetrievalQuery:
    """Current request plus bounded interaction context used for routing."""

    text: str
    conversation_context: str = ""

    @property
    def combined_text(self) -> str:
        return " ".join(part for part in (self.text, self.conversation_context) if part).strip()


@dataclass(frozen=True)
class AgentCandidate:
    """One explainable directory candidate."""

    agent_id: UUID
    name: str
    purpose: str
    status: AgentStatus
    score: float
    score_components: dict[str, float]
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class _QueryFeatures:
    normalized: str
    tokens: frozenset[str]
    bigrams: frozenset[tuple[str, str]]


@dataclass(frozen=True)
class _RecordFeatures:
    record: AgentRecord
    exact_phrases: tuple[str, ...]
    document_tokens: frozenset[str]
    field_bigrams: frozenset[tuple[str, str]]
    sort_name: str


def _query_features(query: RetrievalQuery) -> _QueryFeatures:
    sequence = _tokens(query.combined_text)
    return _QueryFeatures(
        normalized=normalize_agent_text(query.combined_text),
        tokens=frozenset(sequence),
        bigrams=frozenset(zip(sequence, sequence[1:])),
    )


def _record_features(record: AgentRecord) -> _RecordFeatures:
    fields = (record.name, record.purpose, *record.aliases, record.memory_summary)
    normalized_fields = tuple(normalize_agent_text(field) for field in fields if field)
    field_bigrams: set[tuple[str, str]] = set()
    for field in normalized_fields:
        field_tokens = _tokens(field)
        field_bigrams.update(zip(field_tokens, field_tokens[1:]))
    return _RecordFeatures(
        record=record,
        exact_phrases=(record.normalized_name, *record.normalized_aliases),
        document_tokens=frozenset(_tokens(" ".join(fields))),
        field_bigrams=frozenset(field_bigrams),
        sort_name=record.normalized_name,
    )


RecordProvider = Iterable[AgentRecord] | Callable[[], Iterable[AgentRecord]]


class AgentRetriever:
    """Score all local records but expose only a strictly bounded working set."""

    def __init__(
        self,
        records: RecordProvider,
        *,
        now: Callable[[], datetime] | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._records = records if callable(records) else tuple(records)
        self._static_features = (
            None
            if callable(records)
            else tuple(_record_features(record) for record in self._records)
        )
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._settings = settings or Settings()

    def _get_records(self) -> tuple[AgentRecord, ...]:
        source = self._records() if callable(self._records) else self._records
        return tuple(source)

    def _get_features(self) -> tuple[_RecordFeatures, ...]:
        if self._static_features is not None:
            return self._static_features
        return tuple(_record_features(record) for record in self._get_records())

    def _score(
        self,
        query: _QueryFeatures,
        features: _RecordFeatures,
        *,
        now: datetime,
    ) -> AgentCandidate:
        record = features.record

        exact_phrase = any(
            _contains_phrase(query.normalized, phrase)
            for phrase in features.exact_phrases
            if phrase
        )
        exact_match = 0.70 if exact_phrase else 0.0

        overlap = query.tokens & features.document_tokens
        overlap_ratio = len(overlap) / max(1, len(query.tokens))
        token_overlap = min(0.35, 0.35 * overlap_ratio)

        phrase_match = 0.12 if query.bigrams & features.field_bigrams else 0.0

        age_days = max(0.0, (now - record.last_used_at).total_seconds() / 86_400)
        recency = 0.05 * max(0.0, 1.0 - min(age_days, 30.0) / 30.0)
        prior_use = 0.04 * min(1.0, math.log1p(record.use_count) / math.log(101))
        lifecycle = {
            AgentStatus.HOT: 0.0,
            AgentStatus.DORMANT: -0.03,
            AgentStatus.ARCHIVED: -0.08,
        }[record.status]

        components = {
            "exact_match": exact_match,
            "token_overlap": token_overlap,
            "phrase_match": phrase_match,
            "recency": recency,
            "prior_use": prior_use,
            "lifecycle": lifecycle,
        }
        score = max(0.0, min(1.0, sum(components.values())))

        reasons: list[str] = []
        if exact_phrase:
            reasons.append("exact name or alias phrase")
        if overlap:
            reasons.append(f"shared terms: {', '.join(sorted(overlap))}")
        if phrase_match:
            reasons.append("shared multi-word phrase")
        if recency:
            reasons.append("recently used")
        if prior_use:
            reasons.append("prior successful use")
        if record.status is not AgentStatus.HOT:
            reasons.append(f"{record.status.value} lifecycle penalty")

        return AgentCandidate(
            agent_id=record.agent_id,
            name=record.name,
            purpose=record.purpose,
            status=record.status,
            score=round(score, 6),
            score_components={name: round(value, 6) for name, value in components.items()},
            reasons=tuple(reasons) or ("weak metadata overlap",),
        )

    def retrieve(self, query: RetrievalQuery, limit: int | None = None) -> list[AgentCandidate]:
        """Return at most configured top K candidates with deterministic ordering."""

        requested_limit = self._settings.agent_retrieval_top_k if limit is None else limit
        if requested_limit < 1:
            raise ValueError("retrieval limit must be positive")
        effective_limit = min(requested_limit, self._settings.agent_retrieval_top_k)

        features = _query_features(query)
        now = self._now()
        record_features = self._get_features()
        sort_names = {
            record_feature.record.agent_id: record_feature.sort_name
            for record_feature in record_features
        }
        candidates = [
            self._score(features, record_feature, now=now)
            for record_feature in record_features
        ]
        candidates = [
            candidate
            for candidate in candidates
            if candidate.score >= self._settings.agent_retrieval_min_score
        ]
        candidates.sort(
            key=lambda candidate: (
                -candidate.score,
                sort_names[candidate.agent_id],
                str(candidate.agent_id),
            )
        )
        return candidates[:effective_limit]
