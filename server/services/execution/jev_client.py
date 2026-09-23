"""Provider-neutral Jev client interface with deterministic Fake and TypeSafe implementations."""

from __future__ import annotations

import asyncio
import os
import re
import stat
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
from pydantic import BaseModel, ConfigDict, Field

from .activity_card import ActivityCard
from .routing import RoutingAction


class JevProviderError(RuntimeError):
    """Base error for Jev provider calls."""


class JevTimeoutError(JevProviderError):
    """Raised when a Jev request times out."""


class JevRateLimitError(JevProviderError):
    """Raised when encountering 429 rate limit."""


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False, extra="forbid", frozen=True, strict=True
    )


class AgentScore(_FrozenModel):
    """Detailed multi-factor score for an eligible agent card against a user turn."""

    agent_id: UUID
    composite_score: float = Field(ge=0.0, le=1.0)
    affinity_score: float = Field(ge=0.0, le=1.0)
    continuity_score: float = Field(ge=0.0, le=1.0)
    risk_score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    card_digest: str
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0.0)


class JevRoutingDecision(_FrozenModel):
    """Outcome of reducing the shortlist against the interaction request."""

    action: RoutingAction
    recommended_agent_id: UUID | None
    confidence: float = Field(ge=0.0, le=1.0)
    winner_margin: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str
    agreement_with_top_candidate: bool = True
    clarification_prompt: str | None = None
    split_tasks: tuple[str, ...] | None = None
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0.0)


class JevClient(ABC):
    """Provider-neutral abstraction for Jev scoring and reduction."""

    @abstractmethod
    async def score_agent(
        self, card: ActivityCard, interaction_request: str
    ) -> AgentScore:
        """Score an individual agent activity card against the user request."""
        raise NotImplementedError

    @abstractmethod
    async def reduce_shortlist(
        self, shortlist: Sequence[AgentScore], interaction_request: str
    ) -> JevRoutingDecision:
        """Reduce the candidate shortlist to a final routing decision."""
        raise NotImplementedError


class FakeJevClient(JevClient):
    """Deterministic, offline provider for unit tests and local evaluation."""

    def __init__(
        self,
        *,
        inject_timeout: bool = False,
        inject_rate_limit: bool = False,
        inject_failure: bool = False,
        min_reuse_threshold: float = 0.50,
        ambiguity_margin: float = 0.15,
        ambiguity_action: RoutingAction = RoutingAction.ABSTAIN,
    ) -> None:
        self.inject_timeout = inject_timeout
        self.inject_rate_limit = inject_rate_limit
        self.inject_failure = inject_failure
        self.min_reuse_threshold = min_reuse_threshold
        self.ambiguity_margin = ambiguity_margin
        self.ambiguity_action = ambiguity_action

    def _check_injected_failures(self) -> None:
        if self.inject_timeout:
            raise JevTimeoutError("Injected Jev timeout")
        if self.inject_rate_limit:
            raise JevRateLimitError("Injected Jev rate limit 429")
        if self.inject_failure:
            raise JevProviderError("Injected Jev provider error")

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {w.lower() for w in re.findall(r"[A-Za-z0-9_-]+", text) if len(w) > 2}

    async def score_agent(
        self, card: ActivityCard, interaction_request: str
    ) -> AgentScore:
        self._check_injected_failures()

        req_tokens = self._tokenize(interaction_request)
        if not req_tokens:
            return AgentScore(
                agent_id=card.agent_id,
                composite_score=0.1,
                affinity_score=0.1,
                continuity_score=0.1,
                risk_score=0.1,
                reasoning="Empty request tokens",
                card_digest=card.digest,
            )

        name_tokens = self._tokenize(card.name)
        entity_tokens = self._tokenize(" ".join(card.entity_claims))
        purpose_tokens = self._tokenize(f"{card.purpose} {card.durable_summary}")

        matched_tokens: set[str] = set()
        score_points = 0.0

        for rt in req_tokens:
            # Check name match (strongest signal)
            if any(rt in nt or nt in rt for nt in name_tokens):
                score_points += 2.0
                matched_tokens.add(rt)
            # Check entity match
            elif any(rt in et or et in rt for et in entity_tokens):
                score_points += 1.5
                matched_tokens.add(rt)
            # Check purpose match
            elif any(rt in pt or pt in rt for pt in purpose_tokens):
                score_points += 1.0
                matched_tokens.add(rt)

        max_possible = max(1.5, min(len(req_tokens), 3) * 1.5)
        affinity = min(1.0, max(0.05, score_points / max_possible))
        continuity = 0.85 if card.episode_count > 0 else 0.3
        risk = 0.05 if card.status in ("hot", "active") else 0.2
        composite = round(min(1.0, max(0.05, 0.75 * affinity + 0.25 * continuity - 0.1 * risk)), 4)

        reasoning = (
            f"Matched tokens: {', '.join(sorted(matched_tokens)[:4])} (affinity {affinity:.2f})"
            if matched_tokens
            else "Low token overlap"
        )

        return AgentScore(
            agent_id=card.agent_id,
            composite_score=composite,
            affinity_score=round(affinity, 4),
            continuity_score=round(continuity, 4),
            risk_score=round(risk, 4),
            reasoning=reasoning,
            card_digest=card.digest,
            input_tokens=150,
            output_tokens=30,
            latency_ms=1.5,
        )

    async def reduce_shortlist(
        self, shortlist: Sequence[AgentScore], interaction_request: str
    ) -> JevRoutingDecision:
        self._check_injected_failures()

        if not shortlist:
            return JevRoutingDecision(
                action=RoutingAction.CREATE_NEW,
                recommended_agent_id=None,
                confidence=1.0,
                winner_margin=0.0,
                rationale="Empty candidate shortlist",
                agreement_with_top_candidate=True,
                input_tokens=50,
                output_tokens=20,
                latency_ms=1.0,
            )

        sorted_candidates = sorted(shortlist, key=lambda s: s.composite_score, reverse=True)
        top = sorted_candidates[0]

        if top.composite_score < self.min_reuse_threshold:
            return JevRoutingDecision(
                action=RoutingAction.CREATE_NEW,
                recommended_agent_id=None,
                confidence=round(1.0 - top.composite_score, 4),
                winner_margin=0.0,
                rationale="Best candidate below reuse threshold",
                agreement_with_top_candidate=False,
                input_tokens=100,
                output_tokens=25,
                latency_ms=1.5,
            )

        margin = 1.0
        if len(sorted_candidates) > 1:
            runner_up = sorted_candidates[1]
            margin = round(top.composite_score - runner_up.composite_score, 4)
            if margin < self.ambiguity_margin:
                action = self.ambiguity_action
                return JevRoutingDecision(
                    action=action,
                    recommended_agent_id=None if action != RoutingAction.REUSE else top.agent_id,
                    confidence=round(1.0 - margin / self.ambiguity_margin, 4),
                    winner_margin=margin,
                    rationale=f"Winner margin {margin:.3f} below threshold {self.ambiguity_margin}",
                    agreement_with_top_candidate=False,
                    clarification_prompt="Multiple agents could fulfill this request. Please clarify." if action == RoutingAction.ABSTAIN else None,
                    input_tokens=200,
                    output_tokens=40,
                    latency_ms=2.0,
                )

        return JevRoutingDecision(
            action=RoutingAction.REUSE,
            recommended_agent_id=top.agent_id,
            confidence=top.composite_score,
            winner_margin=margin,
            rationale=f"Confident reuse of agent {top.agent_id} (score {top.composite_score:.3f}, margin {margin:.3f})",
            agreement_with_top_candidate=True,
            input_tokens=200,
            output_tokens=40,
            latency_ms=2.0,
        )


class TypeSafeJevClient(JevClient):
    """Client for TypeSafe Jev API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model_id: str | None = None,
        env_path: Path | str | None = None,
        timeout_seconds: float = 30.0,
        min_reuse_threshold: float = 0.45,
        ambiguity_margin: float = 0.05,
        ambiguity_action: RoutingAction = RoutingAction.CREATE_NEW,
    ) -> None:
        env_target = Path(env_path) if env_path else Path(".env")
        if env_target.exists():
            self.verify_env_permissions(env_target)

        resolved_key = api_key or os.environ.get("JEV_API_KEY")
        if not resolved_key:
            raise ValueError("JEV_API_KEY is not configured")

        self._api_key = resolved_key
        raw_base = base_url or os.environ.get("JEV_API_BASE_URL", "https://api.typesafe.ai/v1")
        if "#" in raw_base:
            raw_base = raw_base.split("#")[0]
        self.base_url = raw_base.strip().rstrip("/")
        self.model_id = model_id or os.environ.get("JEV_MODEL_ID", "jev-latest")
        self.timeout_seconds = timeout_seconds
        self.min_reuse_threshold = min_reuse_threshold
        self.ambiguity_margin = ambiguity_margin
        self.ambiguity_action = ambiguity_action
        self._http_client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Return a pooled httpx.AsyncClient to execute parallel JEV Map calls with low latency."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=self.timeout_seconds,
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
            )
        return self._http_client

    async def close(self) -> None:
        if self._http_client is not None and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None

    @staticmethod
    def verify_env_permissions(env_path: Path | str) -> bool:
        """Verify that .env or its resolved target has mode 0600."""
        target = Path(env_path).resolve(strict=True)
        file_mode = stat.S_IMODE(target.stat().st_mode)
        if file_mode != 0o600:
            raise PermissionError(f"{target} permissions must be 0600, found {oct(file_mode)}")
        return True

    async def score_agent(
        self, card: ActivityCard, interaction_request: str
    ) -> AgentScore:
        start_time = time.perf_counter()
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        recent_intents = [ep.request_intent for ep in card.recent_episodes[:3]]
        payload = {
            "model": self.model_id,
            "state": {
                "user_query": interaction_request,
                "agent": {
                    "name": card.name,
                    "purpose": card.purpose,
                    "status": card.status,
                    "summary": card.durable_summary,
                    "recent_episodes": recent_intents,
                },
            },
            "questions": {
                "affinity": {
                    "type": "noul",
                    "instructions": "Is this agent suitable for fulfilling the user query?",
                    "criteria": {
                        "true": "The agent specializes in this task or related domain",
                        "false": "The agent does not handle this task",
                    },
                },
                "continuity": {
                    "type": "noul",
                    "instructions": "Does using this agent provide conversational continuity for this request?",
                    "criteria": {
                        "true": "The agent handled recent related requests",
                        "false": "This is a brand new topic or domain",
                    },
                },
                "risk": {
                    "type": "noul",
                    "instructions": "Is there risk of domain pollution or context cross-talk?",
                    "criteria": {
                        "true": "High risk of cross-talk or conflicting context",
                        "false": "Low risk or clean task boundary",
                    },
                },
            },
        }

        try:
            client = await self._get_client()
            response = await client.post(
                f"{self.base_url}/systemone",
                headers=headers,
                json=payload,
            )
        except httpx.TimeoutException as exc:
            raise JevTimeoutError(f"TypeSafe Jev request timed out: {exc}") from exc
        except Exception as exc:
            raise JevProviderError(f"TypeSafe Jev request failed: {exc}") from exc

        if response.status_code == 429:
            raise JevRateLimitError("TypeSafe Jev rate limit exceeded (429)")
        if response.status_code != 200:
            raise JevProviderError(f"TypeSafe Jev HTTP {response.status_code}: {response.text}")

        data = response.json()
        answers = data.get("answers", {})
        affinity = float(answers.get("affinity", {}).get("noul", 0.1))
        continuity = float(answers.get("continuity", {}).get("noul", 0.1))
        risk = float(answers.get("risk", {}).get("noul", 0.1))

        composite = round(max(0.0, min(1.0, 0.75 * affinity + 0.25 * continuity - 0.10 * risk)), 4)
        usage = data.get("usage", {})
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return AgentScore(
            agent_id=card.agent_id,
            composite_score=composite,
            affinity_score=round(affinity, 4),
            continuity_score=round(continuity, 4),
            risk_score=round(risk, 4),
            reasoning=f"TypeSafe Jev affinity: {affinity:.2f}, continuity: {continuity:.2f}, risk: {risk:.2f}",
            card_digest=card.digest,
            latency_ms=latency_ms,
            input_tokens=int(usage.get("input_tokens", 0) or 0),
            output_tokens=int(usage.get("output_tokens", 0) or 0),
        )

    async def reduce_shortlist(
        self, shortlist: Sequence[AgentScore], interaction_request: str
    ) -> JevRoutingDecision:
        start_time = time.perf_counter()
        if not shortlist:
            return JevRoutingDecision(
                action=RoutingAction.CREATE_NEW,
                recommended_agent_id=None,
                confidence=1.0,
                winner_margin=0.0,
                rationale="Empty candidate shortlist",
                agreement_with_top_candidate=False,
            )

        sorted_candidates = sorted(shortlist, key=lambda s: s.composite_score, reverse=True)
        top = sorted_candidates[0]

        if top.composite_score < self.min_reuse_threshold:
            return JevRoutingDecision(
                action=RoutingAction.CREATE_NEW,
                recommended_agent_id=None,
                confidence=round(1.0 - top.composite_score, 4),
                winner_margin=0.0,
                rationale="Best candidate below reuse threshold",
                agreement_with_top_candidate=False,
                latency_ms=round((time.perf_counter() - start_time) * 1000, 2),
            )

        margin = 1.0
        if len(sorted_candidates) > 1:
            runner_up = sorted_candidates[1]
            margin = round(top.composite_score - runner_up.composite_score, 4)
            if margin < self.ambiguity_margin:
                action = self.ambiguity_action
                return JevRoutingDecision(
                    action=action,
                    recommended_agent_id=None if action != RoutingAction.REUSE else top.agent_id,
                    confidence=round(1.0 - margin / self.ambiguity_margin, 4),
                    winner_margin=margin,
                    rationale=f"Winner margin {margin:.3f} below threshold {self.ambiguity_margin}; {'creating specialized agent' if action == RoutingAction.CREATE_NEW else 'abstaining'} to avoid ambiguity",
                    agreement_with_top_candidate=False,
                    clarification_prompt="Multiple agents could fulfill this request. Please clarify." if action == RoutingAction.ABSTAIN else None,
                    latency_ms=round((time.perf_counter() - start_time) * 1000, 2),
                )

        return JevRoutingDecision(
            action=RoutingAction.REUSE,
            recommended_agent_id=top.agent_id,
            confidence=top.composite_score,
            winner_margin=margin,
            rationale=f"Confident reuse of agent {top.agent_id} (score {top.composite_score:.3f}, margin {margin:.3f})",
            agreement_with_top_candidate=True,
            latency_ms=round((time.perf_counter() - start_time) * 1000, 2),
        )


__all__ = [
    "AgentScore",
    "FakeJevClient",
    "JevClient",
    "JevProviderError",
    "JevRateLimitError",
    "JevRoutingDecision",
    "JevTimeoutError",
    "TypeSafeJevClient",
]
