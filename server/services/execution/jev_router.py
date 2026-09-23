"""Jev Map/Reduce Router with parallel attribution, shortlist selection, and authorization guardrails."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from .activity_card import ActivityCard
from .jev_client import AgentScore, FakeJevClient, JevClient, JevRoutingDecision
from .models import AgentRecord
from .routing import RoutingAction


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False, extra="forbid", frozen=True, strict=True
    )


class JevRoutingContext(_FrozenModel):
    """Execution context and full telemetry emitted by the Jev router."""

    decision: JevRoutingDecision
    authorized_ids: tuple[str, ...] = ()
    map_scores: tuple[AgentScore, ...] = ()
    shortlist: tuple[AgentScore, ...] = ()
    partial_failures: dict[str, str] = Field(default_factory=dict)
    map_latency_ms: float = Field(default=0.0, ge=0.0)
    reduce_latency_ms: float = Field(default=0.0, ge=0.0)
    total_latency_ms: float = Field(default=0.0, ge=0.0)
    api_calls_count: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class JevRouter:
    """Orchestrate Jev Map over eligible agent cards and Reduce to a final decision."""

    def __init__(
        self,
        *,
        client: JevClient | None = None,
        max_concurrency: int = 10,
        top_k: int = 3,
        min_map_threshold: float = 0.50,
        ambiguity_margin: float = 0.15,
    ) -> None:
        self.client = client or FakeJevClient(
            min_reuse_threshold=min_map_threshold,
            ambiguity_margin=ambiguity_margin,
        )
        self.max_concurrency = max_concurrency
        self.top_k = top_k
        self.min_map_threshold = min_map_threshold
        self.ambiguity_margin = ambiguity_margin

    async def route(
        self,
        interaction_request: str,
        roster: Sequence[ActivityCard],
        explicit_binding: UUID | None = None,
    ) -> JevRoutingContext:
        """Route an interaction turn across candidate agent cards using Map/Reduce."""
        start_wall = time.perf_counter()

        # Short-circuit if an explicit binding exists and is in the roster
        if explicit_binding is not None:
            bound_card = next((c for c in roster if c.agent_id == explicit_binding), None)
            if bound_card:
                decision = JevRoutingDecision(
                    action=RoutingAction.REUSE,
                    recommended_agent_id=explicit_binding,
                    confidence=1.0,
                    winner_margin=1.0,
                    rationale=f"Explicit stable binding short-circuit to {explicit_binding}",
                    agreement_with_top_candidate=True,
                )
                return JevRoutingContext(
                    decision=decision,
                    authorized_ids=(str(explicit_binding),),
                    map_scores=(),
                    shortlist=(),
                    total_latency_ms=round((time.perf_counter() - start_wall) * 1000, 2),
                    api_calls_count=0,
                    total_tokens=0,
                )

        # Deterministic eligibility filter (e.g. omit non-active or archived cards)
        eligible_cards = [c for c in roster if getattr(c, "status", "active") != "archived"]

        if not eligible_cards:
            decision = JevRoutingDecision(
                action=RoutingAction.CREATE_NEW,
                recommended_agent_id=None,
                confidence=1.0,
                winner_margin=0.0,
                rationale="No eligible agents available in roster",
                agreement_with_top_candidate=True,
            )
            return JevRoutingContext(
                decision=decision,
                authorized_ids=(),
                map_scores=(),
                shortlist=(),
                total_latency_ms=round((time.perf_counter() - start_wall) * 1000, 2),
                api_calls_count=0,
                total_tokens=0,
            )

        # Map stage: Bounded parallel scoring
        semaphore = asyncio.Semaphore(self.max_concurrency)
        map_start = time.perf_counter()

        async def _score_with_guard(card: ActivityCard) -> AgentScore | tuple[UUID, Exception]:
            async with semaphore:
                try:
                    return await self.client.score_agent(card, interaction_request)
                except Exception as exc:
                    return (card.agent_id, exc)

        scoring_tasks = [_score_with_guard(c) for c in eligible_cards]
        results = await asyncio.gather(*scoring_tasks)
        map_latency = round((time.perf_counter() - map_start) * 1000, 2)

        valid_scores: list[AgentScore] = []
        partial_failures: dict[str, str] = {}
        total_tokens = 0
        api_calls = 0

        for res in results:
            if isinstance(res, AgentScore):
                valid_scores.append(res)
                total_tokens += res.input_tokens + res.output_tokens
                api_calls += 1
            else:
                agent_id, exc = res
                partial_failures[str(agent_id)] = str(exc)

        # Shortlist selection: filter by threshold and sort descending
        qualified_scores = [s for s in valid_scores if s.composite_score >= self.min_map_threshold]
        qualified_scores.sort(key=lambda s: s.composite_score, reverse=True)
        shortlist = tuple(qualified_scores[: self.top_k])

        # Reduce stage: Comparative decision over shortlist
        reduce_start = time.perf_counter()
        reduce_decision = await self.client.reduce_shortlist(
            shortlist, interaction_request
        )
        reduce_latency = round((time.perf_counter() - reduce_start) * 1000, 2)
        total_tokens += reduce_decision.input_tokens + reduce_decision.output_tokens
        api_calls += 1

        total_latency = round((time.perf_counter() - start_wall) * 1000, 2)

        # Authorization: Fail-closed guardrail
        # Only authorize if action is REUSE, an agent was explicitly recommended,
        # and that agent actually scored among valid candidates
        authorized_ids: tuple[str, ...] = ()
        if (
            reduce_decision.action == RoutingAction.REUSE
            and reduce_decision.recommended_agent_id is not None
        ):
            recommended_id = reduce_decision.recommended_agent_id
            if any(s.agent_id == recommended_id for s in valid_scores):
                authorized_ids = (str(recommended_id),)

        return JevRoutingContext(
            decision=reduce_decision,
            authorized_ids=authorized_ids,
            map_scores=tuple(valid_scores),
            shortlist=shortlist,
            partial_failures=partial_failures,
            map_latency_ms=map_latency,
            reduce_latency_ms=reduce_latency,
            total_latency_ms=total_latency,
            api_calls_count=api_calls,
            total_tokens=total_tokens,
        )


__all__ = [
    "JevRouter",
    "JevRoutingContext",
]
