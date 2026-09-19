"""Explicit reuse/create-new/abstain policy over retrieved candidates."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from uuid import UUID

from ...config import Settings
from .retrieval import AgentCandidate, RetrievalQuery


class RoutingAction(str, Enum):
    REUSE = "reuse"
    CREATE_NEW = "create_new"
    ABSTAIN = "abstain"


@dataclass(frozen=True)
class RoutingDecision:
    action: RoutingAction
    agent_id: UUID | None
    confidence: float
    reasons: tuple[str, ...]


class AgentRouter:
    """Convert a bounded candidate set into a safe, explainable action."""

    def __init__(self, *, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()

    def route(
        self,
        query: RetrievalQuery,
        candidates: list[AgentCandidate],
    ) -> RoutingDecision:
        del query  # The deterministic policy acts only on auditable retrieval evidence.

        if not candidates:
            return RoutingDecision(
                action=RoutingAction.CREATE_NEW,
                agent_id=None,
                confidence=1.0,
                reasons=("no candidate met the minimum relevance score",),
            )

        top = candidates[0]
        if top.score < self._settings.agent_route_reuse_threshold:
            return RoutingDecision(
                action=RoutingAction.CREATE_NEW,
                agent_id=None,
                confidence=round(1.0 - top.score, 6),
                reasons=("best candidate was below the reuse threshold",),
            )

        if len(candidates) > 1:
            runner_up = candidates[1]
            gap = top.score - runner_up.score
            if (
                runner_up.score >= self._settings.agent_route_reuse_threshold
                and gap < self._settings.agent_route_ambiguity_margin
            ):
                return RoutingDecision(
                    action=RoutingAction.ABSTAIN,
                    agent_id=None,
                    confidence=round(
                        max(0.0, 1.0 - gap / self._settings.agent_route_ambiguity_margin),
                        6,
                    ),
                    reasons=(
                        "top candidates are materially ambiguous",
                        f"score gap {gap:.3f} is below ambiguity margin",
                    ),
                )

        return RoutingDecision(
            action=RoutingAction.REUSE,
            agent_id=top.agent_id,
            confidence=top.score,
            reasons=("one candidate cleared the reuse threshold and ambiguity margin", *top.reasons),
        )
