"""Offline breadth strategies used in the comparative evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from evals.baseline import render_full_roster
from evals.materialize import EVALUATION_NOW, MaterializedCase
from evals.metrics import RoutingObservation
from server.agents.interaction_agent.agent import CandidateContext, render_agent_candidates
from server.services.execution.models import AgentRecord, AgentStatus
from server.services.execution.retrieval import AgentCandidate, AgentRetriever, RetrievalQuery
from server.services.execution.routing import AgentRouter, RoutingAction


@dataclass(frozen=True)
class StrategyResult:
    action: str
    agent_id: str | None
    ranked_agent_ids: tuple[str, ...] | None
    candidate_count: int
    prompt_text: str
    latency_ms: float


class BreadthStrategy:
    name: str

    def run(self, materialized: MaterializedCase) -> StrategyResult:
        raise NotImplementedError

    def observe(self, materialized: MaterializedCase) -> RoutingObservation:
        result = self.run(materialized)
        case = materialized.case
        return RoutingObservation(
            case_id=case.case_id,
            category=case.category,
            split=case.split,
            expected_action=case.expected_action,
            expected_agent_id=materialized.expected_agent_id,
            action=result.action,
            agent_id=result.agent_id,
            ranked_agent_ids=result.ranked_agent_ids,
            candidate_count=result.candidate_count,
            prompt_characters=len(result.prompt_text),
            prompt_bytes=len(result.prompt_text.encode("utf-8")),
            latency_ms=result.latency_ms,
        )


class CurrentFullRosterStrategy(BreadthStrategy):
    """Measurable proxy for current all-roster/exact-name behavior."""

    name = "current_full_roster_exact_name_proxy"

    def run(self, materialized: MaterializedCase) -> StrategyResult:
        started = perf_counter()
        combined = f"{materialized.case.query} {materialized.case.conversation_context}"
        matches = [record for record in materialized.records if record.name in combined]
        if len(matches) == 1:
            action = RoutingAction.REUSE.value
            agent_id = str(matches[0].agent_id)
        elif len(matches) > 1:
            action = RoutingAction.ABSTAIN.value
            agent_id = None
        else:
            action = RoutingAction.CREATE_NEW.value
            agent_id = None
        prompt = render_full_roster(record.name for record in materialized.records)
        latency_ms = (perf_counter() - started) * 1_000
        return StrategyResult(
            action=action,
            agent_id=agent_id,
            ranked_agent_ids=None,
            candidate_count=len(materialized.records),
            prompt_text=prompt,
            latency_ms=latency_ms,
        )


class RecencyOnlyStrategy(BreadthStrategy):
    """Cheap comparator: retrieve only from the five most recent hot identities."""

    name = "recency_only_top_five"

    def run(self, materialized: MaterializedCase) -> StrategyResult:
        started = perf_counter()
        hot_records = [
            record for record in materialized.records if record.status is AgentStatus.HOT
        ]
        hot_records.sort(
            key=lambda record: (-record.last_used_at.timestamp(), -record.use_count, str(record.agent_id))
        )
        cache = hot_records[:5]
        query = RetrievalQuery(
            materialized.case.query,
            materialized.case.conversation_context,
        )
        candidates = AgentRetriever(cache, now=lambda: EVALUATION_NOW).retrieve(query)
        decision = AgentRouter().route(query, candidates)
        context = CandidateContext(tuple(candidates), decision)
        prompt = render_agent_candidates(context)
        latency_ms = (perf_counter() - started) * 1_000
        return StrategyResult(
            action=decision.action.value,
            agent_id=str(decision.agent_id) if decision.agent_id else None,
            ranked_agent_ids=tuple(str(candidate.agent_id) for candidate in context.prompt_candidates),
            candidate_count=len(context.prompt_candidates),
            prompt_text=prompt,
            latency_ms=latency_ms,
        )


class HybridDirectoryStrategy(BreadthStrategy):
    name = "hybrid_directory"

    def run(self, materialized: MaterializedCase) -> StrategyResult:
        started = perf_counter()
        query = RetrievalQuery(
            materialized.case.query,
            materialized.case.conversation_context,
        )
        candidates = AgentRetriever(
            materialized.records,
            now=lambda: EVALUATION_NOW,
        ).retrieve(query)
        decision = AgentRouter().route(query, candidates)
        context = CandidateContext(tuple(candidates), decision)
        prompt = render_agent_candidates(context)
        latency_ms = (perf_counter() - started) * 1_000
        return StrategyResult(
            action=decision.action.value,
            agent_id=str(decision.agent_id) if decision.agent_id else None,
            ranked_agent_ids=tuple(str(candidate.agent_id) for candidate in context.prompt_candidates),
            candidate_count=len(context.prompt_candidates),
            prompt_text=prompt,
            latency_ms=latency_ms,
        )


def default_breadth_strategies() -> tuple[BreadthStrategy, ...]:
    return (
        CurrentFullRosterStrategy(),
        RecencyOnlyStrategy(),
        HybridDirectoryStrategy(),
    )
