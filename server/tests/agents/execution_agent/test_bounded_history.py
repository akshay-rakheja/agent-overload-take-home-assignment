"""Integration tests for bounded context rehydration in ExecutionAgent."""

from __future__ import annotations

from server.agents.execution_agent.agent import ExecutionAgent
from server.services.execution.context_policy import ExecutionContextPolicy
from server.services.execution.directory import AgentDirectory
from server.services.execution.log_store import ExecutionAgentLogStore


def test_agent_uses_directory_summary_and_bounded_recent_log_without_mutation(tmp_path) -> None:
    directory = AgentDirectory(tmp_path / "roster.json")
    record = directory.create(
        name="Alice partnership",
        purpose="Track the Alice partnership",
        memory_summary="Alice prefers the two-year option.",
    )
    logs = ExecutionAgentLogStore(tmp_path / "logs")
    stable_key = str(record.agent_id)
    for index in range(100):
        logs.record_request(stable_key, f"request {index}")
        logs.record_agent_response(stable_key, f"response {index}")
    before = logs.read_raw_bytes(stable_key)

    agent = ExecutionAgent(
        record.name,
        storage_key=stable_key,
        agent_id=stable_key,
        log_store=logs,
        directory=directory,
        context_policy=ExecutionContextPolicy(max_recent_episodes=3, max_characters=1_200),
    )
    prompt = agent.build_system_prompt_with_history()

    assert "Alice prefers the two-year option." in prompt
    assert "request 99" in prompt
    assert "request 0" not in prompt
    assert agent.last_context_metrics.rendered_characters <= 1_200
    assert logs.read_raw_bytes(stable_key) == before


def test_agent_histories_are_isolated_by_stable_identity(tmp_path) -> None:
    directory = AgentDirectory(tmp_path / "roster.json")
    first = directory.create(name="A B", purpose="First workflow")
    second = directory.create(name="A-B", purpose="Second workflow")
    logs = ExecutionAgentLogStore(tmp_path / "logs")
    logs.record_request(str(first.agent_id), "FIRST-ONLY-CONTEXT")
    logs.record_request(str(second.agent_id), "SECOND-ONLY-CONTEXT")

    first_agent = ExecutionAgent(
        first.name,
        storage_key=str(first.agent_id),
        agent_id=str(first.agent_id),
        log_store=logs,
        directory=directory,
        context_policy=ExecutionContextPolicy(max_recent_episodes=4, max_characters=1_000),
    )
    prompt = first_agent.build_system_prompt_with_history()

    assert "FIRST-ONLY-CONTEXT" in prompt
    assert "SECOND-ONLY-CONTEXT" not in prompt
