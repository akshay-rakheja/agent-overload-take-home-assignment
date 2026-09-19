"""Integration tests for bounded context rehydration in ExecutionAgent."""

from __future__ import annotations

import json

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


def test_migrated_identity_rehydrates_legacy_name_log_then_stable_log(tmp_path) -> None:
    execution_dir = tmp_path / "execution_agents"
    execution_dir.mkdir()
    roster_path = execution_dir / "roster.json"
    roster_path.write_text(json.dumps(["Alice", "Alice"]), encoding="utf-8")
    logs = ExecutionAgentLogStore(execution_dir)
    logs.record_request("Alice", "LEGACY-CONTEXT")
    directory = AgentDirectory(roster_path)
    first, duplicate = directory.list_records()
    logs.record_request(str(first.agent_id), "STABLE-CONTEXT")

    first_agent = ExecutionAgent(
        first.name,
        storage_key=str(first.agent_id),
        agent_id=str(first.agent_id),
        legacy_storage_key=first.legacy_storage_key,
        log_store=logs,
        directory=directory,
        context_policy=ExecutionContextPolicy(max_recent_episodes=4, max_characters=2_000),
    )
    duplicate_agent = ExecutionAgent(
        duplicate.name,
        storage_key=str(duplicate.agent_id),
        agent_id=str(duplicate.agent_id),
        legacy_storage_key=duplicate.legacy_storage_key,
        log_store=logs,
        directory=directory,
        context_policy=ExecutionContextPolicy(max_recent_episodes=4, max_characters=2_000),
    )

    first_prompt = first_agent.build_system_prompt_with_history()
    duplicate_prompt = duplicate_agent.build_system_prompt_with_history()
    assert "LEGACY-CONTEXT" in first_prompt
    assert "STABLE-CONTEXT" in first_prompt
    assert "LEGACY-CONTEXT" not in duplicate_prompt
    assert logs.read_raw_bytes("Alice")


def test_normalization_equivalent_names_keep_distinct_legacy_journals(tmp_path) -> None:
    execution_dir = tmp_path / "execution_agents"
    execution_dir.mkdir()
    roster_path = execution_dir / "roster.json"
    roster_path.write_text(json.dumps(["José", "Jose"], ensure_ascii=False), encoding="utf-8")
    logs = ExecutionAgentLogStore(execution_dir)
    logs.record_request("José", "ACCENTED-CONTEXT")
    logs.record_request("Jose", "PLAIN-CONTEXT")
    directory = AgentDirectory(roster_path)
    accented, plain = directory.list_records()

    def prompt_for(record):
        return ExecutionAgent(
            record.name,
            storage_key=str(record.agent_id),
            agent_id=str(record.agent_id),
            legacy_storage_key=record.legacy_storage_key,
            log_store=logs,
            directory=directory,
            context_policy=ExecutionContextPolicy(
                max_recent_episodes=4,
                max_characters=2_000,
            ),
        ).build_system_prompt_with_history()

    accented_prompt = prompt_for(accented)
    plain_prompt = prompt_for(plain)

    assert "ACCENTED-CONTEXT" in accented_prompt
    assert "PLAIN-CONTEXT" not in accented_prompt
    assert "PLAIN-CONTEXT" in plain_prompt
    assert "ACCENTED-CONTEXT" not in plain_prompt
