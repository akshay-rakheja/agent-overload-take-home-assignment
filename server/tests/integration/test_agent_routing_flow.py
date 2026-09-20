"""Credential-free routing flow tests across directory, retrieval, and dispatch."""

from __future__ import annotations

from server.agents.interaction_agent.agent import build_candidate_context
from server.services.execution.directory import AgentDirectory
from server.services.execution.routing import RoutingAction


def test_paraphrased_alice_follow_up_resolves_existing_identity(tmp_path) -> None:
    directory = AgentDirectory(tmp_path / "roster.json")
    alice = directory.create(
        name="Alice partnership",
        purpose="Track partnership proposal messages with Alice",
        aliases=["Alice"],
    )

    context = build_candidate_context(
        "Did she answer the revised proposal?",
        "Alice was reviewing our partnership proposal.",
        directory=directory,
    )

    assert context.decision.action is RoutingAction.REUSE
    assert context.decision.agent_id == alice.agent_id


def test_novel_bob_task_recommends_creation_without_mutating_directory(tmp_path) -> None:
    directory = AgentDirectory(tmp_path / "roster.json")
    directory.create(name="Alice", purpose="Track Alice correspondence")

    context = build_candidate_context(
        "Start tracking Bob's overdue invoices",
        "",
        directory=directory,
    )

    assert context.decision.action is RoutingAction.CREATE_NEW
    assert [record.name for record in directory.list_records()] == ["Alice"]


def test_ambiguous_candidates_do_not_dispatch_or_create_implicitly(tmp_path) -> None:
    directory = AgentDirectory(tmp_path / "roster.json")
    directory.create(name="Jordan client", purpose="Manage client Jordan", aliases=["Jordan"])
    directory.create(name="Jordan candidate", purpose="Recruit candidate Jordan", aliases=["Jordan"])

    before = directory.list_records()
    context = build_candidate_context("Email Jordan the update", "", directory=directory)

    assert context.decision.action is RoutingAction.ABSTAIN
    assert directory.list_records() == before
