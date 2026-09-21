from __future__ import annotations

import importlib

import pytest

from server.agents.execution_agent.tools import gmail, triggers


APPROVED_MODEL_TOOLS = {
    "task_email_search",
    "gmail_get_contacts",
    "gmail_get_people",
    "gmail_list_drafts",
    "gmail_search_people",
    "listTriggers",
}
APPROVED_COMPOSIO_TOOLS = {
    "GMAIL_GET_PROFILE",
    "GMAIL_FETCH_EMAILS",
    "GMAIL_GET_CONTACTS",
    "GMAIL_GET_PEOPLE",
    "GMAIL_LIST_DRAFTS",
    "GMAIL_SEARCH_PEOPLE",
}
MODEL_MUTATIONS = {
    "gmail_create_draft",
    "gmail_execute_draft",
    "gmail_forward_email",
    "gmail_reply_to_thread",
    "gmail_delete_draft",
    "gmail_delete_message",
    "gmail_add_label",
    "gmail_create_filter",
    "createTrigger",
    "updateTrigger",
    "deleteTrigger",
    "send_draft",
}
COMPOSIO_MUTATIONS = {
    "GMAIL_CREATE_EMAIL_DRAFT",
    "GMAIL_SEND_DRAFT",
    "GMAIL_FORWARD_MESSAGE",
    "GMAIL_REPLY_TO_THREAD",
    "GMAIL_DELETE_DRAFT",
    "GMAIL_ADD_LABEL_TO_EMAIL",
    "GMAIL_CREATE_FILTER",
}


def _policy():
    module = importlib.import_module("server.services.evaluation_lab.policy")
    return module.LabToolPolicy()


@pytest.mark.parametrize("tool_name", sorted(APPROVED_MODEL_TOOLS))
def test_approved_model_tools_are_read_only(tool_name):
    decision = _policy().decide_model_tool(tool_name)
    assert decision.allowed is True
    assert decision.code == "allowed_read_only"


@pytest.mark.parametrize("tool_name", sorted(APPROVED_COMPOSIO_TOOLS))
def test_approved_composio_tools_are_read_only(tool_name):
    decision = _policy().decide_composio_tool(tool_name)
    assert decision.allowed is True
    assert decision.code == "allowed_read_only"


@pytest.mark.parametrize("tool_name", sorted(MODEL_MUTATIONS))
def test_named_model_mutations_are_blocked(tool_name):
    decision = _policy().decide_model_tool(tool_name)
    assert decision.allowed is False
    assert decision.code == "mutation_blocked"


@pytest.mark.parametrize("tool_name", sorted(COMPOSIO_MUTATIONS))
def test_named_composio_mutations_are_blocked(tool_name):
    decision = _policy().decide_composio_tool(tool_name)
    assert decision.allowed is False
    assert decision.code == "mutation_blocked"


@pytest.mark.parametrize(
    ("method", "tool_name"),
    [
        ("decide_model_tool", "gmail_future_bulk_mutation"),
        ("decide_model_tool", "archiveAllTriggers"),
        ("decide_composio_tool", "GMAIL_FUTURE_BULK_MUTATION"),
    ],
)
def test_future_operations_fail_closed(method, tool_name):
    decision = getattr(_policy(), method)(tool_name)
    assert decision.allowed is False
    assert decision.code == "unknown_blocked"


def test_every_current_gmail_and_trigger_tool_has_an_explicit_classification():
    schema_names = {
        schema["function"]["name"]
        for schema in [*gmail.get_schemas(), *triggers.get_schemas()]
    }
    registry_names = {
        *gmail.build_registry("fixture-agent"),
        *triggers.build_registry("fixture-agent"),
    }

    assert schema_names == registry_names
    decisions = {
        name: _policy().decide_model_tool(name) for name in schema_names
    }
    assert all(decision.code != "unknown_blocked" for decision in decisions.values())
    assert {name for name, decision in decisions.items() if decision.allowed} == (
        APPROVED_MODEL_TOOLS & schema_names
    )
