from __future__ import annotations

import asyncio
from types import SimpleNamespace

from server.agents.execution_agent.runtime import ExecutionAgentRuntime
from server.agents.execution_agent.tools.registry import get_tool_registry, get_tool_schemas
from server.agents.interaction_agent import tools as interaction_tools
from server.config import Settings
from server.services.gmail import client as gmail_client


EXPECTED_REJECTION = {
    "error": {
        "code": "lab_mutation_blocked",
        "tool": "gmail_create_draft",
        "reason": "Mutating Gmail and trigger tools are disabled in Evaluation Lab mode.",
    }
}

ALL_MODEL_TOOLS = {
    "task_email_search",
    "gmail_create_draft",
    "gmail_execute_draft",
    "gmail_delete_draft",
    "gmail_forward_email",
    "gmail_reply_to_thread",
    "gmail_get_contacts",
    "gmail_get_people",
    "gmail_list_drafts",
    "gmail_search_people",
    "createTrigger",
    "updateTrigger",
    "listTriggers",
}


def _schema_names(schemas):
    return {schema["function"]["name"] for schema in schemas}


def test_lab_registry_exposes_only_policy_approved_tools():
    settings = Settings(
        server_host="127.0.0.1",
        lab_enabled=True,
        lab_composio_user_id="opaque-lab-user",
    )

    schema_names = _schema_names(get_tool_schemas(settings=settings))
    registry_names = set(get_tool_registry("fixture-agent", settings=settings))

    assert schema_names == registry_names
    assert schema_names == {
        "task_email_search",
        "gmail_get_contacts",
        "gmail_get_people",
        "gmail_list_drafts",
        "gmail_search_people",
        "listTriggers",
    }


def test_non_lab_registry_is_unchanged():
    settings = Settings(lab_enabled=False)

    assert _schema_names(get_tool_schemas(settings=settings)) == ALL_MODEL_TOOLS
    assert set(get_tool_registry("fixture-agent", settings=settings)) == ALL_MODEL_TOOLS


def test_runtime_blocks_direct_registry_bypass_before_callable_runs():
    called = False

    def bypass_callable(**kwargs):
        nonlocal called
        called = True
        return kwargs

    runtime = ExecutionAgentRuntime.__new__(ExecutionAgentRuntime)
    runtime.lab_enabled = True
    runtime.tool_registry = {"gmail_create_draft": bypass_callable}

    success, result = asyncio.run(
        runtime._execute_tool(
            "gmail_create_draft",
            {"recipient_email": "fixture@example.invalid", "body": "fixture"},
        )
    )

    assert success is False
    assert result == EXPECTED_REJECTION
    assert called is False


def test_runtime_allows_approved_read_callable():
    runtime = ExecutionAgentRuntime.__new__(ExecutionAgentRuntime)
    runtime.lab_enabled = True
    runtime.tool_registry = {
        "gmail_get_contacts": lambda **kwargs: {"source": "fixture", **kwargs}
    }

    success, result = asyncio.run(
        runtime._execute_tool("gmail_get_contacts", {"page_token": "fixture-page"})
    )

    assert success is True
    assert result == {"source": "fixture", "page_token": "fixture-page"}


def test_raw_composio_bypass_is_blocked_before_sdk_execution(monkeypatch):
    execute_calls = []
    fake_client = SimpleNamespace(
        client=SimpleNamespace(
            tools=SimpleNamespace(
                execute=lambda *args, **kwargs: execute_calls.append((args, kwargs))
            )
        )
    )
    monkeypatch.setattr(gmail_client, "_CLIENT", fake_client)
    monkeypatch.setattr(
        gmail_client,
        "get_settings",
        lambda: Settings(
            server_host="127.0.0.1",
            lab_enabled=True,
            lab_composio_user_id="opaque-lab-user",
        ),
    )

    result = gmail_client.execute_gmail_tool(
        "GMAIL_CREATE_EMAIL_DRAFT",
        "opaque-lab-user",
        arguments={"recipient_email": "fixture@example.invalid"},
    )

    assert result == {
        "error": {
            "code": "lab_mutation_blocked",
            "tool": "GMAIL_CREATE_EMAIL_DRAFT",
            "reason": "Mutating Gmail and trigger tools are disabled in Evaluation Lab mode.",
        }
    }
    assert execute_calls == []


def test_local_interaction_draft_is_blocked_without_recording(monkeypatch):
    class FailIfUsedLog:
        def record_reply(self, message):  # pragma: no cover - failure sentinel
            raise AssertionError(f"draft was recorded: {message}")

    monkeypatch.setattr(interaction_tools, "get_conversation_log", FailIfUsedLog)
    monkeypatch.setattr(
        interaction_tools,
        "get_settings",
        lambda: Settings(
            server_host="127.0.0.1",
            lab_enabled=True,
            lab_composio_user_id="opaque-lab-user",
        ),
    )

    result = interaction_tools.send_draft(
        "fixture@example.invalid", "Fixture subject", "Fixture body"
    )

    assert result.success is False
    assert result.payload == {
        "error": {
            "code": "lab_mutation_blocked",
            "tool": "send_draft",
            "reason": "Mutating Gmail and trigger tools are disabled in Evaluation Lab mode.",
        }
    }


def test_local_interaction_draft_is_unchanged_outside_lab(monkeypatch):
    recorded = []
    monkeypatch.setattr(
        interaction_tools,
        "get_conversation_log",
        lambda: SimpleNamespace(record_reply=recorded.append),
    )
    monkeypatch.setattr(
        interaction_tools,
        "get_settings",
        lambda: Settings(lab_enabled=False),
    )

    result = interaction_tools.send_draft(
        "fixture@example.invalid", "Fixture subject", "Fixture body"
    )

    assert result.success is True
    assert result.payload == {
        "status": "draft_recorded",
        "to": "fixture@example.invalid",
        "subject": "Fixture subject",
    }
    assert recorded == [
        "To: fixture@example.invalid\nSubject: Fixture subject\n\nFixture body"
    ]
