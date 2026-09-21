"""Privacy-safe Gmail client and email-search trace contracts."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from server.agents.execution_agent.tasks.search_email import tool as email_search
from server.config import Settings
from server.services.evaluation_lab.models import TraceContext, TraceEventKind
from server.services.evaluation_lab.trace import trace_scope
from server.services.gmail import ProcessedEmail
from server.services.gmail import client as gmail_client


class CollectingSink:
    def __init__(self) -> None:
        self.events = []

    def emit(self, event) -> None:
        self.events.append(event)


class FakeComposio:
    def __init__(self, result) -> None:
        self.execute_calls = []
        self.result = result
        self.client = SimpleNamespace(
            tools=SimpleNamespace(execute=self._execute)
        )

    def _execute(self, tool_name, *, user_id, arguments):
        self.execute_calls.append((tool_name, user_id, arguments))
        return self.result


def _trace_context() -> TraceContext:
    return TraceContext(
        run_id=uuid4(),
        turn_id=uuid4(),
        system="enhanced",
        revision="fixture-revision",
        mode="fixture",
    )


def _gmail_events(sink: CollectingSink):
    return [
        event for event in sink.events if event.kind is TraceEventKind.GMAIL_EVIDENCE
    ]


@pytest.mark.parametrize(
    ("tool_name", "policy_code"),
    [
        ("GMAIL_CREATE_EMAIL_DRAFT", "mutation_blocked"),
        ("GMAIL_FUTURE_UNCLASSIFIED", "unknown_blocked"),
    ],
)
def test_gmail_client_traces_policy_rejection_before_sdk_execution(
    monkeypatch,
    tool_name,
    policy_code,
) -> None:
    client = FakeComposio({"private": "provider payload"})
    monkeypatch.setattr(gmail_client, "_CLIENT", client)
    monkeypatch.setattr(
        gmail_client,
        "get_settings",
        lambda: Settings(
            lab_enabled=True,
            lab_composio_user_id="opaque-lab-user",
        ),
    )
    sink = CollectingSink()

    with trace_scope(_trace_context(), sink):
        result = gmail_client.execute_gmail_tool(
            tool_name,
            "opaque-lab-user",
            arguments={"body": "private body", "recipient": "private@example.invalid"},
        )

    event = _gmail_events(sink)[0]
    assert result["error"]["code"] == "lab_mutation_blocked"
    assert client.execute_calls == []
    assert event.payload == {
        "boundary": "gmail_client",
        "operation_name": tool_name,
        "stage": "rejected",
        "allowed": False,
        "policy_code": policy_code,
        "sdk_executed": False,
    }


def test_gmail_client_traces_only_sanitized_read_only_result_facts(monkeypatch) -> None:
    private_markers = (
        "private-message-id",
        "private-thread-id",
        "private@example.invalid",
        "private body",
    )
    provider_result = {
        "items": [
            {
                "message_id": private_markers[0],
                "thread_id": private_markers[1],
                "sender": private_markers[2],
                "body": private_markers[3],
            }
        ],
        "nextPageToken": "private-page-token",
    }
    client = FakeComposio(provider_result)
    monkeypatch.setattr(gmail_client, "_CLIENT", client)
    monkeypatch.setattr(
        gmail_client,
        "get_settings",
        lambda: Settings(
            lab_enabled=True,
            lab_composio_user_id="opaque-lab-user",
        ),
    )
    sink = CollectingSink()

    with trace_scope(_trace_context(), sink):
        result = gmail_client.execute_gmail_tool(
            "GMAIL_FETCH_EMAILS",
            "opaque-lab-user",
            arguments={"query": "from:private@example.invalid"},
        )

    event = _gmail_events(sink)[0]
    serialized_event = json.dumps(event.model_dump(mode="json"))
    assert result is provider_result
    assert client.execute_calls == [
        (
            "GMAIL_FETCH_EMAILS",
            "opaque-lab-user",
            {"query": "from:private@example.invalid", "user_id": "me"},
        )
    ]
    assert event.payload == {
        "boundary": "gmail_client",
        "operation_name": "GMAIL_FETCH_EMAILS",
        "stage": "completed",
        "allowed": True,
        "policy_code": "allowed_read_only",
        "sdk_executed": True,
        "result_count": 1,
        "has_more": True,
    }
    for marker in private_markers:
        assert marker not in serialized_event


def test_gmail_observation_failure_does_not_change_provider_result(monkeypatch) -> None:
    class HostileProviderMapping(dict):
        def get(self, key, default=None):  # pragma: no cover - failure sentinel
            raise RuntimeError("observation failed")

    provider_result = HostileProviderMapping({"status": "fixture"})
    client = FakeComposio(provider_result)
    monkeypatch.setattr(gmail_client, "_CLIENT", client)
    monkeypatch.setattr(
        gmail_client,
        "get_settings",
        lambda: Settings(
            lab_enabled=True,
            lab_composio_user_id="opaque-lab-user",
        ),
    )

    with trace_scope(_trace_context(), CollectingSink()):
        result = gmail_client.execute_gmail_tool(
            "GMAIL_GET_PROFILE",
            "opaque-lab-user",
        )

    assert result is provider_result


def test_email_search_traces_sanitized_facts_from_exact_processed_result(
    monkeypatch,
) -> None:
    private_markers = (
        "private-message-id",
        "private-thread-id",
        "private@example.invalid",
        "private body",
    )
    processed = ProcessedEmail(
        id=private_markers[0],
        thread_id=private_markers[1],
        query="from:private@example.invalid",
        subject="Private subject",
        sender=private_markers[2],
        recipient="recipient@example.invalid",
        timestamp=datetime(2026, 9, 21, tzinfo=timezone.utc),
        label_ids=["INBOX"],
        clean_text=private_markers[3],
        has_attachments=True,
        attachment_count=2,
        attachment_filenames=["private.pdf"],
    )
    raw_result = {"provider": "private provider payload"}
    monkeypatch.setattr(email_search, "execute_gmail_tool", lambda *args, **kwargs: raw_result)
    monkeypatch.setattr(
        email_search,
        "parse_gmail_fetch_response",
        lambda result, **kwargs: ([processed], "private-page-token"),
    )
    monkeypatch.setattr(
        email_search,
        "_LOG_STORE",
        SimpleNamespace(record_action=lambda *args, **kwargs: None),
    )
    queries = []
    emails = {}
    sink = CollectingSink()

    with trace_scope(_trace_context(), sink):
        result = asyncio.run(
            email_search._perform_search(
                arguments={"query": "from:private@example.invalid", "max_results": 5},
                queries=queries,
                emails=emails,
                composio_user_id="opaque-lab-user",
            )
        )

    event = _gmail_events(sink)[-1]
    serialized_event = json.dumps(event.model_dump(mode="json"))
    assert result.status == "success"
    assert result.result_count == 1
    assert result.messages[0].id == private_markers[0]
    assert event.payload == {
        "boundary": "email_search_task",
        "operation_name": "GMAIL_FETCH_EMAILS",
        "stage": "completed",
        "result_count": 1,
        "has_more": True,
        "attachment_count": 2,
    }
    for marker in private_markers:
        assert marker not in serialized_event
