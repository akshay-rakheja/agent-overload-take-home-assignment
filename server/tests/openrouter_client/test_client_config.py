from __future__ import annotations

import asyncio
import json
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from pydantic import ValidationError

from server.config import ModelCallConfig, ModelRole, Settings
from server.openrouter_client.client import OpenRouterError, request_chat_completion
from server.services.evaluation_lab.models import TraceContext, TraceEventKind
from server.services.evaluation_lab.trace import trace_scope


class _CollectingSink:
    def __init__(self) -> None:
        self.events = []

    def emit(self, event) -> None:
        self.events.append(event)


class _FakeAsyncClient:
    response: httpx.Response
    requests: list[dict[str, object]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def post(self, url, *, headers, json, timeout):
        self.__class__.requests.append(
            {"url": url, "headers": headers, "json": json, "timeout": timeout}
        )
        return self.__class__.response


def _response(payload: dict, status_code: int = 200, headers=None) -> httpx.Response:
    request = httpx.Request("POST", "https://router.test/chat/completions")
    return httpx.Response(
        status_code,
        request=request,
        headers=headers,
        content=json.dumps(payload).encode(),
    )


def test_ordinary_settings_keep_claude_for_every_role_without_generation_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in tuple(__import__("os").environ):
        if name.startswith("OPENPOKE_") and ("MODEL" in name or "SEED" in name):
            monkeypatch.delenv(name, raising=False)

    settings = Settings()

    configs = {role: settings.model_call_config(role) for role in ModelRole}
    assert {config.model_id for config in configs.values()} == {
        "anthropic/claude-sonnet-4"
    }
    assert all(config.explicit_payload_fields() == {} for config in configs.values())


def test_role_environment_models_are_independent_and_lab_model_pins_all_roles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = {
        "OPENPOKE_INTERACTION_MODEL": "provider/interaction",
        "OPENPOKE_EXECUTION_MODEL": "provider/execution",
        "OPENPOKE_EMAIL_SEARCH_MODEL": "provider/search",
        "OPENPOKE_SUMMARIZER_MODEL": "provider/summarizer",
        "OPENPOKE_CLASSIFIER_MODEL": "provider/classifier",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    ordinary = Settings()
    assert {
        role.value: ordinary.model_call_config(role).model_id for role in ModelRole
    } == {
        "interaction": "provider/interaction",
        "execution": "provider/execution",
        "email_search": "provider/search",
        "summarizer": "provider/summarizer",
        "classifier": "provider/classifier",
    }

    monkeypatch.setenv("OPENPOKE_LAB_MODEL", "openai/gpt-4.1-mini")
    lab = Settings(
        lab_enabled=True,
        server_host="127.0.0.1",
        lab_composio_user_id="fixture-user",
    )
    configs = [lab.model_call_config(role) for role in ModelRole]
    assert {config.model_id for config in configs} == {"openai/gpt-4.1-mini"}
    assert all(
        config.explicit_payload_fields()
        == {"temperature": 0.0, "top_p": 1.0, "max_tokens": 1000}
        for config in configs
    )
    assert all(config.timeout_seconds == 60.0 for config in configs)
    assert all(config.max_retries == 0 for config in configs)


def test_seed_requires_an_explicit_compatibility_acceptance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENPOKE_LAB_MODEL", "openai/gpt-4.1-mini")
    monkeypatch.setenv("OPENPOKE_LAB_SEED", "1313")
    common = {
        "lab_enabled": True,
        "server_host": "127.0.0.1",
        "lab_composio_user_id": "fixture-user",
    }

    withheld = Settings(**common).model_call_config(ModelRole.INTERACTION)
    assert withheld.seed is None
    assert "seed" not in withheld.explicit_payload_fields()

    monkeypatch.setenv("OPENPOKE_LAB_SEED_COMPATIBLE", "true")
    accepted = Settings(**common).model_call_config(ModelRole.INTERACTION)
    assert accepted.seed == 1313
    assert accepted.explicit_payload_fields()["seed"] == 1313


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model_id": ""},
        {"model_id": "missing-provider"},
        {"model_id": "p/m", "temperature": -0.1},
        {"model_id": "p/m", "top_p": 0},
        {"model_id": "p/m", "max_tokens": 0},
        {"model_id": "p/m", "timeout_seconds": 0},
        {"model_id": "p/m", "max_retries": -1},
    ],
)
def test_invalid_model_configuration_fails_preflight(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ModelCallConfig(**kwargs)


def test_client_sends_only_explicit_compatible_fields_and_returns_raw_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = {
        "id": "generation-1",
        "model": "openai/gpt-4.1-mini",
        "provider": "OpenAI",
        "choices": [{"message": {"content": "done"}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14},
    }
    _FakeAsyncClient.requests = []
    _FakeAsyncClient.response = _response(raw)
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    config = ModelCallConfig(
        model_id="openai/gpt-4.1-mini",
        temperature=0,
        top_p=1,
        max_tokens=1000,
        timeout_seconds=60,
        max_retries=0,
    )

    result = asyncio.run(
        request_chat_completion(
            config=config,
            role=ModelRole.INTERACTION,
            messages=[{"role": "user", "content": "private prompt"}],
            api_key="fixture-key",
            base_url="https://router.test",
        )
    )

    assert result == raw
    sent = _FakeAsyncClient.requests[0]
    assert sent["json"] == {
        "model": "openai/gpt-4.1-mini",
        "messages": [{"role": "user", "content": "private prompt"}],
        "stream": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 1000,
    }
    assert sent["timeout"] == 60.0


def test_legacy_client_call_does_not_add_generation_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = {"choices": [{"message": {"content": "done"}}]}
    _FakeAsyncClient.requests = []
    _FakeAsyncClient.response = _response(raw)
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)

    asyncio.run(
        request_chat_completion(
            model="anthropic/claude-sonnet-4",
            messages=[],
            api_key="fixture-key",
            base_url="https://router.test",
        )
    )

    assert _FakeAsyncClient.requests[0]["json"] == {
        "model": "anthropic/claude-sonnet-4",
        "messages": [],
        "stream": False,
    }


def test_client_emits_sanitized_request_and_response_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw = {
        "model": "openai/gpt-4.1-mini",
        "provider": "OpenAI",
        "context_length": 1_047_576,
        "seed": 1313,
        "choices": [
            {
                "message": {
                    "content": "private output",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "fixture_tool",
                                "arguments": "{not-json",
                            }
                        }
                    ],
                }
            }
        ],
        "usage": {
            "prompt_tokens": 8,
            "completion_tokens": 2,
            "total_tokens": 10,
            "cost": 0.00012,
        },
    }
    _FakeAsyncClient.requests = []
    _FakeAsyncClient.response = _response(raw)
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    sink = _CollectingSink()
    context = TraceContext(
        run_id=uuid4(),
        turn_id=uuid4(),
        system="enhanced",
        revision="fixture",
        mode="offline",
    )
    config = ModelCallConfig(
        model_id="openai/gpt-4.1-mini",
        temperature=0,
        top_p=1,
        max_tokens=1000,
        seed=1313,
        timeout_seconds=60,
        max_retries=0,
    )

    with trace_scope(context, sink):
        asyncio.run(
            request_chat_completion(
                config=config,
                role=ModelRole.CLASSIFIER,
                messages=[{"role": "user", "content": "alice@example.test private"}],
                api_key="fixture-key",
                base_url="https://router.test",
            )
        )

    model_events = [event for event in sink.events if event.kind is TraceEventKind.MODEL_CALL]
    assert [event.payload["stage"] for event in model_events] == ["request", "response"]
    request_payload = model_events[0].payload
    assert request_payload["role"] == "classifier"
    assert request_payload["model"] == "openai/gpt-4.1-mini"
    assert request_payload["provider"] == "openai"
    assert request_payload["generation"] == {
        "max_tokens": 1000,
        "seed": 1313,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    assert "private" not in repr(model_events)
    response_payload = model_events[1].payload
    assert response_payload["provider"] == "OpenAI"
    assert response_payload["context_limit"] == 1_047_576
    assert response_payload["seed_acknowledged"] is True
    assert response_payload["malformed_tool_calls"] == 1
    assert response_payload["retry_count"] == 0
    assert response_payload["failover"] is False
    usage_event = [event for event in sink.events if event.kind is TraceEventKind.USAGE][0]
    cost_event = [event for event in sink.events if event.kind is TraceEventKind.COST][0]
    assert usage_event.payload["role"] == "classifier"
    assert usage_event.payload["prompt_tokens"] == {
        "availability": "available",
        "value": 8,
        "reason": None,
    }
    assert usage_event.payload["completion_tokens"]["value"] == 2
    assert usage_event.payload["cached_tokens"]["value"] is None
    assert cost_event.payload["provider_cost_usd"]["value"] == 0.00012


def test_client_emits_rate_limit_error_without_leaking_provider_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeAsyncClient.requests = []
    _FakeAsyncClient.response = _response(
        {"error": {"message": "alice@example.test private failure"}},
        status_code=429,
        headers={"retry-after": "2", "x-ratelimit-remaining": "0"},
    )
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    sink = _CollectingSink()
    context = TraceContext(
        run_id=uuid4(),
        turn_id=uuid4(),
        system="enhanced",
        revision="fixture",
        mode="offline",
    )

    with trace_scope(context, sink), pytest.raises(OpenRouterError):
        asyncio.run(
            request_chat_completion(
                config=ModelCallConfig(model_id="openai/gpt-4.1-mini", max_retries=0),
                role=ModelRole.EXECUTION,
                messages=[],
                api_key="fixture-key",
                base_url="https://router.test",
            )
        )

    event = sink.events[-1]
    assert event.kind is TraceEventKind.MODEL_CALL
    assert event.payload["stage"] == "error"
    assert event.payload["status_code"] == 429
    assert event.payload["rate_limit"] == {
        "remaining": "0",
        "retry_after_seconds": "2",
    }
    assert "alice" not in repr(event.payload)


def test_success_evidence_keeps_unrequested_seed_unknown_and_captures_rate_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeAsyncClient.requests = []
    _FakeAsyncClient.response = _response(
        {
            "model": "openai/gpt-4.1-mini",
            "provider": "OpenAI",
            "choices": [{"message": {"content": "done"}}],
        },
        headers={"x-ratelimit-limit": "100", "x-ratelimit-remaining": "99"},
    )
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    sink = _CollectingSink()
    context = TraceContext(
        run_id=uuid4(),
        turn_id=uuid4(),
        system="enhanced",
        revision="fixture",
        mode="offline",
    )

    with trace_scope(context, sink):
        asyncio.run(
            request_chat_completion(
                config=ModelCallConfig(model_id="openai/gpt-4.1-mini"),
                role=ModelRole.INTERACTION,
                messages=[],
                api_key="fixture-key",
                base_url="https://router.test",
            )
        )

    response_event = next(
        event
        for event in sink.events
        if event.kind is TraceEventKind.MODEL_CALL
        and event.payload["stage"] == "response"
    )
    assert response_event.payload["seed_acknowledged"] is None
    assert response_event.payload["rate_limit"] == {
        "limit": "100",
        "remaining": "99",
    }
