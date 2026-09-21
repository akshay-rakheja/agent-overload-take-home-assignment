from __future__ import annotations

import asyncio
import json
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from server.config import ModelCallConfig, ModelRole, Settings, get_settings
from server.openrouter_client.client import OpenRouterError, request_chat_completion
from server.services.evaluation_lab.models import TraceContext, TraceEventKind
from server.services.evaluation_lab.trace import JsonlTraceStore, consolidate_trace, trace_scope


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


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("OPENPOKE_LAB_TEMPERATURE", "bad"),
        ("OPENPOKE_LAB_TOP_P", "nan"),
        ("OPENPOKE_LAB_MAX_TOKENS", "1.5"),
        ("OPENPOKE_LAB_TIMEOUT_SECONDS", "inf"),
        ("OPENPOKE_LAB_MAX_RETRIES", "bad"),
    ],
)
def test_explicit_invalid_lab_environment_values_fail_preflight(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises((ValueError, ValidationError)):
        Settings(
            lab_enabled=True,
            lab_model="openai/gpt-4.1-mini",
            server_host="127.0.0.1",
            lab_composio_user_id="fixture-user",
        )


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
    assert response_payload["application_retry_count"] == 0
    assert response_payload["provider_failover"] is None
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


def test_provider_routing_facts_are_distinct_from_model_author_and_app_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = {
        "model": "openai/gpt-4.1-mini",
        "provider": "Azure",
        "provider_metadata": {
            "seed": 1313,
            "context_length": 8192,
            "retry_count": 3,
            "failover": True,
            "routing": ["Azure", "OpenAI"],
        },
        "choices": [],
    }
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

    with trace_scope(context, sink):
        asyncio.run(
            request_chat_completion(
                config=ModelCallConfig(
                    model_id="openai/gpt-4.1-mini", seed=1313, max_retries=0
                ),
                role=ModelRole.INTERACTION,
                messages=[],
                api_key="fixture-key",
                base_url="https://router.test",
            )
        )

    response = next(
        event.payload
        for event in sink.events
        if event.kind is TraceEventKind.MODEL_CALL and event.payload["stage"] == "response"
    )
    assert response["requested_model_author"] == "openai"
    assert response["requested_provider_policy"] is None
    assert response["actual_provider"] == "Azure"
    assert response["application_retry_count"] == 0
    assert response["provider_retry_count"] == 3
    assert response["provider_failover"] is True
    assert tuple(response["provider_routing"]) == ("Azure", "OpenAI")
    assert response["seed_acknowledged"] is True
    assert response["context_limit"] == 8192


def test_missing_routing_and_seed_echo_remain_unknown_not_negative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeAsyncClient.response = _response(
        {"provider": "Azure", "choices": [{"message": {"content": "done"}}]}
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
                config=ModelCallConfig(model_id="openai/gpt-4.1-mini", seed=1313),
                role=ModelRole.INTERACTION,
                messages=[],
                api_key="fixture-key",
                base_url="https://router.test",
            )
        )

    response = next(
        event.payload
        for event in sink.events
        if event.kind is TraceEventKind.MODEL_CALL and event.payload["stage"] == "response"
    )
    assert response["actual_provider"] == "Azure"
    assert response["provider_failover"] is None
    assert response["provider_retry_count"] is None
    assert response["seed_acknowledged"] is None


def test_usage_observability_failure_does_not_change_successful_raw_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from server.openrouter_client import client as client_module

    raw = {"choices": [], "usage": {"cost": 10**400}}
    _FakeAsyncClient.response = _response(raw)
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    monkeypatch.setattr(
        client_module,
        "emit_usage_evidence",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("observer failed")),
    )

    result = asyncio.run(
        request_chat_completion(
            config=ModelCallConfig(model_id="openai/gpt-4.1-mini"),
            role=ModelRole.INTERACTION,
            messages=[],
            api_key="fixture-key",
            base_url="https://router.test",
        )
    )

    assert result is raw or result == raw


def test_http_error_retains_reported_usage_and_cost_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeAsyncClient.response = _response(
        {
            "error": {"message": "private"},
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 2,
                "total_tokens": 7,
                "cost": 0.01,
            },
        },
        status_code=429,
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
                config=ModelCallConfig(model_id="openai/gpt-4.1-mini"),
                role=ModelRole.EXECUTION,
                messages=[],
                api_key="fixture-key",
                base_url="https://router.test",
            )
        )

    usage = next(event.payload for event in sink.events if event.kind is TraceEventKind.USAGE)
    cost = next(event.payload for event in sink.events if event.kind is TraceEventKind.COST)
    assert usage["prompt_tokens"]["value"] == 5
    assert usage["completion_tokens"]["value"] == 2
    assert cost["provider_cost_usd"]["value"] == 0.01


def test_retryable_http_error_retains_failed_attempt_usage_before_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = [
        _response(
            {
                "error": {"message": "private"},
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 2,
                    "total_tokens": 7,
                    "cost": 0.01,
                },
            },
            status_code=429,
        ),
        _response(
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 3,
                    "total_tokens": 13,
                    "cost": 0.02,
                },
            }
        ),
    ]

    class RetryingClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, *_args, **_kwargs):
            return responses.pop(0)

    monkeypatch.setattr(httpx, "AsyncClient", RetryingClient)
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
                config=ModelCallConfig(
                    model_id="openai/gpt-4.1-mini", max_retries=1
                ),
                role=ModelRole.EXECUTION,
                messages=[],
                api_key="fixture-key",
                base_url="https://router.test",
            )
        )

    result = consolidate_trace(sink.events)
    assert result.usage.input_tokens.value == 15
    assert result.usage.output_tokens.value == 5
    assert result.cost.amount.value == 0.03


def test_real_client_usage_round_trips_jsonl_consolidation_and_trace_api(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from server.routes import api_router

    raw = {
        "choices": [{"message": {"content": "done"}}],
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 4,
            "cached_tokens": 3,
            "total_tokens": 15,
            "cost": 0.001,
        },
    }
    root = tmp_path / ".lab" / "traces"
    store = JsonlTraceStore(root)
    context = TraceContext(
        run_id=uuid4(),
        turn_id=uuid4(),
        system="enhanced",
        revision="fixture",
        mode="offline",
    )
    with monkeypatch.context() as transport_patch:
        _FakeAsyncClient.response = _response(raw)
        transport_patch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
        with trace_scope(context, store):
            returned = asyncio.run(
                request_chat_completion(
                    config=ModelCallConfig(model_id="openai/gpt-4.1-mini"),
                    role=ModelRole.INTERACTION,
                    messages=[],
                    api_key="fixture-key",
                    base_url="https://router.test",
                )
            )
    assert returned == raw
    stored = store.read(context.run_id)
    consolidated = consolidate_trace(stored)
    assert consolidated.usage.input_tokens.value == 11
    assert consolidated.usage.output_tokens.value == 4
    assert consolidated.cost.amount.value == 0.001

    settings = Settings(
        lab_enabled=True,
        lab_composio_user_id="fixture-user",
        server_host="127.0.0.1",
        lab_trace_root=root,
    )
    app = FastAPI()
    app.include_router(api_router)
    app.dependency_overrides[get_settings] = lambda: settings
    response = TestClient(app, client=("127.0.0.1", 40_000)).get(
        f"/api/v1/lab/runs/{context.run_id}/trace"
    )

    assert response.status_code == 200
    assert response.json()["result"]["usage"]["input_tokens"]["value"] == 11
    assert len(response.json()["events"]) >= 3
