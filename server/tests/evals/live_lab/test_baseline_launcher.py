from __future__ import annotations

import asyncio
import json
import os
import secrets
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import httpx

from evals.live_lab.baseline_launcher import (
    ObservationSink,
    _build_baseline_payload,
    _build_model_configs,
    wrap_async_call,
    wrap_sync_call,
)
from evals.live_lab import baseline_launcher
from evals.live_lab.baseline_observer import run_baseline_turn
from evals.live_lab.fixtures import build_fixture_manifest, materialize_baseline
from evals.live_lab.processes import ManagedProcess
from evals.live_lab.raw_observation import (
    BaselineObservation,
    BaselineTurnRequest,
    RawModelCall,
)
from server.services.evaluation_lab.baseline_adapter import adapt_baseline
from server.services.evaluation_lab.models import Availability


PROJECT_ROOT = Path(__file__).resolve().parents[4]
BASELINE_WORKTREE = PROJECT_ROOT.parent / "openpoke-evaluation-baseline"


def test_baseline_role_configs_pin_the_same_measured_generation_policy() -> None:
    configs = _build_model_configs("openai/gpt-4.1-mini")

    assert set(configs) == {
        "interaction",
        "execution",
        "email_search",
        "summarizer",
        "classifier",
    }
    assert all(
        config
        == {
            "model_id": "openai/gpt-4.1-mini",
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 1000,
            "timeout_seconds": 60.0,
            "max_retries": 0,
        }
        for config in configs.values()
    )


def test_baseline_payload_withholds_seed_until_compatibility_is_accepted() -> None:
    kwargs = {
        "model": "historical/default",
        "messages": [{"role": "user", "content": "fixture"}],
        "system": "system",
        "tools": [{"function": {"name": "fixture_tool"}}],
    }
    withheld = _build_model_configs(
        "openai/gpt-4.1-mini", seed=1313, seed_compatible=False
    )["interaction"]
    accepted = _build_model_configs(
        "openai/gpt-4.1-mini", seed=1313, seed_compatible=True
    )["interaction"]

    withheld_payload = _build_baseline_payload(kwargs, withheld)
    accepted_payload = _build_baseline_payload(kwargs, accepted)

    assert withheld_payload == {
        "model": "openai/gpt-4.1-mini",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "fixture"},
        ],
        "stream": False,
        "tools": [{"function": {"name": "fixture_tool"}}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 1000,
    }
    assert accepted_payload["seed"] == 1313


@pytest.mark.parametrize("model_id", ["/", "/model", "provider/", "missing-provider", "p m/model"])
def test_baseline_rejects_the_same_invalid_model_ids_as_enhanced(model_id: str) -> None:
    with pytest.raises(ValueError, match="provider/model"):
        _build_model_configs(model_id)


def test_baseline_public_append_redacts_adversarial_metadata_but_private_raw_is_exact(
    tmp_path: Path,
) -> None:
    sink = ObservationSink(tmp_path / "events.jsonl", tmp_path / "private")
    raw = {
        "provider": "Bearer FAKE_REVIEW_SECRET",
        "model": "alice@example.test",
        "choices": [],
    }

    async def response(**_kwargs):
        return raw

    wrapped = wrap_async_call(
        "interaction_model",
        response,
        sink,
        model_config=_build_model_configs("openai/gpt-4.1-mini")["interaction"],
    )
    returned = asyncio.run(wrapped(messages=[]))

    public = (tmp_path / "events.jsonl").read_text(encoding="utf-8")
    exported = sink.read_events()
    event = exported[-1]["model_call"]
    private = (tmp_path / "private" / f"{event['response_sha256']}.json").read_text(
        encoding="utf-8"
    )
    assert returned is raw
    assert "FAKE_REVIEW_SECRET" not in public
    assert "alice@example.test" not in public
    assert "FAKE_REVIEW_SECRET" not in json.dumps(exported)
    assert "alice@example.test" not in json.dumps(exported)
    assert "Bearer FAKE_REVIEW_SECRET" in private
    assert "alice@example.test" in private


@pytest.mark.parametrize(
    ("routing", "expected"),
    [
        (
            {
                "attempts": [
                    {
                        "provider": "azure-east",
                        "status": "rate_limited",
                        "status_code": 429,
                        "error_code": "RATE_LIMITED",
                        "error": "PRIVATE_MAILBOX_TEXT",
                        "prompt": "PRIVATE_PROMPT_TEXT",
                        "completion": "PRIVATE_OUTPUT_TEXT",
                    }
                ],
                "retry_count": 1,
                "failover": True,
            },
            {
                "attempts": [
                    {
                        "provider": "azure-east",
                        "status": "rate_limited",
                        "status_code": 429,
                        "error_code": "RATE_LIMITED",
                    }
                ],
                "retry_count": 1,
                "failover": True,
            },
        ),
        (
            [
                {
                    "provider_id": "openai-primary",
                    "status": "success",
                    "status_code": 200,
                    "completion": "PRIVATE_OUTPUT_TEXT",
                },
                "PRIVATE_MAILBOX_TEXT",
            ],
            {
                "attempts": [
                    {
                        "provider_id": "openai-primary",
                        "status": "success",
                        "status_code": 200,
                    }
                ]
            },
        ),
        ("PRIVATE_MAILBOX_TEXT", None),
    ],
)
def test_baseline_provider_routing_exports_only_typed_safe_facts_and_keeps_private_raw(
    tmp_path: Path, routing, expected
) -> None:
    sink = ObservationSink(tmp_path / "events.jsonl", tmp_path / "private")
    raw = {"choices": [], "provider_metadata": {"routing": routing}}

    async def response(**_kwargs):
        return raw

    wrapped = wrap_async_call(
        "interaction_model",
        response,
        sink,
        model_config=_build_model_configs("openai/gpt-4.1-mini")["interaction"],
    )
    returned = asyncio.run(wrapped(messages=[]))

    event = sink.read_events()[-1]["model_call"]
    private = (tmp_path / "private" / f"{event['response_sha256']}.json").read_text(
        encoding="utf-8"
    )
    public = (tmp_path / "events.jsonl").read_text(encoding="utf-8")
    assert returned is raw
    assert event["provider_routing"] == expected
    for marker in (
        "PRIVATE_MAILBOX_TEXT",
        "PRIVATE_PROMPT_TEXT",
        "PRIVATE_OUTPUT_TEXT",
    ):
        assert marker not in public
    assert json.loads(private) == raw


@pytest.mark.parametrize(
    "outcome",
    [RuntimeError("transport failed"), {"choices": []}],
)
def test_measured_baseline_summarizer_helper_makes_one_request(outcome) -> None:
    calls = []

    class HistoricalError(RuntimeError):
        pass

    async def request(**kwargs):
        calls.append(kwargs)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    call = baseline_launcher._single_request_summarizer(request, HistoricalError)
    prompt = SimpleNamespace(system_prompt="system", messages=[])

    with pytest.raises((RuntimeError, HistoricalError)):
        asyncio.run(call(prompt, "openai/gpt-4.1-mini", "fixture-key"))

    assert len(calls) == 1


def test_baseline_transport_preserves_nested_provider_metadata_headers_and_timeout_cause(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    request = httpx.Request("POST", "https://fixture.invalid/chat/completions")

    class FakeClient:
        outcome = httpx.Response(
            200,
            request=request,
            json={
                "model": "openai/gpt-4.1-mini",
                "provider": "Azure",
                "provider_metadata": {
                    "seed": 1313,
                    "context_length": 8192,
                    "retry_count": 3,
                    "failover": True,
                    "routing": {
                        "attempts": [
                            {"provider_id": "Azure"},
                            {"provider_id": "OpenAI"},
                        ]
                    },
                },
                "choices": [],
            },
            headers={"retry-after": "2", "x-ratelimit-remaining": "0"},
        )

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, *_args, **_kwargs):
            if isinstance(self.outcome, BaseException):
                raise self.outcome
            return self.outcome

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(
        baseline_launcher,
        "_BASELINE_MODEL_CONFIGS",
        _build_model_configs(
            "openai/gpt-4.1-mini", seed=1313, seed_compatible=True
        ),
    )
    sink = ObservationSink(tmp_path / "events.jsonl", tmp_path / "private")
    wrapped = wrap_async_call(
        "interaction_model",
        baseline_launcher._transport("https://fixture.invalid", "interaction_model"),
        sink,
        model_config=baseline_launcher._BASELINE_MODEL_CONFIGS["interaction"],
    )

    raw = asyncio.run(wrapped(messages=[], api_key="fixture"))
    evidence = sink.read_events()[-1]["model_call"]
    assert raw["provider"] == "Azure"
    assert evidence["requested_model_author"] == "openai"
    assert evidence["requested_provider_policy"] is None
    assert evidence["actual_provider"] == "Azure"
    assert evidence["provider_retry_count"] == 3
    assert evidence["provider_failover"] is True
    assert evidence["provider_routing"] == {
        "attempts": [
            {"provider_id": "Azure"},
            {"provider_id": "OpenAI"},
        ]
    }
    assert evidence["seed_acknowledged"] is True
    assert evidence["context_limit"] == 8192
    assert evidence["rate_limit"] == {
        "remaining": "0",
        "retry_after_seconds": "2",
    }

    timeout = httpx.ReadTimeout("fixture")
    FakeClient.outcome = timeout
    with pytest.raises(Exception) as caught:
        asyncio.run(wrapped(messages=[], api_key="fixture"))
    failed = sink.read_events()[-1]["model_call"]
    assert caught.value.__cause__ is timeout
    assert failed["timeout"] is True


def test_baseline_transport_reports_application_retries_separately(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    request = httpx.Request("POST", "https://fixture.invalid/chat/completions")

    class FakeClient:
        outcomes = [
            httpx.ReadTimeout("retryable"),
            httpx.Response(200, request=request, json={"choices": []}),
        ]

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, *_args, **_kwargs):
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    configs = _build_model_configs("openai/gpt-4.1-mini")
    configs["interaction"]["max_retries"] = 1
    monkeypatch.setattr(baseline_launcher, "_BASELINE_MODEL_CONFIGS", configs)

    sink = ObservationSink(tmp_path / "events.jsonl", tmp_path / "private")
    wrapped = wrap_async_call(
        "interaction_model",
        baseline_launcher._transport("https://fixture.invalid", "interaction_model"),
        sink,
        model_config=configs["interaction"],
    )

    asyncio.run(wrapped(messages=[], api_key="fixture"))
    evidence = sink.read_events()[-1]["model_call"]

    assert evidence["application_retry_count"] == 1
    assert evidence["provider_retry_count"] is None
    assert evidence["provider_failover"] is None


@pytest.mark.parametrize(
    "terminal",
    ["success", "charged_error", "timeout", "cancellation", "keyboard_interrupt"],
)
def test_baseline_transport_preserves_every_retry_attempt_without_double_counting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, terminal: str
) -> None:
    request = httpx.Request("POST", "https://fixture.invalid/chat/completions")

    def charged_response(status: int, cost: float) -> httpx.Response:
        return httpx.Response(
            status,
            request=request,
            headers={"retry-after": "2"} if status == 429 else {},
            json={
                "error": {"code": status} if status >= 400 else None,
                "choices": [],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 3,
                    "total_tokens": 5,
                    "cost": cost,
                },
            },
        )

    terminal_error: BaseException | None = None
    outcomes: list[httpx.Response | BaseException] = [charged_response(429, 0.01)]
    if terminal == "success":
        outcomes.append(charged_response(200, 0.02))
    elif terminal == "charged_error":
        outcomes.append(charged_response(400, 0.02))
    elif terminal == "timeout":
        outcomes.append(httpx.ReadTimeout("fixture"))
    elif terminal == "cancellation":
        terminal_error = asyncio.CancelledError("fixture")
        outcomes.append(terminal_error)
    else:
        terminal_error = KeyboardInterrupt("fixture")
        outcomes.append(terminal_error)

    clock = {"now": 100, "steps": [7, 11]}

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, *_args, **_kwargs):
            outcome = outcomes.pop(0)
            clock["now"] += clock["steps"].pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(baseline_launcher.time, "perf_counter_ns", lambda: clock["now"])
    configs = _build_model_configs("openai/gpt-4.1-mini")
    configs["interaction"]["max_retries"] = 1
    monkeypatch.setattr(baseline_launcher, "_BASELINE_MODEL_CONFIGS", configs)

    class SlowObservationSink(ObservationSink):
        def append(self, event):
            clock["now"] += 1_000_000
            return super().append(event)

    sink = SlowObservationSink(tmp_path / "events.jsonl", tmp_path / "private")
    wrapped = wrap_async_call(
        "interaction_model",
        baseline_launcher._transport("https://fixture.invalid", "interaction_model"),
        sink,
        model_config=configs["interaction"],
    )

    if terminal == "success":
        asyncio.run(wrapped(messages=[], api_key="fixture"))
    else:
        with pytest.raises(BaseException) as caught:
            asyncio.run(wrapped(messages=[], api_key="fixture"))
        if terminal_error is not None:
            assert caught.value is terminal_error

    events = [event for event in sink.read_events() if event["kind"] == "model_call"]
    calls = [event["model_call"] for event in events]
    expected_statuses = {
        "success": [429, 200],
        "charged_error": [429, 400],
        "timeout": [429, None],
        "cancellation": [429, None],
        "keyboard_interrupt": [429, None],
    }
    expected_costs = {
        "success": [0.01, 0.02],
        "charged_error": [0.01, 0.02],
        "timeout": [0.01, None],
        "cancellation": [0.01, None],
        "keyboard_interrupt": [0.01, None],
    }
    assert len(calls) == 2
    assert [call["attempt"] for call in calls] == [0, 1]
    assert calls[0]["call_id"] == calls[1]["call_id"]
    assert [call["elapsed_ms"] for call in calls] == [0.000007, 0.000011]
    assert [call["status_code"] for call in calls] == expected_statuses[terminal]
    assert [call["usage"]["provider_cost_usd"]["value"] for call in calls] == expected_costs[
        terminal
    ]
    assert calls[0]["rate_limit"] == {"retry_after_seconds": "2"}
    assert calls[0]["error_type"] == "HTTPStatusError"
    assert calls[1]["timeout"] is (terminal == "timeout")
    if terminal_error is not None:
        assert calls[1]["error_type"] == type(terminal_error).__name__
    first_private = json.loads(
        (tmp_path / "private" / f"{calls[0]['response_sha256']}.json").read_text(
            encoding="utf-8"
        )
    )
    assert first_private["usage"]["cost"] == 0.01

    if terminal in {"timeout", "cancellation", "keyboard_interrupt"}:
        raw_calls = tuple(RawModelCall.model_validate(call) for call in calls)
        result = adapt_baseline(
            BaselineObservation(
                run_id="fixture",
                prompt_xml_sha256="fixture",
                prompt_characters=0,
                exposed_names=(),
                roster_before=(),
                roster_after=(),
                journal_hashes_before={},
                journal_hashes_after={},
                inferred_action="unobservable",
                inferred_name=None,
                inference_reason="terminal transport failure",
                final_response=None,
                raw_model_calls=raw_calls,
                errors=(),
            )
        )
        assert result.usage.input_tokens.availability is Availability.UNAVAILABLE
        assert result.usage.known_input_tokens_subtotal.value == 2
        assert result.usage.known_output_tokens_subtotal.value == 3
        assert result.usage.known_total_tokens_subtotal.value == 5
        assert result.cost.amount.availability is Availability.UNAVAILABLE
        assert result.cost.known_amount_subtotal.value == 0.01


@pytest.mark.parametrize(
    ("status_code", "terminal_error"),
    [
        (200, asyncio.CancelledError("fixture")),
        (429, KeyboardInterrupt("fixture")),
    ],
)
def test_baseline_response_parsing_base_exception_records_exact_attempt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    status_code: int,
    terminal_error: BaseException,
) -> None:
    clock = {"now": 100}

    class InterruptedJsonResponse(httpx.Response):
        def json(self, **_kwargs):
            clock["now"] += 13
            raise terminal_error

    response = InterruptedJsonResponse(
        status_code,
        request=httpx.Request("POST", "https://fixture.invalid/chat/completions"),
        content=b"fixture",
    )

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, *_args, **_kwargs):
            return response

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(baseline_launcher.time, "perf_counter_ns", lambda: clock["now"])
    configs = _build_model_configs("openai/gpt-4.1-mini")
    monkeypatch.setattr(baseline_launcher, "_BASELINE_MODEL_CONFIGS", configs)
    sink = ObservationSink(tmp_path / "events.jsonl", tmp_path / "private")
    wrapped = wrap_async_call(
        "interaction_model",
        baseline_launcher._transport("https://fixture.invalid", "interaction_model"),
        sink,
        model_config=configs["interaction"],
    )

    with pytest.raises(BaseException) as caught:
        asyncio.run(wrapped(messages=[], api_key="fixture"))

    assert caught.value is terminal_error
    calls = [event["model_call"] for event in sink.read_events()]
    assert len(calls) == 1
    assert calls[0]["attempt"] == 0
    assert calls[0]["status_code"] == status_code
    assert calls[0]["error_type"] == type(terminal_error).__name__
    assert calls[0]["transport_error_type"] == type(terminal_error).__name__
    assert calls[0]["elapsed_ms"] == 0.000013
    assert calls[0]["usage"]["provider_cost_usd"]["value"] is None


def test_baseline_error_response_retains_reported_usage_for_reconciliation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    request = httpx.Request("POST", "https://fixture.invalid/chat/completions")

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, *_args, **_kwargs):
            return httpx.Response(
                400,
                request=request,
                json={
                    "error": {"code": 400},
                    "usage": {
                        "prompt_tokens": 5,
                        "completion_tokens": 2,
                        "total_tokens": 7,
                        "cost": 0.01,
                    },
                },
            )

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    configs = _build_model_configs("openai/gpt-4.1-mini")
    monkeypatch.setattr(baseline_launcher, "_BASELINE_MODEL_CONFIGS", configs)
    sink = ObservationSink(tmp_path / "events.jsonl", tmp_path / "private")
    wrapped = wrap_async_call(
        "interaction_model",
        baseline_launcher._transport("https://fixture.invalid", "interaction_model"),
        sink,
        model_config=configs["interaction"],
    )

    with pytest.raises(Exception):
        asyncio.run(wrapped(messages=[], api_key="fixture"))
    usage = sink.read_events()[-1]["model_call"]["usage"]

    assert usage["prompt_tokens"] == {
        "availability": "available",
        "reason": None,
        "value": 5,
    }
    assert usage["provider_cost_usd"] == {
        "availability": "available",
        "reason": None,
        "value": 0.01,
    }


def test_baseline_total_mismatch_warning_is_retained(tmp_path: Path) -> None:
    sink = ObservationSink(tmp_path / "events.jsonl", tmp_path / "private")

    async def response(**_kwargs):
        return {
            "choices": [],
            "usage": {
                "prompt_tokens": 4,
                "completion_tokens": 1,
                "total_tokens": 99,
            },
        }

    wrapped = wrap_async_call(
        "interaction_model",
        response,
        sink,
        model_config=_build_model_configs("openai/gpt-4.1-mini")["interaction"],
    )
    asyncio.run(wrapped(messages=[]))

    assert sink.read_events()[-1]["model_call"]["usage_warnings"] == [
        "total_tokens mismatch: provider=99 calculated=5"
    ]


def test_baseline_phase_installer_wraps_real_total_and_gmail_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 100}
    events = []

    class Sink:
        def append(self, event):
            clock["now"] += 1_000_000
            events.append(event)

    class Runtime:
        async def execute(self, value):
            clock["now"] += 7
            return value

    gmail_module = SimpleNamespace()

    def execute_gmail_tool(value):
        clock["now"] += 11
        return value

    gmail_module.execute_gmail_tool = execute_gmail_tool
    runtime_module = SimpleNamespace(InteractionAgentRuntime=Runtime)
    monkeypatch.setattr(baseline_launcher.time, "perf_counter_ns", lambda: clock["now"])

    baseline_launcher._install_phase_wrappers(
        Sink(),
        interaction_runtime=runtime_module,
        gmail_modules=(gmail_module,),
    )

    assert asyncio.run(Runtime().execute("turn")) == "turn"
    assert gmail_module.execute_gmail_tool("gmail") == "gmail"
    assert [(event["phase"], event["elapsed_ns"]) for event in events] == [
        ("total_run", 7),
        ("gmail_tool", 11),
    ]


def test_baseline_phase_wrapper_emits_on_cancellation_and_reraises_exact_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 10}
    events = []
    error = asyncio.CancelledError()

    class Sink:
        def append(self, event):
            clock["now"] += 1_000_000
            events.append(event)

    async def cancelled():
        clock["now"] += 5
        raise error

    monkeypatch.setattr(baseline_launcher.time, "perf_counter_ns", lambda: clock["now"])
    wrapped = baseline_launcher.wrap_async_phase_call("total_run", cancelled, Sink())

    with pytest.raises(asyncio.CancelledError) as caught:
        asyncio.run(wrapped())

    assert caught.value is error
    assert events[0]["phase"] == "total_run"
    assert events[0]["elapsed_ns"] == 5
    assert events[0]["error_type"] == "CancelledError"


class _FailingSink:
    def __init__(self, *, fail_store: bool = False, fail_append: bool = False) -> None:
        self.fail_store = fail_store
        self.fail_append = fail_append

    def store_private(self, value):
        if self.fail_store:
            raise OSError("private storage unavailable")
        return "0" * 64

    def append(self, event):
        if self.fail_append:
            raise OSError("event storage unavailable")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _fake_response_queue(agent_name: str) -> list[dict]:
    return [
        {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "dispatch-1",
                                "type": "function",
                                "function": {
                                    "name": "send_message_to_agent",
                                    "arguments": json.dumps(
                                        {"agent_name": agent_name, "instructions": "inspect fixture"}
                                    ),
                                },
                            }
                        ],
                    }
                }
            ]
        },
        {"choices": [{"message": {"content": "fixture execution complete"}}]},
        {"choices": [{"message": {"content": "fixture turn complete"}}]},
    ]


_FAKE_SERVER = r'''
import json, socket, sys, time
host, port, queue_path = "127.0.0.1", int(sys.argv[1]), sys.argv[2]
queue = json.loads(open(queue_path, encoding="utf-8").read())
last_response = queue[-1]
index = 0
server = socket.socket()
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((host, port))
server.listen()
while True:
    connection, _ = server.accept()
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = connection.recv(65536)
        if not chunk:
            break
        data += chunk
    head, _, body = data.partition(b"\r\n\r\n")
    first = head.split(b"\r\n", 1)[0]
    length = 0
    for line in head.split(b"\r\n")[1:]:
        if line.lower().startswith(b"content-length:"):
            length = int(line.split(b":", 1)[1].strip())
    while len(body) < length:
        body += connection.recv(65536)
    if first.startswith(b"GET "):
        payload = b"{}"
        status = b"204 No Content"
    else:
        request_payload = json.loads(body[:length] or b"{}")
        system_text = ""
        non_system_text = ""
        for message in request_payload.get("messages", []):
            if message.get("role") == "system":
                system_text = str(message.get("content", ""))
            else:
                non_system_text += str(message.get("content", ""))
        if "Agent Name:" in system_text:
            component = "execution"
        elif "<new_agent_message>" in non_system_text:
            component = "interaction_agent"
        else:
            component = "interaction_user"
        match = next(
            (item for item in queue if item.get("_component") in (None, component)),
            last_response,
        )
        response = match
        if not response.get("_repeat") and response in queue:
            queue.remove(response)
        time.sleep(float(response.get("_delay", 0)))
        status = str(response.get("_status", "200 OK")).encode()
        payload = json.dumps(response.get("_body", response)).encode()
    headers = (
        b"HTTP/1.1 " + status + b"\r\nContent-Type: application/json\r\n"
        + b"Content-Length: " + str(len(payload)).encode() + b"\r\nConnection: close\r\n\r\n"
    )
    try:
        connection.sendall(headers + payload)
    except OSError:
        pass
    connection.close()
'''


def _process_env() -> dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(PROJECT_ROOT),
        "PYTHONDONTWRITEBYTECODE": "1",
        "NO_PROXY": "127.0.0.1,localhost",
    }


def _run_real_turn(
    tmp_path: Path,
    *,
    size: int,
    selected_name: str,
    responses: list[dict] | None = None,
    timeout_seconds: float = 10,
    late_grace_seconds: float = 0,
    execution_timeout_seconds: float | None = None,
    run_second_turn: bool = False,
):
    manifest = build_fixture_manifest(seed=1313, roster_size=size)
    data_dir = tmp_path / "fixture" / "server" / "data"
    materialize_baseline(manifest, data_dir)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    queue_path = tmp_path / "responses.json"
    queue_path.write_text(
        json.dumps(responses or _fake_response_queue(selected_name)),
        encoding="utf-8",
    )
    fake_port = _free_port()
    readiness_nonce = secrets.token_hex(16)
    fake = ManagedProcess(
        argv=[sys.executable, "-c", _FAKE_SERVER, str(fake_port), str(queue_path)],
        env=_process_env(),
        cwd=tmp_path,
        pid_file=tmp_path / "fake.pid",
        stdout_path=tmp_path / "fake.out",
        stderr_path=tmp_path / "fake.err",
    )
    baseline = ManagedProcess(
        argv=[
            sys.executable,
            "-m",
            "evals.live_lab.baseline_launcher",
            "--worktree",
            str(BASELINE_WORKTREE),
            "--host",
            "127.0.0.1",
            "--port",
            "8001",
            "--data-dir",
            str(data_dir),
            "--run-dir",
            str(run_dir),
            "--fake-base-url",
            f"http://127.0.0.1:{fake_port}/v1",
            "--readiness-nonce",
            readiness_nonce,
        ]
        + (
            ["--execution-timeout-seconds", str(execution_timeout_seconds)]
            if execution_timeout_seconds is not None
            else []
        ),
        env=_process_env(),
        cwd=PROJECT_ROOT,
        pid_file=tmp_path / "baseline.pid",
        stdout_path=tmp_path / "baseline.out",
        stderr_path=tmp_path / "baseline.err",
        private_dir=run_dir / "private" / "process-streams",
        readiness_host="127.0.0.1",
        readiness_port=8001,
        readiness_nonce=readiness_nonce,
    )
    fake.start()
    fake.wait_ready(f"http://127.0.0.1:{fake_port}/health", timeout=5)
    baseline.start()
    try:
        # A cold checkout can spend over a minute importing individual historical
        # modules from the host filesystem. This is one bounded launch, not a retry.
        baseline.wait_ready(
            f"http://127.0.0.1:8001/__live_lab_ready__/{readiness_nonce}",
            timeout=120,
        )
        observation = asyncio.run(
            run_baseline_turn(
                BaselineTurnRequest(
                    run_id=f"size-{size}",
                    data_dir=str(data_dir),
                    event_path=str(run_dir / "events.jsonl"),
                    user_message="inspect the fixture",
                    timeout_seconds=timeout_seconds,
                    late_grace_seconds=late_grace_seconds,
                )
            )
        )
        if run_second_turn:
            second = asyncio.run(
                run_baseline_turn(
                    BaselineTurnRequest(
                        run_id=f"size-{size}-second",
                        data_dir=str(data_dir),
                        event_path=str(run_dir / "events.jsonl"),
                        user_message="second turn must not consume late work",
                        timeout_seconds=0.2,
                    )
                )
            )
            return manifest, observation, second, run_dir
        return manifest, observation, run_dir
    finally:
        baseline.stop(timeout=5)
        fake.stop(timeout=5)


def test_sync_wrapper_calls_original_once_and_returns_same_object(tmp_path: Path) -> None:
    sink = ObservationSink(tmp_path / "events.jsonl", tmp_path / "private")
    returned = [{"role": "user", "content": "private mailbox text"}]
    calls = 0

    def original(value):
        nonlocal calls
        calls += 1
        return returned

    wrapped = wrap_sync_call("interaction_prompt", original, sink)

    assert wrapped("input") is returned
    assert calls == 1
    assert "private mailbox text" not in (tmp_path / "events.jsonl").read_text()
    assert any((tmp_path / "private").iterdir())


def test_async_wrapper_calls_original_once_and_reraises_same_error(tmp_path: Path) -> None:
    sink = ObservationSink(tmp_path / "events.jsonl", tmp_path / "private")
    error = RuntimeError("context limit: private mailbox body")
    calls = 0

    async def original(**kwargs):
        nonlocal calls
        calls += 1
        raise error

    wrapped = wrap_async_call("execution_model", original, sink)

    with pytest.raises(RuntimeError) as caught:
        asyncio.run(wrapped(model="fake-model", messages=[]))

    assert caught.value is error
    assert calls == 1
    exported = (tmp_path / "events.jsonl").read_text()
    assert "private mailbox body" not in exported
    assert "error_sha256" in exported


def test_tool_wrapper_exports_name_but_hashes_sensitive_arguments(tmp_path: Path) -> None:
    sink = ObservationSink(tmp_path / "events.jsonl", tmp_path / "private")
    wrapped = wrap_sync_call("interaction_tool", lambda *_: object(), sink)

    wrapped(
        "send_message_to_agent",
        {"agent_name": "Known", "instructions": "private mailbox body"},
    )
    event = sink.read_events()[0]

    assert event["tool_call"]["arguments"]["agent_name"] == "Known"
    assert event["tool_call"]["arguments"]["instructions"]["characters"] == 20
    assert "private mailbox body" not in (tmp_path / "events.jsonl").read_text()


def test_interaction_prompt_event_has_full_roster_exposure(tmp_path: Path) -> None:
    sink = ObservationSink(tmp_path / "events.jsonl", tmp_path / "private")
    names = tuple(f"Agent {index:04d}" for index in range(1000))
    prompt = "<active_agents>\n" + "\n".join(f'<agent name="{name}" />' for name in names) + "\n</active_agents>"
    wrapped = wrap_sync_call("interaction_prompt", lambda *_: [{"role": "user", "content": prompt}], sink)

    wrapped("turn", "history")
    events = sink.read_events()

    assert events[0]["exposed_names"] == list(names)
    assert events[0]["prompt_characters"] == len(prompt)
    assert events[0]["prompt_sha256"]


def test_execution_prompt_records_unbounded_history_size(tmp_path: Path) -> None:
    sink = ObservationSink(tmp_path / "events.jsonl", tmp_path / "private")
    short = wrap_sync_call("execution_prompt", lambda: "system\n# Execution History\nold", sink)
    long = wrap_sync_call("execution_prompt", lambda: "system\n# Execution History\n" + "entry\n" * 10_000, sink)

    short()
    long()
    events = sink.read_events()

    assert events[1]["prompt_characters"] > events[0]["prompt_characters"]
    assert events[1]["prompt_characters"] > 50_000


def test_async_pre_call_observation_failure_does_not_prevent_original() -> None:
    returned = object()
    calls = 0

    async def original(**kwargs):
        nonlocal calls
        calls += 1
        return returned

    wrapped = wrap_async_call("interaction_model", original, _FailingSink(fail_store=True))

    assert asyncio.run(wrapped(model="fake", messages=[])) is returned
    assert calls == 1


def test_async_success_observation_failure_does_not_replace_return() -> None:
    returned = {"choices": []}
    wrapped = wrap_async_call(
        "interaction_model",
        lambda **kwargs: asyncio.sleep(0, result=returned),
        _FailingSink(fail_append=True),
    )

    assert asyncio.run(wrapped(model="fake", messages=[])) is returned


def test_async_observation_failure_does_not_replace_original_exception() -> None:
    original_error = RuntimeError("original failure")

    async def original(**kwargs):
        raise original_error

    wrapped = wrap_async_call("interaction_model", original, _FailingSink(fail_store=True))

    with pytest.raises(RuntimeError) as caught:
        asyncio.run(wrapped(model="fake", messages=[]))
    assert caught.value is original_error


def test_sync_observation_failure_does_not_replace_return_or_exception() -> None:
    returned = object()
    assert wrap_sync_call(
        "interaction_prompt",
        lambda: returned,
        _FailingSink(fail_store=True),
    )() is returned

    original_error = RuntimeError("sync original failure")

    def fail():
        raise original_error

    with pytest.raises(RuntimeError) as caught:
        wrap_sync_call("interaction_prompt", fail, _FailingSink(fail_append=True))()
    assert caught.value is original_error


@pytest.mark.parametrize(
    ("size", "selection", "expected_action"),
    [
        (10, "AI Video Newsletter Curator", "reuse"),
        (100, "Calendar Deadline Tracker", "create_new"),
        (500, "Instagram Security Monitor", "reuse"),
        (1000, "Instagram Engagement Digest", "reuse"),
    ],
)
def test_real_historical_path_exposes_full_roster_and_infers_action(
    tmp_path: Path,
    size: int,
    selection: str,
    expected_action: str,
) -> None:
    manifest, observation, run_dir = _run_real_turn(
        tmp_path,
        size=size,
        selected_name=selection,
    )
    events = ObservationSink(run_dir / "events.jsonl", run_dir / "private").read_events()

    assert observation.exposed_names == tuple(agent.name for agent in manifest.agents)
    assert observation.inferred_action == expected_action
    assert observation.inferred_name == selection
    assert observation.final_response in {"fixture execution complete", "fixture turn complete"}
    assert any(
        event.get("kind") == "phase_timing" and event.get("phase") == "total_run"
        for event in events
    )
    assert any(timing.phase == "total_run" for timing in observation.raw_phase_timings)
    assert not observation.errors


def test_real_historical_execution_prompt_keeps_complete_seeded_history(tmp_path: Path) -> None:
    _, observation, run_dir = _run_real_turn(
        tmp_path,
        size=10,
        selected_name="AI Video Newsletter Curator",
    )
    events = ObservationSink(run_dir / "events.jsonl", run_dir / "private").read_events()
    execution_prompt = next(event for event in events if event["kind"] == "execution_prompt")
    private_prompt = (run_dir / "private" / f"{execution_prompt['prompt_sha256']}.json").read_text()

    assert observation.inferred_action == "reuse"
    assert execution_prompt["prompt_characters"] > 500_000
    assert "DEPTH-START-SENTINEL" in private_prompt
    assert "DEPTH-END-SENTINEL" in private_prompt


def test_real_historical_context_limit_is_preserved_as_failure(tmp_path: Path) -> None:
    responses = [
        {
            "_status": "400 Bad Request",
            "_body": {"error": "context length exceeded"},
        }
    ]

    _, observation, _ = _run_real_turn(
        tmp_path,
        size=10,
        selected_name="AI Video Newsletter Curator",
        responses=responses,
        timeout_seconds=0.3,
    )

    assert observation.inferred_action == "unobservable"
    assert observation.final_response is None
    assert {error.code for error in observation.errors} == {"timeout", "model_error"}
    assert observation.raw_model_calls[0].error_type == "OpenRouterError"


def test_real_historical_late_response_is_not_promoted_to_success(tmp_path: Path) -> None:
    responses = _fake_response_queue("AI Video Newsletter Curator")
    responses[0] = {"_delay": 0.35, "_body": responses[0]}

    _, observation, _ = _run_real_turn(
        tmp_path,
        size=10,
        selected_name="AI Video Newsletter Curator",
        responses=responses,
        timeout_seconds=0.1,
        late_grace_seconds=2,
    )

    assert observation.inferred_action == "unobservable"
    assert observation.final_response is not None
    assert {error.code for error in observation.errors} >= {"timeout", "late_response"}


def test_late_failed_turn_taints_context_and_cannot_satisfy_next_turn(tmp_path: Path) -> None:
    responses = _fake_response_queue("AI Video Newsletter Curator")
    responses[0] = {"_delay": 0.35, "_body": responses[0]}

    _, first, second, _ = _run_real_turn(
        tmp_path,
        size=10,
        selected_name="AI Video Newsletter Curator",
        responses=responses,
        timeout_seconds=0.1,
        late_grace_seconds=2,
        run_second_turn=True,
    )

    assert first.inferred_action == "unobservable"
    assert any(error.code == "late_response" for error in first.errors)
    assert second.inferred_action == "unobservable"
    assert second.final_response is None
    assert any(error.code == "tainted_process" for error in second.errors)


def test_real_historical_malformed_tool_args_remain_inconclusive(tmp_path: Path) -> None:
    responses = _fake_response_queue("AI Video Newsletter Curator")
    responses[0]["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = "{bad-json"
    responses[1] = {"choices": [{"message": {"content": "Please clarify the workflow."}}]}

    _, observation, _ = _run_real_turn(
        tmp_path,
        size=10,
        selected_name="AI Video Newsletter Curator",
        responses=responses,
    )

    assert observation.inferred_action == "unobservable"
    assert observation.inferred_name is None
    assert any(error.code == "failed_observation" for error in observation.errors)


def test_real_historical_execution_timeout_is_preserved_as_failure(tmp_path: Path) -> None:
    dispatch = _fake_response_queue("AI Video Newsletter Curator")[0]
    wait_response = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "wait-1",
                            "type": "function",
                            "function": {
                                "name": "wait",
                                "arguments": json.dumps({"reason": "await execution"}),
                            },
                        }
                    ],
                }
            }
        ]
    }
    responses = [
        {"_component": "interaction_user", "_body": dispatch},
        {
            "_component": "interaction_user",
            "_repeat": True,
            "_body": wait_response,
        },
        {
            "_component": "execution",
            "_delay": 0.5,
            "_body": {"choices": [{"message": {"content": "too late"}}]},
        },
        {
            "_component": "interaction_agent",
            "_body": {"choices": [{"message": {"content": "execution timeout preserved"}}]},
        },
    ]

    _, observation, _ = _run_real_turn(
        tmp_path,
        size=10,
        selected_name="AI Video Newsletter Curator",
        responses=responses,
        timeout_seconds=3,
        execution_timeout_seconds=0.1,
    )

    assert observation.inferred_action == "unobservable"
    assert any(
        error.phase == "execution" and error.code == "model_error"
        for error in observation.errors
    )
