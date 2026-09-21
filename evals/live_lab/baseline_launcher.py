"""Launch the pinned historical app with in-memory observation wrappers."""

from __future__ import annotations

import argparse
import functools
import hashlib
import html
import importlib
import inspect
import json
import re
import sys
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import httpx

from server.services.evaluation_lab.usage import normalize_usage
from server.services.evaluation_lab.trace import TraceTiming
from server.services.evaluation_lab.redaction import redact_value

from .revisions import (
    APPROVED_OVERLAY_PATHS,
    HISTORICAL_BASE_SHA,
    verify_baseline_revision,
)


_AGENT_NAME = re.compile(r'<agent\s+name="([^"]*)"\s*/>')
_COMPONENTS = {
    "interaction_model": "interaction",
    "execution_model": "execution",
    "email_search_model": "email_search",
    "summarizer_model": "summarizer",
    "classifier_model": "classifier",
}
_BASELINE_MODEL_CONFIGS: dict[str, dict[str, Any]] = {}
_BASELINE_TIMINGS: ContextVar[tuple[TraceTiming, ...]] = ContextVar(
    "baseline_observation_timings", default=()
)
_BASELINE_TRANSPORT_EVIDENCE: ContextVar[dict[str, Any] | None] = ContextVar(
    "baseline_transport_evidence", default=None
)


@contextmanager
def _baseline_timing():
    timing = TraceTiming(time.perf_counter_ns)
    token = _BASELINE_TIMINGS.set((*_BASELINE_TIMINGS.get(), timing))
    try:
        yield timing
    finally:
        timing.finish()
        _BASELINE_TIMINGS.reset(token)


def _build_model_configs(
    model_id: str,
    *,
    seed: int | None = None,
    seed_compatible: bool = False,
) -> dict[str, dict[str, Any]]:
    """Create one immutable-in-practice measured policy for every baseline role."""

    if not isinstance(model_id, str) or "/" not in model_id or any(
        character.isspace() for character in model_id
    ) or model_id.startswith("/") or model_id.endswith("/"):
        raise ValueError("model_id must be a provider/model identifier")
    common: dict[str, Any] = {
        "model_id": model_id,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 1000,
        "timeout_seconds": 60.0,
        "max_retries": 0,
    }
    if seed is not None and seed_compatible:
        common["seed"] = seed
    return {component: dict(common) for component in _COMPONENTS.values()}


def _build_baseline_payload(
    kwargs: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Build the historical request without editing its checked-out source."""

    messages = list(kwargs.get("messages") or [])
    system = kwargs.get("system")
    if isinstance(system, str) and system:
        messages = [{"role": "system", "content": system}, *messages]
    payload: dict[str, Any] = {
        "model": config["model_id"],
        "messages": messages,
        "stream": False,
    }
    tools = kwargs.get("tools")
    if tools:
        payload["tools"] = tools
    for field in ("temperature", "top_p", "max_tokens", "seed"):
        if field in config and config[field] is not None:
            payload[field] = config[field]
    return payload


def _canonical(value: Any) -> bytes:
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    except Exception:
        rendered = repr(value)
    return rendered.encode("utf-8", errors="replace")


class ObservationSink:
    """Write sanitized events and hashed private payloads to separate paths."""

    def __init__(
        self,
        event_path: Path,
        private_dir: Path,
        *,
        owner_path: Path | None = None,
        process_nonce: str | None = None,
    ) -> None:
        self.event_path = Path(event_path)
        self.private_dir = Path(private_dir)
        self.event_path.parent.mkdir(parents=True, exist_ok=True)
        self.private_dir.mkdir(parents=True, exist_ok=True)
        self.owner_path = Path(owner_path) if owner_path is not None else None
        self.process_nonce = process_nonce
        self._lock = threading.Lock()

    def store_private(self, value: Any) -> str:
        payload = _canonical(value)
        digest = hashlib.sha256(payload).hexdigest()
        target = self.private_dir / f"{digest}.json"
        if not target.exists():
            target.write_bytes(payload)
        return digest

    def append(self, event: Mapping[str, Any]) -> None:
        public_event = redact_value(dict(event))
        if not isinstance(public_event, dict):
            raise ValueError("baseline public event must be a JSON object")
        if self.process_nonce is not None:
            public_event["process_nonce"] = self.process_nonce
        if self.owner_path is not None:
            try:
                owner = json.loads(self.owner_path.read_text(encoding="utf-8"))
            except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
                owner = None
            if (
                isinstance(owner, dict)
                and owner.get("process_nonce") == self.process_nonce
                and isinstance(owner.get("owner_token"), str)
            ):
                public_event["owner_token"] = owner["owner_token"]
        line = json.dumps(public_event, sort_keys=True, separators=(",", ":"), default=str)
        with self._lock:
            with self.event_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def read_events(self) -> list[dict[str, Any]]:
        try:
            lines = self.event_path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return []
        return [json.loads(line) for line in lines]


def _prompt_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value:
        final = value[-1]
        if isinstance(final, dict) and isinstance(final.get("content"), str):
            return final["content"]
    return None


def _public_tool_arguments(
    tool_name: str,
    arguments: Mapping[str, Any],
    sink: ObservationSink,
) -> dict[str, Any]:
    public: dict[str, Any] = {}
    for key, value in arguments.items():
        if tool_name == "send_message_to_agent" and key == "agent_name" and isinstance(value, str):
            public[key] = value
        elif isinstance(value, str):
            public[key] = {
                "characters": len(value),
                "sha256": sink.store_private(value),
            }
        elif value is None or isinstance(value, (bool, int, float)):
            public[key] = value
        else:
            payload = _canonical(value)
            public[key] = {
                "bytes": len(payload),
                "sha256": sink.store_private(value),
            }
    return public


def _observation_failure(sink: ObservationSink, stage: str, exc: BaseException) -> None:
    """Best-effort metadata only; observation must never affect baseline control flow."""

    try:
        sink.append(
            {
                "kind": "observation_failure",
                "stage": stage,
                "error_type": type(exc).__name__,
            }
        )
    except BaseException:
        pass


def _best_effort_observe(
    sink: ObservationSink,
    stage: str,
    operation: Callable[[], Any],
) -> Any | None:
    observations = tuple(
        (timing, started)
        for timing in _BASELINE_TIMINGS.get()
        if (started := timing._observation_started()) is not None
    )
    try:
        return operation()
    except BaseException as exc:
        _observation_failure(sink, stage, exc)
        return None
    finally:
        for timing, started in observations:
            timing._observation_finished(started)


def _sync_event(component: str, result: Any, elapsed_ms: float, sink: ObservationSink) -> dict[str, Any]:
    prompt = _prompt_text(result)
    digest = sink.store_private(result)
    if component == "interaction_prompt" and prompt is not None:
        return {
            "kind": "interaction_prompt",
            "elapsed_ms": elapsed_ms,
            "prompt_sha256": digest,
            "prompt_characters": len(prompt),
            "exposed_names": [html.unescape(name) for name in _AGENT_NAME.findall(prompt)],
        }
    if component == "active_roster" and prompt is not None:
        return {
            "kind": "active_roster",
            "elapsed_ms": elapsed_ms,
            "roster_sha256": digest,
            "roster_characters": len(prompt),
            "exposed_names": [html.unescape(name) for name in _AGENT_NAME.findall(prompt)],
        }
    if component == "execution_prompt" and prompt is not None:
        return {
            "kind": "execution_prompt",
            "elapsed_ms": elapsed_ms,
            "prompt_sha256": digest,
            "prompt_characters": len(prompt),
        }
    return {"kind": component, "elapsed_ms": elapsed_ms, "result_sha256": digest}


def wrap_sync_call(component: str, original: Callable[..., Any], sink: ObservationSink):
    @functools.wraps(original)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        caught: BaseException | None = None
        with _baseline_timing() as timing:
            try:
                result = original(*args, **kwargs)
            except BaseException as error:
                caught = error
                timing.finish()
                elapsed_ms = timing.elapsed_ns / 1_000_000
            else:
                timing.finish()
                elapsed_ms = timing.elapsed_ns / 1_000_000
        if caught is not None:
            def observe_error() -> None:
                error_digest = sink.store_private(
                    {"error_type": type(caught).__name__, "message": str(caught)}
                )
                sink.append(
                    {
                        "kind": component,
                        "elapsed_ms": elapsed_ms,
                        "error_type": type(caught).__name__,
                        "error_sha256": error_digest,
                    }
                )

            _best_effort_observe(sink, f"{component}:exception", observe_error)
            raise caught
        elapsed = elapsed_ms

        def observe_success() -> None:
            if component == "interaction_tool":
                name = args[0] if args else kwargs.get("name", "")
                arguments = args[1] if len(args) > 1 else kwargs.get("arguments", {})
                malformed = False
                raw_digest = sink.store_private(arguments)
                if isinstance(arguments, str):
                    try:
                        normalized = json.loads(arguments) if arguments.strip() else {}
                    except json.JSONDecodeError:
                        normalized = {}
                        malformed = True
                elif isinstance(arguments, dict):
                    normalized = arguments
                else:
                    normalized = {}
                    malformed = True
                sink.append(
                    {
                        "kind": "tool_call",
                        "elapsed_ms": elapsed,
                        "tool_call": {
                            "name": str(name),
                            "arguments": _public_tool_arguments(str(name), normalized, sink),
                            "raw_arguments_sha256": raw_digest,
                            "malformed": malformed,
                        },
                    }
                )
            else:
                sink.append(_sync_event(component, result, elapsed, sink))

        _best_effort_observe(sink, f"{component}:success", observe_success)
        return result

    return wrapped


def _phase_event(
    phase: str, timing: TraceTiming, error: BaseException | None
) -> dict[str, Any]:
    return {
        "kind": "phase_timing",
        "phase": phase,
        "started_monotonic_ns": timing.started_monotonic_ns,
        "finished_monotonic_ns": timing.finished_monotonic_ns,
        "elapsed_ns": timing.elapsed_ns,
        "error_type": type(error).__name__ if error is not None else None,
    }


def wrap_async_phase_call(
    phase: str, original: Callable[..., Any], sink: ObservationSink
):
    @functools.wraps(original)
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        caught: BaseException | None = None
        with _baseline_timing() as timing:
            try:
                result = original(*args, **kwargs)
                if inspect.isawaitable(result):
                    result = await result
            except BaseException as error:
                caught = error
            finally:
                timing.finish()
        _best_effort_observe(
            sink,
            f"{phase}:timing",
            lambda: sink.append(_phase_event(phase, timing, caught)),
        )
        if caught is not None:
            raise caught
        return result

    return wrapped


def wrap_sync_phase_call(
    phase: str, original: Callable[..., Any], sink: ObservationSink
):
    @functools.wraps(original)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        caught: BaseException | None = None
        with _baseline_timing() as timing:
            try:
                result = original(*args, **kwargs)
            except BaseException as error:
                caught = error
            finally:
                timing.finish()
        _best_effort_observe(
            sink,
            f"{phase}:timing",
            lambda: sink.append(_phase_event(phase, timing, caught)),
        )
        if caught is not None:
            raise caught
        return result

    return wrapped


def _install_phase_wrappers(
    sink: ObservationSink,
    *,
    interaction_runtime: Any,
    gmail_modules: Sequence[Any],
) -> None:
    interaction_runtime.InteractionAgentRuntime.execute = wrap_async_phase_call(
        "total_run",
        interaction_runtime.InteractionAgentRuntime.execute,
        sink,
    )
    gmail_original = next(
        (
            module.execute_gmail_tool
            for module in gmail_modules
            if hasattr(module, "execute_gmail_tool")
        ),
        None,
    )
    if gmail_original is None:
        return
    gmail_wrapper = wrap_sync_phase_call("gmail_tool", gmail_original, sink)
    for module in gmail_modules:
        if hasattr(module, "execute_gmail_tool"):
            module.execute_gmail_tool = gmail_wrapper


def _tool_names(tools: Any) -> list[str]:
    names: list[str] = []
    if not isinstance(tools, list):
        return names
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            names.append(function["name"])
    return names


def _response_tool_call_count(response: Any) -> int | None:
    if not isinstance(response, dict) or not isinstance(response.get("choices"), list):
        return None
    count = 0
    for choice in response["choices"]:
        if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
            return None
        tool_calls = choice["message"].get("tool_calls")
        if tool_calls is None:
            continue
        if not isinstance(tool_calls, list):
            return None
        count += len(tool_calls)
    return count


def _malformed_tool_call_count(response: Any) -> int:
    if not isinstance(response, dict) or not isinstance(response.get("choices"), list):
        return 0
    malformed = 0
    for choice in response["choices"]:
        message = choice.get("message") if isinstance(choice, dict) else None
        calls = message.get("tool_calls") if isinstance(message, dict) else None
        if calls is None:
            continue
        if not isinstance(calls, list):
            malformed += 1
            continue
        for call in calls:
            function = call.get("function") if isinstance(call, dict) else None
            arguments = function.get("arguments") if isinstance(function, dict) else None
            if not isinstance(function, dict) or not isinstance(function.get("name"), str):
                malformed += 1
            elif isinstance(arguments, dict):
                continue
            elif isinstance(arguments, str):
                try:
                    if not isinstance(json.loads(arguments) if arguments.strip() else {}, dict):
                        malformed += 1
                except json.JSONDecodeError:
                    malformed += 1
            else:
                malformed += 1
    return malformed


def _nested_value(payload: object, *paths: tuple[str, ...]) -> object | None:
    for path in paths:
        current = payload
        for part in path:
            if not isinstance(current, Mapping) or part not in current:
                break
            current = current[part]
        else:
            return current
    return None


def _rate_limit_evidence(headers: Mapping[str, Any]) -> dict[str, str]:
    normalized = {str(key).casefold(): str(value) for key, value in headers.items()}
    evidence: dict[str, str] = {}
    for header, target in (
        ("retry-after", "retry_after_seconds"),
        ("x-ratelimit-limit", "limit"),
        ("x-ratelimit-remaining", "remaining"),
        ("x-ratelimit-reset", "reset"),
    ):
        if header in normalized:
            evidence[target] = normalized[header]
    return evidence


def _model_evidence(
    model_component: str,
    kwargs: Mapping[str, Any],
    *,
    result: Any | None,
    elapsed_ms: float,
    request_digest: str,
    config: Mapping[str, Any] | None,
    transport_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    policy = dict(config or {})
    transport = dict(transport_evidence or {})
    transport_response = transport.get("response_payload")
    observed_result = (
        result
        if isinstance(result, Mapping)
        else transport_response
        if isinstance(transport_response, Mapping)
        else None
    )
    requested_model = str(policy.get("model_id") or kwargs.get("model") or "")
    requested_seed = policy.get("seed")
    response_model = observed_result.get("model") if observed_result is not None else None
    provider = observed_result.get("provider") if observed_result is not None else None
    provider_seed = _nested_value(
        observed_result,
        ("seed",),
        ("metadata", "seed"),
        ("provider_metadata", "seed"),
    )
    context_limit = _nested_value(
        observed_result,
        ("context_length",),
        ("context_limit",),
        ("top_provider", "context_length"),
        ("provider_metadata", "context_length"),
    )
    provider_retry_count = _nested_value(
        observed_result,
        ("provider_metadata", "retry_count"),
        ("routing", "retry_count"),
    )
    provider_failover = _nested_value(
        observed_result,
        ("provider_metadata", "failover"),
        ("routing", "failover"),
    )
    provider_routing = _nested_value(
        observed_result,
        ("provider_metadata", "routing"),
        ("routing", "attempts"),
    )
    application_retry_count = transport.get("application_retry_count")
    if not isinstance(application_retry_count, int) or isinstance(
        application_retry_count, bool
    ):
        application_retry_count = 0
    generation = {
        key: policy[key]
        for key in ("temperature", "top_p", "max_tokens", "seed")
        if key in policy
    }
    usage_warnings: list[str] = []
    usage = normalize_usage(observed_result, warnings=usage_warnings)
    return {
        "component": model_component,
        "model": response_model if isinstance(response_model, str) else requested_model,
        "provider": provider if isinstance(provider, str) else None,
        "requested_model_author": requested_model.split("/", 1)[0],
        "requested_provider_policy": None,
        "actual_provider": provider if isinstance(provider, str) else None,
        "generation": generation,
        "context_limit": context_limit if isinstance(context_limit, int) else None,
        "requested_seed": requested_seed if isinstance(requested_seed, int) else None,
        "provider_seed": provider_seed if isinstance(provider_seed, int) else None,
        "seed_acknowledged": (
            provider_seed == requested_seed
            if requested_seed is not None and isinstance(provider_seed, int)
            else None
        ),
        "retry_count": application_retry_count,
        "application_retry_count": application_retry_count,
        "provider_retry_count": (
            provider_retry_count if isinstance(provider_retry_count, int) else None
        ),
        "max_retries": int(policy.get("max_retries", 0)),
        "provider_failover": (
            provider_failover if isinstance(provider_failover, bool) else None
        ),
        "provider_routing": (
            provider_routing
            if isinstance(provider_routing, (list, dict, str))
            else None
        ),
        "failover": provider_failover if isinstance(provider_failover, bool) else None,
        "timeout": bool(transport.get("timeout", False)),
        "timeout_seconds": float(policy.get("timeout_seconds", 60.0)),
        "status_code": transport.get("status_code"),
        "rate_limit": transport.get("rate_limit", {}),
        "malformed_tool_calls": _malformed_tool_call_count(observed_result),
        "usage": usage.model_dump(mode="json"),
        "usage_warnings": usage_warnings,
        "elapsed_ms": elapsed_ms,
        "request_sha256": request_digest,
        "message_count": len(kwargs.get("messages") or []),
        "tool_names": _tool_names(kwargs.get("tools")),
    }


def wrap_async_call(
    component: str,
    original: Callable[..., Any],
    sink: ObservationSink,
    *,
    model_config: Mapping[str, Any] | None = None,
):
    @functools.wraps(original)
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        request_digest = _best_effort_observe(
            sink,
            f"{component}:request",
            lambda: sink.store_private({"args": args, "kwargs": kwargs}),
        )
        if not isinstance(request_digest, str):
            request_digest = ""
        model_component = _COMPONENTS.get(component)
        caught: BaseException | None = None
        transport_token = _BASELINE_TRANSPORT_EVIDENCE.set(None)
        try:
            with _baseline_timing() as timing:
                try:
                    result = original(*args, **kwargs)
                    if inspect.isawaitable(result):
                        result = await result
                except BaseException as error:
                    caught = error
                    timing.finish()
                    elapsed = timing.elapsed_ns / 1_000_000
                else:
                    timing.finish()
                    elapsed = timing.elapsed_ns / 1_000_000
            transport_evidence = _BASELINE_TRANSPORT_EVIDENCE.get()
        finally:
            _BASELINE_TRANSPORT_EVIDENCE.reset(transport_token)
        if caught is not None:
            def observe_error() -> None:
                error_digest = sink.store_private(
                    {"error_type": type(caught).__name__, "message": str(caught)}
                )
                if model_component:
                    evidence = _model_evidence(
                        model_component,
                        kwargs,
                        result=None,
                        elapsed_ms=elapsed,
                        request_digest=request_digest,
                        config=model_config,
                        transport_evidence=transport_evidence,
                    )
                    evidence.update(
                        {
                            "error_type": type(caught).__name__,
                            "timeout": bool(
                                (transport_evidence or {}).get("timeout")
                                or isinstance(caught, (TimeoutError, httpx.TimeoutException))
                            ),
                        }
                    )
                    sink.append(
                        {
                            "kind": "model_call",
                            "model_call": evidence,
                            "error_sha256": error_digest,
                        }
                    )
                else:
                    sink.append(
                        {
                            "kind": component,
                            "elapsed_ms": elapsed,
                            "error_type": type(caught).__name__,
                            "error_sha256": error_digest,
                        }
                    )

            _best_effort_observe(sink, f"{component}:exception", observe_error)
            raise caught

        def observe_success() -> None:
            response_digest = sink.store_private(result)
            if model_component:
                choices = result.get("choices") if isinstance(result, dict) else None
                evidence = _model_evidence(
                    model_component,
                    kwargs,
                    result=result,
                    elapsed_ms=elapsed,
                    request_digest=request_digest,
                    config=model_config,
                    transport_evidence=transport_evidence,
                )
                evidence.update(
                    {
                        "response_sha256": response_digest,
                        "response_choice_count": len(choices) if isinstance(choices, list) else 0,
                        "response_tool_call_count": _response_tool_call_count(result),
                    }
                )
                sink.append(
                    {
                        "kind": "model_call",
                        "model_call": evidence,
                    }
                )
            else:
                sink.append(
                    {
                        "kind": component,
                        "elapsed_ms": elapsed,
                        "request_sha256": request_digest,
                        "response_sha256": response_digest,
                    }
                )

        _best_effort_observe(sink, f"{component}:success", observe_success)
        return result

    return wrapped


def _purge_server_modules() -> None:
    for name in tuple(sys.modules):
        if name == "server" or name.startswith("server."):
            del sys.modules[name]


def _configure_historical_state(
    data_dir: Path,
    model_id: str,
    *,
    seed: int | None = None,
    seed_compatible: bool = False,
) -> None:
    global _BASELINE_MODEL_CONFIGS

    _BASELINE_MODEL_CONFIGS = _build_model_configs(
        model_id, seed=seed, seed_compatible=seed_compatible
    )
    config = importlib.import_module("server.config")
    settings = config.Settings(
        server_host="127.0.0.1",
        openrouter_api_key="live-lab-fake-key",
        lab_enabled=True,
        lab_composio_user_id="live-lab-local-user",
        interaction_agent_model=model_id,
        execution_agent_model=model_id,
        execution_agent_search_model=model_id,
        summarizer_model=model_id,
        email_classifier_model=model_id,
        conversation_summary_threshold=0,
    )
    config.get_settings = lambda: settings

    memory_module = importlib.import_module(
        "server.services.conversation.summarization.working_memory_log"
    )
    memory_module._working_memory_log = memory_module.WorkingMemoryLog(
        data_dir / "conversation" / "poke_working_memory.log"
    )
    conversation_module = importlib.import_module("server.services.conversation.log")
    conversation_module._conversation_log = conversation_module.ConversationLog(
        data_dir / "conversation" / "poke_conversation.log"
    )
    roster_module = importlib.import_module("server.services.execution.roster")
    roster_module._agent_roster = roster_module.AgentRoster(
        data_dir / "execution_agents" / "roster.json"
    )
    logs_module = importlib.import_module("server.services.execution.log_store")
    logs_module._execution_agent_logs = logs_module.ExecutionAgentLogStore(
        data_dir / "execution_agents"
    )
    timezone_module = importlib.import_module("server.services.timezone_store")
    timezone_module._timezone_store = timezone_module.TimezoneStore(data_dir / "timezone.txt")


def _transport(fake_base_url: str, component: str):
    client_module = importlib.import_module("server.openrouter_client.client")
    role = _COMPONENTS[component]
    config = _BASELINE_MODEL_CONFIGS[role]

    async def request_chat_completion(**kwargs: Any) -> Any:
        _BASELINE_TRANSPORT_EVIDENCE.set(None)
        url = f"{fake_base_url.rstrip('/')}/chat/completions"
        payload = _build_baseline_payload(kwargs, config)
        timeout_seconds = float(config["timeout_seconds"])
        max_retries = int(config["max_retries"])
        async with httpx.AsyncClient() as client:
            for attempt in range(max_retries + 1):
                try:
                    response = await client.post(
                        url,
                        headers=client_module._headers(api_key=kwargs.get("api_key")),
                        json=payload,
                        timeout=timeout_seconds,
                    )
                except httpx.HTTPError as exc:
                    _BASELINE_TRANSPORT_EVIDENCE.set(
                        {
                            "application_retry_count": attempt,
                            "timeout": isinstance(exc, httpx.TimeoutException),
                            "rate_limit": {},
                            "status_code": None,
                        }
                    )
                    if attempt < max_retries:
                        continue
                    raise client_module.OpenRouterError(
                        f"OpenRouter request failed: {exc}"
                    ) from exc
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    try:
                        response_payload = response.json()
                    except Exception:
                        response_payload = None
                    _BASELINE_TRANSPORT_EVIDENCE.set(
                        {
                            "application_retry_count": attempt,
                            "timeout": False,
                            "rate_limit": _rate_limit_evidence(response.headers),
                            "response_payload": response_payload,
                            "status_code": response.status_code,
                        }
                    )
                    if attempt < max_retries and response.status_code in {
                        408,
                        409,
                        429,
                        500,
                        502,
                        503,
                        504,
                    }:
                        continue
                    client_module._handle_response_error(exc)
                _BASELINE_TRANSPORT_EVIDENCE.set(
                    {
                        "application_retry_count": attempt,
                        "timeout": False,
                        "rate_limit": _rate_limit_evidence(response.headers),
                        "status_code": response.status_code,
                    }
                )
                result = response.json()
                if not isinstance(result, dict):
                    raise client_module.OpenRouterError(
                        "OpenRouter response must be a JSON object"
                    )
                return result
        raise client_module.OpenRouterError("OpenRouter request failed: unknown error")

    return request_chat_completion


def _single_request_summarizer(
    request: Callable[..., Any], error_type: type[BaseException]
):
    """Replace only the measured historical summarizer's outer retry loop."""

    async def call(prompt: Any, model: str, api_key: str | None) -> str:
        response = request(
            model=model,
            messages=prompt.messages,
            system=prompt.system_prompt,
            api_key=api_key,
        )
        if inspect.isawaitable(response):
            response = await response
        choices = response.get("choices") or [] if isinstance(response, dict) else []
        if not choices:
            raise error_type("OpenRouter response missing choices")
        message = choices[0].get("message") or {}
        content = (message.get("content") or "").strip()
        if not content:
            raise error_type("OpenRouter response missing content")
        return content

    return call


def install_baseline_wrappers(
    sink: ObservationSink,
    *,
    fake_base_url: str,
    execution_timeout_seconds: float = 90.0,
) -> None:
    """Patch all imported call sites before importing ``server.app``."""

    interaction_agent = importlib.import_module("server.agents.interaction_agent.agent")
    interaction_runtime = importlib.import_module("server.agents.interaction_agent.runtime")
    interaction_tools = importlib.import_module("server.agents.interaction_agent.tools")
    execution_agent = importlib.import_module("server.agents.execution_agent.agent")
    execution_runtime = importlib.import_module("server.agents.execution_agent.runtime")
    email_search = importlib.import_module("server.agents.execution_agent.tasks.search_email.tool")
    summarizer = importlib.import_module("server.services.conversation.summarization.summarizer")
    classifier = importlib.import_module("server.services.gmail.importance_classifier")
    gmail_client = importlib.import_module("server.services.gmail.client")
    gmail_package = importlib.import_module("server.services.gmail")
    gmail_tools = importlib.import_module("server.agents.execution_agent.tools.gmail")
    gmail_internal = importlib.import_module(
        "server.agents.execution_agent.tasks.search_email.gmail_internal"
    )
    interaction_tools._EXECUTION_BATCH_MANAGER.timeout_seconds = execution_timeout_seconds
    _install_phase_wrappers(
        sink,
        interaction_runtime=interaction_runtime,
        gmail_modules=(
            gmail_client,
            gmail_package,
            gmail_tools,
            gmail_internal,
            email_search,
        ),
    )

    active_wrapper = wrap_sync_call(
        "active_roster", interaction_agent._render_active_agents, sink
    )
    interaction_agent._render_active_agents = active_wrapper
    prompt_wrapper = wrap_sync_call(
        "interaction_prompt", interaction_agent.prepare_message_with_history, sink
    )
    interaction_agent.prepare_message_with_history = prompt_wrapper
    interaction_runtime.prepare_message_with_history = prompt_wrapper

    original_parse_tool_calls = interaction_runtime.InteractionAgentRuntime._parse_tool_calls

    @functools.wraps(original_parse_tool_calls)
    def parse_tool_calls_with_rejections(runtime: Any, raw_tool_calls: Any) -> Any:
        parsed = original_parse_tool_calls(runtime, raw_tool_calls)

        def observe_rejections() -> None:
            for tool_call in parsed:
                arguments = getattr(tool_call, "arguments", {})
                if isinstance(arguments, dict) and "__invalid_arguments__" in arguments:
                    sink.append(
                        {
                            "kind": "tool_call",
                            "tool_call": {
                                "name": str(getattr(tool_call, "name", "")),
                                "arguments": {},
                                "raw_arguments_sha256": sink.store_private(raw_tool_calls),
                                "malformed": True,
                            },
                        }
                    )

        _best_effort_observe(sink, "interaction_tool_parse:rejected", observe_rejections)
        return parsed

    interaction_runtime.InteractionAgentRuntime._parse_tool_calls = parse_tool_calls_with_rejections

    execution_prompt = wrap_sync_call(
        "execution_prompt", execution_agent.ExecutionAgent.build_system_prompt_with_history, sink
    )
    execution_agent.ExecutionAgent.build_system_prompt_with_history = execution_prompt
    tool_wrapper = wrap_sync_call("interaction_tool", interaction_tools.handle_tool_call, sink)
    interaction_tools.handle_tool_call = tool_wrapper
    interaction_runtime.handle_tool_call = tool_wrapper

    interaction_runtime.request_chat_completion = wrap_async_call(
        "interaction_model",
        _transport(fake_base_url, "interaction_model"),
        sink,
        model_config=_BASELINE_MODEL_CONFIGS["interaction"],
    )
    execution_runtime.request_chat_completion = wrap_async_call(
        "execution_model",
        _transport(fake_base_url, "execution_model"),
        sink,
        model_config=_BASELINE_MODEL_CONFIGS["execution"],
    )
    email_search.request_chat_completion = wrap_async_call(
        "email_search_model",
        _transport(fake_base_url, "email_search_model"),
        sink,
        model_config=_BASELINE_MODEL_CONFIGS["email_search"],
    )
    summarizer_transport = wrap_async_call(
        "summarizer_model",
        _transport(fake_base_url, "summarizer_model"),
        sink,
        model_config=_BASELINE_MODEL_CONFIGS["summarizer"],
    )
    summarizer.request_chat_completion = summarizer_transport
    summarizer._call_openrouter = _single_request_summarizer(
        summarizer_transport,
        summarizer.OpenRouterError,
    )
    classifier.request_chat_completion = wrap_async_call(
        "classifier_model",
        _transport(fake_base_url, "classifier_model"),
        sink,
        model_config=_BASELINE_MODEL_CONFIGS["classifier"],
    )


def build_historical_app(
    *,
    worktree: Path,
    data_dir: Path,
    run_dir: Path,
    fake_base_url: str,
    model_id: str,
    process_nonce: str,
    seed: int | None = None,
    seed_compatible: bool = False,
    execution_timeout_seconds: float = 90.0,
):
    """Import the baseline package only after process-local hooks are ready."""

    _purge_server_modules()
    root = str(worktree.resolve(strict=True))
    sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != Path(__file__).resolve().parents[2]]
    sys.path.insert(0, root)
    _configure_historical_state(
        data_dir.resolve(strict=True),
        model_id,
        seed=seed,
        seed_compatible=seed_compatible,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    context_path = run_dir / "process_context.json"
    previous_nonce: str | None = None
    try:
        previous_context = json.loads(context_path.read_text(encoding="utf-8"))
        if isinstance(previous_context, dict) and isinstance(
            previous_context.get("process_nonce"), str
        ):
            previous_nonce = previous_context["process_nonce"]
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
        pass
    if previous_nonce != process_nonce:
        (run_dir / "active_turn.json").unlink(missing_ok=True)
        (run_dir / "tainted.json").unlink(missing_ok=True)
    context_path.write_text(
        json.dumps({"process_nonce": process_nonce}, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    sink = ObservationSink(
        run_dir / "events.jsonl",
        run_dir / "private",
        owner_path=run_dir / "active_turn.json",
        process_nonce=process_nonce,
    )
    install_baseline_wrappers(
        sink,
        fake_base_url=fake_base_url,
        execution_timeout_seconds=execution_timeout_seconds,
    )
    app_module = importlib.import_module("server.app")
    app_module.app.router.on_startup.clear()
    app_module.app.router.on_shutdown.clear()

    async def live_lab_ready() -> dict[str, str]:
        return {"nonce": process_nonce}

    app_module.app.add_api_route(
        f"/__live_lab_ready__/{process_nonce}",
        live_lab_ready,
        methods=["GET"],
        include_in_schema=False,
    )
    return app_module.app


def _file_hashes(data_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    if not data_dir.exists():
        return hashes
    for path in sorted(data_dir.rglob("*")):
        if path.is_file():
            hashes[path.relative_to(data_dir).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _preflight(args: argparse.Namespace) -> int:
    worktree = Path(args.worktree)
    revision = verify_baseline_revision(
        worktree,
        expected_base=HISTORICAL_BASE_SHA,
        allowed_paths=APPROVED_OVERLAY_PATHS,
    )
    data_dir = Path(args.data_dir) if args.data_dir else worktree / "server" / "data"
    roster_path = data_dir / "execution_agents" / "roster.json"
    roster_count = 0
    if roster_path.is_file():
        roster = json.loads(roster_path.read_text(encoding="utf-8"))
        roster_count = len(roster) if isinstance(roster, list) else 0
    model_id = args.model_id
    configs = _build_model_configs(
        model_id,
        seed=args.seed,
        seed_compatible=args.seed_compatible,
    )
    print(
        json.dumps(
            {
                "base_sha": revision.base_sha,
                "fixture_hashes": _file_hashes(data_dir),
                "model_ids": {
                    "classifier": model_id,
                    "email_search": model_id,
                    "execution": model_id,
                    "interaction": model_id,
                    "summarizer": model_id,
                },
                "model_configs": configs,
                "overlay_diff_sha256": revision.overlay_diff_sha256,
                "overlay_head_sha": revision.overlay_head_sha,
                "port": args.port,
                "roster_count": roster_count,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--data-dir")
    parser.add_argument("--run-dir")
    parser.add_argument("--fake-base-url", default="http://127.0.0.1:8999/v1")
    parser.add_argument("--model-id", default="openai/gpt-4.1-mini")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--seed-compatible", action="store_true")
    parser.add_argument("--execution-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--readiness-nonce")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.host != "127.0.0.1" or args.port != 8001:
        raise ValueError("historical live-lab server is pinned to 127.0.0.1:8001")
    if args.preflight_only:
        return _preflight(args)
    if not args.data_dir or not args.run_dir:
        raise ValueError("--data-dir and --run-dir are required unless --preflight-only is used")
    if not args.readiness_nonce:
        raise ValueError("--readiness-nonce is required for a measured historical launch")
    verify_baseline_revision(
        Path(args.worktree),
        expected_base=HISTORICAL_BASE_SHA,
        allowed_paths=APPROVED_OVERLAY_PATHS,
    )
    app = build_historical_app(
        worktree=Path(args.worktree),
        data_dir=Path(args.data_dir),
        run_dir=Path(args.run_dir),
        fake_base_url=args.fake_base_url,
        model_id=args.model_id,
        process_nonce=args.readiness_nonce,
        seed=args.seed,
        seed_compatible=args.seed_compatible,
        execution_timeout_seconds=args.execution_timeout_seconds,
    )
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_config=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ObservationSink",
    "build_historical_app",
    "install_baseline_wrappers",
    "main",
    "wrap_async_call",
    "wrap_async_phase_call",
    "wrap_sync_call",
    "wrap_sync_phase_call",
]
