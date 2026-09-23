from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

import httpx

from ..config import ModelCallConfig, ModelRole, get_settings
from ..services.evaluation_lab.models import TraceEventKind
from ..services.evaluation_lab.budget import current_cost_controller
from ..services.evaluation_lab.routing_evidence import safe_provider_routing
from ..services.evaluation_lab.trace import emit_trace
from ..services.evaluation_lab.usage import (
    emit_unavailable_usage_evidence,
    emit_usage_evidence,
)

OpenRouterBaseURL = "https://openrouter.ai/api/v1"


class OpenRouterError(RuntimeError):
    """Raised when the OpenRouter API returns an error response."""


def _headers(*, api_key: Optional[str] = None) -> Dict[str, str]:
    settings = get_settings()
    key = (api_key or settings.openrouter_api_key or "").strip()
    if not key:
        raise OpenRouterError("Missing OpenRouter API key")

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    return headers


def _build_messages(messages: List[Dict[str, str]], system: Optional[str]) -> List[Dict[str, str]]:
    if system:
        return [{"role": "system", "content": system}, *messages]
    return messages


def _handle_response_error(exc: httpx.HTTPStatusError) -> None:
    response = exc.response
    detail: str
    try:
        payload = response.json()
        detail = payload.get("error") or payload.get("message") or json.dumps(payload)
    except Exception:
        detail = response.text
    raise OpenRouterError(f"OpenRouter request failed ({response.status_code}): {detail}") from exc


def _provider_from_model(model: str) -> str:
    return model.split("/", 1)[0]


def _provider_key(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _tool_names(tools: Optional[List[Dict[str, Any]]]) -> list[str]:
    names: list[str] = []
    for tool in tools or []:
        function = tool.get("function") if isinstance(tool, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        if isinstance(name, str) and name:
            names.append(name)
    return names


def _malformed_tool_call_count(payload: object) -> int:
    if not isinstance(payload, dict):
        return 0
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return 0
    malformed = 0
    for choice in choices:
        message = choice.get("message") if isinstance(choice, dict) else None
        calls = message.get("tool_calls") if isinstance(message, dict) else None
        if calls is None:
            continue
        if not isinstance(calls, list):
            malformed += 1
            continue
        for call in calls:
            function = call.get("function") if isinstance(call, dict) else None
            if not isinstance(function, dict) or not isinstance(function.get("name"), str):
                malformed += 1
                continue
            arguments = function.get("arguments")
            if isinstance(arguments, dict):
                continue
            if not isinstance(arguments, str):
                malformed += 1
                continue
            try:
                normalized = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError:
                malformed += 1
                continue
            if not isinstance(normalized, dict):
                malformed += 1
    return malformed


def _nested_value(payload: object, *paths: tuple[str, ...]) -> object | None:
    for path in paths:
        current = payload
        for part in path:
            if not isinstance(current, dict) or part not in current:
                break
            current = current[part]
        else:
            return current
    return None


def _rate_limit_evidence(response: httpx.Response) -> dict[str, str]:
    evidence: dict[str, str] = {}
    for header, target in (
        ("retry-after", "retry_after_seconds"),
        ("x-ratelimit-limit", "limit"),
        ("x-ratelimit-remaining", "remaining"),
        ("x-ratelimit-reset", "reset"),
    ):
        value = response.headers.get(header)
        if value is not None:
            evidence[target] = value
    return evidence


def _emit_request_evidence(
    *,
    role: ModelRole | None,
    config: ModelCallConfig | None,
    model: str,
    messages: List[Dict[str, str]],
    tools: Optional[List[Dict[str, Any]]],
    call_id: str,
) -> None:
    emit_trace(
        TraceEventKind.MODEL_CALL,
        {
            "stage": "request",
            "call_id": call_id,
            "attempt": 0,
            "role": role.value if role is not None else None,
            "model": model,
            "provider": _provider_from_model(model),
            "requested_model_author": _provider_from_model(model),
            "requested_provider_policy": None,
            "generation": config.explicit_payload_fields() if config is not None else {},
            "message_count": len(messages),
            "tool_names": _tool_names(tools),
            "requested_seed": config.seed if config is not None else None,
            "timeout_seconds": config.timeout_seconds if config is not None else 60.0,
            "max_retries": config.max_retries if config is not None else 0,
        },
    )


def _emit_retry_attempt_start(*, call_id: str, attempt: int) -> None:
    """Mark a retry as expected before starting transport work.

    The initial attempt is represented by the request event itself.  Later
    attempts need their own start marker so an in-flight or cancelled retry
    cannot make a partial aggregate appear complete.
    """

    emit_trace(
        TraceEventKind.MODEL_CALL,
        {
            "stage": "attempt_start",
            "call_id": call_id,
            "attempt": attempt,
        },
    )


def _emit_response_evidence(
    *,
    role: ModelRole | None,
    config: ModelCallConfig | None,
    requested_model: str,
    response: dict[str, Any],
    http_response: httpx.Response,
    retry_count: int,
    call_id: str,
    attempt: int,
) -> None:
    response_model = response.get("model")
    provider = response.get("provider")
    acknowledged_seed = _nested_value(
        response,
        ("seed",),
        ("metadata", "seed"),
        ("provider_metadata", "seed"),
    )
    requested_seed = config.seed if config is not None else None
    context_limit = _nested_value(
        response,
        ("context_length",),
        ("context_limit",),
        ("top_provider", "context_length"),
        ("provider_metadata", "context_length"),
    )
    provider_retry_count = _nested_value(
        response,
        ("provider_metadata", "retry_count"),
        ("routing", "retry_count"),
    )
    provider_failover = _nested_value(
        response,
        ("provider_metadata", "failover"),
        ("routing", "failover"),
    )
    provider_routing = _nested_value(
        response,
        ("provider_metadata", "routing"),
        ("routing", "attempts"),
    )
    emit_trace(
        TraceEventKind.MODEL_CALL,
        {
            "stage": "response",
            "call_id": call_id,
            "attempt": attempt,
            "role": role.value if role is not None else None,
            "model": response_model if isinstance(response_model, str) else requested_model,
            "requested_model": requested_model,
            "provider": provider if isinstance(provider, str) else None,
            "requested_model_author": _provider_from_model(requested_model),
            "requested_provider_policy": None,
            "actual_provider": provider if isinstance(provider, str) else None,
            "generation": config.explicit_payload_fields() if config is not None else {},
            "context_limit": context_limit if isinstance(context_limit, int) else None,
            "requested_seed": requested_seed,
            "provider_seed": acknowledged_seed if isinstance(acknowledged_seed, int) else None,
            "seed_acknowledged": (
                acknowledged_seed == requested_seed
                if requested_seed is not None and isinstance(acknowledged_seed, int)
                else None
            ),
            "retry_count": retry_count,
            "application_retry_count": retry_count,
            "provider_retry_count": (
                provider_retry_count if isinstance(provider_retry_count, int) else None
            ),
            "provider_failover": (
                provider_failover if isinstance(provider_failover, bool) else None
            ),
            "provider_routing": safe_provider_routing(provider_routing),
            "failover": (
                provider_failover if isinstance(provider_failover, bool) else None
            ),
            "timeout": False,
            "status_code": http_response.status_code,
            "rate_limit": _rate_limit_evidence(http_response),
            "malformed_tool_calls": _malformed_tool_call_count(response),
        },
    )


def _emit_error_evidence(
    *,
    role: ModelRole | None,
    model: str,
    config: ModelCallConfig | None,
    error: BaseException,
    retry_count: int,
    response: httpx.Response | None = None,
    call_id: str,
    attempt: int,
) -> None:
    emit_trace(
        TraceEventKind.MODEL_CALL,
        {
            "stage": "error",
            "call_id": call_id,
            "attempt": attempt,
            "role": role.value if role is not None else None,
            "model": model,
            "provider": _provider_from_model(model),
            "requested_model_author": _provider_from_model(model),
            "requested_provider_policy": None,
            "actual_provider": None,
            "generation": config.explicit_payload_fields() if config is not None else {},
            "requested_seed": config.seed if config is not None else None,
            "retry_count": retry_count,
            "application_retry_count": retry_count,
            "provider_retry_count": None,
            "max_retries": config.max_retries if config is not None else 0,
            "provider_failover": None,
            "provider_routing": None,
            "failover": None,
            "timeout": isinstance(error, httpx.TimeoutException),
            "status_code": response.status_code if response is not None else None,
            "rate_limit": _rate_limit_evidence(response) if response is not None else {},
            "error_type": type(error).__name__,
        },
    )


def _emit_usage_best_effort(
    response: object,
    *,
    role: ModelRole | None,
    call_id: str,
    attempt: int,
    observation_status: str,
) -> None:
    role_name = role.value if role is not None else "unassigned"
    try:
        emit_usage_evidence(
            response,
            role=role_name,
            call_id=call_id,
            attempt=attempt,
            observation_status=observation_status,
        )
    except Exception as exc:
        emit_trace(
            TraceEventKind.OBSERVABILITY_WARNING,
            {
                "boundary": "model_usage",
                "role": role_name,
                "call_id": call_id,
                "attempt": attempt,
                "error_type": type(exc).__name__,
            },
        )
        emit_unavailable_usage_evidence(
            role=role_name,
            call_id=call_id,
            attempt=attempt,
            observation_status="observer_degraded",
        )


async def request_chat_completion(
    *,
    model: str | None = None,
    config: ModelCallConfig | None = None,
    role: ModelRole | None = None,
    messages: List[Dict[str, str]],
    system: Optional[str] = None,
    api_key: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    base_url: str = OpenRouterBaseURL,
) -> Dict[str, Any]:
    """Request a chat completion and return the raw JSON payload."""

    if config is None and model is None:
        raise ValueError("model or config is required")
    if config is not None and model is not None and config.model_id != model:
        raise ValueError("model and config.model_id must match")
    resolved_model = config.model_id if config is not None else str(model)

    payload: Dict[str, object] = {
        "model": resolved_model,
        "messages": _build_messages(messages, system),
        "stream": False,
    }
    if tools:
        payload["tools"] = tools
    if config is not None:
        payload.update(config.explicit_payload_fields())

    url = f"{base_url.rstrip('/')}/chat/completions"
    timeout_seconds = config.timeout_seconds if config is not None else 60.0
    max_retries = config.max_retries if config is not None else 0
    call_id = uuid4().hex
    _emit_request_evidence(
        role=role,
        config=config,
        model=resolved_model,
        messages=messages,
        tools=tools,
        call_id=call_id,
    )

    async with httpx.AsyncClient() as client:
        for attempt in range(max_retries + 1):
            if attempt:
                _emit_retry_attempt_start(call_id=call_id, attempt=attempt)
            cost_controller = current_cost_controller()
            reservation = (
                cost_controller.reserve_call(
                    payload=payload,
                    max_tokens=config.max_tokens if config is not None else 1000,
                    logical_call_id=UUID(call_id),
                    attempt=attempt,
                )
                if cost_controller is not None
                else None
            )
            try:
                response = await client.post(
                    url,
                    headers=_headers(api_key=api_key),
                    json=payload,
                    timeout=timeout_seconds,
                )
            except asyncio.CancelledError as exc:
                if cost_controller is not None and reservation is not None:
                    cost_controller.settle_unreported(reservation, outcome="cancelled")
                _emit_usage_best_effort(
                    None,
                    role=role,
                    call_id=call_id,
                    attempt=attempt,
                    observation_status="cancelled",
                )
                _emit_error_evidence(
                    role=role,
                    model=resolved_model,
                    config=config,
                    error=exc,
                    retry_count=attempt,
                    call_id=call_id,
                    attempt=attempt,
                )
                raise
            except httpx.HTTPError as exc:
                if cost_controller is not None and reservation is not None:
                    cost_controller.settle_unreported(reservation, outcome="failed")
                _emit_usage_best_effort(
                    None,
                    role=role,
                    call_id=call_id,
                    attempt=attempt,
                    observation_status="transport_error",
                )
                _emit_error_evidence(
                    role=role,
                    model=resolved_model,
                    config=config,
                    error=exc,
                    retry_count=attempt,
                    call_id=call_id,
                    attempt=attempt,
                )
                if attempt < max_retries:
                    continue
                raise OpenRouterError(f"OpenRouter request failed: {exc}") from exc
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                try:
                    error_payload = response.json()
                except Exception:
                    error_payload = None
                if cost_controller is not None and reservation is not None:
                    if error_payload is not None:
                        cost_controller.settle_response(
                            reservation, error_payload, outcome="failed"
                        )
                    else:
                        cost_controller.settle_unreported(
                            reservation, outcome="failed"
                        )
                if error_payload is not None:
                    _emit_usage_best_effort(
                        error_payload,
                        role=role,
                        call_id=call_id,
                        attempt=attempt,
                        observation_status="http_error",
                    )
                else:
                    _emit_usage_best_effort(
                        None,
                        role=role,
                        call_id=call_id,
                        attempt=attempt,
                        observation_status="malformed_error_response",
                    )
                _emit_error_evidence(
                    role=role,
                    model=resolved_model,
                    config=config,
                    error=exc,
                    retry_count=attempt,
                    response=response,
                    call_id=call_id,
                    attempt=attempt,
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
                    retry_header = response.headers.get("retry-after")
                    try:
                        backoff = float(retry_header) if retry_header else min(2.0 ** attempt + 1.5, 10.0)
                    except (ValueError, TypeError):
                        backoff = min(2.0 ** attempt + 1.5, 10.0)
                    await asyncio.sleep(backoff)
                    continue
                _handle_response_error(exc)
            try:
                raw_response = response.json()
            except asyncio.CancelledError as exc:
                if cost_controller is not None and reservation is not None:
                    cost_controller.settle_unreported(reservation, outcome="cancelled")
                _emit_usage_best_effort(
                    None,
                    role=role,
                    call_id=call_id,
                    attempt=attempt,
                    observation_status="cancelled",
                )
                _emit_error_evidence(
                    role=role,
                    model=resolved_model,
                    config=config,
                    error=exc,
                    retry_count=attempt,
                    response=response,
                    call_id=call_id,
                    attempt=attempt,
                )
                raise
            except Exception as exc:
                if cost_controller is not None and reservation is not None:
                    cost_controller.settle_unreported(reservation, outcome="failed")
                _emit_usage_best_effort(
                    None,
                    role=role,
                    call_id=call_id,
                    attempt=attempt,
                    observation_status="malformed_success_response",
                )
                _emit_error_evidence(
                    role=role,
                    model=resolved_model,
                    config=config,
                    error=exc,
                    retry_count=attempt,
                    response=response,
                    call_id=call_id,
                    attempt=attempt,
                )
                raise
            if not isinstance(raw_response, dict):
                if cost_controller is not None and reservation is not None:
                    cost_controller.settle_response(
                        reservation, raw_response, outcome="failed"
                    )
                error = OpenRouterError("OpenRouter response must be a JSON object")
                _emit_error_evidence(
                    role=role,
                    model=resolved_model,
                    config=config,
                    error=error,
                    retry_count=attempt,
                    response=response,
                    call_id=call_id,
                    attempt=attempt,
                )
                _emit_usage_best_effort(
                    raw_response,
                    role=role,
                    call_id=call_id,
                    attempt=attempt,
                    observation_status="malformed_success_response",
                )
                raise error
            _emit_response_evidence(
                role=role,
                config=config,
                requested_model=resolved_model,
                response=raw_response,
                http_response=response,
                retry_count=attempt,
                call_id=call_id,
                attempt=attempt,
            )
            _emit_usage_best_effort(
                raw_response,
                role=role,
                call_id=call_id,
                attempt=attempt,
                observation_status="success",
            )
            if cost_controller is not None and reservation is not None:
                cost_controller.settle_response(
                    reservation, raw_response, outcome="success"
                )
            return raw_response

    raise OpenRouterError("OpenRouter request failed: unknown error")


__all__ = ["OpenRouterError", "request_chat_completion", "OpenRouterBaseURL"]
