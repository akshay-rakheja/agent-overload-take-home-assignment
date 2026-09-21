"""Deterministic recursive redaction for every trace export boundary."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit, urlunsplit


REDACTED = "[REDACTED]"
REDACTED_EMAIL = "[REDACTED_EMAIL]"

_EMAIL = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
_BEARER = re.compile(r"(?i)\bbearer\s+[A-Z0-9._~+/=-]+")
_SECRET = re.compile(
    r"(?i)(?:\bsk-[A-Z0-9_-]{8,}|\b(?:api[_ -]?key|access[_ -]?token|client[_ -]?secret)\b\s*[:=]\s*\S+)"
)

_SECRET_KEYS = frozenset(
    {
        "authorization",
        "proxy_authorization",
        "api_key",
        "x_api_key",
        "auth_config_id",
        "oauth_code",
        "authorization_code",
        "access_token",
        "refresh_token",
        "id_token",
        "client_secret",
        "password",
        "secret",
        "token",
    }
)
_MAIL_KEYS = frozenset(
    {
        "email",
        "email_address",
        "address",
        "from",
        "to",
        "cc",
        "bcc",
        "sender",
        "recipient",
        "message_id",
        "thread_id",
        "gmail_message_id",
        "gmail_thread_id",
        "snippet",
        "body",
        "raw_body",
        "html_body",
    }
)
_ERROR_KEYS = frozenset(
    {
        "provider_error",
        "provider_exception",
        "exception",
        "error_message",
        "error_detail",
    }
)
_QUERY_KEYS = frozenset({"query", "query_string", "raw_query"})
_HEADER_KEYS = frozenset({"headers", "request_headers", "response_headers"})


def _normalized_key(key: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")


def _redact_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return REDACTED
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return value
    host = parsed.hostname
    if parsed.port:
        host = f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path, "", ""))


def _redact_string(value: str) -> str:
    if _BEARER.search(value) or _SECRET.search(value):
        return REDACTED
    if value.startswith(("http://", "https://")):
        value = _redact_url(value)
    return _EMAIL.sub(REDACTED_EMAIL, value)


def _redact_headers(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): REDACTED for key in sorted(value, key=lambda item: str(item))}
    return REDACTED


def redact_value(
    value: object, *, _key: str | None = None, _parent_key: str | None = None
) -> Any:
    """Return a sorted, JSON-safe copy with sensitive observations removed."""

    key = _normalized_key(_key) if _key is not None else None
    parent_key = _normalized_key(_parent_key) if _parent_key is not None else None
    if key in _SECRET_KEYS or key in _MAIL_KEYS or key in _ERROR_KEYS or key in _QUERY_KEYS:
        return REDACTED
    if key == "code" and parent_key in {"authorization", "oauth", "oauth_response"}:
        return REDACTED
    if key in _HEADER_KEYS:
        return _redact_headers(value)

    if isinstance(value, Mapping):
        return {
            str(item_key): redact_value(
                value[item_key], _key=str(item_key), _parent_key=key
            )
            for item_key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (list, tuple)):
        return [redact_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [redact_value(item) for item in sorted(value, key=repr)]
    if isinstance(value, str):
        return _redact_string(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, BaseException):
        return REDACTED
    return REDACTED


__all__ = ["REDACTED", "REDACTED_EMAIL", "redact_value"]
