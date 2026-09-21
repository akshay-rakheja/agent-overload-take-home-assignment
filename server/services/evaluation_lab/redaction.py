"""Deterministic recursive redaction for every trace export boundary."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit, urlunsplit


REDACTED = "[REDACTED]"
REDACTED_EMAIL = "[REDACTED_EMAIL]"

_EMAIL = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
_BEARER = re.compile(r"(?i)\bbearer\s+[A-Z0-9._~+/=-]+")
_SECRET = re.compile(
    r"(?i)(?:\bsk-[A-Z0-9_-]{8,}|"
    r"\b(?:api[_ -]?key|access[_ -]?token|client[_ -]?secret|password|token|secret|key)"
    r"\b\s*[:=]\s*\S+)"
)

_SECRET_KEYS = frozenset(
    {
        "authorization",
        "proxyauthorization",
        "apikey",
        "xapikey",
        "authconfigid",
        "oauthcode",
        "authorizationcode",
        "accesstoken",
        "refreshtoken",
        "idtoken",
        "clientsecret",
        "password",
        "secret",
        "token",
    }
)
_MAIL_KEYS = frozenset(
    {
        "email",
        "emailaddress",
        "address",
        "from",
        "to",
        "cc",
        "bcc",
        "sender",
        "recipient",
        "messageid",
        "threadid",
        "gmailmessageid",
        "gmailthreadid",
        "snippet",
        "body",
        "rawbody",
        "htmlbody",
    }
)
_ERROR_KEYS = frozenset(
    {
        "providererror",
        "providerexception",
        "exception",
        "errormessage",
        "errordetail",
    }
)
_QUERY_KEYS = frozenset({"query", "querystring", "rawquery"})
_HEADER_KEYS = frozenset({"headers", "requestheaders", "responseheaders"})
_SECRET_SUFFIXES = (
    "apikey",
    "authconfigid",
    "oauthcode",
    "authorizationcode",
    "token",
    "clientsecret",
    "password",
    "secret",
)
_MAIL_SUFFIXES = ("address", "emailaddress", "messageid", "threadid")


def _canonical_key(key: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(key).lower())


def _matches_key(
    key: str | None, exact: frozenset[str], suffixes: tuple[str, ...] = ()
) -> bool:
    return key is not None and (key in exact or key.endswith(suffixes))


def _is_address_key(key: str | None) -> bool:
    return key is not None and "address" in key


def _is_oauth_wrapper(key: str | None) -> bool:
    return key is not None and ("oauth" in key or "authorization" in key)


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
    if value.lower().startswith(("http://", "https://")):
        value = _redact_url(value)
    return _EMAIL.sub(REDACTED_EMAIL, value)


def contains_secret_material(value: str) -> bool:
    """Return whether free text contains a recognized bearer or secret form."""

    return bool(_BEARER.search(value) or _SECRET.search(value))


def _redact_headers(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): REDACTED for key in sorted(value, key=lambda item: str(item))}
    return REDACTED


def redact_value(
    value: object, *, _key: str | None = None, _parent_key: str | None = None
) -> Any:
    """Return a sorted, JSON-safe copy with sensitive observations removed."""

    key = _canonical_key(_key) if _key is not None else None
    parent_key = _canonical_key(_parent_key) if _parent_key is not None else None
    if (
        _matches_key(key, _SECRET_KEYS, _SECRET_SUFFIXES)
        or _matches_key(key, _MAIL_KEYS, _MAIL_SUFFIXES)
        or _is_address_key(key)
        or _matches_key(key, _ERROR_KEYS)
        or _matches_key(key, _QUERY_KEYS)
    ):
        return REDACTED
    if key == "code" and _is_oauth_wrapper(parent_key):
        return REDACTED
    if _matches_key(key, _HEADER_KEYS, ("headers",)):
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
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON numbers must be finite")
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, BaseException):
        return REDACTED
    return REDACTED


__all__ = [
    "REDACTED",
    "REDACTED_EMAIL",
    "contains_secret_material",
    "redact_value",
]
