"""Typed, privacy-safe extraction of provider routing facts."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_CODE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")
_STATUSES = {
    "success",
    "error",
    "failed",
    "timeout",
    "rate_limited",
    "unavailable",
}


def _safe_record(value: Mapping[str, object]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key in ("provider", "provider_id"):
        observed = value.get(key)
        if isinstance(observed, str) and _IDENTIFIER.fullmatch(observed):
            safe[key] = observed
    for key in ("attempt_count", "retry_count"):
        observed = value.get(key)
        if (
            isinstance(observed, int)
            and not isinstance(observed, bool)
            and 0 <= observed <= 1_000
        ):
            safe[key] = observed
    status_code = value.get("status_code")
    if (
        isinstance(status_code, int)
        and not isinstance(status_code, bool)
        and 100 <= status_code <= 599
    ):
        safe["status_code"] = status_code
    for key in ("code", "error_code"):
        observed = value.get(key)
        if isinstance(observed, int) and not isinstance(observed, bool):
            safe[key] = observed
        elif isinstance(observed, str) and _CODE.fullmatch(observed):
            safe[key] = observed
    status = value.get("status")
    if isinstance(status, str) and status in _STATUSES:
        safe["status"] = status
    for key in ("failover", "retried", "success"):
        observed = value.get(key)
        if isinstance(observed, bool):
            safe[key] = observed
    attempts = value.get("attempts")
    if isinstance(attempts, Sequence) and not isinstance(
        attempts, (str, bytes, bytearray)
    ):
        safe_attempts = [
            item
            for candidate in attempts
            if isinstance(candidate, Mapping)
            and (item := _safe_record(candidate))
        ]
        if safe_attempts:
            safe["attempts"] = safe_attempts
    return safe


def safe_provider_routing(value: object) -> dict[str, Any] | None:
    """Return only allowlisted routing facts; never arbitrary provider prose."""

    if isinstance(value, Mapping):
        return _safe_record(value) or None
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        attempts = [
            item
            for candidate in value
            if isinstance(candidate, Mapping)
            and (item := _safe_record(candidate))
        ]
        return {"attempts": attempts} if attempts else None
    return None


__all__ = ["safe_provider_routing"]
