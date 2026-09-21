from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import status
from fastapi.responses import JSONResponse

from ...config import Settings, get_settings
from ...logging_config import logger
from ...models import GmailConnectPayload, GmailDisconnectPayload, GmailStatusPayload
from ...utils import error_response
from ..evaluation_lab import LabToolPolicy, lab_tool_rejection


_CLIENT_LOCK = threading.Lock()
_CLIENT: Optional[Any] = None

_PROFILE_CACHE: Dict[str, Dict[str, Any]] = {}
_PROFILE_CACHE_LOCK = threading.Lock()
_ACTIVE_USER_ID_LOCK = threading.Lock()
_ACTIVE_USER_ID: Optional[str] = None


def _normalized(value: Optional[str]) -> str:
    return (value or "").strip()


def _value(obj: Any, *names: str) -> Any:
    """Read the first populated snake/camel field from an SDK object or mapping."""

    for name in names:
        if isinstance(obj, dict) and name in obj and obj[name] is not None:
            return obj[name]
        try:
            candidate = getattr(obj, name)
        except (AttributeError, TypeError):
            continue
        if candidate is not None:
            return candidate
    return None


def _listed_accounts(result: Any) -> list[Any]:
    items = _value(result, "items", "data")
    if isinstance(items, (list, tuple)):
        return list(items)
    return []


def _is_active_account(account: Any) -> bool:
    status_value = str(_value(account, "status") or "").upper()
    return status_value in {"CONNECTED", "SUCCESS", "SUCCESSFUL", "ACTIVE", "COMPLETED"}


def _active_account_for(client: Any, user_id: str, auth_config_id: str) -> Any:
    result = client.connected_accounts.list(
        user_ids=[user_id],
        auth_config_ids=[auth_config_id],
        statuses=["ACTIVE"],
    )
    normalized_user_id = _normalized(user_id)
    return next(
        (
            account
            for account in _listed_accounts(result)
            if _is_active_account(account)
            and _normalized(_value(account, "user_id", "userId"))
            == normalized_user_id
        ),
        None,
    )


def _set_active_gmail_user_id(user_id: Optional[str]) -> None:
    sanitized = _normalized(user_id)
    with _ACTIVE_USER_ID_LOCK:
        global _ACTIVE_USER_ID
        _ACTIVE_USER_ID = sanitized or None


def get_active_gmail_user_id() -> Optional[str]:
    with _ACTIVE_USER_ID_LOCK:
        return _ACTIVE_USER_ID


def _gmail_import_client():
    from composio import Composio  # type: ignore
    return Composio


# Get or create a singleton Composio client instance with thread-safe initialization
def _get_composio_client(settings: Optional[Settings] = None):
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT

    with _CLIENT_LOCK:
        if _CLIENT is None:
            resolved_settings = settings or get_settings()
            Composio = _gmail_import_client()
            api_key = resolved_settings.composio_api_key
            try:
                _CLIENT = Composio(api_key=api_key) if api_key else Composio()
            except TypeError as exc:
                if api_key:
                    raise RuntimeError(
                        "Installed Composio SDK does not accept the api_key argument; upgrade the SDK or remove COMPOSIO_API_KEY."
                    ) from exc
                _CLIENT = Composio()
    return _CLIENT


def _extract_email(obj: Any) -> Optional[str]:
    if obj is None:
        return None
    direct_keys = (
        "email",
        "email_address",
        "emailAddress",
        "user_email",
        "provider_email",
        "account_email",
    )
    for key in direct_keys:
        try:
            val = getattr(obj, key)
            if isinstance(val, str) and "@" in val:
                return val
        except Exception:
            pass
        if isinstance(obj, dict):
            val = obj.get(key)
            if isinstance(val, str) and "@" in val:
                return val
    if isinstance(obj, dict):
        email_addresses = obj.get("emailAddresses")
        if isinstance(email_addresses, (list, tuple)):
            for entry in email_addresses:
                if isinstance(entry, dict):
                    candidate = entry.get("value") or entry.get("email") or entry.get("emailAddress")
                    if isinstance(candidate, str) and "@" in candidate:
                        return candidate
                elif isinstance(entry, str) and "@" in entry:
                    return entry
    if isinstance(obj, dict):
        nested_paths = (
            ("profile", "email"),
            ("profile", "emailAddress"),
            ("user", "email"),
            ("data", "email"),
            ("data", "user", "email"),
            ("provider_profile", "email"),
        )
        for path in nested_paths:
            current: Any = obj
            for segment in path:
                if isinstance(current, dict) and segment in current:
                    current = current[segment]
                else:
                    current = None
                    break
            if isinstance(current, str) and "@" in current:
                return current
    return None


def _cache_profile(user_id: str, profile: Dict[str, Any]) -> None:
    sanitized = _normalized(user_id)
    if not sanitized or not isinstance(profile, dict):
        return
    with _PROFILE_CACHE_LOCK:
        _PROFILE_CACHE[sanitized] = {
            "profile": profile,
            "cached_at": datetime.utcnow().isoformat(),
        }


def _get_cached_profile(user_id: Optional[str]) -> Optional[Dict[str, Any]]:
    sanitized = _normalized(user_id)
    if not sanitized:
        return None
    with _PROFILE_CACHE_LOCK:
        payload = _PROFILE_CACHE.get(sanitized)
        if payload and isinstance(payload.get("profile"), dict):
            return payload["profile"]
    return None


def _clear_cached_profile(user_id: Optional[str] = None) -> None:
    with _PROFILE_CACHE_LOCK:
        if user_id:
            _PROFILE_CACHE.pop(_normalized(user_id), None)
        else:
            _PROFILE_CACHE.clear()


def _fetch_profile_from_composio(user_id: Optional[str]) -> Optional[Dict[str, Any]]:
    sanitized = _normalized(user_id)
    if not sanitized:
        return None
    try:
        result = execute_gmail_tool("GMAIL_GET_PROFILE", sanitized, arguments={"user_id": "me"})
    except RuntimeError:
        logger.warning("Gmail profile lookup failed")
        return None
    except Exception:  # pragma: no cover - defensive
        logger.error("Unexpected error fetching Gmail profile")
        return None

    profile: Optional[Dict[str, Any]] = None
    if isinstance(result, dict):
        if isinstance(result.get("data"), dict):
            profile = result["data"]
        elif isinstance(result.get("profile"), dict):
            profile = result["profile"]
        elif isinstance(result.get("response_data"), dict):
            profile = result["response_data"]
        elif isinstance(result.get("items"), list):
            for item in result["items"]:
                if not isinstance(item, dict):
                    continue
                data_dict = item.get("data")
                if isinstance(data_dict, dict):
                    if isinstance(data_dict.get("response_data"), dict):
                        profile = data_dict["response_data"]
                    elif isinstance(data_dict.get("profile"), dict):
                        profile = data_dict["profile"]
                    else:
                        profile = data_dict
                elif isinstance(item.get("response_data"), dict):
                    profile = item["response_data"]
                elif isinstance(item.get("profile"), dict):
                    profile = item["profile"]
                if isinstance(profile, dict):
                    break
        elif result.get("successful") is True and isinstance(result.get("result"), dict):
            profile = result.get("result")  # type: ignore[assignment]
        elif all(not isinstance(result.get(key), dict) for key in ("data", "profile", "result")):
            profile = result if result else None

    if isinstance(profile, dict):
        _cache_profile(sanitized, profile)
        return profile

    logger.warning("Received unexpected Gmail profile response")
    return None


# Start Gmail OAuth connection process and return redirect URL
def initiate_connect(payload: GmailConnectPayload, settings: Settings) -> JSONResponse:
    auth_config_id = payload.auth_config_id or settings.composio_gmail_auth_config_id or ""
    if not auth_config_id:
        return error_response(
            "Gmail connection is not configured.",
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    requested_user_id = _normalized(payload.user_id)
    if settings.lab_enabled:
        configured_user_id = _normalized(settings.lab_composio_user_id)
        if requested_user_id and requested_user_id != configured_user_id:
            return error_response(
                "The requested Gmail user is not authorized for Evaluation Lab mode.",
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        user_id = configured_user_id
    else:
        user_id = requested_user_id or f"web-{os.getpid()}"

    _set_active_gmail_user_id(user_id)
    _clear_cached_profile(user_id)
    try:
        client = _get_composio_client(settings)
        active_account = _active_account_for(client, user_id, auth_config_id)
        req = active_account or client.connected_accounts.link(
            user_id=user_id,
            auth_config_id=auth_config_id,
        )
        data = {
            "ok": True,
            "redirect_url": _value(req, "redirect_url", "redirectUrl"),
            "connection_request_id": _value(
                req,
                "connection_request_id",
                "connectionRequestId",
                "id",
                "connected_account_id",
                "connectedAccountId",
            ),
            "user_id": _value(req, "user_id", "userId") or user_id,
        }
        return JSONResponse(data)
    except Exception:
        logger.error("Gmail connection request failed")
        return error_response(
            "Failed to connect Gmail.",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


# Check Gmail connection status and retrieve user account information
def fetch_status(payload: GmailStatusPayload) -> JSONResponse:
    connection_request_id = _normalized(payload.connection_request_id)
    user_id = _normalized(payload.user_id)

    if not connection_request_id and not user_id:
        return error_response(
            "Missing connection_request_id or user_id",
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    try:
        client = _get_composio_client()
        account: Any = None
        if connection_request_id:
            try:
                account = client.connected_accounts.wait_for_connection(connection_request_id, timeout=2.0)
            except Exception:
                try:
                    account = client.connected_accounts.get(connection_request_id)
                except Exception:
                    account = None
        if account is None and user_id:
            try:
                items = client.connected_accounts.list(
                    user_ids=[user_id], toolkit_slugs=["GMAIL"], statuses=["ACTIVE"]
                )
                data = _listed_accounts(items)
                if data:
                    account = data[0]
            except Exception:
                account = None
        status_value = None
        email = None
        connected = False
        profile: Optional[Dict[str, Any]] = None
        profile_source = "none"

        account_user_id = None
        if account is not None:
            status_value = _value(account, "status")
            connected = _is_active_account(account)
            email = _extract_email(account)
            account_user_id = _value(account, "user_id", "userId")

        if not user_id and account_user_id:
            user_id = _normalized(account_user_id)

        if connected and user_id:
            cached_profile = _get_cached_profile(user_id)
            if cached_profile:
                profile = cached_profile
                profile_source = "cache"
            else:
                fetched_profile = _fetch_profile_from_composio(user_id)
                if fetched_profile:
                    profile = fetched_profile
                    profile_source = "fetched"
            if profile and not email:
                email = _extract_email(profile)
        elif user_id:
            _clear_cached_profile(user_id)

        _set_active_gmail_user_id(user_id)

        return JSONResponse(
            {
                "ok": True,
                "connected": bool(connected),
                "status": status_value or "UNKNOWN",
                "email": email,
                "user_id": user_id,
                "profile": profile,
                "profile_source": profile_source,
            }
        )
    except Exception:
        logger.error("Gmail connection status lookup failed")
        return error_response(
            "Failed to fetch connection status",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


def disconnect_account(payload: GmailDisconnectPayload) -> JSONResponse:
    connection_id = _normalized(payload.connection_id) or _normalized(payload.connection_request_id)
    user_id = _normalized(payload.user_id)

    if not connection_id and not user_id:
        return error_response(
            "Missing connection_id or user_id",
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    try:
        client = _get_composio_client()
    except Exception:
        logger.error("Gmail disconnect client initialization failed")
        return error_response(
            "Failed to disconnect Gmail",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    removed_ids: list[str] = []
    errors: list[str] = []
    affected_user_ids: set[str] = set()

    def _delete_connection(identifier: str) -> None:
        sanitized_id = _normalized(identifier)
        if not sanitized_id:
            return
        try:
            connection = client.connected_accounts.get(sanitized_id)
        except Exception:
            connection = None
        try:
            client.connected_accounts.delete(sanitized_id)
            removed_ids.append(sanitized_id)
            if connection is not None:
                if hasattr(connection, "user_id"):
                    affected_user_ids.add(_normalized(getattr(connection, "user_id", None)))
                elif isinstance(connection, dict):
                    affected_user_ids.add(_normalized(connection.get("user_id")))
        except Exception:  # pragma: no cover - depends on remote state
            logger.error("Failed to remove Gmail connection")
            errors.append("Unable to remove a Gmail connection.")

    if connection_id:
        _delete_connection(connection_id)
    else:
        try:
            items = client.connected_accounts.list(user_ids=[user_id], toolkit_slugs=["GMAIL"])
            data = _listed_accounts(items)
        except Exception:  # pragma: no cover - dependent on SDK
            logger.error("Failed to list Gmail connections")
            return error_response(
                "Failed to disconnect Gmail",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        if data:
            for entry in data:
                candidate = None
                candidate_user_id = None
                candidate = _value(entry, "id", "connection_id", "connectionId")
                candidate_user_id = _value(entry, "user_id", "userId")
                if candidate:
                    if candidate_user_id:
                        affected_user_ids.add(_normalized(candidate_user_id))
                    _delete_connection(candidate)

    if user_id:
        affected_user_ids.add(user_id)

    for uid in list(affected_user_ids):
        if uid:
            _clear_cached_profile(uid)
            if get_active_gmail_user_id() == uid:
                _set_active_gmail_user_id(None)

    if errors and not removed_ids:
        return error_response(
            "Failed to disconnect Gmail",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="; ".join(errors),
        )

    payload = {
        "ok": True,
        "disconnected": bool(removed_ids),
        "removed_connection_ids": removed_ids,
    }
    if not removed_ids:
        payload["message"] = "No Gmail connection found"

    if errors:
        payload["warnings"] = errors
    return JSONResponse(payload)


def _normalize_tool_response(result: Any) -> Dict[str, Any]:
    payload_dict: Optional[Dict[str, Any]] = None
    try:
        if hasattr(result, "model_dump"):
            payload_dict = result.model_dump()  # type: ignore[assignment]
        elif hasattr(result, "dict"):
            payload_dict = result.dict()  # type: ignore[assignment]
    except Exception:
        payload_dict = None

    if payload_dict is None:
        try:
            if hasattr(result, "model_dump_json"):
                payload_dict = json.loads(result.model_dump_json())
        except Exception:
            payload_dict = None

    if payload_dict is None:
        if isinstance(result, dict):
            payload_dict = result
        elif isinstance(result, list):
            payload_dict = {"items": result}
        else:
            payload_dict = {"error": "Unexpected Gmail provider response."}

    return payload_dict


# Execute Gmail operations through Composio SDK with error handling
def execute_gmail_tool(
    tool_name: str,
    composio_user_id: str,
    *,
    arguments: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    settings = get_settings()
    if settings.lab_enabled:
        configured_user_id = _normalized(settings.lab_composio_user_id)
        if _normalized(composio_user_id) != configured_user_id:
            return {
                "error": {
                    "code": "lab_user_mismatch",
                    "tool": tool_name,
                    "reason": "The Gmail user is not authorized for Evaluation Lab mode.",
                }
            }
        decision = LabToolPolicy().decide_composio_tool(tool_name)
        if not decision.allowed:
            return lab_tool_rejection(tool_name, decision)

    prepared_arguments: Dict[str, Any] = {}
    if isinstance(arguments, dict):
        for key, value in arguments.items():
            if value is not None:
                prepared_arguments[key] = value

    prepared_arguments.setdefault("user_id", "me")

    try:
        client = _get_composio_client()
        result = client.client.tools.execute(
            tool_name,
            user_id=composio_user_id,
            arguments=prepared_arguments,
        )
        return _normalize_tool_response(result)
    except Exception:
        logger.error("Gmail tool execution failed", extra={"tool": tool_name})
        raise RuntimeError("Gmail tool execution failed") from None
