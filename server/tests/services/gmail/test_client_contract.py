from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from server.config import Settings
from server.models import GmailConnectPayload, GmailDisconnectPayload, GmailStatusPayload
from server.services.gmail import client as gmail_client


FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "composio_link_responses.json"


def _body(response) -> dict:
    return json.loads(response.body)


class FakeConnectedAccounts:
    def __init__(self, *, link_response=None, accounts=()):
        self.link_response = link_response
        self.accounts = list(accounts)
        self.link_calls: list[tuple[str, str]] = []
        self.list_calls: list[dict] = []
        self.wait_calls: list[tuple[str, float]] = []
        self.deleted: list[str] = []

    def initiate(self, **kwargs):  # pragma: no cover - a failure sentinel
        raise AssertionError(f"deprecated initiate called: {sorted(kwargs)}")

    def link(self, user_id: str, auth_config_id: str):
        self.link_calls.append((user_id, auth_config_id))
        if isinstance(self.link_response, Exception):
            raise self.link_response
        return self.link_response

    def list(self, **kwargs):
        self.list_calls.append(kwargs)
        return SimpleNamespace(items=self.accounts)

    def wait_for_connection(self, request_id: str, timeout: float):
        self.wait_calls.append((request_id, timeout))
        return self.accounts[0] if self.accounts else None

    def get(self, connection_id: str):
        return next((item for item in self.accounts if getattr(item, "id", None) == connection_id), None)

    def delete(self, connection_id: str):
        self.deleted.append(connection_id)


class FakeComposio:
    def __init__(self, connected_accounts: FakeConnectedAccounts, *, execute_result=None):
        self.connected_accounts = connected_accounts
        self.execute_calls: list[tuple[str, str, dict]] = []
        self._execute_result = execute_result
        self.client = SimpleNamespace(tools=SimpleNamespace(execute=self._execute))

    def _execute(self, tool_name: str, *, user_id: str, arguments: dict):
        self.execute_calls.append((tool_name, user_id, arguments))
        if isinstance(self._execute_result, Exception):
            raise self._execute_result
        return self._execute_result or {}


@pytest.fixture(autouse=True)
def reset_gmail_state(monkeypatch):
    monkeypatch.setattr(gmail_client, "_CLIENT", None)
    gmail_client._clear_cached_profile()
    gmail_client._set_active_gmail_user_id(None)


@pytest.mark.parametrize("case_name", ["snake_case", "camel_case"])
def test_connect_uses_link_and_normalizes_link_response(case_name, monkeypatch):
    fixture = json.loads(FIXTURES.read_text(encoding="utf-8"))[case_name]
    response_object = SimpleNamespace(**fixture)
    accounts = FakeConnectedAccounts(link_response=response_object)
    monkeypatch.setattr(gmail_client, "_CLIENT", FakeComposio(accounts))
    requested_user = fixture.get("user_id") or fixture["userId"]

    response = gmail_client.initiate_connect(
        GmailConnectPayload(user_id=requested_user),
        Settings(composio_gmail_auth_config_id="fixture-auth-config"),
    )

    payload = _body(response)
    assert response.status_code == 200
    assert payload["redirect_url"] == fixture.get("redirect_url", fixture.get("redirectUrl"))
    assert payload["connection_request_id"] == fixture.get(
        "connection_request_id", fixture.get("connectionRequestId")
    )
    assert payload["user_id"] == requested_user
    assert accounts.link_calls == [(requested_user, "fixture-auth-config")]


def test_lab_settings_require_an_opaque_user_id():
    with pytest.raises(ValidationError, match="OPENPOKE_LAB_COMPOSIO_USER_ID"):
        Settings(
            server_host="127.0.0.1",
            lab_enabled=True,
            lab_composio_user_id=None,
        )


def test_lab_settings_parse_environment_without_changing_model_defaults(monkeypatch):
    monkeypatch.setenv("OPENPOKE_LAB_ENABLED", "true")
    monkeypatch.setenv("OPENPOKE_LAB_COMPOSIO_USER_ID", "opaque-env-user")
    monkeypatch.setenv("OPENPOKE_HOST", "127.0.0.1")

    settings = Settings(server_host="127.0.0.1")

    assert settings.lab_enabled is True
    assert settings.lab_composio_user_id == "opaque-env-user"
    assert settings.interaction_agent_model == "anthropic/claude-sonnet-4"
    assert settings.execution_agent_model == "anthropic/claude-sonnet-4"
    assert settings.execution_agent_search_model == "anthropic/claude-sonnet-4"
    assert settings.summarizer_model == "anthropic/claude-sonnet-4"
    assert settings.email_classifier_model == "anthropic/claude-sonnet-4"


def test_lab_connect_uses_configured_user_and_rejects_conflicting_payload(monkeypatch):
    accounts = FakeConnectedAccounts(
        link_response=SimpleNamespace(
            redirect_url="https://fixture.invalid/connect/lab",
            id="request_lab",
        )
    )
    monkeypatch.setattr(gmail_client, "_CLIENT", FakeComposio(accounts))
    settings = Settings(
        server_host="127.0.0.1",
        lab_enabled=True,
        lab_composio_user_id="opaque-lab-user",
        composio_gmail_auth_config_id="fixture-auth-config",
    )

    accepted = gmail_client.initiate_connect(GmailConnectPayload(), settings)
    rejected = gmail_client.initiate_connect(
        GmailConnectPayload(user_id="different-user"), settings
    )

    assert _body(accepted)["user_id"] == "opaque-lab-user"
    assert rejected.status_code == 400
    assert _body(rejected) == {
        "ok": False,
        "error": "The requested Gmail user is not authorized for Evaluation Lab mode.",
    }
    assert accounts.link_calls == [("opaque-lab-user", "fixture-auth-config")]


def test_connect_reuses_active_account_for_stable_user_without_new_link(monkeypatch):
    active = SimpleNamespace(id="account_active", user_id="opaque-user", status="ACTIVE")
    accounts = FakeConnectedAccounts(accounts=[active])
    monkeypatch.setattr(gmail_client, "_CLIENT", FakeComposio(accounts))

    response = gmail_client.initiate_connect(
        GmailConnectPayload(user_id="opaque-user"),
        Settings(composio_gmail_auth_config_id="fixture-auth-config"),
    )

    assert _body(response) == {
        "ok": True,
        "redirect_url": None,
        "connection_request_id": "account_active",
        "user_id": "opaque-user",
    }
    assert accounts.link_calls == []


def test_connect_ignores_foreign_active_account_before_matching_camel_user(monkeypatch):
    foreign = SimpleNamespace(
        id="account_foreign",
        user_id="different-user",
        status="ACTIVE",
    )
    matching = SimpleNamespace(
        id="account_matching",
        userId="opaque-user",
        status="ACTIVE",
    )
    accounts = FakeConnectedAccounts(accounts=[foreign, matching])
    monkeypatch.setattr(gmail_client, "_CLIENT", FakeComposio(accounts))

    response = gmail_client.initiate_connect(
        GmailConnectPayload(user_id="opaque-user"),
        Settings(composio_gmail_auth_config_id="fixture-auth-config"),
    )

    assert _body(response) == {
        "ok": True,
        "redirect_url": None,
        "connection_request_id": "account_matching",
        "user_id": "opaque-user",
    }
    assert accounts.link_calls == []


def test_connect_relinks_when_no_active_account_exists(monkeypatch):
    accounts = FakeConnectedAccounts(
        link_response=SimpleNamespace(
            redirect_url="https://fixture.invalid/connect/reconnect",
            id="request_reconnect",
        )
    )
    monkeypatch.setattr(gmail_client, "_CLIENT", FakeComposio(accounts))

    response = gmail_client.initiate_connect(
        GmailConnectPayload(user_id="opaque-user"),
        Settings(composio_gmail_auth_config_id="fixture-auth-config"),
    )

    assert _body(response)["connection_request_id"] == "request_reconnect"
    assert accounts.link_calls == [("opaque-user", "fixture-auth-config")]


def test_status_supports_request_polling_and_stable_user_lookup(monkeypatch):
    active = SimpleNamespace(id="account_active", user_id="opaque-user", status="ACTIVE")
    accounts = FakeConnectedAccounts(accounts=[active])
    monkeypatch.setattr(gmail_client, "_CLIENT", FakeComposio(accounts))
    monkeypatch.setattr(
        gmail_client,
        "_fetch_profile_from_composio",
        lambda user_id: {"display_name": "Fixture User"},
    )

    polled = gmail_client.fetch_status(
        GmailStatusPayload(connection_request_id="request_pending")
    )
    looked_up = gmail_client.fetch_status(GmailStatusPayload(user_id="opaque-user"))

    assert _body(polled)["connected"] is True
    assert _body(polled)["user_id"] == "opaque-user"
    assert accounts.wait_calls == [("request_pending", 2.0)]
    assert _body(looked_up)["connected"] is True
    assert accounts.list_calls[-1]["user_ids"] == ["opaque-user"]


def test_profile_lookup_and_disconnect_contracts(monkeypatch):
    active = SimpleNamespace(id="account_active", user_id="opaque-user", status="ACTIVE")
    accounts = FakeConnectedAccounts(accounts=[active])
    client = FakeComposio(
        accounts,
        execute_result={"data": {"email": "fixture.user@example.invalid"}},
    )
    monkeypatch.setattr(gmail_client, "_CLIENT", client)

    status_response = gmail_client.fetch_status(GmailStatusPayload(user_id="opaque-user"))
    disconnect_response = gmail_client.disconnect_account(
        GmailDisconnectPayload(user_id="opaque-user")
    )

    status_payload = _body(status_response)
    assert status_payload["profile"] == {"email": "fixture.user@example.invalid"}
    assert status_payload["profile_source"] == "fetched"
    assert _body(disconnect_response) == {
        "ok": True,
        "disconnected": True,
        "removed_connection_ids": ["account_active"],
    }
    assert accounts.deleted == ["account_active"]
    assert gmail_client.get_active_gmail_user_id() is None


def test_missing_auth_config_is_sanitized():
    response = gmail_client.initiate_connect(
        GmailConnectPayload(user_id="opaque-user"),
        Settings(composio_gmail_auth_config_id=None),
    )

    assert response.status_code == 400
    assert _body(response) == {
        "ok": False,
        "error": "Gmail connection is not configured.",
    }


def test_provider_failure_is_redacted_from_json_and_logs(monkeypatch, caplog):
    marker_values = (
        "auth-config-secret",
        "https://oauth.invalid/private",
        "provider-key-secret",
        "mailbox@example.invalid",
    )
    provider_error = RuntimeError(" ".join(marker_values))
    accounts = FakeConnectedAccounts(link_response=provider_error)
    monkeypatch.setattr(gmail_client, "_CLIENT", FakeComposio(accounts))

    with caplog.at_level(logging.ERROR, logger="openpoke.server"):
        response = gmail_client.initiate_connect(
            GmailConnectPayload(user_id="opaque-user"),
            Settings(composio_gmail_auth_config_id="auth-config-secret"),
        )

    serialized = json.dumps(_body(response)) + caplog.text
    assert response.status_code == 500
    assert _body(response) == {"ok": False, "error": "Failed to connect Gmail."}
    for marker in marker_values:
        assert marker not in serialized


def test_raw_provider_failure_is_redacted(monkeypatch, caplog):
    marker_values = ("https://provider.invalid/private", "provider-key-secret")
    client = FakeComposio(
        FakeConnectedAccounts(),
        execute_result=RuntimeError(" ".join(marker_values)),
    )
    monkeypatch.setattr(gmail_client, "_CLIENT", client)
    monkeypatch.setattr(
        gmail_client,
        "get_settings",
        lambda: Settings(lab_enabled=False),
    )

    with caplog.at_level(logging.ERROR, logger="openpoke.server"):
        with pytest.raises(RuntimeError, match="Gmail tool execution failed") as exc_info:
            gmail_client.execute_gmail_tool("GMAIL_GET_PROFILE", "opaque-user")

    serialized = str(exc_info.value) + caplog.text
    for marker in marker_values:
        assert marker not in serialized
