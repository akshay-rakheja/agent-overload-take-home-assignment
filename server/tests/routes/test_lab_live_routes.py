"""Tests for live Evaluation Lab routes: preflight, gmail status, and gmail link."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.config import Settings, get_settings
from server.routes import api_router


def _settings(tmp_path: Path, *, enabled: bool) -> Settings:
    return Settings(
        lab_enabled=enabled,
        lab_composio_user_id="fixture-user" if enabled else None,
        server_host="127.0.0.1",
        lab_run_root=tmp_path / ".lab" / "runs",
        lab_trace_root=tmp_path / ".lab" / "traces",
        lab_revision="feat/live-agent-overload-evaluation-lab",
        lab_model="openai/gpt-4.1-mini",
    )


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    app = FastAPI()
    app.include_router(api_router)
    settings = _settings(tmp_path, enabled=True)
    app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(app, client=("127.0.0.1", 40_000))


def test_lab_preflight_returns_valid_contract(client: TestClient) -> None:
    resp = client.get("/api/v1/lab/preflight")
    assert resp.status_code == 200
    data = resp.json()

    assert data["schema_version"] == 1
    assert isinstance(data["runnable"], bool)
    assert isinstance(data["blockers"], list)
    assert isinstance(data["warnings"], list)
    assert "baseline" in data
    assert "enhanced" in data
    assert "fixture_equivalence" in data
    assert "gmail_safety" in data
    assert "budget" in data

    # Verify baseline and enhanced fields
    assert isinstance(data["baseline"]["reachable"], bool)
    assert data["enhanced"]["reachable"] is True
    assert data["enhanced"]["revision"] == "feat/live-agent-overload-evaluation-lab"
    assert data["enhanced"]["model"] == "openai/gpt-4.1-mini"

    # Verify fixture equivalence
    assert data["fixture_equivalence"]["equivalent"] is True

    # Verify gmail safety
    assert isinstance(data["gmail_safety"]["connected"], bool)
    assert data["gmail_safety"]["read_only"] is True

    # Verify budget
    assert isinstance(data["budget"]["safe"], bool)
    assert data["budget"]["remaining_usd"] is not None

    # Safety invariant: no raw email addresses or OAuth URLs in text fields
    serialized = json.dumps(data)
    assert "@" not in serialized
    assert "http://" not in serialized or "http://127.0.0.1" in serialized
    assert "oauth" not in serialized.lower() or "oauth connection initiated" in serialized.lower()


def test_lab_gmail_status_contract(client: TestClient) -> None:
    resp = client.get("/api/v1/lab/gmail/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["schema_version"] == 1
    assert isinstance(data["connected"], bool)
    assert data["read_only"] is True
    assert isinstance(data["message"], str)
    assert "@" not in data["message"]


def test_lab_gmail_link_contract(client: TestClient) -> None:
    with patch("server.services.gmail.client.initiate_connect") as mock_connect:
        mock_response = MagicMock()
        mock_response.body = json.dumps({
            "ok": True,
            "redirect_url": "https://accounts.google.com/o/oauth2/auth?client_id=fake",
            "connection_request_id": "req-123",
        }).encode("utf-8")
        mock_connect.return_value = mock_response

        resp = client.post("/api/v1/lab/gmail/link")
        assert resp.status_code == 200
        data = resp.json()
        assert data["schema_version"] == 1
        assert data["available"] is True
        assert data["message"] == "OAuth connection initiated. Open the URL printed in the operator terminal."
        # Crucial security assertion: redirect_url must NEVER be returned in response body
        assert "redirect_url" not in data
        assert "oauth2" not in json.dumps(data)


def test_lab_disabled_returns_404(tmp_path: Path) -> None:
    app = FastAPI()
    app.include_router(api_router)
    app.dependency_overrides[get_settings] = lambda: _settings(tmp_path, enabled=False)
    client = TestClient(app, base_url="http://127.0.0.1")

    assert client.get("/api/v1/lab/preflight").status_code == 404
    assert client.get("/api/v1/lab/gmail/status").status_code == 404
    assert client.post("/api/v1/lab/gmail/link").status_code == 404


def test_lab_remote_host_returns_403(tmp_path: Path) -> None:
    app = FastAPI()
    app.include_router(api_router)
    app.dependency_overrides[get_settings] = lambda: _settings(tmp_path, enabled=True)
    client = TestClient(app, base_url="http://192.168.1.50")

    assert client.get("/api/v1/lab/preflight").status_code == 403
    assert client.get("/api/v1/lab/gmail/status").status_code == 403
    assert client.post("/api/v1/lab/gmail/link").status_code == 403
