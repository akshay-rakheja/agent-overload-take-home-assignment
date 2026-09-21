"""Server CLI safety checks."""

from __future__ import annotations

import pytest

from server import server
from server.config import Settings


def test_lab_enabled_cli_rejects_unsafe_host_override(monkeypatch) -> None:
    settings = Settings(
        lab_enabled=True,
        lab_composio_user_id="opaque-fixture-user",
        server_host="127.0.0.1",
    )
    monkeypatch.setattr(server, "get_settings", lambda: settings)
    monkeypatch.setattr(server.uvicorn, "run", lambda *args, **kwargs: pytest.fail("unsafe bind started"))
    monkeypatch.setattr("sys.argv", ["openpoke-server", "--host", "0.0.0.0"])

    with pytest.raises(SystemExit):
        server.main()
