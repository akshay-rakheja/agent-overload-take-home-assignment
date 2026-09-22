"""Shared pytest configuration for credential-free tests."""

from __future__ import annotations

import sys
from pathlib import Path


import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(autouse=True)
def _clean_lab_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENPOKE_LAB_ENABLED", raising=False)
    from server.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()

