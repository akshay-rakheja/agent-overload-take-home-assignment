"""Tests for local Evaluation Lab multi-process lifecycle manager."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

from evals.live_lab.lifecycle import LabLifecycle, ProcessConfig


def test_lifecycle_start_and_stop_order() -> None:
    events: list[str] = []

    mock_keep_awake = MagicMock()
    mock_keep_awake.start.side_effect = lambda: events.append("start:keep_awake")
    mock_keep_awake.stop.side_effect = lambda **_: events.append("stop:keep_awake")
    mock_keep_awake.is_running = True

    mock_baseline = MagicMock()
    mock_baseline.start.side_effect = lambda: events.append("start:baseline")
    mock_baseline.wait_ready.side_effect = lambda url, timeout: events.append(f"ready:baseline:{url}")
    mock_baseline.stop.side_effect = lambda **_: events.append("stop:baseline")
    mock_baseline.is_running = True

    mock_enhanced = MagicMock()
    mock_enhanced.start.side_effect = lambda: events.append("start:enhanced")
    mock_enhanced.wait_ready.side_effect = lambda url, timeout: events.append(f"ready:enhanced:{url}")
    mock_enhanced.stop.side_effect = lambda **_: events.append("stop:enhanced")
    mock_enhanced.is_running = True

    mock_ui = MagicMock()
    mock_ui.start.side_effect = lambda: events.append("start:ui")
    mock_ui.wait_ready.side_effect = lambda url, timeout: events.append(f"ready:ui:{url}")
    mock_ui.stop.side_effect = lambda **_: events.append("stop:ui")
    mock_ui.is_running = True

    lifecycle = LabLifecycle(
        keep_awake_proc=mock_keep_awake,
        baseline_proc=mock_baseline,
        enhanced_proc=mock_enhanced,
        ui_proc=mock_ui,
    )

    lifecycle.start(readiness_timeout=1.0)
    assert events == [
        "start:keep_awake",
        "start:baseline",
        "ready:baseline:http://127.0.0.1:8001/api/v1/health",
        "start:enhanced",
        "ready:enhanced:http://127.0.0.1:8002/api/v1/health",
        "start:ui",
        "ready:ui:http://127.0.0.1:3000/lab",
    ]

    events.clear()
    lifecycle.stop(stop_timeout=1.0)
    # Stop order: UI, enhanced, baseline, keep-awake
    assert events == [
        "stop:ui",
        "stop:enhanced",
        "stop:baseline",
        "stop:keep_awake",
    ]


def test_lifecycle_exceptional_cleanup_on_start_failure() -> None:
    events: list[str] = []

    mock_keep_awake = MagicMock()
    mock_keep_awake.start.side_effect = lambda: events.append("start:keep_awake")
    mock_keep_awake.stop.side_effect = lambda **_: events.append("stop:keep_awake")

    mock_baseline = MagicMock()
    mock_baseline.start.side_effect = lambda: events.append("start:baseline")
    mock_baseline.wait_ready.side_effect = lambda url, timeout: events.append("ready:baseline")
    mock_baseline.stop.side_effect = lambda **_: events.append("stop:baseline")

    mock_enhanced = MagicMock()
    mock_enhanced.start.side_effect = RuntimeError("enhanced crashed on startup")
    mock_enhanced.stop.side_effect = lambda **_: events.append("stop:enhanced")

    mock_ui = MagicMock()
    mock_ui.stop.side_effect = lambda **_: events.append("stop:ui")

    lifecycle = LabLifecycle(
        keep_awake_proc=mock_keep_awake,
        baseline_proc=mock_baseline,
        enhanced_proc=mock_enhanced,
        ui_proc=mock_ui,
    )

    with pytest.raises(RuntimeError, match="enhanced crashed"):
        lifecycle.start(readiness_timeout=1.0)

    # In exceptional cleanup, all already-started processes are stopped in reverse order
    assert "start:keep_awake" in events
    assert "start:baseline" in events
    assert "stop:baseline" in events
    assert "stop:keep_awake" in events


def test_lifecycle_status_reporting() -> None:
    mock_keep_awake = MagicMock(is_running=True, pid=101)
    mock_baseline = MagicMock(is_running=True, pid=102)
    mock_enhanced = MagicMock(is_running=False, pid=None)
    mock_ui = MagicMock(is_running=False, pid=None)

    lifecycle = LabLifecycle(
        keep_awake_proc=mock_keep_awake,
        baseline_proc=mock_baseline,
        enhanced_proc=mock_enhanced,
        ui_proc=mock_ui,
    )

    status = lifecycle.status()
    assert status["keep_awake"]["running"] is True
    assert status["keep_awake"]["pid"] == 101
    assert status["baseline"]["running"] is True
    assert status["enhanced"]["running"] is False
    assert status["ui"]["running"] is False
