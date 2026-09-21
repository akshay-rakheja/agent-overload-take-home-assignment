from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

import pytest

from evals.live_lab.processes import ManagedProcess, redact_stream


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _server_argv(port: int) -> list[str]:
    script = (
        "import socket; s=socket.socket(); "
        "s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); "
        f"s.bind(('127.0.0.1',{port})); s.listen(); "
        "\nwhile True:\n c,_=s.accept(); c.recv(4096); "
        "c.sendall(b'HTTP/1.1 204 No Content\\r\\nContent-Length: 0\\r\\n\\r\\n'); c.close()"
    )
    return [sys.executable, "-c", script]


def test_process_becomes_ready_and_stops_with_sigterm(tmp_path: Path) -> None:
    port = _free_port()
    marker = tmp_path / "terminated"
    script = (
        "import socket,signal,pathlib; "
        f"m=pathlib.Path({str(marker)!r}); "
        "signal.signal(signal.SIGTERM,lambda *_:(m.write_text('term'),exit(0))); "
        "s=socket.socket(); s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); "
        f"s.bind(('127.0.0.1',{port})); s.listen(); "
        "\nwhile True:\n c,_=s.accept(); c.recv(4096); "
        "c.sendall(b'HTTP/1.1 204 No Content\\r\\nContent-Length: 0\\r\\n\\r\\n'); c.close()"
    )
    process = ManagedProcess(
        argv=[sys.executable, "-c", script],
        env={"PATH": os.environ.get("PATH", "")},
        cwd=tmp_path,
        pid_file=tmp_path / "server.pid",
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
    )

    process.start()
    process.wait_ready("http://127.0.0.1:%d" % port, timeout=5)
    process.stop(timeout=5)

    assert marker.read_text() == "term"
    assert not (tmp_path / "server.pid").exists()


def test_duplicate_live_pid_is_refused(tmp_path: Path) -> None:
    pid_file = tmp_path / "server.pid"
    first = ManagedProcess(
        argv=_server_argv(_free_port()),
        env={},
        cwd=tmp_path,
        pid_file=pid_file,
        stdout_path=tmp_path / "one.out",
        stderr_path=tmp_path / "one.err",
    )
    second = ManagedProcess(
        argv=_server_argv(_free_port()),
        env={},
        cwd=tmp_path,
        pid_file=pid_file,
        stdout_path=tmp_path / "two.out",
        stderr_path=tmp_path / "two.err",
    )
    first.start()
    try:
        with pytest.raises(RuntimeError, match="already running"):
            second.start()
    finally:
        first.stop(timeout=5)


def test_stale_pid_is_removed_before_start(tmp_path: Path) -> None:
    pid_file = tmp_path / "server.pid"
    pid_file.write_text("99999999\n", encoding="ascii")
    process = ManagedProcess(
        argv=[sys.executable, "-c", "pass"],
        env={},
        cwd=tmp_path,
        pid_file=pid_file,
        stdout_path=tmp_path / "out",
        stderr_path=tmp_path / "err",
    )

    process.start()
    process.stop(timeout=5)

    assert not pid_file.exists()


def test_redaction_covers_bearer_auth_and_oauth_markers() -> None:
    raw = (
        "Authorization: Bearer fake-token-123\n"
        "OPENROUTER_API_KEY=fake-secret\n"
        "oauth_token: oauth-secret\n"
        "client_secret=client-secret-value\n"
    )

    redacted = redact_stream(raw)

    assert "fake-token-123" not in redacted
    assert "fake-secret" not in redacted
    assert "oauth-secret" not in redacted
    assert "client-secret-value" not in redacted
    assert redacted.count("[REDACTED]") == 4


def test_readiness_timeout_stops_process(tmp_path: Path) -> None:
    process = ManagedProcess(
        argv=[sys.executable, "-c", "import time; time.sleep(30)"],
        env={},
        cwd=tmp_path,
        pid_file=tmp_path / "server.pid",
        stdout_path=tmp_path / "out",
        stderr_path=tmp_path / "err",
    )
    process.start()

    with pytest.raises(TimeoutError, match="ready"):
        process.wait_ready("http://127.0.0.1:%d" % _free_port(), timeout=0.15)

    process.stop(timeout=5)
    assert not (tmp_path / "server.pid").exists()
