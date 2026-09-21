from __future__ import annotations

import os
import socket
import sys
from concurrent.futures import ThreadPoolExecutor
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

    assert not (tmp_path / "server.pid").exists()
    assert process.pid is None or not process.is_running


def test_stream_exports_are_metadata_only_and_raw_text_is_private(tmp_path: Path) -> None:
    secret = 'provider body {"access_token":"nested-secret","mailbox":"private body"}'
    process = ManagedProcess(
        argv=[sys.executable, "-c", f"import sys; print({secret!r}); print({secret!r}, file=sys.stderr)"],
        env={},
        cwd=tmp_path,
        pid_file=tmp_path / "server.pid",
        stdout_path=tmp_path / "stdout.json",
        stderr_path=tmp_path / "stderr.json",
        private_dir=tmp_path / "private",
    )

    process.start()
    process.wait(timeout=5)
    process.stop(timeout=5)

    exported = (tmp_path / "stdout.json").read_text() + (tmp_path / "stderr.json").read_text()
    assert "nested-secret" not in exported
    assert "private body" not in exported
    assert '"byte_count"' in exported
    assert '"sha256"' in exported
    private = "".join(path.read_text() for path in (tmp_path / "private").iterdir())
    assert "nested-secret" in private
    assert "private body" in private


def test_unrelated_listener_is_rejected_before_child_start(tmp_path: Path) -> None:
    port = _free_port()
    unrelated = ManagedProcess(
        argv=_server_argv(port),
        env={},
        cwd=tmp_path,
        pid_file=tmp_path / "unrelated.pid",
        stdout_path=tmp_path / "unrelated.out",
        stderr_path=tmp_path / "unrelated.err",
    )
    unrelated.start()
    unrelated.wait_ready(f"http://127.0.0.1:{port}", timeout=5)
    child = ManagedProcess(
        argv=[sys.executable, "-c", "import time; time.sleep(30)"],
        env={},
        cwd=tmp_path,
        pid_file=tmp_path / "child.pid",
        stdout_path=tmp_path / "child.out",
        stderr_path=tmp_path / "child.err",
        readiness_host="127.0.0.1",
        readiness_port=port,
        readiness_nonce="owned-child",
    )
    try:
        with pytest.raises(RuntimeError, match="occupied"):
            child.start()
        assert not (tmp_path / "child.pid").exists()
    finally:
        unrelated.stop(timeout=5)


def test_concurrent_start_has_one_atomic_owner(tmp_path: Path) -> None:
    pid_file = tmp_path / "shared.pid"
    processes = [
        ManagedProcess(
            argv=[sys.executable, "-c", "import time; time.sleep(30)"],
            env={},
            cwd=tmp_path,
            pid_file=pid_file,
            stdout_path=tmp_path / f"{index}.out",
            stderr_path=tmp_path / f"{index}.err",
        )
        for index in range(2)
    ]

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(process.start) for process in processes]
        outcomes = []
        for future in futures:
            try:
                future.result()
                outcomes.append("started")
            except RuntimeError:
                outcomes.append("refused")
    try:
        assert sorted(outcomes) == ["refused", "started"]
    finally:
        for process in processes:
            process.stop(timeout=5)


def test_refused_duplicate_stop_cannot_remove_owner_marker(tmp_path: Path) -> None:
    pid_file = tmp_path / "shared.pid"
    owner = ManagedProcess(
        argv=[sys.executable, "-c", "import time; time.sleep(30)"],
        env={},
        cwd=tmp_path,
        pid_file=pid_file,
        stdout_path=tmp_path / "owner.out",
        stderr_path=tmp_path / "owner.err",
    )
    duplicate = ManagedProcess(
        argv=[sys.executable, "-c", "pass"],
        env={},
        cwd=tmp_path,
        pid_file=pid_file,
        stdout_path=tmp_path / "duplicate.out",
        stderr_path=tmp_path / "duplicate.err",
    )
    owner.start()
    try:
        with pytest.raises(RuntimeError, match="already running"):
            duplicate.start()
        duplicate.stop(timeout=1)
        assert pid_file.exists()
        assert owner.is_running
    finally:
        owner.stop(timeout=5)


def test_owner_token_mismatch_preserves_foreign_marker(tmp_path: Path) -> None:
    pid_file = tmp_path / "server.pid"
    process = ManagedProcess(
        argv=[sys.executable, "-c", "import time; time.sleep(30)"],
        env={},
        cwd=tmp_path,
        pid_file=pid_file,
        stdout_path=tmp_path / "out",
        stderr_path=tmp_path / "err",
    )
    process.start()
    pid_file.write_text('{"pid":99999999,"owner_token":"foreign"}\n', encoding="utf-8")

    process.stop(timeout=5)

    assert pid_file.exists()
    pid_file.unlink()
