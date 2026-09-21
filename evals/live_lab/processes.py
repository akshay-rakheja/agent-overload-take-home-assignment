"""Managed subprocesses scoped to one live-lab run."""

from __future__ import annotations

import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import IO, Mapping, Sequence


_REDACTIONS = (
    re.compile(r"(?i)(Authorization\s*:\s*Bearer\s+)[^\s]+"),
    re.compile(r"(?i)((?:OPENROUTER_API_KEY|API[_-]?KEY)\s*=\s*)[^\s]+"),
    re.compile(r"(?i)((?:oauth[_-]?token)\s*[:=]\s*)[^\s]+"),
    re.compile(r"(?i)((?:client[_-]?secret)\s*=\s*)[^\s]+"),
)


def redact_stream(value: str) -> str:
    redacted = value
    for pattern in _REDACTIONS:
        redacted = pattern.sub(lambda match: match.group(1) + "[REDACTED]", redacted)
    return redacted


def _pid_is_live(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class ManagedProcess:
    """Start one explicit argv/env process and own its PID and streams."""

    def __init__(
        self,
        *,
        argv: Sequence[str],
        env: Mapping[str, str],
        cwd: Path,
        pid_file: Path,
        stdout_path: Path,
        stderr_path: Path,
    ) -> None:
        if not argv or not all(isinstance(item, str) and item for item in argv):
            raise ValueError("argv must be a non-empty explicit string sequence")
        self.argv = tuple(argv)
        self.env = dict(env)
        self.cwd = Path(cwd)
        self.pid_file = Path(pid_file)
        self.stdout_path = Path(stdout_path)
        self.stderr_path = Path(stderr_path)
        self._process: subprocess.Popen[str] | None = None
        self._threads: list[threading.Thread] = []

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    def _guard_pid_file(self) -> None:
        try:
            pid = int(self.pid_file.read_text(encoding="ascii").strip())
        except FileNotFoundError:
            return
        except (ValueError, OSError):
            self.pid_file.unlink(missing_ok=True)
            return
        if _pid_is_live(pid):
            raise RuntimeError(f"managed process already running with PID {pid}")
        self.pid_file.unlink(missing_ok=True)

    def _drain(self, stream: IO[str], target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            for chunk in iter(stream.readline, ""):
                handle.write(redact_stream(chunk))
                handle.flush()
        stream.close()

    def start(self) -> None:
        if self._process is not None and self._process.poll() is None:
            raise RuntimeError("managed process instance is already running")
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)
        self._guard_pid_file()
        self._process = subprocess.Popen(
            list(self.argv),
            cwd=str(self.cwd),
            env=self.env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            shell=False,
        )
        self.pid_file.write_text(f"{self._process.pid}\n", encoding="ascii")
        assert self._process.stdout is not None and self._process.stderr is not None
        self._threads = [
            threading.Thread(
                target=self._drain,
                args=(self._process.stdout, self.stdout_path),
                daemon=True,
            ),
            threading.Thread(
                target=self._drain,
                args=(self._process.stderr, self.stderr_path),
                daemon=True,
            ),
        ]
        for thread in self._threads:
            thread.start()

    def wait_ready(self, url: str, *, timeout: float) -> None:
        if self._process is None:
            raise RuntimeError("process has not been started")
        deadline = time.monotonic() + timeout
        last_error = "not attempted"
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        while time.monotonic() < deadline:
            returncode = self._process.poll()
            if returncode is not None:
                raise RuntimeError(f"process exited before readiness with status {returncode}")
            try:
                with opener.open(url, timeout=min(0.25, max(0.01, timeout))) as response:
                    if response.status < 500:
                        return
            except (OSError, urllib.error.URLError) as exc:
                last_error = str(exc)
            time.sleep(0.02)
        raise TimeoutError(f"process did not become ready at {url}: {last_error}")

    def stop(self, *, timeout: float) -> None:
        process = self._process
        if process is None:
            self.pid_file.unlink(missing_ok=True)
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=max(timeout, 1.0))
        for thread in self._threads:
            thread.join(timeout=1.0)
        self.pid_file.unlink(missing_ok=True)


__all__ = ["ManagedProcess", "redact_stream"]
