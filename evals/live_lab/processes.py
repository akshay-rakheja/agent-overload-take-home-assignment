"""Managed subprocesses scoped to one live-lab run."""

from __future__ import annotations

import os
import fcntl
import hashlib
import json
import re
import secrets
import socket
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
        private_dir: Path | None = None,
        readiness_host: str | None = None,
        readiness_port: int | None = None,
        readiness_nonce: str | None = None,
    ) -> None:
        if not argv or not all(isinstance(item, str) and item for item in argv):
            raise ValueError("argv must be a non-empty explicit string sequence")
        self.argv = tuple(argv)
        self.env = dict(env)
        self.cwd = Path(cwd)
        self.pid_file = Path(pid_file)
        self.stdout_path = Path(stdout_path)
        self.stderr_path = Path(stderr_path)
        self.private_dir = Path(private_dir) if private_dir is not None else self.pid_file.parent / "private"
        self.readiness_host = readiness_host
        self.readiness_port = readiness_port
        self.readiness_nonce = readiness_nonce
        if (readiness_host is None) != (readiness_port is None):
            raise ValueError("readiness_host and readiness_port must be provided together")
        if readiness_nonce is not None and readiness_host is None:
            raise ValueError("readiness_nonce requires an owned readiness address")
        self.owner_token = secrets.token_hex(16)
        self.lock_file = self.pid_file.with_name(self.pid_file.name + ".lock")
        self._process: subprocess.Popen[str] | None = None
        self._threads: list[threading.Thread] = []
        self._lock_handle: IO[str] | None = None

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _acquire_owner_lock(self) -> None:
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        handle = self.lock_file.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise RuntimeError("managed process already running") from exc
        self._lock_handle = handle

    def _release_owner_lock(self) -> None:
        handle = self._lock_handle
        if handle is None:
            return
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            self._lock_handle = None

    def _read_marker(self) -> tuple[int, str | None] | None:
        try:
            raw = self.pid_file.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            try:
                return int(raw), None
            except ValueError:
                return None
        if not isinstance(parsed, dict) or not isinstance(parsed.get("pid"), int):
            return None
        token = parsed.get("owner_token")
        return parsed["pid"], token if isinstance(token, str) else None

    def _guard_pid_file(self) -> None:
        marker = self._read_marker()
        if marker is None:
            self.pid_file.unlink(missing_ok=True)
            return
        pid, _ = marker
        if _pid_is_live(pid):
            raise RuntimeError(f"managed process already running with PID {pid}")
        self.pid_file.unlink(missing_ok=True)

    def _port_is_occupied(self) -> bool:
        if self.readiness_host is None or self.readiness_port is None:
            return False
        with socket.socket() as probe:
            probe.settimeout(0.2)
            return probe.connect_ex((self.readiness_host, self.readiness_port)) == 0

    def _write_marker(self, pid: int) -> None:
        payload = json.dumps(
            {"owner_token": self.owner_token, "pid": pid},
            sort_keys=True,
            separators=(",", ":"),
        ) + "\n"
        descriptor = os.open(
            self.pid_file,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)

    def _remove_marker_if_owned(self) -> None:
        marker = self._read_marker()
        if marker is not None and marker[1] == self.owner_token:
            self.pid_file.unlink(missing_ok=True)

    def _drain(self, stream: IO[str], target: Path, stream_name: str) -> None:
        self.private_dir.mkdir(parents=True, exist_ok=True)
        private_path = self.private_dir / f"{self.owner_token}-{stream_name}.log"
        digest = hashlib.sha256()
        byte_count = 0
        with private_path.open("a", encoding="utf-8") as handle:
            for chunk in iter(stream.readline, ""):
                encoded = chunk.encode("utf-8", errors="replace")
                digest.update(encoded)
                byte_count += len(encoded)
                handle.write(chunk)
                handle.flush()
        stream.close()
        target.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "byte_count": byte_count,
            "event_type": "stream_closed",
            "sha256": digest.hexdigest(),
            "stream": stream_name,
        }
        target.write_text(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

    def start(self) -> None:
        if self._process is not None and self._process.poll() is None:
            raise RuntimeError("managed process instance is already running")
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)
        self._acquire_owner_lock()
        try:
            self._guard_pid_file()
            if self._port_is_occupied():
                raise RuntimeError(
                    f"readiness port already occupied: {self.readiness_host}:{self.readiness_port}"
                )
            self.private_dir.mkdir(parents=True, exist_ok=True)
            private_stdout = self.private_dir / f"{self.owner_token}-stdout.log"
            private_stderr = self.private_dir / f"{self.owner_token}-stderr.log"
            stdout_handle = private_stdout.open("a", encoding="utf-8")
            stderr_handle = private_stderr.open("a", encoding="utf-8")
            try:
                self._process = subprocess.Popen(
                    list(self.argv),
                    cwd=str(self.cwd),
                    env=self.env,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    text=True,
                    bufsize=1,
                    shell=False,
                    start_new_session=True,
                )
            finally:
                stdout_handle.close()
                stderr_handle.close()
            self._write_marker(self._process.pid)
        except BaseException:
            if self._process is not None and self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=5)
            self._process = None
            self._remove_marker_if_owned()
            self._release_owner_lock()
            raise

    def wait_ready(self, url: str, *, timeout: float) -> None:
        if self._process is None:
            raise RuntimeError("process has not been started")
        deadline = time.monotonic() + timeout
        last_error = "not attempted"
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            while time.monotonic() < deadline:
                returncode = self._process.poll()
                if returncode is not None:
                    raise RuntimeError(f"process exited before readiness with status {returncode}")
                try:
                    with opener.open(url, timeout=min(0.25, max(0.01, timeout))) as response:
                        payload = response.read()
                        if response.status < 500:
                            if self.readiness_nonce is None:
                                return
                            try:
                                decoded = json.loads(payload)
                            except (json.JSONDecodeError, UnicodeDecodeError):
                                decoded = None
                            if isinstance(decoded, dict) and decoded.get("nonce") == self.readiness_nonce:
                                return
                            last_error = "readiness nonce mismatch"
                except (OSError, urllib.error.URLError) as exc:
                    last_error = str(exc)
                time.sleep(0.02)
            raise TimeoutError(f"process did not become ready at {url}: {last_error}")
        except BaseException:
            self.stop(timeout=min(max(timeout, 0.1), 5.0))
            raise

    def wait(self, *, timeout: float) -> int:
        if self._process is None:
            raise RuntimeError("process has not been started")
        return self._process.wait(timeout=timeout)

    def _export_stream_metadata(self) -> None:
        for stream_name, target in (("stdout", self.stdout_path), ("stderr", self.stderr_path)):
            private_path = self.private_dir / f"{self.owner_token}-{stream_name}.log"
            byte_count = 0
            digest = hashlib.sha256()
            if private_path.exists():
                with private_path.open("rb") as handle:
                    while chunk := handle.read(65536):
                        digest.update(chunk)
                        byte_count += len(chunk)
            target.parent.mkdir(parents=True, exist_ok=True)
            metadata = {
                "byte_count": byte_count,
                "event_type": "stream_closed",
                "sha256": digest.hexdigest(),
                "stream": stream_name,
            }
            target.write_text(
                json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )

    def stop(self, *, timeout: float) -> None:
        process = self._process
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=max(timeout, 1.0))
        self._remove_marker_if_owned()
        self._release_owner_lock()
        self._export_stream_metadata()
        self._process = None


__all__ = ["ManagedProcess", "redact_stream", "_pid_is_live"]
