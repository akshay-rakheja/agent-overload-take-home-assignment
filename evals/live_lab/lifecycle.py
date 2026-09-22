"""Lifecycle manager for local Evaluation Lab processes and keep-awake coordination."""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .processes import ManagedProcess, _pid_is_live


@dataclass(frozen=True)
class ProcessConfig:
    name: str
    argv: tuple[str, ...]
    cwd: Path
    env: Mapping[str, str]
    port: int | None
    readiness_url: str | None


def _inspect_proc(proc: Any) -> tuple[bool, int | None]:
    if proc is None:
        return False, None
    if getattr(proc, "is_running", False):
        return True, getattr(proc, "pid", None)
    pid_file = getattr(proc, "pid_file", None)
    if pid_file is not None and isinstance(pid_file, Path) and pid_file.exists():
        try:
            raw = pid_file.read_text(encoding="utf-8").strip()
            data = json.loads(raw) if raw.startswith("{") else {"pid": int(raw)}
            pid = data.get("pid")
            if isinstance(pid, int) and _pid_is_live(pid):
                return True, pid
        except Exception:
            pass
    return False, None


class LabLifecycle:
    """Manages start, readiness, status, and ordered shutdown of lab processes."""

    def __init__(
        self,
        *,
        keep_awake_proc: Any = None,
        baseline_proc: Any = None,
        enhanced_proc: Any = None,
        ui_proc: Any = None,
    ) -> None:
        self.keep_awake_proc = keep_awake_proc
        self.baseline_proc = baseline_proc
        self.enhanced_proc = enhanced_proc
        self.ui_proc = ui_proc

    def start(self, *, readiness_timeout: float = 30.0) -> None:
        started: list[tuple[str, Any]] = []

        try:
            # 1. Keep-awake
            if self.keep_awake_proc is not None:
                self.keep_awake_proc.start()
                started.append(("keep_awake", self.keep_awake_proc))

            # 2. Baseline backend
            if self.baseline_proc is not None:
                self.baseline_proc.start()
                started.append(("baseline", self.baseline_proc))
                self.baseline_proc.wait_ready(
                    "http://127.0.0.1:8001/api/v1/health",
                    timeout=readiness_timeout,
                )

            # 3. Enhanced backend
            if self.enhanced_proc is not None:
                self.enhanced_proc.start()
                started.append(("enhanced", self.enhanced_proc))
                self.enhanced_proc.wait_ready(
                    "http://127.0.0.1:8002/api/v1/health",
                    timeout=readiness_timeout,
                )

            # 4. Web UI
            if self.ui_proc is not None:
                self.ui_proc.start()
                started.append(("ui", self.ui_proc))
                self.ui_proc.wait_ready(
                    "http://127.0.0.1:3000/lab",
                    timeout=readiness_timeout,
                )

        except BaseException:
            # Exceptional-finally: tear down any started processes in reverse order
            for _, proc in reversed(started):
                try:
                    proc.stop(timeout=5.0)
                except Exception:
                    pass
            raise

    def stop(self, *, stop_timeout: float = 5.0) -> None:
        # Stop order: UI, enhanced, baseline, keep-awake
        procs = [
            ("ui", self.ui_proc),
            ("enhanced", self.enhanced_proc),
            ("baseline", self.baseline_proc),
            ("keep_awake", self.keep_awake_proc),
        ]
        for name, proc in procs:
            if proc is None:
                continue
            try:
                proc.stop(timeout=stop_timeout)
            except Exception:
                pass
            pid_file = getattr(proc, "pid_file", None)
            if pid_file is not None and isinstance(pid_file, Path) and pid_file.exists():
                try:
                    raw = pid_file.read_text(encoding="utf-8").strip()
                    data = json.loads(raw) if raw.startswith("{") else {"pid": int(raw)}
                    pid = data.get("pid")
                    if isinstance(pid, int) and _pid_is_live(pid):
                        try:
                            os.kill(pid, 15)
                            deadline = time.monotonic() + stop_timeout
                            while time.monotonic() < deadline and _pid_is_live(pid):
                                time.sleep(0.05)
                            if _pid_is_live(pid):
                                os.kill(pid, 9)
                                time.sleep(0.05)
                        except OSError:
                            pass
                except Exception:
                    pass
                pid_file.unlink(missing_ok=True)
                lock_file = getattr(proc, "lock_file", None)
                if lock_file is not None and isinstance(lock_file, Path):
                    lock_file.unlink(missing_ok=True)

    def status(self) -> dict[str, dict[str, Any]]:
        ka_running, ka_pid = _inspect_proc(self.keep_awake_proc)
        base_running, base_pid = _inspect_proc(self.baseline_proc)
        enh_running, enh_pid = _inspect_proc(self.enhanced_proc)
        ui_running, ui_pid = _inspect_proc(self.ui_proc)
        return {
            "keep_awake": {
                "running": ka_running,
                "pid": ka_pid,
            },
            "baseline": {
                "running": base_running,
                "pid": base_pid,
                "port": 8001,
            },
            "enhanced": {
                "running": enh_running,
                "pid": enh_pid,
                "port": 8002,
            },
            "ui": {
                "running": ui_running,
                "pid": ui_pid,
                "port": 3000,
            },
        }


def create_default_lifecycle(
    *,
    base_dir: Path,
    runtime_state_dir: Path,
) -> LabLifecycle:
    """Instantiate ManagedProcesses for keep-awake, baseline, enhanced, and Next.js UI."""
    runtime_state_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = runtime_state_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    baseline_dir = base_dir.parent / "openpoke-evaluation-baseline"
    enhanced_dir = base_dir
    web_dir = enhanced_dir / "web"

    # Keep-awake (caffeinate on macOS if available, otherwise python no-op loop)
    caffeinate_bin = shutil.which("caffeinate")
    if caffeinate_bin:
        keep_awake_argv = [caffeinate_bin, "-dimsu"]
    else:
        keep_awake_argv = [sys.executable, "-c", "import time; time.sleep(86400)"]

    keep_awake = ManagedProcess(
        argv=keep_awake_argv,
        env={"PATH": os.environ.get("PATH", "")},
        cwd=enhanced_dir,
        pid_file=runtime_state_dir / "keep_awake.pid",
        stdout_path=logs_dir / "keep_awake.stdout.log",
        stderr_path=logs_dir / "keep_awake.stderr.log",
    )

    baseline_proc = ManagedProcess(
        argv=[
            str(baseline_dir / ".venv" / "bin" / "python"),
            "-m",
            "uvicorn",
            "server.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8001",
        ],
        env=dict(os.environ),
        cwd=baseline_dir,
        pid_file=runtime_state_dir / "baseline.pid",
        stdout_path=logs_dir / "baseline.stdout.log",
        stderr_path=logs_dir / "baseline.stderr.log",
        readiness_host="127.0.0.1",
        readiness_port=8001,
    )

    enhanced_proc = ManagedProcess(
        argv=[
            str(enhanced_dir / ".venv" / "bin" / "python"),
            "-m",
            "uvicorn",
            "server.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8002",
        ],
        env=dict(os.environ),
        cwd=enhanced_dir,
        pid_file=runtime_state_dir / "enhanced.pid",
        stdout_path=logs_dir / "enhanced.stdout.log",
        stderr_path=logs_dir / "enhanced.stderr.log",
        readiness_host="127.0.0.1",
        readiness_port=8002,
    )

    ui_proc = ManagedProcess(
        argv=["npm", "run", "start"],
        env={**dict(os.environ), "PORT": "3000", "HOST": "127.0.0.1"},
        cwd=web_dir,
        pid_file=runtime_state_dir / "ui.pid",
        stdout_path=logs_dir / "ui.stdout.log",
        stderr_path=logs_dir / "ui.stderr.log",
        readiness_host="127.0.0.1",
        readiness_port=3000,
    )

    return LabLifecycle(
        keep_awake_proc=keep_awake,
        baseline_proc=baseline_proc,
        enhanced_proc=enhanced_proc,
        ui_proc=ui_proc,
    )


__all__ = [
    "LabLifecycle",
    "ProcessConfig",
    "create_default_lifecycle",
]
