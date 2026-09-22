"""Lifecycle manager for local Evaluation Lab processes and keep-awake coordination."""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .processes import ManagedProcess


@dataclass(frozen=True)
class ProcessConfig:
    name: str
    argv: tuple[str, ...]
    cwd: Path
    env: Mapping[str, str]
    port: int | None
    readiness_url: str | None


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
                    "http://127.0.0.1:8001/api/lab/preflight",
                    timeout=readiness_timeout,
                )

            # 3. Enhanced backend
            if self.enhanced_proc is not None:
                self.enhanced_proc.start()
                started.append(("enhanced", self.enhanced_proc))
                self.enhanced_proc.wait_ready(
                    "http://127.0.0.1:8002/api/lab/preflight",
                    timeout=readiness_timeout,
                )

            # 4. Web UI
            if self.ui_proc is not None:
                self.ui_proc.start()
                started.append(("ui", self.ui_proc))
                self.ui_proc.wait_ready(
                    "http://127.0.0.1:3000/api/lab/preflight",
                    timeout=readiness_timeout,
                )

        except BaseException:
            # Exceptional cleanup: stop started in reverse order
            for name, proc in reversed(started):
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
            if proc is not None:
                try:
                    proc.stop(timeout=stop_timeout)
                except Exception:
                    pass

    def status(self) -> dict[str, dict[str, Any]]:
        return {
            "keep_awake": {
                "running": bool(self.keep_awake_proc and getattr(self.keep_awake_proc, "is_running", False)),
                "pid": getattr(self.keep_awake_proc, "pid", None),
            },
            "baseline": {
                "running": bool(self.baseline_proc and getattr(self.baseline_proc, "is_running", False)),
                "pid": getattr(self.baseline_proc, "pid", None),
                "port": 8001,
            },
            "enhanced": {
                "running": bool(self.enhanced_proc and getattr(self.enhanced_proc, "is_running", False)),
                "pid": getattr(self.enhanced_proc, "pid", None),
                "port": 8002,
            },
            "ui": {
                "running": bool(self.ui_proc and getattr(self.ui_proc, "is_running", False)),
                "pid": getattr(self.ui_proc, "pid", None),
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
