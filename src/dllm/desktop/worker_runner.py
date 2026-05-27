"""QProcess wrapper around `dllm.client.worker`.

Spawns the worker as a subprocess so the GUI stays responsive while training
runs. Parses well-known log lines (`registered worker_id=...`, `inner round=N
avg_loss=...`, `async sync applied round=...`) into Qt signals the UI can
listen to without scraping raw stdout.

In dev mode the subprocess is `sys.executable -m dllm.client.worker`. When
the desktop app is bundled by PyInstaller (no Python module path), the same
binary re-execs itself with `--worker-mode` and `dllm.desktop.main:main`
delegates to `dllm.client.worker.main()`.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal

# Parsing patterns for the worker's structured-ish log lines. Keep these
# narrow so the GUI is robust to log-format drift — anything that doesn't
# match just flows through to the raw log pane.
_RE_REGISTERED = re.compile(
    r"registered worker_id=(?P<wid>\d+) round=(?P<round>\d+) world_size=(?P<ws>\d+) inner=(?P<inner>\d+)"
)
_RE_INNER = re.compile(
    r"inner round=(?P<round>\d+) avg_loss=(?P<loss>[\d.]+) steps=(?P<steps>\d+)\s+(?P<tps>[\d.]+) tok/s"
)
_RE_VAL = re.compile(r"val round=(?P<round>\d+) loss=(?P<loss>[\d.]+)")
_RE_SYNC = re.compile(r"async sync applied round=(?P<round>\d+)")
_RE_RETUNE = re.compile(
    r"\[TIER-AWARE\] coord re-tuned inner_steps (?P<old>\d+) -> (?P<new>\d+)"
)
_RE_REREGISTER = re.compile(r"\[REREGISTER\] resumed as worker_id=(?P<wid>\d+)")
_RE_POWER = re.compile(r"power=(?P<watts>\d+)W")


class WorkerRunner(QObject):
    """Owns a QProcess running `dllm.client.worker`. Cross-platform.

    Lifecycle:
        runner = WorkerRunner(parent=window)
        runner.line_received.connect(log_panel.append)
        runner.registered.connect(on_registered)
        runner.inner_completed.connect(on_inner_done)
        runner.start(coord_url="https://dllm.planetbass.de", country="DE",
                     preset="300M", device="cuda", max_rounds=1000)
        ...
        runner.stop(grace_seconds=10)
    """

    # High-level events, parsed from log lines.
    started = Signal()
    registered = Signal(int, int, int, int)  # worker_id, round, world_size, inner_steps
    inner_completed = Signal(int, float, int, float)  # round, loss, steps, tok/s
    val_reported = Signal(int, float)  # round, loss
    sync_applied = Signal(int)  # new round
    retune_applied = Signal(int, int)  # old, new
    reregistered = Signal(int)  # new worker_id
    power_sample = Signal(float)  # watts

    # Raw events for the log panel.
    line_received = Signal(str)
    error_line = Signal(str)

    # Lifecycle.
    process_started = Signal()
    process_stopped = Signal(int)  # exit code

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._proc: QProcess | None = None

    # ------------------------------------------------------------------
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.state() != QProcess.NotRunning

    def start(
        self,
        *,
        coord_url: str,
        country: str,
        preset: str = "300M",
        device: str = "cuda",
        max_rounds: int = 1000,
        require_gpu: bool = True,
        extra_args: list[str] | None = None,
        cwd: Path | None = None,
        identity_key: Path | None = None,
    ) -> None:
        """Spawn the worker subprocess. Raises if already running."""
        if self.is_running():
            raise RuntimeError("worker process already running")

        argv = self._build_argv(
            coord_url=coord_url,
            country=country,
            preset=preset,
            device=device,
            max_rounds=max_rounds,
            require_gpu=require_gpu,
            extra_args=extra_args or [],
        )

        env = QProcessEnvironment.systemEnvironment()
        if require_gpu:
            # The worker checks DLLM_GPU_REQUIRED via --require-gpu too, but
            # set both so anything spawned downstream sees the requirement.
            env.insert("DLLM_GPU_REQUIRED", "1")
        # Force unbuffered Python output so the UI sees log lines in real
        # time. Without this, line-buffering can hold up to 8 KB at a time
        # before the GUI sees anything.
        env.insert("PYTHONUNBUFFERED", "1")
        # The CLI worker creates / reads `.dllm/identity.key` relative to
        # cwd. Override via env var honored by `dllm.shared.identity` if
        # the user passed a custom path (desktop client uses the per-user
        # data dir instead of cwd-relative).
        if identity_key is not None:
            env.insert("DLLM_IDENTITY_KEY", str(identity_key))

        proc = QProcess(self)
        proc.setProgram(argv[0])
        proc.setArguments(argv[1:])
        proc.setProcessEnvironment(env)
        if cwd is not None:
            proc.setWorkingDirectory(str(cwd))
        # Merge stderr → stdout so a single line stream carries everything;
        # we tag warning/error lines by content rather than by stream.
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(self._drain_stdout)
        proc.started.connect(self._on_started)
        proc.finished.connect(self._on_finished)
        proc.errorOccurred.connect(self._on_error)

        self._proc = proc
        proc.start()

    def stop(self, grace_seconds: float = 10.0) -> None:
        """SIGTERM the worker; if it doesn't exit within `grace_seconds`,
        SIGKILL. No-op if not running.
        """
        if not self.is_running():
            return
        assert self._proc is not None
        self._proc.terminate()
        # Block briefly on the event loop's behalf — short waits are fine
        # because we're being called from the UI thread on user click.
        if not self._proc.waitForFinished(int(grace_seconds * 1000)):
            self._proc.kill()
            self._proc.waitForFinished(3000)

    # ------------------------------------------------------------------
    def _build_argv(
        self,
        *,
        coord_url: str,
        country: str,
        preset: str,
        device: str,
        max_rounds: int,
        require_gpu: bool,
        extra_args: list[str],
    ) -> list[str]:
        # Three execution flavours, picked at runtime:
        #
        #   1) Frozen launcher + bootstrapped runtime present
        #      → spawn the runtime's python.exe with -m dllm.client.worker.
        #        This is the production volunteer flow: lean launcher
        #        +`%APPDATA%/dllm/runtime/python.exe`.
        #
        #   2) Frozen launcher + NO runtime
        #      → fall back to `sys.executable --worker-mode`, which works
        #        only if the bundle includes torch (heavy-bundle build).
        #        Lean builds raise instead — MainWindow should block start
        #        until bootstrap completes, so this branch only triggers
        #        on a corrupt install.
        #
        #   3) Dev mode (not frozen)
        #      → `python -m dllm.client.worker` against the venv that's
        #        running the GUI. Skips bootstrap.
        from . import runtime_manager  # local import: avoid cycle on cold start

        if getattr(sys, "frozen", False):
            if runtime_manager.is_installed():
                py = str(runtime_manager.runtime_python())
                argv = [py, "-u", "-m", "dllm.client.worker"]
            else:
                # No runtime yet but we have a heavy bundle — re-exec self.
                argv = [sys.executable, "--worker-mode"]
        else:
            argv = [sys.executable, "-u", "-m", "dllm.client.worker"]
        argv += [
            "--coord", coord_url,
            "--preset", preset,
            "--country", country,
            "--device", device,
            "--max-rounds", str(max_rounds),
        ]
        if require_gpu:
            argv.append("--require-gpu")
        argv += extra_args
        return argv

    # -- QProcess slots -------------------------------------------------
    def _on_started(self) -> None:
        self.process_started.emit()
        self.started.emit()

    def _on_finished(self, exit_code: int, _exit_status) -> None:  # type: ignore[no-untyped-def]
        self.process_stopped.emit(int(exit_code))
        self._proc = None

    def _on_error(self, err) -> None:  # type: ignore[no-untyped-def]
        # QProcess.ProcessError doesn't include a textual reason; pull it
        # off the process before it goes away.
        msg = self._proc.errorString() if self._proc else str(err)
        self.error_line.emit(f"[runner] process error: {msg}")

    def _drain_stdout(self) -> None:
        if self._proc is None:
            return
        data = bytes(self._proc.readAllStandardOutput())
        if not data:
            return
        text = data.decode("utf-8", errors="replace")
        for raw in text.splitlines():
            line = raw.rstrip()
            if not line:
                continue
            self.line_received.emit(line)
            self._parse_line(line)

    # -- log-line → high-level signal -----------------------------------
    def _parse_line(self, line: str) -> None:
        m = _RE_REGISTERED.search(line)
        if m:
            self.registered.emit(
                int(m["wid"]), int(m["round"]), int(m["ws"]), int(m["inner"])
            )
            return
        m = _RE_INNER.search(line)
        if m:
            self.inner_completed.emit(
                int(m["round"]),
                float(m["loss"]),
                int(m["steps"]),
                float(m["tps"]),
            )
            # power lives on the same line; emit separately if present
            mp = _RE_POWER.search(line)
            if mp:
                self.power_sample.emit(float(mp["watts"]))
            return
        m = _RE_VAL.search(line)
        if m:
            self.val_reported.emit(int(m["round"]), float(m["loss"]))
            return
        m = _RE_SYNC.search(line)
        if m:
            self.sync_applied.emit(int(m["round"]))
            return
        m = _RE_RETUNE.search(line)
        if m:
            self.retune_applied.emit(int(m["old"]), int(m["new"]))
            return
        m = _RE_REREGISTER.search(line)
        if m:
            self.reregistered.emit(int(m["wid"]))
            return
