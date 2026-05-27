"""Main window for the dllm contributor GUI.

Layout:
    ┌──────────────────────────────────────────────────────────────────┐
    │  dllm contributor                                  ● connected   │
    │  Help train the EU sovereign open LLM with your GPU.             │
    ├──────────────────────────────────────────────────────────────────┤
    │  status                                                          │
    │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐                             │
    │  │ round│ │ val  │ │ tok/s│ │ power│                             │
    │  │  238 │ │ 3.42 │ │ 3.7k │ │ 152W │                             │
    │  └──────┘ └──────┘ └──────┘ └──────┘                             │
    │  worker_id: 7 · rounds contributed: 12 · last sync: 1 min ago    │
    ├──────────────────────────────────────────────────────────────────┤
    │  settings                                                        │
    │  country [DE ▾]  preset [300M ▾]   [ Start contributing ]        │
    ├──────────────────────────────────────────────────────────────────┤
    │  log                                                             │
    │  ╔════════════════════════════════════════════════════════════╗  │
    │  ║ 18:11:32 registered worker_id=0 round=78 ...               ║  │
    │  ║ 18:12:36 pulled state at round=78 ...                      ║  │
    │  ╚════════════════════════════════════════════════════════════╝  │
    └──────────────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .paths import identity_key_path, user_data_dir
from .worker_runner import WorkerRunner

# EU/EEA + UK ISO-3166-1 alpha-2 codes. Tier-D residency attestation is
# scope-out for v0 — we trust the user's pick. eIDAS-based verification
# lands in Phase 2 per PLAN §5.5.
_EU_COUNTRIES = [
    "AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "ES", "FI",
    "FR", "GR", "HR", "HU", "IE", "IT", "LT", "LU", "LV", "MT",
    "NL", "PL", "PT", "RO", "SE", "SI", "SK",
    # EEA
    "IS", "LI", "NO",
]

# Public coord. For Phase 0/1 we point everyone at the same VPS coord; later
# the desktop client lets the user pick / discovers via DNS-SD or similar.
_DEFAULT_COORD = "https://dllm.planetbass.de"
_DEFAULT_PRESET = "300M"

# Log pane keeps this many recent lines so contributors with a 7-day session
# don't OOM the GUI process.
_LOG_RING_LINES = 2000


def _tile(parent: QWidget, label: str) -> tuple[QFrame, QLabel, QLabel]:
    """One metric tile. Returns (frame, value_label, sub_label)."""
    frame = QFrame(parent)
    frame.setFrameShape(QFrame.StyledPanel)
    frame.setStyleSheet(
        "QFrame { background: palette(base); border: 1px solid palette(mid);"
        " border-radius: 6px; padding: 8px; }"
    )
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(8, 6, 8, 6)
    lay.setSpacing(2)

    lab = QLabel(label.upper(), frame)
    lab.setStyleSheet("color: palette(mid); font-size: 10pt;")
    val = QLabel("—", frame)
    f = QFont(); f.setPointSize(20); f.setBold(True); val.setFont(f)
    sub = QLabel("—", frame)
    sub.setStyleSheet("color: palette(mid); font-size: 9pt;")

    lay.addWidget(lab)
    lay.addWidget(val)
    lay.addWidget(sub)
    return frame, val, sub


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("dllm contributor")
        self.resize(820, 640)

        self.runner = WorkerRunner(self)
        self.runner.line_received.connect(self._on_log_line)
        self.runner.error_line.connect(self._on_log_line)
        self.runner.registered.connect(self._on_registered)
        self.runner.inner_completed.connect(self._on_inner_completed)
        self.runner.val_reported.connect(self._on_val_reported)
        self.runner.sync_applied.connect(self._on_sync_applied)
        self.runner.retune_applied.connect(self._on_retune_applied)
        self.runner.reregistered.connect(self._on_reregistered)
        self.runner.power_sample.connect(self._on_power_sample)
        self.runner.process_started.connect(self._on_process_started)
        self.runner.process_stopped.connect(self._on_process_stopped)

        # Session counters reset on each Start.
        self._rounds_contributed = 0
        self._worker_id: int | None = None
        self._last_sync_at: datetime.datetime | None = None
        self._best_val: float | None = None
        self._build_ui()

        # Heartbeat: refresh "last sync N ago" once a second so the UI
        # doesn't look frozen between training rounds.
        self._heartbeat = QTimer(self)
        self._heartbeat.setInterval(1000)
        self._heartbeat.timeout.connect(self._tick)
        self._heartbeat.start()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(12)

        # ---- header -----------------------------------------------------
        header_row = QHBoxLayout()
        h_title = QLabel("dllm contributor")
        h_title.setStyleSheet("font-size: 18pt; font-weight: 600;")
        header_row.addWidget(h_title)
        header_row.addStretch(1)
        self.status_dot = QLabel("● idle")
        self.status_dot.setStyleSheet("color: palette(mid);")
        header_row.addWidget(self.status_dot)
        root.addLayout(header_row)

        sub = QLabel(
            "Help train the EU sovereign open LLM with your GPU. "
            "Your hardware contributes one inner DiLoCo loop per round."
        )
        sub.setStyleSheet("color: palette(mid); font-size: 10pt;")
        sub.setWordWrap(True)
        root.addWidget(sub)

        # ---- metric tiles -----------------------------------------------
        grid = QGridLayout()
        grid.setSpacing(10)
        self.tile_round, self.tile_round_val, self.tile_round_sub = _tile(central, "round")
        self.tile_val, self.tile_val_val, self.tile_val_sub = _tile(central, "val loss")
        self.tile_tps, self.tile_tps_val, self.tile_tps_sub = _tile(central, "throughput")
        self.tile_pow, self.tile_pow_val, self.tile_pow_sub = _tile(central, "power")
        grid.addWidget(self.tile_round, 0, 0)
        grid.addWidget(self.tile_val, 0, 1)
        grid.addWidget(self.tile_tps, 0, 2)
        grid.addWidget(self.tile_pow, 0, 3)
        root.addLayout(grid)

        # contributor-meta strip
        self.meta_label = QLabel(
            "Not yet registered. Click Start contributing to join the cohort."
        )
        self.meta_label.setStyleSheet("color: palette(mid); font-size: 10pt;")
        self.meta_label.setWordWrap(True)
        root.addWidget(self.meta_label)

        # ---- settings ---------------------------------------------------
        settings = QHBoxLayout()
        settings.setSpacing(8)
        settings.addWidget(QLabel("country"))
        self.country = QComboBox()
        self.country.addItems(_EU_COUNTRIES)
        self.country.setCurrentText("DE")
        settings.addWidget(self.country)

        settings.addWidget(QLabel("preset"))
        self.preset = QComboBox()
        # Only 300M is meaningfully running on the public coord today; smoke
        # is for self-test. Full list returns once a presets-on-coord
        # discovery endpoint lands.
        self.preset.addItems(["300M", "smoke"])
        self.preset.setCurrentText(_DEFAULT_PRESET)
        settings.addWidget(self.preset)

        settings.addStretch(1)

        self.btn_start = QPushButton("Start contributing")
        self.btn_start.setMinimumWidth(180)
        self.btn_start.setStyleSheet(
            "QPushButton { background: palette(highlight); color: palette(highlighted-text);"
            " padding: 8px 16px; font-weight: 600; border-radius: 4px; }"
            " QPushButton:disabled { background: palette(mid); }"
        )
        self.btn_start.clicked.connect(self._on_start_clicked)
        settings.addWidget(self.btn_start)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop_clicked)
        settings.addWidget(self.btn_stop)

        root.addLayout(settings)

        # ---- log pane ---------------------------------------------------
        log_label = QLabel("worker log")
        log_label.setStyleSheet("color: palette(mid); font-size: 10pt;")
        root.addWidget(log_label)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(_LOG_RING_LINES)
        mono = QFont("Consolas" if sys.platform == "win32" else "Menlo")
        mono.setStyleHint(QFont.Monospace)
        mono.setPointSize(9)
        self.log.setFont(mono)
        root.addWidget(self.log, 1)

        # ---- footer -----------------------------------------------------
        footer = QHBoxLayout()
        footer.addWidget(QLabel(f"data dir: {user_data_dir()}"))
        footer.addStretch(1)
        footer.addWidget(QLabel("v0 · early access"))
        root.addLayout(footer)

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------
    def _on_start_clicked(self) -> None:
        # Reset session state.
        self._rounds_contributed = 0
        self._worker_id = None
        self._last_sync_at = None
        self._best_val = None
        self.tile_round_val.setText("…")
        self.tile_round_sub.setText("connecting…")
        self.tile_val_val.setText("—")
        self.tile_val_sub.setText("—")
        self.tile_tps_val.setText("—")
        self.tile_tps_sub.setText("—")
        self.tile_pow_val.setText("—")
        self.tile_pow_sub.setText("—")
        self.meta_label.setText("Starting worker…")
        self.log.clear()

        try:
            self.runner.start(
                coord_url=_DEFAULT_COORD,
                country=self.country.currentText(),
                preset=self.preset.currentText(),
                device="cuda",  # macOS users on MPS need a future device picker
                max_rounds=10_000,
                require_gpu=True,
                identity_key=identity_key_path(),
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "dllm", f"Could not start worker:\n{exc}")
            return
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.status_dot.setText("● starting")
        self.status_dot.setStyleSheet("color: palette(highlight);")

    def _on_stop_clicked(self) -> None:
        self.btn_stop.setEnabled(False)
        self.status_dot.setText("● stopping")
        self.runner.stop(grace_seconds=15.0)

    # ------------------------------------------------------------------
    # Runner signal handlers
    # ------------------------------------------------------------------
    def _on_process_started(self) -> None:
        self.status_dot.setText("● connecting")
        self.status_dot.setStyleSheet("color: palette(highlight);")
        self._append_log("[gui] worker process started")

    def _on_process_stopped(self, exit_code: int) -> None:
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status_dot.setText(
            f"● stopped (exit {exit_code})" if exit_code != 0 else "● stopped"
        )
        self.status_dot.setStyleSheet("color: palette(mid);")
        self.meta_label.setText(
            f"Worker exited with code {exit_code}. "
            "Click Start contributing to resume."
        )
        self._append_log(f"[gui] worker process exited (code {exit_code})")

    def _on_registered(
        self, worker_id: int, round_no: int, world_size: int, inner_steps: int
    ) -> None:
        self._worker_id = worker_id
        self.tile_round_val.setText(str(round_no))
        self.tile_round_sub.setText(f"world={world_size} · inner={inner_steps}")
        self.meta_label.setText(
            f"worker_id={worker_id} · round={round_no} · world_size={world_size}"
            f" · inner_steps={inner_steps}"
        )
        self.status_dot.setText("● contributing")
        self.status_dot.setStyleSheet("color: #16a34a;")

    def _on_reregistered(self, new_wid: int) -> None:
        self._worker_id = new_wid
        self._append_log(f"[gui] auto-reregistered as worker_id={new_wid}")

    def _on_inner_completed(
        self, round_no: int, loss: float, steps: int, tok_per_s: float
    ) -> None:
        self.tile_round_val.setText(str(round_no))
        self.tile_round_sub.setText(f"{steps} inner steps")
        self.tile_tps_val.setText(self._fmt_tok_per_s(tok_per_s))
        self.tile_tps_sub.setText("tok/s · inner loop")

    def _on_val_reported(self, round_no: int, loss: float) -> None:
        self.tile_val_val.setText(f"{loss:.3f}")
        # Perplexity sub-line is a more human-friendly framing.
        import math
        ppl = math.exp(loss) if loss < 30 else float("inf")
        self.tile_val_sub.setText(f"perplexity {ppl:.1f}")
        if self._best_val is None or loss < self._best_val:
            self._best_val = loss

    def _on_sync_applied(self, new_round: int) -> None:
        self._rounds_contributed += 1
        self._last_sync_at = datetime.datetime.now()
        self.tile_round_val.setText(str(new_round))
        self.meta_label.setText(
            f"worker_id={self._worker_id} · contributed {self._rounds_contributed} round(s)"
            + (f" · best val {self._best_val:.3f}" if self._best_val is not None else "")
        )

    def _on_retune_applied(self, old_steps: int, new_steps: int) -> None:
        self._append_log(
            f"[gui] coord re-tuned inner_steps {old_steps} → {new_steps}"
        )

    def _on_power_sample(self, watts: float) -> None:
        self.tile_pow_val.setText(f"{watts:.0f} W")
        self.tile_pow_sub.setText("last inner loop")

    # ------------------------------------------------------------------
    def _on_log_line(self, line: str) -> None:
        self._append_log(line)

    def _append_log(self, line: str) -> None:
        # QPlainTextEdit handles maxBlockCount internally → bounded memory.
        self.log.appendPlainText(line)

    def _tick(self) -> None:
        if self._last_sync_at is None or self._worker_id is None:
            return
        ago = (datetime.datetime.now() - self._last_sync_at).total_seconds()
        ago_s = self._fmt_seconds(ago)
        self.meta_label.setText(
            f"worker_id={self._worker_id} · contributed {self._rounds_contributed} round(s)"
            + (f" · best val {self._best_val:.3f}" if self._best_val is not None else "")
            + f" · last sync {ago_s} ago"
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _fmt_tok_per_s(v: float) -> str:
        if v >= 1_000_000:
            return f"{v / 1_000_000:.2f}M"
        if v >= 1_000:
            return f"{v / 1000:.1f}k"
        return f"{v:.0f}"

    @staticmethod
    def _fmt_seconds(s: float) -> str:
        s = int(s)
        if s < 60:
            return f"{s}s"
        if s < 3600:
            return f"{s // 60}m {s % 60:02d}s"
        return f"{s // 3600}h {(s % 3600) // 60:02d}m"

    # ------------------------------------------------------------------
    def closeEvent(self, ev) -> None:  # type: ignore[override]
        # Politely stop the worker before quitting so the coord sees a clean
        # exit (no orphan registration for the auto-evictor to clean up).
        if self.runner.is_running():
            self._append_log("[gui] window closed; stopping worker…")
            self.runner.stop(grace_seconds=10.0)
        super().closeEvent(ev)


def run_app() -> int:
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("dllm contributor")
    app.setOrganizationName("dllm")
    app.setOrganizationDomain("dllm.planetbass.de")
    w = MainWindow()
    w.show()
    return app.exec()
