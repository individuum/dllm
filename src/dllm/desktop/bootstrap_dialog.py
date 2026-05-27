"""First-launch bootstrap dialog.

Shown when the lean launcher detects there's no usable runtime under
<user_data>/runtime/. Runs `runtime_manager.install_runtime()` in a worker
thread so the GUI stays responsive while pip churns through the ~2 GB
torch download.

UI states:
    initial   →  "Setup required — download ~2 GB to start contributing"
                  [ Install ]  [ Cancel ]
    running   →  progress bar + status text streamed from pip
                  [ Cancel ]
    success   →  "Ready to contribute!"
                  [ OK ]
    error     →  red banner with the exception message
                  [ Retry ]  [ Cancel ]
"""
from __future__ import annotations

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from . import runtime_manager


class _BootstrapWorker(QObject):
    """Runs runtime_manager.install_runtime in its own thread + relays
    progress callbacks to the GUI via signals."""

    progress = Signal(str, float)  # message, fraction
    finished = Signal(dict)  # marker data
    failed = Signal(str)  # error message

    def run(self) -> None:
        try:
            marker = runtime_manager.install_runtime(
                progress=lambda msg, frac: self.progress.emit(msg, frac)
            )
            self.finished.emit(marker)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class BootstrapDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("dllm contributor — first-time setup")
        self.setMinimumWidth(520)
        self._worker: _BootstrapWorker | None = None
        self._thread: QThread | None = None
        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(10)

        title = QLabel("First-time setup")
        title.setStyleSheet("font-size: 14pt; font-weight: 600;")
        root.addWidget(title)

        explain = QLabel(
            "To contribute, dllm needs to download PyTorch + the CUDA runtime "
            "(about 2 GB) from PyTorch's official servers. This is a one-time "
            "download per machine — subsequent launches start instantly."
        )
        explain.setWordWrap(True)
        explain.setStyleSheet("color: palette(mid);")
        root.addWidget(explain)

        self.status = QLabel("Ready to begin.")
        self.status.setStyleSheet("font-size: 10pt;")
        root.addWidget(self.status)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setVisible(False)
        root.addWidget(self.progress)

        self.error_banner = QLabel("")
        self.error_banner.setStyleSheet(
            "color: var(--warn, #d97706); padding: 8px;"
            "background: rgba(217,119,6,0.10); border-radius: 4px;"
        )
        self.error_banner.setWordWrap(True)
        self.error_banner.setVisible(False)
        root.addWidget(self.error_banner)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_cancel)
        self.btn_install = QPushButton("Install")
        self.btn_install.setDefault(True)
        self.btn_install.setStyleSheet(
            "QPushButton { background: palette(highlight); color: palette(highlighted-text);"
            " padding: 6px 14px; font-weight: 600; border-radius: 4px; }"
        )
        self.btn_install.clicked.connect(self._on_install_clicked)
        btn_row.addWidget(self.btn_install)
        root.addLayout(btn_row)

    # ------------------------------------------------------------------
    def _on_install_clicked(self) -> None:
        # Start the bootstrap on a worker thread so the dialog stays responsive.
        self.btn_install.setEnabled(False)
        self.btn_install.setText("Installing…")
        self.error_banner.setVisible(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.status.setText("Preparing runtime…")

        self._thread = QThread(self)
        self._worker = _BootstrapWorker()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.start()

    def _on_progress(self, message: str, fraction: float) -> None:
        self.status.setText(message)
        self.progress.setValue(max(0, min(100, int(fraction * 100))))

    def _on_finished(self, marker: dict) -> None:
        self.status.setText(
            f"Runtime ready (torch {marker.get('torch_version', '?')}). "
            "Click OK to start contributing."
        )
        self.progress.setValue(100)
        self.btn_install.setText("OK")
        self.btn_install.setEnabled(True)
        try:
            self.btn_install.clicked.disconnect()
        except RuntimeError:
            pass
        self.btn_install.clicked.connect(self.accept)
        self.btn_cancel.setVisible(False)

    def _on_failed(self, msg: str) -> None:
        self.status.setText("Setup failed.")
        self.error_banner.setText(f"⚠ {msg}")
        self.error_banner.setVisible(True)
        self.progress.setVisible(False)
        self.btn_install.setText("Retry")
        self.btn_install.setEnabled(True)
