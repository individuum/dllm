"""Entry point for the dllm contributor GUI.

Two roles, dispatched by the leading argv flag:

  ``--worker-mode``  Delegate to the headless `dllm.client.worker` CLI. This
                     branch exists so a PyInstaller-bundled binary can re-exec
                     itself as the worker subprocess instead of needing a
                     separate `python -m dllm.client.worker` (which doesn't
                     work inside a frozen single-file bundle).

  default            Launch the PySide6 GUI (`run_app`).

Dev mode (`pip install -e .[desktop]` + `dllm-desktop`) always goes through
the second branch; the worker subprocess uses ``sys.executable -m
dllm.client.worker`` as before.
"""
from __future__ import annotations

import sys


def main() -> int:
    # NOTE: must use ABSOLUTE imports here, not relative — when this module
    # is the entry point of a PyInstaller bundle it has no parent package,
    # so `from ..client.worker` raises ImportError at runtime. Absolute
    # imports work in both the dev path (`python -m dllm.desktop.main`)
    # and the frozen bundle.
    if len(sys.argv) > 1 and sys.argv[1] == "--worker-mode":
        # Strip the --worker-mode shim so argparse in worker.main() sees a
        # vanilla argv. Everything after is passed through unchanged.
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        from dllm.client.worker import main as worker_main

        worker_main()
        return 0
    # Default: launch the GUI. Imported lazily so the --worker-mode path
    # never has to pay PySide6's ~200 MB resident-memory tax.
    from dllm.desktop.main_window import run_app

    return run_app()


if __name__ == "__main__":
    sys.exit(main())
