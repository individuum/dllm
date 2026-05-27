"""Contributor desktop client.

A thin PySide6 GUI around `dllm.client.worker`. Goal: a volunteer with a
gaming GPU should be able to install one app, click Start, and contribute
compute to the EU sovereign LLM project — no command line, no Python venv,
no podman.

Phase 0 of the easy-install effort. Pairs with the `[desktop]` extra and
`dllm-desktop` entry point in pyproject.toml. PyInstaller bundling for
Windows / macOS lives in `scripts/build_desktop.py` (planned).
"""
