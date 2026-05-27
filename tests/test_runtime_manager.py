"""Tests for the runtime_manager (pure helpers + isolated subprocess paths).

We don't exercise install_runtime() here because it would actually download
PyTorch from the internet — that's the desktop client's first-launch
integration test, not unit coverage. Helpers + state-checks are testable
without network.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest import mock

import pytest

PySide6 = pytest.importorskip("PySide6")

from dllm.desktop import runtime_manager  # noqa: E402


def test_runtime_dir_is_under_user_data_dir(tmp_path: Path) -> None:
    """`runtime_dir()` lives inside the user_data_dir, not the repo cwd —
    so a contributor's runtime survives the launcher being moved / reinstalled.
    """
    rt = runtime_manager.runtime_dir()
    # Whatever the OS, the path ends with /dllm/runtime
    assert rt.name == "runtime"
    assert rt.parent.name == "dllm"


def test_runtime_python_path_is_platform_appropriate() -> None:
    py = runtime_manager.runtime_python()
    if sys.platform == "win32":
        assert py.name.lower() == "python.exe"
        # Windows venvs put it under Scripts/
        assert py.parent.name == "Scripts"
    else:
        # POSIX venvs use bin/python3
        assert py.parent.name == "bin"


def test_is_installed_false_when_marker_missing(tmp_path: Path, monkeypatch) -> None:
    """No marker → not installed, regardless of any stray python.exe."""
    monkeypatch.setattr(runtime_manager, "runtime_dir", lambda: tmp_path / "rt")
    monkeypatch.setattr(runtime_manager, "runtime_marker", lambda: tmp_path / "rt" / runtime_manager._INSTALLED_MARKER)
    assert runtime_manager.is_installed() is False


def test_is_installed_false_when_marker_malformed(tmp_path: Path, monkeypatch) -> None:
    """Marker exists but doesn't contain torch_version → not installed.
    Defends against partial installs that wrote half-baked markers.
    """
    rt = tmp_path / "rt"
    rt.mkdir()
    marker = rt / runtime_manager._INSTALLED_MARKER
    marker.write_text("{}", encoding="utf-8")  # no torch_version key

    monkeypatch.setattr(runtime_manager, "runtime_dir", lambda: rt)
    monkeypatch.setattr(runtime_manager, "runtime_marker", lambda: marker)
    monkeypatch.setattr(runtime_manager, "runtime_python", lambda: rt / "Scripts" / "python.exe")
    assert runtime_manager.is_installed() is False


def test_is_installed_false_when_python_missing(tmp_path: Path, monkeypatch) -> None:
    """Marker says torch is installed but python.exe was deleted (user
    cleared their AppData) → must report not installed.
    """
    rt = tmp_path / "rt"
    rt.mkdir()
    marker = rt / runtime_manager._INSTALLED_MARKER
    marker.write_text(json.dumps({"torch_version": "2.11.0"}), encoding="utf-8")
    fake_py = rt / "Scripts" / "python.exe"  # does not exist

    monkeypatch.setattr(runtime_manager, "runtime_dir", lambda: rt)
    monkeypatch.setattr(runtime_manager, "runtime_marker", lambda: marker)
    monkeypatch.setattr(runtime_manager, "runtime_python", lambda: fake_py)
    assert runtime_manager.is_installed() is False


def test_is_installed_true_when_marker_and_python_present(tmp_path: Path, monkeypatch) -> None:
    """Happy path: marker + python.exe → installed."""
    rt = tmp_path / "rt"
    py_dir = rt / "Scripts"
    py_dir.mkdir(parents=True)
    fake_py = py_dir / "python.exe"
    fake_py.write_text("not really python", encoding="utf-8")
    marker = rt / runtime_manager._INSTALLED_MARKER
    marker.write_text(json.dumps({"torch_version": "2.11.0"}), encoding="utf-8")

    monkeypatch.setattr(runtime_manager, "runtime_dir", lambda: rt)
    monkeypatch.setattr(runtime_manager, "runtime_marker", lambda: marker)
    monkeypatch.setattr(runtime_manager, "runtime_python", lambda: fake_py)
    assert runtime_manager.is_installed() is True


def test_runtime_info_returns_empty_when_no_marker(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(runtime_manager, "runtime_marker", lambda: tmp_path / "missing")
    assert runtime_manager.runtime_info() == {}


def test_verify_torch_available_returns_none_for_nonexistent_python() -> None:
    """Guards against calling subprocess on a python.exe that's not there
    (e.g. mid-install or after user-deletion)."""
    fake = Path("/nonexistent/python.exe")
    assert runtime_manager.verify_torch_available(fake) is None
