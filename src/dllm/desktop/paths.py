"""Cross-platform per-user paths for the desktop client.

We don't want to write to the install directory (Program Files / Applications
are read-only for unprivileged users on Windows + macOS respectively). All
mutable state — Ed25519 identity key, downloaded training data shards,
worker logs — lives in the OS's standard user data directory by default.

Users with constrained C:/system drives can override the data location at
first-launch setup; the choice is persisted to QSettings (see
`bootstrap_dialog`). Resolution order each call:

  1. ``DLLM_DATA_DIR`` env var — for CI, testing, custom workflows.
  2. QSettings("data_dir") — what the user picked at first-launch setup.
  3. Platform default (%APPDATA%, ~/Library/Application Support, $XDG_DATA_HOME).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _platform_default_data_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        xdg = os.environ.get("XDG_DATA_HOME")
        base = Path(xdg) if xdg else Path.home() / ".local" / "share"
    return base / "dllm"


def _settings_data_dir() -> Path | None:
    """Read the user's chosen data dir from QSettings. Returns None when
    PySide6 is unavailable (e.g. headless / tests without GUI) or when
    the user hasn't picked one yet.
    """
    try:
        from PySide6.QtCore import QSettings
    except ImportError:
        return None
    s = QSettings()
    raw = s.value("data_dir", "", type=str)
    if not raw:
        return None
    return Path(str(raw))


def user_data_dir() -> Path:
    """Return the per-user dllm data directory, creating it if needed.

    See module docstring for resolution order. Always returns an existing
    directory.
    """
    override = os.environ.get("DLLM_DATA_DIR")
    if override:
        p = Path(override) / "dllm" if not override.endswith("dllm") else Path(override)
    else:
        chosen = _settings_data_dir()
        p = chosen if chosen else _platform_default_data_dir()
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_user_data_dir(path: Path | str) -> None:
    """Persist the user's data-dir choice. Called from the first-launch
    bootstrap dialog after the user picks an install location.
    """
    from PySide6.QtCore import QSettings

    QSettings().setValue("data_dir", str(path))


def user_log_dir() -> Path:
    """Per-user log directory. Separate from data so users can wipe logs
    without losing their identity / cached shards.
    """
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        p = base / "dllm" / "logs"
    elif sys.platform == "darwin":
        p = Path.home() / "Library" / "Logs" / "dllm"
    else:
        xdg = os.environ.get("XDG_STATE_HOME")
        base = Path(xdg) if xdg else Path.home() / ".local" / "state"
        p = base / "dllm" / "logs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def identity_key_path() -> Path:
    """Where the Ed25519 contributor identity is persisted.

    NOTE: the CLI worker (`dllm.shared.identity.load_or_create_identity`)
    writes to `./.dllm/identity.key` (cwd-relative). The desktop client
    overrides that path to keep it portable across cwd changes — see
    main.py's `_configure_identity_path()`.
    """
    return user_data_dir() / "identity.key"


def settings_dir() -> Path:
    """For QSettings to use as backing storage (when we wire it). Today
    QSettings auto-locates per OS; this is here for future explicit use.
    """
    return user_data_dir() / "settings"


def cached_data_dir() -> Path:
    """Future home of streamed training data shards. v0 still expects
    `data/cache/train.bin` in the worker's cwd; the shard-streaming follow-up
    will move that download here.
    """
    p = user_data_dir() / "data"
    p.mkdir(parents=True, exist_ok=True)
    return p
