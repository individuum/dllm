"""Cross-platform per-user paths for the desktop client.

We don't want to write to the install directory (Program Files / Applications
are read-only for unprivileged users on Windows + macOS respectively). All
mutable state — Ed25519 identity key, downloaded training data shards,
worker logs — lives in the OS's standard user data directory.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def user_data_dir() -> Path:
    """Return the per-user dllm data directory, creating it if needed.

    - Windows: %APPDATA%/dllm  (e.g. C:/Users/foo/AppData/Roaming/dllm)
    - macOS:   ~/Library/Application Support/dllm
    - Linux:   $XDG_DATA_HOME/dllm  (default ~/.local/share/dllm)
    """
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        xdg = os.environ.get("XDG_DATA_HOME")
        base = Path(xdg) if xdg else Path.home() / ".local" / "share"
    p = base / "dllm"
    p.mkdir(parents=True, exist_ok=True)
    return p


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
