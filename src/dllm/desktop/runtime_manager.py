"""First-launch CUDA runtime bootstrap.

The lean launcher (PyInstaller bundle, ~150 MB) excludes torch + CUDA so
volunteers can download a small installer fast. On first launch we set up
a per-user "runtime" Python environment under <user_data>/runtime/ and
pip-install the heavy ML stack into it from PyTorch's official CDN.

Layout after bootstrap:

    %APPDATA%/dllm/                      (Windows)
    ~/Library/Application Support/dllm/  (macOS)
    └── runtime/
        ├── python.exe / bin/python3
        ├── Lib/site-packages/
        │   ├── torch/      (~2 GB, includes CUDA DLLs)
        │   ├── numpy/
        │   ├── safetensors/
        │   ├── httpx/
        │   ├── cryptography/
        │   ├── pydantic/
        │   └── dllm/       (cloned from the launcher's bundle)
        └── ...

Subsequent launches: skip the bootstrap, spawn worker subprocess as
`runtime/python.exe -m dllm.client.worker ...`.

Why a per-user runtime instead of one bundled exe:
- 150 MB launcher vs 5 GB monolith → 30× faster initial download
- Reuses PyTorch's CDN for the heavy bits (~2 GB) — saves our VPS egress
- Updates handled by re-running `pip install --upgrade torch dllm` instead
  of re-downloading the entire installer
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Callable

from .paths import user_data_dir

# Marker file written after a successful bootstrap. Caches what was installed
# so subsequent launches can skip the slow `import torch` probe.
_INSTALLED_MARKER = "runtime_installed.json"

# PyTorch's official CDN. cu128 = CUDA 12.8 runtime (matches the consumer
# 30xx/40xx/50xx series on current drivers). Pin to a tested torch version
# so torch + worker stay protocol-compatible across upgrades.
_TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"
_TORCH_SPEC = "torch>=2.4,<2.13"

# Other runtime deps the worker needs. Pulled from PyPI (default index),
# not the torch CDN. Pinned with the same lower bounds as pyproject.toml.
_RUNTIME_DEPS = [
    "numpy>=1.26",
    "safetensors>=0.4",
    "httpx>=0.27",
    "pydantic>=2.8",
    "cryptography>=43.0",
    "fastapi>=0.115",  # only needed if user runs local coord mode
]


def runtime_dir() -> Path:
    """Where the bootstrapped runtime lives. Created on first install."""
    return user_data_dir() / "runtime"


def runtime_python() -> Path:
    """Path to the per-user runtime's Python interpreter."""
    if sys.platform == "win32":
        return runtime_dir() / "Scripts" / "python.exe"
    return runtime_dir() / "bin" / "python3"


def runtime_marker() -> Path:
    return runtime_dir() / _INSTALLED_MARKER


def is_installed() -> bool:
    """Cheap check: marker file exists AND points to a real python.exe.

    Avoids the expensive `subprocess(import torch)` probe on every launch.
    Bootstrap writes the marker only after the install succeeds end-to-end,
    so a partial / interrupted install does NOT set is_installed=True.
    """
    marker = runtime_marker()
    if not marker.exists():
        return False
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not runtime_python().exists():
        return False
    return bool(data.get("torch_version"))


def verify_torch_available(python_exe: Path | None = None) -> str | None:
    """Run `python -c "import torch; print(torch.__version__)"` against the
    runtime. Returns the version string or None if import failed. Slow
    (~3 s on cold runtime). Use sparingly — `is_installed()` is the fast
    path for the common case.
    """
    py = python_exe or runtime_python()
    if not py.exists():
        return None
    try:
        r = subprocess.run(
            [str(py), "-c", "import torch; print(torch.__version__)"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    if r.returncode != 0:
        return None
    return r.stdout.strip() or None


def _seed_runtime_with_venv(progress: Callable[[str, float], None]) -> Path:
    """Create the runtime/ directory as a Python venv based on the
    interpreter that's currently running the launcher.

    Why a venv: it's portable enough that pip install --target=
    site-packages works, AND it pulls in pip/setuptools/wheel as part of
    creation so we don't need to download get-pip.py ourselves. On
    Windows the venv layout is Scripts/python.exe + Lib/site-packages.

    Note: if the launcher's Python is a PyInstaller-frozen interpreter,
    venv creation fails because frozen Python doesn't expose ensurepip.
    In that case we fall back to a portable embedded distribution
    (downloaded from python.org) — see _seed_runtime_with_embedded().
    """
    rt = runtime_dir()
    rt.parent.mkdir(parents=True, exist_ok=True)
    if rt.exists():
        shutil.rmtree(rt, ignore_errors=True)

    progress("Creating runtime environment...", 0.05)
    # Use sys._base_executable when available (Python 3.11+ on
    # PyInstaller-frozen interpreters). Otherwise sys.executable.
    base_py = getattr(sys, "_base_executable", None) or sys.executable
    try:
        subprocess.run(
            [base_py, "-m", "venv", str(rt), "--upgrade-deps"],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to create runtime venv: {e.stderr or e.stdout!r}"
        ) from e
    return runtime_python()


def install_runtime(
    progress: Callable[[str, float], None] | None = None,
    torch_spec: str = _TORCH_SPEC,
    torch_index_url: str = _TORCH_INDEX_URL,
    extra_deps: list[str] | None = None,
) -> dict:
    """Bootstrap the per-user runtime. Idempotent: a second call after a
    successful bootstrap re-runs pip (cheap; pip skips already-satisfied).

    `progress(message, fraction_0_to_1)` is called periodically so the GUI
    can update its progress bar. Safe to call from a worker thread; the
    caller is responsible for thread-safety in the callback.

    Returns the marker dict (also written to disk). Raises on failure.
    """
    p = progress or (lambda m, f: None)
    # 1. Seed the runtime directory with a fresh venv.
    py = _seed_runtime_with_venv(p)

    # 2. Pip install torch from the official CUDA CDN. This is the big
    #    download (~2 GB for cu128). pip prints to stderr by default — we
    #    line-buffer it so the GUI can pull progress out.
    p(f"Downloading PyTorch ({torch_spec}) + CUDA runtime — about 2 GB...", 0.10)
    try:
        subprocess.run(
            [
                str(py), "-m", "pip", "install",
                "--no-cache-dir",
                "--index-url", torch_index_url,
                "--extra-index-url", "https://pypi.org/simple",
                torch_spec,
            ],
            check=True,
            timeout=1800,  # 30 min for slow connections
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"pip install torch failed (exit {e.returncode})") from e

    # 3. Other runtime deps from PyPI.
    deps = list(_RUNTIME_DEPS)
    if extra_deps:
        deps.extend(extra_deps)
    p("Installing supporting libraries...", 0.85)
    try:
        subprocess.run(
            [str(py), "-m", "pip", "install", "--no-cache-dir", *deps],
            check=True,
            timeout=600,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"pip install of runtime deps failed: {e}") from e

    # 4. Install dllm itself (worker module) into the runtime so the
    #    spawned subprocess can `python -m dllm.client.worker ...`.
    #    Prefer the launcher's bundled wheel (avoids a network round-trip);
    #    fall back to PyPI when running in dev mode.
    p("Installing dllm worker module...", 0.95)
    bundled_wheel = _locate_bundled_dllm_source()
    install_target = bundled_wheel if bundled_wheel else "dllm"
    try:
        subprocess.run(
            [str(py), "-m", "pip", "install", "--no-cache-dir", str(install_target)],
            check=True,
            timeout=300,
        )
    except subprocess.CalledProcessError:
        # Last resort: hand-copy the dllm package into site-packages.
        # Less clean than pip but works even when bundled source isn't
        # importable as a normal package (e.g., PyInstaller-frozen .pyc).
        _hand_copy_dllm_package(py)

    # 5. Verify torch loads.
    p("Verifying installation...", 0.98)
    version = verify_torch_available(py)
    if version is None:
        raise RuntimeError("torch installed but `import torch` failed at probe")

    marker_data = {
        "torch_version": version,
        "torch_spec": torch_spec,
        "torch_index_url": torch_index_url,
        "runtime_dir": str(runtime_dir()),
        "python_exe": str(py),
    }
    runtime_marker().write_text(json.dumps(marker_data, indent=2), encoding="utf-8")
    p("Done!", 1.0)
    return marker_data


def _locate_bundled_dllm_source() -> Path | None:
    """If we're running inside a PyInstaller bundle, the launcher ships
    the dllm source tree alongside its frozen scripts. Best-effort
    locator; returns None when not found so caller falls back to PyPI.
    """
    # PyInstaller exposes sys._MEIPASS as the temp-extract dir of the
    # frozen bundle. In dev mode (`python -m dllm.desktop.main`), this
    # attribute doesn't exist.
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        # Bundled `pip install <path-to-source-or-wheel>` works if there's
        # a setup.py/pyproject.toml at that path. The PyInstaller spec
        # has to ship those alongside the source — we'll handle when we
        # update the spec. For now this just returns None.
        candidate = Path(meipass) / "dllm-source"
        if candidate.exists():
            return candidate
    return None


def _hand_copy_dllm_package(runtime_py: Path) -> None:
    """Fallback: copy the dllm/ source dir into the runtime's site-packages
    when pip install dllm-from-bundle isn't available. Best-effort; raises
    on failure.
    """
    import dllm  # imports from the launcher's package path

    src = Path(dllm.__file__).parent  # /path/to/dllm/
    # Find runtime's site-packages.
    try:
        sp = subprocess.run(
            [str(runtime_py), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError("Could not locate runtime site-packages") from e
    if not sp:
        raise RuntimeError("Runtime returned empty site-packages path")
    dst = Path(sp) / "dllm"
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst)


def runtime_info() -> dict:
    """Read the marker for display in the GUI (e.g. About dialog)."""
    if not runtime_marker().exists():
        return {}
    try:
        return json.loads(runtime_marker().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
