# PyInstaller spec for the dllm contributor GUI.
#
# Build:
#     cd <repo>
#     pip install pyinstaller
#     pyinstaller packaging/desktop/dllm_desktop.spec --clean --noconfirm
#
# Output: dist/dllm-contributor/  (one-folder mode — required for PyTorch's
# many CUDA DLLs to load cleanly). The executable is dist/dllm-contributor/
# dllm-contributor[.exe]. Total size ~2 GB on Windows (most of it torch.lib
# CUDA wheels).
#
# One-folder vs one-file
#   PyTorch's CUDA runtime, NumPy's BLAS, and PySide6's Qt plugins all hate
#   being squeezed into a single .exe (lazy DLL loading, native dependency
#   resolution, etc.). One-folder mode unpacks everything to disk at build
#   time → reliable, easier to debug, easier to inspect what shipped.
#
# This spec is platform-agnostic. macOS builds will produce a `.app` bundle
# automatically when run on macOS.

import sys
from pathlib import Path

block_cipher = None
# Anchor to the spec file's location rather than cwd. PyInstaller chdirs
# into the spec's directory while parsing, so cwd != where the user typed
# `pyinstaller packaging/desktop/dllm_desktop.spec`.
project_root = Path(SPECPATH).resolve().parent.parent


# ---------------------------------------------------------------------------
# Analysis: what to scan + which packages to force-include
# ---------------------------------------------------------------------------
# Lean-launcher mode: torch + numpy + safetensors live in the per-user
# runtime (bootstrap_dialog pip-installs them on first launch). The
# launcher only contains the GUI + bootstrap orchestration; it spawns the
# runtime's python.exe to do the actual training.
hidden_imports = [
    "dllm.desktop.main",
    "dllm.desktop.main_window",
    "dllm.desktop.worker_runner",
    "dllm.desktop.bootstrap_dialog",
    "dllm.desktop.runtime_manager",
    "dllm.desktop.paths",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    # cryptography (Ed25519 identity) ships C-extensions PyInstaller
    # doesn't always discover via static analysis.
    "cryptography.hazmat.primitives.asymmetric.ed25519",
    "cryptography.hazmat.primitives.serialization",
]

# Data files to bundle alongside the binary. The dashboard.html and the
# `src/dllm/` tree itself are included so runtime_manager._hand_copy_dllm_package
# can copy them into the bootstrapped runtime's site-packages when pip
# install dllm fails (e.g. offline / private network).
datas = [
    (str(project_root / "src" / "dllm" / "coord" / "dashboard.html"), "dllm/coord"),
]


# ---------------------------------------------------------------------------
# Heavy exclusions — the LEAN launcher must not bundle the ML stack
# ---------------------------------------------------------------------------
# torch + cuDNN + cuBLAS together weigh ~4 GB. We push them to the runtime
# install on first launch (PyTorch's CDN). The launcher then weighs in at
# ~150-200 MB — small enough for github releases / a fast download.
excludes = [
    # The ML stack — runtime_manager pip-installs these into the per-user
    # runtime, NOT into the launcher.
    "torch", "torchvision", "torchaudio",
    "numpy",
    "safetensors",
    "tokenizers", "datasets",
    "huggingface_hub", "transformers",
    # Misc data-science deps that pip might pull in transitively
    "scipy", "sklearn", "pandas", "matplotlib",
    # Notebook / IDE extras the worker never touches
    "notebook", "ipykernel", "ipython", "jupyter_core", "jupyterlab",
    # Test frameworks
    "pytest", "_pytest", "pytest_asyncio", "pytest_cov", "pytest_qt",
    # uvicorn + fastapi are coord-only; lean launcher is GUI + worker spawn
    "uvicorn", "fastapi", "starlette",
]


a = Analysis(
    [str(project_root / "src" / "dllm" / "desktop" / "main.py")],
    pathex=[str(project_root / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Console=False on Windows → no terminal window pops up alongside the GUI.
# (Workers spawned via QProcess still log to stdout, captured by the GUI.)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="dllm-contributor",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX-compressed PyTorch DLLs sometimes fail to load
    console=False,
    icon=None,  # TODO: add brand icon once we have one
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="dllm-contributor",
)


# ---------------------------------------------------------------------------
# macOS .app wrapper
# ---------------------------------------------------------------------------
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="dllm-contributor.app",
        icon=None,  # TODO
        bundle_identifier="de.planetbass.dllm.contributor",
        info_plist={
            "NSHighResolutionCapable": True,
            # PySide6/Qt needs this on macOS 12+ for non-blocking subprocesses
            "LSEnvironment": {"QT_LOGGING_RULES": "qt.qpa.input.tablet=false"},
            "CFBundleShortVersionString": "0.0.1",
            "CFBundleVersion": "0.0.1",
            "LSApplicationCategoryType": "public.app-category.utilities",
            # Apple notarization will require this when we get to Phase 2
            "NSCameraUsageDescription": "",  # placeholder
        },
    )
