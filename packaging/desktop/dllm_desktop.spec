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
# PyInstaller's static analysis catches most imports, but a few are dynamic
# and need hidden_imports. Worker spawns torch.distributed, datasets, etc.
# only optionally — listing them explicitly hedges against runtime "module
# not found".
hidden_imports = [
    "dllm.client.worker",
    "dllm.coord.server",  # only if a user happens to want local-coord too
    "dllm.shared.identity",
    "dllm.shared.protocol",
    "dllm.shared.serialize",
    "dllm.data.loader",
    "dllm.desktop.main_window",
    "dllm.desktop.worker_runner",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    # safetensors picks its backend at runtime
    "safetensors.torch",
    # tokenizers / datasets aren't strictly needed for the *worker* path,
    # but harmless to include + lets the GUI offer a "prepare data shard"
    # feature later without a rebuild.
]

# Data files to bundle alongside the binary. The coord dashboard.html is
# only needed when running coord-mode locally; harmless on contributor
# install (~50 KB). Use absolute paths anchored at project_root because
# PyInstaller resolves relative paths from the spec's directory.
datas = [
    (str(project_root / "src" / "dllm" / "coord" / "dashboard.html"), "dllm/coord"),
]


# ---------------------------------------------------------------------------
# Heavy exclusions — keep the bundle from ballooning past necessary
# ---------------------------------------------------------------------------
excludes = [
    # Test frameworks
    "pytest", "_pytest", "pytest_asyncio", "pytest_cov", "pytest_qt",
    # Notebook / data-science extras the worker never touches
    "notebook", "ipykernel", "ipython", "jupyter_core", "jupyterlab",
    "matplotlib", "scipy", "sklearn", "pandas",
    # Translation / SFT helpers (separate dep tree, not on worker path)
    "datasets", "tokenizers",
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
