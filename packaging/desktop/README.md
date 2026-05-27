# Desktop client build

PyInstaller bundle of the PySide6 contributor GUI. Output is a one-folder
binary distribution: a `dllm-contributor[.exe]` plus ~2 GB of supporting
DLLs / dylibs / Qt plugins.

## Prerequisites

- A working Python install that can already run `dllm-desktop` from a clone
  (i.e. `pip install -e .[desktop]` worked locally).
- For GPU support in the bundle, PyTorch needs its CUDA wheels at build time.
  The installed `torch` package on the build machine is bundled as-is, so
  use a CUDA-enabled wheel (the default for `pip install torch` on Windows /
  Linux; macOS is CPU/MPS).
- PyInstaller: `pip install pyinstaller`

## Build

```bash
cd <repo>
pyinstaller packaging/desktop/dllm_desktop.spec --clean --noconfirm
```

Output: `dist/dllm-contributor/`. Launch the `dllm-contributor[.exe]` inside.

## What's intentionally NOT in the bundle

- **Training data shards** (~12 GB). The desktop client expects the worker's
  training data to be available via the upcoming coord-side `/shard?worker_id=N`
  endpoint. v0 still requires a local `data/cache/train.bin`; a follow-up
  patch wires per-worker shard streaming.
- **CUDA toolkit / NVIDIA drivers.** Volunteer is expected to have a working
  NVIDIA driver install on Windows / Linux. PyTorch's wheel includes the
  CUDA runtime DLLs but not the driver itself.
- **Code signing.** v0 is unsigned. Windows SmartScreen + macOS Gatekeeper
  will warn the user on first launch ("More info → Run anyway"). Signing
  (Authenticode + Apple notarization) is a Phase 1 wrap-up task.
- **Auto-update.** Operator currently re-downloads on each release. Sparkle
  / Squirrel integration is Phase 2.

## Notes / footguns

- **One-folder, not one-file**: PyTorch's many CUDA DLLs and Qt's plugin
  hierarchy resist single-file packing. One-folder works reliably; the
  trade-off is the unpacked tree is visible to the user.
- **UPX off**: UPX-compressing torch's CUDA DLLs has historically caused
  load failures. Spec disables UPX for both binaries and library files.
- **--clean is essential** when iterating: PyInstaller's caches can hide
  changes to the spec or to `dllm.desktop` until you wipe `build/` + `dist/`.
- **Output size**: expect ~2 GB on Windows (mostly PyTorch CUDA wheels +
  Qt). macOS is similar without CUDA but with Qt frameworks. Trimming with
  `excludes` in the spec is the only lever; PyTorch itself is non-negotiable
  for the worker path.
