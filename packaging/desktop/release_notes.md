# v0.0.1-alpha — first public contributor build

**EuroDLLM desktop contributor**, early access. Pre-built for **Windows 64-bit**;
macOS users build from source until I have a Mac handy to PyInstall on.

## What this is

A double-click installer that turns your idle GPU into a node in the EuroDLLM
distributed training run at [dllm.planetbass.de](https://dllm.planetbass.de).
Pick your country, click **Start contributing**, watch the loss curve drop.

## What it does

Per outer round (~8–10 min on a RTX 3060):
- Downloads ~600 MB of model state from the coord
- Runs your local "inner loop": 200 DiLoCo training steps
- Uploads a ~150 MB compressed delta back
- Auto-resyncs if the cohort moves on without you (stale-round recovery)
- Auto-reregisters if the coord evicts you for inactivity (timeout recovery)

Throughout: tracks energy used + estimated electricity cost, with **negative
€/kWh values supported** for PV / dynamic-tariff users whose marginal compute
literally earns money on sunny days.

## Hardware requirements

- **GPU**: NVIDIA, ≥6 GB VRAM. Tested on RTX 3060 12 GB.
- **Driver**: any recent NVIDIA driver (CUDA runtime is bundled).
- **OS**: Windows 10 / 11 64-bit.
- **Disk**: ~2.5 GB free for the install + identity + cached state.
- **Network**: home broadband; ~750 MB per round each way.

NOT supported in v0 (will land in v1):
- macOS (build from source today, signed .app coming)
- AMD GPUs (PyTorch ROCm is Linux-only)
- < 6 GB VRAM (model won't fit)

## Security warnings on first launch

This build is **unsigned**. Windows SmartScreen will show:

> **Windows protected your PC** — Microsoft Defender SmartScreen prevented an
> unrecognized app from starting…

Click **More info → Run anyway**. (Code signing is on the Phase 1 roadmap;
costs €€ + admin time we haven't spent yet.)

## Privacy + data handling

- No telemetry beyond what the coord sees: your country code, GPU name, VRAM
  size, public key, and per-round metrics (loss, throughput, power).
- Your training data is **public Wikipedia + Project Gutenberg + EuroParl +
  JRC-Acquis + PleIAs PD Books** (5 EU langs, ~6 B tokens). No personal data,
  no scraped private content.
- Identity key (Ed25519, signs your deltas) at `%APPDATA%/dllm/identity.key`
  — back this up if you want contribution continuity across re-installs.

## Known limitations

- ~12 GB training data shard is **not bundled** — v0 expects you to pre-fetch
  `data/cache/train.bin` in the cwd. Coord-side `/shard?worker_id=N` streaming
  is the next priority and will land before any public push beyond this alpha.
- The v0 GUI uses a fixed coord URL (`https://dllm.planetbass.de`). Future
  versions will let you point at private coords.

## License

Apache 2.0 (code). Weights when released: EU Residents Community License.
See [PLAN.md §5.2](https://github.com/individuum/dllm/blob/main/PLAN.md#52-licensing).

## Building from source

```bash
git clone https://github.com/individuum/dllm
cd dllm
pip install -e .[desktop]
dllm-desktop   # launches the GUI
```

Or build your own bundle:
```bash
pip install pyinstaller
pyinstaller packaging/desktop/dllm_desktop.spec --clean --noconfirm
# Output: dist/dllm-contributor/
```
