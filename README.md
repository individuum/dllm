# EuroDLLM

Volunteer-trained, EU-jurisdiction, contributor-owned open language model.

See [PLAN.md](PLAN.md) for the full design. This is **Phase 0**: a working prototype
of [DiLoCo](https://arxiv.org/abs/2311.08105)-style inner/outer distributed training
on one machine, ready to extend to WAN.

## Status

Phase 0 — prototype. Not for production. No EU residency layer yet, no auth, no Byzantine
handling. Just the training-loop bones.

## Quickstart

Requires Python 3.11+ and (optionally, but strongly recommended) a CUDA GPU.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .

# 1. prepare the smoke-test corpus (TinyShakespeare, ~1MB tokenized)
python -m dllm.data.prepare

# 2. run the smoke test: 1 coordinator + 2 workers, 124M-equivalent model
python -m dllm.scripts.smoke_test
```

The smoke test runs ~50 outer cycles and prints a loss curve. Loss should drop
from ~10 to ~5 within a few minutes on a single 4090, or 15–30 minutes on CPU.

## Layout

```
core/        model.py, config.py — the transformer + size presets
data/        prepare.py, loader.py — corpus prep and mmap loading
shared/      protocol.py, serialize.py — wire types + safetensors helpers
coord/       server.py, state_store.py, outer.py — coordinator
client/      worker.py — worker entry point
scripts/     smoke_test.py — local end-to-end test
```

## Contributing

If you have a GPU sitting idle, you can donate compute to the live training run
at [dllm.planetbass.de](https://dllm.planetbass.de):

### Easy mode — download the desktop client

Pre-built installer (Windows, 64-bit, ~2 GB):
**[Latest release on GitHub](https://github.com/individuum/dllm/releases/latest)**

Unzip, run `dllm-contributor.exe`, pick your country, click **Start contributing**.
The client connects outbound-only over HTTPS — no router or firewall config needed.
Identity persists at `%APPDATA%/dllm/identity.key` so your contribution credit
follows you across reboots.

⚠ The v0 binary is **unsigned**. Windows SmartScreen will warn on first launch —
choose "More info → Run anyway". Signed installers + macOS .app builds land in
Phase 1.

### Power-user mode — CLI worker

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .[desktop]

# Worker connects to the public coord
python -m dllm.client.worker \
    --coord https://dllm.planetbass.de \
    --preset 300M --country DE --device cuda --require-gpu \
    --max-rounds 1000
```

macOS workers use `--device mps`. Linux + NVIDIA GPUs use `--device cuda`.

### What it costs

Per round (~8-10 min on a RTX 3060):
- ~750 MB network traffic (state pull + delta upload)
- ~22 Wh of electricity (~150 W × 10 min) — about €0.007 at €0.30/kWh

The desktop client surfaces both as you contribute and supports negative €/kWh
inputs for PV / dynamic-tariff users whose marginal compute can actually earn
money.

## License

Apache 2.0 for code. Weights (when released) will be under the EU Residents Community
License — see PLAN.md §5.2.
