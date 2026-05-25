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

## License

Apache 2.0 for code. Weights (when released) will be under the EU Residents Community
License — see PLAN.md §5.2.
