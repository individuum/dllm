# CLAUDE.md — notes for future sessions

## Live deployment

- **Coordinator**: `https://dllm.planetbass.de` (Phase 1 VPS, see [infra/README.md](infra/README.md))
- Health: `GET /health` → `{ok, round}`. Status: `GET /status` → `{current_round, n_registered, n_submitted, waiting_for, last_val_loss, flops_total}`.
- **Running preset**: `124M` (~200 MB bf16 state at `/state`, `x-codec: bf16`, delta codec `q8`).
- World size + inner steps are returned by `POST /register` — observed `inner_steps=500` with `world_size=2` on 2026-05-25.

## Local setup (macOS, no system Python ≥3.11)

The repo requires Python ≥3.11. System Python was 3.9; `uv` was installed via Homebrew and used to bootstrap:

```bash
brew install uv
uv venv --python 3.12 .venv
uv pip install -e ".[data]"   # [data] needed only for dllm.data.prepare
```

`torch 2.12` with MPS works on Apple Silicon (verified on M5). Use `--device mps --require-gpu` on the worker.

Data prep (`python -m dllm.data.prepare`) downloads 7 EU-language Wikipedia samples and writes `data/cache/{train,val}.bin` + `tokenizer.json` (32k BPE vocab — matches `ModelConfig.vocab_size`). Takes a few minutes. The `.bin` shards are gitignored; identity key persists at `.dllm/identity.key`.

Run the worker:
```bash
.venv/bin/python -m dllm.client.worker \
    --coord https://dllm.planetbass.de \
    --preset 124M --country DE --device mps --require-gpu \
    --max-rounds 200
```

## Fixed: stale-round resync (was: worker exits on stale-round rejection)

**Bug** (2026-05-25): M5 worker registered at round 9, ran 500 inner steps in ~5 min while the two faster 3060 workers advanced the coord to round 12. Coord returned `{accepted: false, reason: 'stale: coord round=12, worker=9', next_round: 12}` and the worker's `_apply_sync_result` bailed with `sync rejected`; `run()` logged `sync failed; stopping early` and exited 0. Val had dropped locally 7.06 → 6.70 before the bail, so inner work was real — the worker just couldn't keep up with the cadence and had no resync path.

**Fix** (commit after `9c74ea6`): `_async_sync_io` now treats a stale-round rejection as a resync trigger — it fetches `/state` at the coord's current round and returns it with `resync=True`. `_apply_sync_result` then both updates `last_ref` and reloads the local model from the new consensus (preventing unbounded drift across many missed rounds). Optimizer state is preserved across the resync — matches DiLoCo paper guidance, the momentum is still useful gradient signal. Coverage in `tests/test_coord_api.py::test_worker_resync_on_stale_round_rejection`.

Net effect: a slow worker (e.g. M5 in a 3060-dominated fleet) skips the missed outer steps, catches up to consensus, and keeps contributing — no longer exits 0 mid-run.

Future tier-aware work (PLAN.md §3): per-worker `inner_steps` so anchor pods do 500+ while individual slow workers do fewer, eliminating the need for resync in the steady state. The resync path is the stopgap until that lands.

## Pointers

- Architecture + roadmap: [PLAN.md](PLAN.md) (Phase 0/1/2, EU compliance, tier-aware scheduling).
- VPS deploy recipe: [infra/README.md](infra/README.md). Coordinator runs under `podman` via systemd unit `dllm-coord.service`, fronted by nginx + Let's Encrypt.
- Presets defined in [src/dllm/core/config.py](src/dllm/core/config.py): `smoke` (~10M), `124M`, `1B`, `7B`. Vocab is `32768` everywhere — worker tokenizer must match.
