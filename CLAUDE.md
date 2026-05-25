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

## Known issue: worker exits on stale-round rejection

**Observed 2026-05-25 against the live coord.** M5 worker registered at round 9, ran 500 inner steps in ~5 min, but in that window the other two workers advanced the coord to round 12. The coord returned `{accepted: false, reason: 'stale: coord round=12, worker=9', next_round: 12}` and the worker's `_apply_sync_result` (worker.py:431-435) bails with `sync rejected` → `run()` logs `sync failed; stopping early` (worker.py:467-469) and exits 0.

Val loss did drop locally (7.06 → 6.70 over one round) before the bail, so the inner loop is doing real work — the worker just can't keep up with the cadence and has no resync path.

**Suggested fix** (~10 lines in [src/dllm/client/worker.py](src/dllm/client/worker.py)): when `_async_sync_io` sees `accepted=false` with a `next_round`, GET `/state` at that round, return it as a "resync" result, and have `_apply_sync_result` update `last_ref` (and the local model + optimizer state? — needs design call) instead of returning False. Effectively: skip the missed deltas, catch up to the consensus, keep going. This makes the worker tolerant of being the slow node in a heterogeneous fleet — exactly the Phase 1+ scenario.

Open questions for that PR:
- Resync should probably also `load_into_model(self.model, new_state)` to avoid the local θ drifting arbitrarily far from consensus.
- Should the rejected delta count toward FLOPs reporting? Currently `flops_total` is coord-side, so no — but worth confirming the coord doesn't double-bill on the retry.
- Heterogeneous-fleet scheduling (PLAN.md §3 anchor pods vs individual workers) suggests tier-aware `inner_steps` per worker; the resync path is the bare-minimum stopgap until that lands.

## Pointers

- Architecture + roadmap: [PLAN.md](PLAN.md) (Phase 0/1/2, EU compliance, tier-aware scheduling).
- VPS deploy recipe: [infra/README.md](infra/README.md). Coordinator runs under `podman` via systemd unit `dllm-coord.service`, fronted by nginx + Let's Encrypt.
- Presets defined in [src/dllm/core/config.py](src/dllm/core/config.py): `smoke` (~10M), `124M`, `1B`, `7B`. Vocab is `32768` everywhere — worker tokenizer must match.
