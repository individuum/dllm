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

## Fixed: HTTP transient-error handling + graceful 404 (commit 7752222)

**Bug** (2026-05-25, three distinct crashes in one M5 session against `dllm.planetbass.de`):

| crash | call | failure | uncaught exception |
|---|---|---|---|
| 1 | mid-loop `GET /state` | truncated at 197/201 MB | `httpx.RemoteProtocolError` |
| 2 | `POST /delta` after long inner loop | coord deregistered us → 404 | `httpx.HTTPStatusError` |
| 3 | initial `pull_state()` | truncated at 98/201 MB | `httpx.RemoteProtocolError` |

Every `/state` and `/delta` call in [worker.py](src/dllm/client/worker.py) was unguarded — one nginx/podman/TLS hiccup mid-stream killed the worker. With a 200 MB state and a long-haul link, the truncation rate is high enough that any sustained run hit it.

**Fix**: module-level `retry_http()` with exponential backoff (1/2/4s, 4 attempts). Retries `RemoteProtocolError`, `ReadError`/`ReadTimeout`, `ConnectError`/`ConnectTimeout`, `WriteError`/`WriteTimeout`, `PoolTimeout`, and HTTP 5xx. **Passes 4xx through** — 404 on `/delta` is now caught at the call site, logged as `coord deregistered worker_id=X. Stopping.`, and the main loop exits cleanly via `dropped: True` instead of raising. Coverage: [tests/test_retry_http.py](tests/test_retry_http.py) (6 cases including the exact truncated-body failure observed live).

## Open: M5 deregistered before first delta

**Observed after the retry fix landed.** M5 worker registered cleanly, downloaded state, ran auto-tuned inner loop, then `POST /delta` got 404 because the coord had already timed it out. Timing breakdown:

| stage | wall clock |
|---|---|
| initial `GET /state` (192 MiB over WAN) | ~70 s |
| auto-tune benchmark (5 warmup + 30 measured steps) | ~15 s |
| first real inner loop (MPS JIT cold) | ~120 s |
| val + sign + `POST /delta` | ~30 s |
| **time-to-first-heartbeat** | **~4 min** |

The coord's dereg timeout is shorter than that, so a sole M5 worker can't ever contribute to round 1.

**Second-order finding: auto-tune is misleading on MPS.** The benchmark measures **4,710 tok/s** after 5 warmup steps; observed first-inner-loop throughput on the same model was **392 tok/s** (~12× slower). The second inner loop in the same process runs at ~3,500 tok/s — close to the benchmark. So MPS pays a large JIT-warmup tax that only 5 benchmark steps don't pay. Either the benchmark needs >50 warmup steps on MPS, or the cadence calculation should discard the first inner loop's wall clock.

**Two complementary fixes**:
1. *Worker-side*: on `POST /delta` 404, automatically re-register and resume (~20 LOC in `run()` + `_apply_sync_result`). Turns the graceful "Stopping" into a transparent recovery.
2. *Coord-side*: longer worker-inactivity timeout, or treat an in-flight `GET /state` as a heartbeat. Right value is probably max(2× pull_state time, 3× target_round_seconds).

Either fix alone unblocks M5; both together make the system robust to similar imbalances at other tiers.

## Observed: per-worker val_loss skew across heterogeneous cohort (2026-05-26)

After flipping `--world-size=2` and getting both 3060 + M5 contributing, dashboard val
jumped 4.55 → 5.49. Not a regression of the consensus model — an averaging artifact:

| round | 3060 val | M5 val | mean reported |
|---|---|---|---|
| 135 | 4.67 | 6.68 | **5.67** |
| 136 | 4.59 | 6.39 | **5.49** |
| 137 | 4.51 | (pending) | — |

M5 joined at round 135 and is doing its first long inner loops on a stale local model
(it pulled state once at join and hasn't resynced since). Its per-worker val reflects
*its own local θ*, not the consensus θ that the outer step produces. The outer step
averages **gradients** not losses, so the post-step cohort model's true val is close to
the 3060's number (~4.5), not the reported mean (5.49).

`last_val_loss` on `/status` is `mean(worker val_losses)` and is therefore misleading
in a heterogeneous cohort where workers' local θ diverge. The new dashboard's "active
workers" table surfaces per-worker val_loss so this is visible at a glance.

**Root cause was actually deeper:** `ShardLoader` partitioned `val.bin` by
`worker_id % world_size`. With world_size=2 the M5 was validating on the *second
half* of val.bin — a different EU language slice than the 3060 was using. Per-token
entropy varies meaningfully across the de/fr/es/it/en/nl/pl languages, so worker
val_losses live on different distributions and aren't comparable. Fixed by hard-coding
`val_loader = ShardLoader(val.bin, worker_id=0, world_size=1)` in the worker — every
worker now validates against the full val.bin. Test: `test_worker_val_loader_spans_full_file_regardless_of_world_size`.

After the val-shard fix lands, both workers' val should land within noise (~±0.1) of
each other and the averaged `last_val_loss` becomes a meaningful cohort signal again.

Future improvements:
1. Coord runs val on the consensus model itself after each outer step — slow on CPU
   container but truthful. ~10× simpler than weighting schemes.
2. Weight `last_val_loss` by how recently each worker resynced its `last_ref` (still
   matters for the per-round drift artifact even with shared val).

## Pointers

- Architecture + roadmap: [PLAN.md](PLAN.md) (Phase 0/1/2, EU compliance, tier-aware scheduling).
- VPS deploy recipe: [infra/README.md](infra/README.md). Coordinator runs under `podman` via systemd unit `dllm-coord.service`, fronted by nginx + Let's Encrypt.
- Presets defined in [src/dllm/core/config.py](src/dllm/core/config.py): `smoke` (~10M), `124M`, `1B`, `7B`. Vocab is `32768` everywhere — worker tokenizer must match.
