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

## Corpus v2 — 5B tokens, 5 languages, EU-compliant (2026-05-26)

The original `prepare.py` pulled small Wikipedia samples (7 langs, ~600 M tokens).
Replaced with a multi-source streaming pipeline targeting ~5 B tokens across
DE/FR/EN/IT/ES from four open sources:

| source | per-lang chars | license | role |
|---|---|---|---|
| Wikipedia | 2.8 G | CC-BY-SA 4.0 | encyclopedic register, bulk |
| Project Gutenberg | 800 M | Public Domain | literary register, balance |
| EuroParl | 200 M | PD (EU institutional) | political / formal register |
| JRC-Acquis | 120 M | PD (EU institutional) | legal / administrative register |

**Architecture**: `dllm/data/sources/{wikipedia,gutenberg,europarl,jrc_acquis}.py`
each expose `iter_docs(lang, char_budget)` + `license_info()` + `supported_langs()`.
The orchestrator in `prepare.py` round-robins across (source, lang) pairs so
`train.bin` has uniform mix throughout (critical for `ShardLoader` partitioning).

**Memory**: streaming tokenization with a `ShardWriter` that flushes uint16 tokens
to disk every 100 MB — 5 B tokens never sit in RAM.

**Compliance**: `manifest.json` records per-source license + URL + per-(source,lang)
token counts + AI Act Art. 53 disclosure + DSM Art. 4 / GDPR posture. Sufficient
for the EU AI Act "detailed summary" requirement.

**Run**: `python -m dllm.data.prepare` (full run, multi-hour download + tokenize)
or `python -m dllm.data.prepare --dry-run` (probe each source for reachability).
The token / BPE / shard format is identical to v1 — only the *content* of
`train.bin`/`val.bin` changes. Re-tokenizing means current model checkpoints
become incompatible (vocab IDs shift), so plan a fresh training run when
swapping in the new corpus.

## Data prep speedup: parallelize both stages (2026-05-26)

The v3 corpus (~5.5 B tokens) initially took **6 hours** on a 32-core box —
~240 k tok/s, one core active. Two cumulative fixes brought it to **~30-60 min**.

**Fix 1: batched + threaded tokenization.** The naive
`for doc in stream: tokenizer.encode(doc)` loop pinned the `tokenizers`
Rust impl to one thread regardless of CPU count: Python held the GIL
between encode calls and the Rayon thread pool couldn't parallelize across
docs. Replaced with `tokenizer.encode_batch(batch, batch_size=256)` which
releases the GIL and uses all Rayon threads. Throughput jumped to ~350 k tok/s
but the bottleneck shifted to the producer.

**Fix 2: parallel HF producers.** PleIAs books are ~33 k tokens/doc, ~100 ms
each from one HTTP stream → producer fed only ~10 docs/sec, tokenizer idle
95 %. Refactored `tokenize_streaming` to spawn one Python thread per
(source, lang) pair (~25 for v3). GIL releases during socket I/O so they
download concurrently. Throughput ~750 k tok/s, ETA ~2 hours.

**Operator notes:**
- `--reuse-tokenizer` flag skips Phase 1+2 (BPE training) on re-runs; the
  tokenizer.json on disk is reused as-is.
- Set `RAYON_NUM_THREADS=N` to cap Rust thread pool (default = all cores).
- Tokenization batch size: `--tokenize-batch-size 256` (default). Larger =
  more memory, more GIL release per batch; smaller = lower latency.

**Remaining bottleneck:** network bandwidth (sum of 25 HF streams). To go
below ~30 min, pre-download parquet files to local disk first and switch
streaming=False. Not done yet.

## Data prep resume (`--resume`, 2026-05-27)

`prepare.py --resume` recovers an interrupted Phase 3 without re-running
Phase 1+2. Phase 1+2 (BPE training) was already resumable via
`--reuse-tokenizer`; this closes the loop on Phase 3.

**How it works**: `tokenize_streaming` writes
`data/cache/prepare_progress.json` (atomic tmp + rename) after every
batch, recording per-(source, lang) emitted chars + tokens. On
`--resume`, the orchestrator:
1. Pins `args.sources` to the existing `manifest.json`'s `source_keys`
   (new top-level field, format_version bumped to 0.0.4) so a recovery
   run can't accidentally widen/narrow the corpus shape.
2. Opens `ShardWriter` in append mode (preserves existing train.bin
   bytes; `_total` is pre-loaded from `path.stat().st_size // 2`).
3. Reduces each per-pair budget by chars already emitted: `iter_docs`
   is called with `max(0, alloc - emitted)`.
4. Deletes any stale `val.bin` so Phase 4 re-splits cleanly.
5. Removes `prepare_progress.json` on successful Phase 5 completion.

**Known limitation — prefix duplication**: source `iter_docs` is
stateless and restarts from the first doc each call. So the resumed
run re-fetches the prefix (chars 0..emitted) and appends a duplicated
copy of it to train.bin. The chars-not-yet-reached at crash time
remain unreached. Net effect: total token count lands within ~10 % of
a clean run (the test pins this), but the corpus contains some
near-duplicate prefix per source. Acceptable for "don't lose your
work" recovery; for a clean rebuild, omit `--resume` and start over.

Test coverage: [tests/test_prepare.py::test_resume_appends_to_existing_train_bin_within_10pct_of_fresh](tests/test_prepare.py).

## Tier-aware scheduling (2026-05-27)

PLAN.md §3's "per-worker `inner_steps`" promise — coord-side adaptive
scheduler that sizes each worker's inner loop so all workers finish in
~target_round_seconds regardless of GPU class. Replaces the stopgap
worker-side `--auto-tune-steps` flag (which still works as a fallback).

**Wire change**: `DeltaAck` gains an optional `inner_steps: int | None`.
When the coord has re-tuned a worker, the next /delta response carries
the new value; the worker applies it before its next inner loop. Backward
compatible — workers that ignore the field stay on whatever they got at
register time.

**Coord side** ([server.py](src/dllm/coord/server.py) `_maybe_retune_worker`):
1. Workers report `tokens_per_sec` on every /delta (already did, dashboard
   used it for cohort throughput).
2. Coord computes `proposed = clamp(round(tps * target / (batch * seq)),
   [min_inner_steps, max_inner_steps])` per worker.
3. Only fires when `|proposed - current| / current > retune_threshold`
   (default 20%) — kills dashboard churn from tok/s noise.
4. Stores per-worker `inner_steps` in `self.workers[wid]`; surfaced on
   `/workers` for the dashboard's active-workers table.
5. FLOPs accounting reworked: each round's contribution is
   `sum(worker_inner_steps for worker in deltas_this_round) * batch * seq`,
   not the old uniform `world_size * default_steps * batch * seq`.
   Captures the per-worker imbalance correctly.

**Coord CLI**: `--tier-aware`, `--target-round-seconds 600`,
`--min-inner-steps 50`, `--max-inner-steps 2000`, `--retune-threshold 0.20`,
`--flops-alarm-threshold 5e24`.

**Worker side** ([worker.py](src/dllm/client/worker.py) `_apply_sync_result`):
applies `result["inner_steps"]` (carried from ack) by overwriting
`self.inner_steps` between inner loops. Mutation is safe there because
`run()` always drains the previous sync BEFORE starting the next
`run_inner()`.

**Dashboard**: workers table grew an "inner steps" column; current-round
table gains a "tier-aware scheduling" row showing target_round_seconds.

Coverage: 5 tests in `test_coord_api.py` (per-worker assignment,
backward-compat off path, retune threshold, /status exposure, FLOPs
account correctness under tier-aware).

## Worker auto-reregister on /delta 404 (2026-05-27)

CLAUDE.md "Open: M5 deregistered before first delta" — eliminated.
Previously the coord-side eviction (worker inactive too long) caused
the worker's next /delta to return 404, and the worker logged "Stopping"
and exited 0. Now `run()` detects `result["dropped"]`, calls
`_reregister_and_resync()` which:

1. Re-registers with the same Ed25519 pubkey — gets a fresh `worker_id`
   from the coord, but signed-delta verification still works because the
   key didn't change.
2. Pulls current consensus state into the local model + last_ref.
3. Rebuilds the train loader with the new worker_id so the
   `worker_id % world_size` shard slice is correct.
4. Skips the wasted-delta submission for the iteration it was evicted on
   (no point — coord doesn't know what state we built it against).
5. Continues from the next inner loop.

Coverage: `test_worker_reregister_resumes_with_fresh_id` — registers,
manually evicts at the coord, calls `_reregister_and_resync`, verifies
the worker got a new id with the same pubkey.

Net effect: an M5 doing a slow first inner loop (~4 min before any
heartbeat lands) can now be evicted by the coord without bringing the
worker down. It quietly re-registers, resyncs to the current round, and
contributes from the next outer step.

## FLOPs alarm (2026-05-27)

PLAN.md §5.3 requirement: alarm at 5×10²⁴ FLOPs to stay safely under the
EU AI Act systemic-risk threshold (10²⁵). Default `--flops-alarm-threshold
5e24` on the coord; `/status` includes `flops_alarm_threshold` so the
dashboard knows what bar to compare against; banner appears at the top
of the dashboard when `flops_total >= flops_alarm_threshold`. Operator
can override via `--flops-alarm-threshold 0` (disable) or any other
positive value.

For Phase 0–1 we're nowhere close (current run is at ~10¹⁷ FLOPs). The
guardrail matters for the 7B Phase 2 and 70B Phase 3 runs.

## Contributor desktop client (2026-05-27)

PySide6 GUI wrapping `dllm.client.worker` so non-CLI volunteers can
contribute. Lives at [src/dllm/desktop/](src/dllm/desktop/). Install via
`pip install -e .[desktop]`, launch with `dllm-desktop` (or `python -m
dllm.desktop.main`).

**Architecture**:
- `main.py` — entry point. Detects `--worker-mode` shim for future
  PyInstaller bundle (frozen binary re-execs itself as the worker
  subprocess instead of relying on `python -m`).
- `main_window.py` — `MainWindow`: country picker, preset picker,
  Start/Stop, 4 metric tiles (round, val_loss, throughput, power), log
  pane with 2000-line ring buffer.
- `worker_runner.py` — `WorkerRunner(QObject)`: wraps `QProcess` running
  the worker subprocess; parses well-known log lines into Qt signals
  (`registered`, `inner_completed`, `val_reported`, `sync_applied`,
  `retune_applied`, `reregistered`, `power_sample`).
- `paths.py` — per-user data + log dirs (Windows %APPDATA%, macOS
  ~/Library/Application Support, Linux XDG). Identity key persists at
  `<data_dir>/identity.key` instead of cwd-relative
  `./.dllm/identity.key` — survives reinstalls + cwd changes.

**Identity migration**: the CLI worker now honors `DLLM_IDENTITY_KEY`
env var (additive to `load_or_create_identity(path)`), which the desktop
client sets to the per-user path. Backward compatible — CLI users
without the env var keep using `./.dllm/identity.key`.

**Test coverage**: 7 regex tests in `test_desktop_runner.py` lock down
the log-line → signal contract. PySide6 is `importorskip`-gated so the
test module just skips on core-only installs.

**PyInstaller bundle** lives at [packaging/desktop/dllm_desktop.spec](packaging/desktop/dllm_desktop.spec).
One-folder mode (PyTorch CUDA + Qt resist single-file packing). Build with
`pyinstaller packaging/desktop/dllm_desktop.spec --clean --noconfirm`. Output
is ~2 GB on Windows. v0 unsigned; signing + auto-update are Phase 1 wrap-up.

**What's missing for full Phase 1 (volunteer-facing release)**:
- Coord-side `/shard?worker_id=N&world_size=K` endpoint so the GUI
  doesn't require a pre-existing local `data/cache/train.bin` (12 GB)
- System tray / menu-bar integration (pause/resume from the dock)
- Code signing (Authenticode + Apple notarization)
- Auto-update (Sparkle on macOS, Squirrel on Windows)
- Brand icon
- License + residency-attestation flow (placeholder for eIDAS later)

## Pointers

- Architecture + roadmap: [PLAN.md](PLAN.md) (Phase 0/1/2, EU compliance, tier-aware scheduling).
- VPS deploy recipe: [infra/README.md](infra/README.md). Coordinator runs under `podman` via systemd unit `dllm-coord.service`, fronted by nginx + Let's Encrypt.
- Presets defined in [src/dllm/core/config.py](src/dllm/core/config.py): `smoke` (~10M), `124M`, `1B`, `7B`. Vocab is `32768` everywhere — worker tokenizer must match.
