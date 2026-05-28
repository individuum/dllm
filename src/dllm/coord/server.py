"""Phase 0.5 coordinator.

Holds the global model state; collects pseudo-gradient deltas from N workers;
applies a Nesterov outer step when all workers in the current round have submitted.
Supports bf16 state transport + q8 delta transport, periodic disk checkpoints,
val-loss aggregation, and cumulative FLOPs accounting.

This module keeps the `CoordinatorState` core (construction, the request-handling
API surface, and state serialization) plus the `main` entry point. The class is
assembled from cohesive mixins:
    - `RoundsMixin`     (coord/rounds.py)     — timeout thread, eviction, outer step
    - `SchedulingMixin` (coord/scheduling.py) — FLOPs, world_size, tier-aware pacing
    - `HistoryMixin`    (coord/history.py)    — dashboard history persistence
The FastAPI route layer lives in `coord/routes.py`; `create_app` is re-exported
here so `from dllm.coord.server import CoordinatorState, create_app, main`
keeps working (the `dllm-coord` console entry point + the test suite rely on it).
"""
from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import torch
import uvicorn
from fastapi import HTTPException

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

from ..core import PRESETS, ModelConfig, Transformer
from ..core.config import TrainConfig
from ..shared.identity import pubkey_from_hex, verify_delta, verify_deregister
from ..shared.protocol import DeltaAck, RegisterRequest, RegisterResponse, RoundStatus
from ..shared.version import PROTOCOL_VERSION
from ..shared.serialize import (
    DeltaCodec,
    StateCodec,
    deserialize_delta,
    model_state,
    serialize_state,
)
from .history import HistoryMixin
from .persistence import find_latest, load_checkpoint
from .rounds import RoundsMixin
from .routes import create_app
from .scheduling import SchedulingMixin

log = logging.getLogger("dllm.coord")

__all__ = ["CoordinatorState", "create_app", "main"]


class CoordinatorState(RoundsMixin, SchedulingMixin, HistoryMixin):
    """All mutable global state; guarded by `lock`."""

    def __init__(
        self,
        preset_name: str,
        world_size: int,
        train_cfg: TrainConfig,
        device: str = "cpu",
        state_codec: StateCodec = "bf16",
        delta_codec: DeltaCodec = "q8",
        checkpoint_dir: Path | None = None,
        checkpoint_every: int = 10,
        resume: bool = True,
        require_signed_deltas: bool = False,
        round_timeout_seconds: float = 900.0,
        min_workers: int = 1,
        straggler_grace_seconds: float = 0.0,
        straggler_backoff: float = 0.5,
        worker_inactive_timeout_seconds: float = 1800.0,
        enable_timeout_thread: bool = True,
        tier_aware: bool = False,
        target_round_seconds: float = 600.0,
        min_inner_steps: int = 50,
        max_inner_steps: int = 2000,
        retune_threshold: float = 0.20,
        flops_alarm_threshold: float = 5e24,
        max_active_workers: int = 0,
        val_spike_hold_factor: float = 1.25,
        val_spike_max_holds: int = 3,
    ) -> None:
        if preset_name not in PRESETS:
            raise ValueError(f"unknown preset {preset_name!r}; have {list(PRESETS)}")
        self.preset_name = preset_name
        self.cfg: ModelConfig = PRESETS[preset_name]
        self.train_cfg = train_cfg
        # Dynamic world_size (Choice A): `world_size` arg is now the FLOOR.
        # Actual quorum target tracks len(active workers) at round boundaries.
        # min_world_size=1 + zero workers = world_size 1 = round just waits.
        # min_world_size=2 + one worker = world_size 2 = round waits for a
        # second registration before quorum can close (legacy behaviour at
        # `--world-size 2`).
        self.min_world_size = max(1, world_size)
        self.world_size = self.min_world_size
        self.device = torch.device(device)
        self.state_codec: StateCodec = state_codec
        self.delta_codec: DeltaCodec = delta_codec
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_every = checkpoint_every
        self.require_signed_deltas = require_signed_deltas
        # Round-level timeout / min-quorum: if a round has been open >
        # round_timeout_seconds and we have >= min_workers deltas, force the
        # outer step with whoever submitted. Lets the cohort survive worker
        # dropouts without operator intervention. Late deltas hit the
        # resync path on the worker side.
        self.round_timeout_seconds = max(0.0, round_timeout_seconds)
        self.min_workers = max(1, min(min_workers, world_size))
        # Straggler grace: once >= min_workers have submitted (quorum met), wait
        # at most this long for the remaining (slower) workers before forcing
        # the outer step — so a fast worker never idles longer than this for a
        # slow peer. The straggler's late delta hits the worker-side resync
        # path and it contributes opportunistically when it can keep pace.
        # 0 = disabled (only the hard round_timeout force-advances).
        self.straggler_grace_seconds = max(0.0, straggler_grace_seconds)
        # Wall-clock when the current round first reached the min_workers
        # quorum; None until then. Reset each round. Drives the grace timer.
        self.quorum_met_at: float | None = None
        # Adaptive pace (AIMD): when a worker is force-advanced past (straggled),
        # multiply its inner_steps by this factor so it does less next round and
        # can keep pace. Tier-aware re-grows it gradually (capped — see
        # _PACE_GROW) when it keeps pace, so it converges to the most work it can
        # finish per round without holding up the fast cohort.
        self.straggler_backoff = min(1.0, max(0.05, straggler_backoff))
        # Auto-evict registrations inactive longer than this. Same physical
        # GPU coming back from a crash/restart re-registers cleanly with a
        # fresh worker_id; the old ghost is dropped so the dashboard "active
        # workers" table doesn't show stale entries forever and so
        # world_size matches the count of actually-contributing workers.
        # 0 disables auto-eviction.
        self.worker_inactive_timeout_seconds = max(0.0, worker_inactive_timeout_seconds)
        # Tier-aware scheduling (PLAN §3 / CLAUDE.md "tier-aware"). When ON,
        # each worker's `inner_steps` is recomputed from its measured tok/s so
        # all workers finish inner loops in ~target_round_seconds regardless
        # of GPU class. Fast workers do more steps; slow ones do fewer.
        # Without this, a fast worker idles waiting for the slowest, or the
        # slow one gets evicted by the round timeout — both wasteful.
        self.tier_aware = tier_aware
        self.target_round_seconds = max(60.0, target_round_seconds)
        self.min_inner_steps = max(1, min_inner_steps)
        self.max_inner_steps = max(self.min_inner_steps, max_inner_steps)
        # Only re-assign when the new value differs from the current one by
        # more than `retune_threshold` (fractional). Stops dashboard churn
        # from jitter — a worker reporting 3700 → 3680 tok/s won't flip
        # inner_steps every round.
        self.retune_threshold = max(0.0, retune_threshold)
        # EU AI Act systemic-risk threshold is 10²⁵ FLOPs (PLAN §5.3); we
        # alarm at half that so the operator has time to plan a soft landing
        # / CoP Safety chapter prep / AI Office notification window.
        self.flops_alarm_threshold = max(0.0, flops_alarm_threshold)
        # Hard cap on simultaneous active registrations. The 8 GB Netcup VPS
        # OOMs around N=4 for the 300M model: each in-flight delta is ~1.25
        # GB fp32 in self.deltas, and the outer step transiently doubles
        # that during averaging. With the bf16-coord + in-place-average
        # optimizations we get peak RSS ≈ 4.45 + 1.25·N GB; OOM-killer fires
        # near 7 GB on this VPS. 0 disables the cap (no limit; relies on
        # operator to size sensibly for the host).
        self.max_active_workers = max(0, max_active_workers)
        # Headline-val spike guard. The dashboard's `last_val_loss` is the
        # consensus-min across the round's reporters (the full-seq worker
        # tracking consensus has the lowest loss). But a round SOLO-closed by a
        # single worker reading structurally high — a short-seq_len M5, or a
        # fresh worker that resynced right after a coord restart before the
        # full-seq 3060 rejoined — yanks the headline up even though the
        # consensus model didn't regress (the next multi-worker round drops
        # straight back). When a round has < 2 reporters AND the candidate is a
        # jump of more than this factor over the last headline, HOLD the
        # previous headline for the dashboard + history. `mean_val_loss` and the
        # per-worker /workers values always carry the true numbers. Default 1.25
        # (a +25% jump): comfortably above normal round-to-round jitter (~±10%),
        # below the observed +34%..+57% short-seq / fresh-join solo bias (round
        # 525's +34.5% would slip past the old 1.4 default). NOTE: with the
        # worker-side consensus-val fix (worker validates last_ref, not its
        # drifted local θ) this guard is now mostly belt-and-suspenders. Set
        # <= 1.0 (or max-holds 0) to disable. Bounded by val_spike_max_holds so a *genuine*
        # sustained regression is surfaced after that many consecutive holds
        # instead of being masked forever.
        self.val_spike_hold_factor = float(val_spike_hold_factor)
        self.val_spike_max_holds = max(0, int(val_spike_max_holds))
        # Ring buffer of (round, val_loss, flops_total, ts) appended at each
        # outer step. Powers the dashboard chart at GET /.
        # Persisted to <checkpoint_dir>/history.jsonl (append-only NDJSON)
        # so the chart survives coord restarts.
        self.history: deque[dict] = deque(maxlen=500)
        self._history_path: Path | None = (
            self.checkpoint_dir / "history.jsonl" if self.checkpoint_dir else None
        )

        torch.manual_seed(train_cfg.seed)
        self.model = Transformer(self.cfg).to(self.device)
        # Coord stores the consensus model in bf16. Halves model + SGD
        # momentum-buffer memory vs fp32 (1.25 GB → 600 MB each for 300M
        # params), critical on the 8 GB VPS. Workers receive bf16 state
        # over the wire anyway (state_codec defaults to bf16), and the
        # DiLoCo outer step's accuracy is robust to bf16 momentum at
        # outer_lr=0.7 + momentum=0.9 (DiLoCo paper §5).
        # Init in fp32 first so the random init is bit-identical with
        # workers' models (which also init fp32 then autocast to bf16),
        # then convert in-place to bf16 for residency.
        self.model.to(torch.bfloat16)
        self.outer_opt = torch.optim.SGD(
            self.model.parameters(),
            lr=train_cfg.outer_lr,
            momentum=train_cfg.outer_momentum,
            nesterov=True,
        )

        self.lock = threading.Lock()
        self.round = 0
        self.next_worker_id = 0
        self.workers: dict[int, dict[str, Any]] = {}
        self.deltas: dict[int, dict[int, dict[str, torch.Tensor]]] = {0: {}}
        self.val_losses: dict[int, list[float]] = {0: []}
        self.power_watts_per_round: dict[int, list[float]] = {0: []}
        self.tokens_per_sec_per_round: dict[int, list[float]] = {0: []}
        self.last_val_loss: float | None = None  # consensus-tracker (min across workers)
        self.mean_val_loss: float | None = None  # cohort mean (secondary; skewed in heterogeneous cohorts)
        self._val_hold_count = 0  # consecutive solo-spike rounds the headline has been held (see val_spike_*)
        self.last_power_watts: float | None = None  # COHORT sum of last round's workers
        self.last_power_watts_per_worker: float | None = None  # mean for reference
        self.last_n_reporting_workers: int = 0
        self.last_tokens_per_sec: float | None = None  # sum across last round's workers
        self.flops_total: float = 0.0  # cumulative estimate
        self.energy_wh_total: float = 0.0  # cumulative cohort energy
        self.round_started_at = time.time()
        self._state_bytes: bytes | None = None

        self._timeout_stop = threading.Event()
        self._timeout_thread: threading.Thread | None = None
        if enable_timeout_thread and (
            self.round_timeout_seconds > 0 or self.straggler_grace_seconds > 0
        ):
            self._timeout_thread = threading.Thread(
                target=self._timeout_loop,
                daemon=True,
                name="dllm-coord-timeout",
            )
            self._timeout_thread.start()

        if resume and self.checkpoint_dir is not None:
            latest = find_latest(self.checkpoint_dir)
            if latest is not None:
                try:
                    meta = load_checkpoint(latest, self.model, self.outer_opt)
                    self.round = int(meta.get("round", 0)) + 1
                    self.flops_total = float(meta.get("flops_total", 0.0))
                    self.energy_wh_total = float(meta.get("energy_wh_total", 0.0))
                    # Carry the headline across restarts (like flops/energy) so
                    # the dashboard doesn't blank to n/a, AND so the spike guard
                    # has a baseline on the first post-restart round — the exact
                    # window where a lone fresh worker tends to solo-close.
                    _lv = meta.get("last_val_loss")
                    self.last_val_loss = float(_lv) if _lv is not None else None
                    self.deltas = {self.round: {}}
                    self.val_losses = {self.round: []}
                    self.power_watts_per_round = {self.round: []}
                    self.tokens_per_sec_per_round = {self.round: []}
                    log.info("resumed from %s -> starting at round %d", latest, self.round)
                except (KeyError, ValueError, RuntimeError, OSError) as e:
                    log.warning(
                        "checkpoint %s incompatible or unreadable (%s); "
                        "starting fresh at round 0",
                        latest,
                        e,
                    )
            # Always try to populate history, even if no compatible checkpoint
            self._load_history_from_disk()
            self._backfill_history_from_checkpoint_metas()
            # Recover cumulative energy from history if we didn't resume from
            # a checkpoint (or if the checkpoint predates the energy field).
            if self.energy_wh_total == 0.0 and self.history:
                for entry in reversed(self.history):
                    e = entry.get("energy_wh_total")
                    if e:
                        self.energy_wh_total = float(e)
                        log.info(
                            "recovered cumulative energy_wh_total=%.2f from history",
                            self.energy_wh_total,
                        )
                        break

    # -- helpers --------------------------------------------------------------

    def _serialize_state(self) -> bytes:
        if self._state_bytes is None:
            self._state_bytes = serialize_state(model_state(self.model), codec=self.state_codec)
        return self._state_bytes

    def _invalidate_state_cache(self) -> None:
        self._state_bytes = None

    def stop(self) -> None:
        """Signal the timeout thread to exit. Idempotent; thread is daemon anyway."""
        self._timeout_stop.set()

    # -- API used by FastAPI handlers ----------------------------------------

    def register(self, req: RegisterRequest) -> RegisterResponse:
        with self.lock:
            # Version handshake FIRST: a client on incompatible code could send
            # deltas built against different assumptions, so reject before any
            # other check. 426 Upgrade Required is the right code. Older clients
            # send no protocol_version (None) and are turned away the same way.
            if req.protocol_version != PROTOCOL_VERSION:
                got = req.protocol_version or "none"
                log.warning(
                    "[REGISTER] rejected (version mismatch): coord=%s client=%s "
                    "country=%s gpu=%s",
                    PROTOCOL_VERSION, got, req.country, req.gpu,
                )
                raise HTTPException(
                    status_code=426,
                    detail=(
                        f"protocol version mismatch: coord={PROTOCOL_VERSION} "
                        f"client={got}. Update the client (git pull && "
                        "pip install -e .) and relaunch."
                    ),
                )
            if req.preset != self.preset_name:
                raise HTTPException(
                    status_code=400,
                    detail=f"preset mismatch: coord={self.preset_name}, worker={req.preset}",
                )
            # Capacity check (memory-driven on small VPSes). HTTP 429 is the
            # right code: "we appreciate the offer but we're full, try again
            # after a worker leaves". 0 disables. Auto-evict eventually frees
            # slots if a registered worker goes silent for
            # `worker_inactive_timeout_seconds`.
            if (
                self.max_active_workers > 0
                and len(self.workers) >= self.max_active_workers
            ):
                log.warning(
                    "[REGISTER] rejected (cap full): %d/%d active workers; "
                    "country=%s gpu=%s preset=%s",
                    len(self.workers),
                    self.max_active_workers,
                    req.country,
                    req.gpu,
                    req.preset,
                )
                raise HTTPException(
                    status_code=429,
                    detail=(
                        f"cohort full: {len(self.workers)}/{self.max_active_workers} "
                        "active workers. Try again later when a slot frees up "
                        "(workers are auto-evicted after inactivity)."
                    ),
                )
            # parse pubkey eagerly so bad keys fail at registration, not at /delta
            parsed_pubkey = None
            if req.pubkey:
                try:
                    parsed_pubkey = pubkey_from_hex(req.pubkey)
                except (ValueError, TypeError) as e:
                    if self.require_signed_deltas:
                        raise HTTPException(400, f"invalid pubkey: {e}") from None
            elif self.require_signed_deltas:
                raise HTTPException(400, "pubkey required when signed deltas enforced")

            wid = self.next_worker_id
            self.next_worker_id += 1
            # Per-worker inner_steps: starts at the global default and gets
            # retuned by submit_delta() once the worker reports tokens_per_sec.
            #
            # shard_index: assigned by _recompute_world_size_locked() at the
            # next round boundary. Initialized to wid for the worker's first
            # round so the very first inner loop has a sensible shard
            # partition; gets a contiguous remap (so eviction doesn't leave
            # gaps in the shard space) at next outer-step end.
            self.workers[wid] = {
                "pubkey_hex": req.pubkey,
                "pubkey": parsed_pubkey,
                "country": req.country,
                "gpu": req.gpu,
                "vram_gb": req.vram_gb,
                "ram_gb": req.ram_gb,
                "registered_at": time.time(),
                "inner_steps": self.train_cfg.inner_steps,
                "shard_index": wid,
                "shard_world_size": self.world_size,
                "protocol_version": req.protocol_version,
                "pace_factor": 1.0,  # AIMD pace; <1 after straggling, recovers on keep-pace
            }
            log.info(
                "register worker=%d country=%s gpu=%s preset=%s",
                wid,
                req.country,
                req.gpu,
                req.preset,
            )
            return RegisterResponse(
                worker_id=wid,
                current_round=self.round,
                world_size=self.world_size,
                seed=self.train_cfg.seed,
                inner_steps=self.train_cfg.inner_steps,
                seq_len=self.train_cfg.seq_len,
                micro_batch_size=self.train_cfg.micro_batch_size,
                state_codec=self.state_codec,
                delta_codec=self.delta_codec,
                require_signed_deltas=self.require_signed_deltas,
            )

    # -- voluntary deregister ---------------------------------------------

    def deregister(
        self, worker_id: int, ts_unix: int, signature_b64: str
    ) -> dict:
        """Worker voluntarily leaves the cohort (clean Ctrl+C / Colab stop
        / atexit). Signed with the worker's Ed25519 key so peer A can't
        kick peer B off the coord.

        Coord rejects replayed signatures older than 5 min — narrow window
        keeps clock skew tolerable without exposing the endpoint to
        long-term replay.

        Caller does not hold lock.
        """
        with self.lock:
            ws = self.workers.get(worker_id)
            if ws is None:
                # Already gone (auto-evicted, prior deregister, never
                # registered). Idempotent — return success.
                return {"removed": False, "reason": "not registered"}

            # Replay-protection: ts must be within ±300s of now.
            now = int(time.time())
            if abs(now - ts_unix) > 300:
                raise HTTPException(
                    400,
                    f"timestamp drift too large: {ts_unix} vs server {now}",
                )

            pubkey = ws.get("pubkey")
            if self.require_signed_deltas:
                if pubkey is None:
                    raise HTTPException(401, "worker registered without pubkey")
                if not verify_deregister(pubkey, worker_id, ts_unix, signature_b64):
                    raise HTTPException(401, "signature verification failed")

            log.info(
                "[DEREGISTER] worker_id=%d voluntarily left (country=%s gpu=%s)",
                worker_id,
                ws.get("country", "??"),
                ws.get("gpu", "??"),
            )
            del self.workers[worker_id]
            # If the worker had submitted a delta in the current round but the
            # round hasn't closed, drop the delta so quorum reflects who's
            # actually still around.
            self.deltas.get(self.round, {}).pop(worker_id, None)
            # Recompute world_size on the way out so the surviving cohort can
            # close the current round at the new (smaller) quorum.
            self._recompute_world_size_locked()
            # And if that shrink just satisfied quorum for the surviving
            # deltas, close the round now (same deadlock-avoidance as the
            # eviction path).
            self._maybe_close_round_locked()
            return {"removed": True}

    def status(self, heartbeat_worker_id: int | None = None) -> RoundStatus:
        with self.lock:
            # A worker long-polling /status (waiting for the round to advance)
            # passes its id so we refresh last_seen_ts — otherwise a worker
            # that's legitimately blocked waiting for slow peers gets
            # auto-evicted for "inactivity" even though it's alive and just
            # finished an inner loop. /delta is the normal heartbeat; this
            # makes /status one too for the long-poll case.
            if heartbeat_worker_id is not None and heartbeat_worker_id in self.workers:
                self.workers[heartbeat_worker_id]["last_seen_ts"] = time.time()
            n_sub = len(self.deltas.get(self.round, {}))
            return RoundStatus(
                current_round=self.round,
                n_registered=len(self.workers),
                n_submitted=n_sub,
                waiting_for=max(0, self.world_size - n_sub),
                world_size=self.world_size,
                min_world_size=self.min_world_size,
                max_active_workers=self.max_active_workers,
                last_val_loss=self.last_val_loss,
                mean_val_loss=self.mean_val_loss,
                flops_total=self.flops_total,
                flops_alarm_threshold=self.flops_alarm_threshold,
                round_open_seconds=time.time() - self.round_started_at,
                round_timeout_seconds=self.round_timeout_seconds,
                min_workers=self.min_workers,
                target_round_seconds=self.target_round_seconds if self.tier_aware else 0.0,
                tier_aware=self.tier_aware,
                protocol_version=PROTOCOL_VERSION,
                energy_wh_total=self.energy_wh_total,
                last_power_watts=self.last_power_watts,
                last_power_watts_per_worker=self.last_power_watts_per_worker,
                last_n_reporting_workers=self.last_n_reporting_workers,
                last_tokens_per_sec=self.last_tokens_per_sec,
            )

    def list_workers(self) -> list[dict]:
        """Snapshot per-worker stats for the /workers endpoint + dashboard."""
        with self.lock:
            return [
                {
                    "worker_id": wid,
                    "country": w.get("country", "XX"),
                    "gpu": w.get("gpu", "unknown"),
                    "vram_gb": w.get("vram_gb", 0),
                    "ram_gb": w.get("ram_gb", 0),
                    "registered_at": w.get("registered_at", 0.0),
                    "rounds_contributed": w.get("rounds_contributed", 0),
                    "last_seen_ts": w.get("last_seen_ts"),
                    "last_round": w.get("last_round"),
                    "last_val_loss": w.get("last_val_loss"),
                    "last_power_watts": w.get("last_power_watts"),
                    "last_tokens_per_sec": w.get("last_tokens_per_sec"),
                    "inner_steps": w.get("inner_steps"),
                    "shard_index": w.get("shard_index"),
                    "shard_world_size": w.get("shard_world_size"),
                    "protocol_version": w.get("protocol_version"),
                }
                for wid, w in sorted(self.workers.items())
            ]

    def state_blob(self) -> tuple[bytes, int]:
        with self.lock:
            return self._serialize_state(), self.round

    def submit_delta(
        self,
        worker_id: int,
        claimed_round: int,
        blob: bytes,
        val_loss: float | None = None,
        power_watts: float | None = None,
        tokens_per_sec: float | None = None,
        tokens_per_step: int | None = None,
        signature_b64: str | None = None,
    ) -> DeltaAck:
        with self.lock:
            if worker_id not in self.workers:
                raise HTTPException(404, f"unknown worker_id {worker_id}")
            if claimed_round != self.round:
                # Carry the worker's current inner_steps — if it was force-
                # advanced past (a straggler), _penalize_stragglers_locked just
                # cut this value; the worker applies it on its resync so its next
                # loop is shorter and it can keep pace (AIMD).
                return DeltaAck(
                    accepted=False,
                    reason=f"stale: coord round={self.round}, worker={claimed_round}",
                    next_round=self.round,
                    inner_steps=self.workers[worker_id].get("inner_steps"),
                )
            # signature check (binds worker_id + round + sha256(body) to the pubkey
            # registered at /register time, defeating spoof + replay + tamper)
            pubkey = self.workers[worker_id].get("pubkey")
            if self.require_signed_deltas or signature_b64 is not None:
                if signature_b64 is None:
                    return DeltaAck(accepted=False, reason="missing signature")
                if pubkey is None:
                    return DeltaAck(accepted=False, reason="worker registered without pubkey")
                if not verify_delta(pubkey, worker_id, claimed_round, blob, signature_b64):
                    log.warning(
                        "REJECTED bad signature from worker_id=%d round=%d", worker_id, claimed_round
                    )
                    return DeltaAck(accepted=False, reason="signature verification failed")
            delta = deserialize_delta(blob, codec=self.delta_codec)
            self.deltas[self.round][worker_id] = delta
            # Start the straggler-grace timer the moment this round first hits
            # the min_workers quorum — the grace counts from quorum, not round
            # open, so the fast quorum waits a bounded time for slow peers.
            effective_min = max(1, min(self.min_workers, self.world_size))
            if self.quorum_met_at is None and len(self.deltas[self.round]) >= effective_min:
                self.quorum_met_at = time.time()
            if val_loss is not None:
                self.val_losses[self.round].append(val_loss)
            if power_watts is not None and power_watts > 0:
                self.power_watts_per_round.setdefault(self.round, []).append(power_watts)
            if tokens_per_sec is not None and tokens_per_sec > 0:
                self.tokens_per_sec_per_round.setdefault(self.round, []).append(tokens_per_sec)
            # Per-worker stats — power the dashboard's "active workers" table so
            # "registered but not contributing" (the classic M5/slow-worker case)
            # is obvious at a glance instead of requiring a journald grep.
            ws = self.workers[worker_id]
            ws["rounds_contributed"] = ws.get("rounds_contributed", 0) + 1
            ws["last_seen_ts"] = time.time()
            ws["last_round"] = claimed_round
            if val_loss is not None:
                ws["last_val_loss"] = val_loss
            if power_watts is not None and power_watts > 0:
                ws["last_power_watts"] = power_watts
            if tokens_per_sec is not None and tokens_per_sec > 0:
                ws["last_tokens_per_sec"] = tokens_per_sec
            # Worker's real tokens-per-optimizer-step (micro_batch × seq_len).
            # Stored per-worker so tier-aware retune AND FLOPs accounting size
            # against the worker's actual batch, not the coord's config (#67).
            if tokens_per_step is not None and tokens_per_step > 0:
                ws["tokens_per_step"] = int(tokens_per_step)
            log.info(
                "delta round=%d worker=%d (%d/%d)%s%s",
                self.round,
                worker_id,
                len(self.deltas[self.round]),
                self.world_size,
                f" val_loss={val_loss:.4f}" if val_loss is not None else "",
                f" power={power_watts:.0f}W" if power_watts is not None else "",
            )
            # Tier-aware retune: now that we have this worker's measured tok/s,
            # see if a different inner_steps would land closer to
            # target_round_seconds. Only return a new value when the change is
            # material (avoid dashboard churn on noise).
            new_inner = self._maybe_retune_worker(worker_id, tokens_per_sec)
            # Dynamic sharding (Choice B): always send the current shard
            # assignment in the ack so the worker can detect changes and
            # rebuild its ShardLoader on its next round if needed. Cheap
            # (16 bytes JSON); the worker is the one that compares to its
            # own state and only acts on diffs.
            shard_index = ws.get("shard_index")
            shard_world_size = ws.get("shard_world_size")
            ready = len(self.deltas[self.round]) >= self.world_size
            if ready:
                self._outer_step_locked()
                return DeltaAck(
                    accepted=True,
                    next_round=self.round,
                    inner_steps=new_inner,
                    shard_index=shard_index,
                    shard_world_size=shard_world_size,
                )
            return DeltaAck(
                accepted=True,
                inner_steps=new_inner,
                shard_index=shard_index,
                shard_world_size=shard_world_size,
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="smoke", choices=list(PRESETS))
    ap.add_argument("--world-size", type=int, default=2)
    ap.add_argument("--inner-steps", type=int, default=50)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--micro-batch-size", type=int, default=16)
    ap.add_argument("--inner-lr", type=float, default=3e-4)
    ap.add_argument("--outer-lr", type=float, default=0.7)
    ap.add_argument("--outer-momentum", type=float, default=0.9)
    ap.add_argument("--max-rounds", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--state-codec", default="bf16", choices=["fp32", "bf16"])
    ap.add_argument("--delta-codec", default="q8", choices=["fp32", "q8"])
    ap.add_argument("--checkpoint-dir", default="coord/state")
    ap.add_argument("--checkpoint-every", type=int, default=10)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument(
        "--require-signed-deltas",
        action="store_true",
        help="Reject /delta submissions without a valid Ed25519 signature (recommended)",
    )
    ap.add_argument(
        "--round-timeout-seconds",
        type=float,
        default=900.0,
        help="Max wall-clock for a round; if exceeded with >= min-workers deltas, force-advance.",
    )
    ap.add_argument(
        "--min-workers",
        type=int,
        default=1,
        help="Minimum deltas needed for a timed-out round to advance (clamped to [1, world_size]).",
    )
    ap.add_argument(
        "--straggler-grace-seconds",
        type=float,
        default=0.0,
        help=(
            "Once >= min-workers have submitted, wait at most this long for the "
            "remaining (slower) workers before force-advancing the round. Caps how "
            "long a fast worker idles for a straggler; the straggler's late delta "
            "resyncs and it contributes opportunistically. 0 = disabled (only the "
            "hard --round-timeout-seconds force-advances)."
        ),
    )
    ap.add_argument(
        "--straggler-backoff",
        type=float,
        default=0.5,
        help=(
            "AIMD multiplicative-decrease factor: a worker force-advanced past "
            "(straggler) has its inner_steps multiplied by this so it does less "
            "next round and can keep pace. Tier-aware re-grows it gradually when "
            "it keeps up. 1.0 = disabled (no backoff)."
        ),
    )
    ap.add_argument(
        "--worker-inactive-timeout-seconds",
        type=float,
        default=1800.0,
        help=(
            "Auto-evict worker registrations inactive for longer than this. "
            "Prevents ghost workers (same GPU re-registering after crashes) "
            "from piling up in the dashboard's active-workers table. 0 disables."
        ),
    )
    ap.add_argument(
        "--tier-aware",
        action="store_true",
        help=(
            "Coord-side tier-aware scheduling: assign each worker a "
            "per-worker inner_steps so every worker finishes its inner loop "
            "in ~--target-round-seconds regardless of GPU class. Lets a fast "
            "3060 and a slow M5 share the same round cadence without eviction. "
            "See PLAN §3."
        ),
    )
    ap.add_argument(
        "--target-round-seconds",
        type=float,
        default=600.0,
        help="Per-worker wall clock target for the inner loop when --tier-aware is on.",
    )
    ap.add_argument(
        "--min-inner-steps",
        type=int,
        default=50,
        help="Clamp lower bound for tier-aware inner_steps (DiLoCo paper guidance).",
    )
    ap.add_argument(
        "--max-inner-steps",
        type=int,
        default=2000,
        help="Clamp upper bound for tier-aware inner_steps (drift control).",
    )
    ap.add_argument(
        "--retune-threshold",
        type=float,
        default=0.20,
        help=(
            "Tier-aware retune fires only when |Δ inner_steps| / current "
            "exceeds this fraction. Avoids dashboard churn from tok/s noise."
        ),
    )
    ap.add_argument(
        "--flops-alarm-threshold",
        type=float,
        default=5e24,
        help=(
            "Cumulative FLOPs at which /status surfaces an alarm flag. "
            "Default 5×10²⁴ — half of the EU AI Act systemic-risk threshold "
            "(10²⁵), buys planning time for CoP Safety chapter prep + "
            "AI Office notification. 0 disables."
        ),
    )
    ap.add_argument(
        "--max-active-workers",
        type=int,
        default=0,
        help=(
            "Hard cap on simultaneous active registrations. /register "
            "returns HTTP 429 when at cap. Memory-driven on small VPSes — "
            "for the 300M model on 8 GB RAM, practical safe cap is ~4 "
            "(each fp32 delta is ~1.25 GB). 0 = uncapped."
        ),
    )
    ap.add_argument(
        "--val-spike-hold-factor",
        type=float,
        default=1.25,
        help=(
            "Headline-val spike guard: when a round is SOLO-closed (< 2 "
            "reporters) and its consensus-min val jumps more than this factor "
            "over the last headline, hold the previous headline for the "
            "dashboard/history (a short-seq or fresh-join worker solo-closing "
            "shouldn't read as a model regression). mean_val_loss + per-worker "
            "values stay truthful. <= 1.0 disables."
        ),
    )
    ap.add_argument(
        "--val-spike-max-holds",
        type=int,
        default=3,
        help=(
            "Max consecutive rounds the headline-val spike guard will hold "
            "before accepting the high value as real, so a genuine sustained "
            "regression is never masked forever. 0 disables the guard."
        ),
    )
    ap.add_argument("--log-level", default="info")
    args = ap.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    train_cfg = TrainConfig(
        seq_len=args.seq_len,
        micro_batch_size=args.micro_batch_size,
        inner_steps=args.inner_steps,
        inner_lr=args.inner_lr,
        outer_lr=args.outer_lr,
        outer_momentum=args.outer_momentum,
        max_outer_rounds=args.max_rounds,
        seed=args.seed,
    )

    ckpt_dir = Path(args.checkpoint_dir).resolve() if args.checkpoint_dir else None

    state = CoordinatorState(
        preset_name=args.preset,
        world_size=args.world_size,
        train_cfg=train_cfg,
        device=args.device,
        state_codec=args.state_codec,
        delta_codec=args.delta_codec,
        checkpoint_dir=ckpt_dir,
        checkpoint_every=args.checkpoint_every,
        resume=not args.no_resume,
        require_signed_deltas=args.require_signed_deltas,
        round_timeout_seconds=args.round_timeout_seconds,
        min_workers=args.min_workers,
        straggler_grace_seconds=args.straggler_grace_seconds,
        straggler_backoff=args.straggler_backoff,
        worker_inactive_timeout_seconds=args.worker_inactive_timeout_seconds,
        tier_aware=args.tier_aware,
        target_round_seconds=args.target_round_seconds,
        min_inner_steps=args.min_inner_steps,
        max_inner_steps=args.max_inner_steps,
        retune_threshold=args.retune_threshold,
        flops_alarm_threshold=args.flops_alarm_threshold,
        max_active_workers=args.max_active_workers,
        val_spike_hold_factor=args.val_spike_hold_factor,
        val_spike_max_holds=args.val_spike_max_holds,
    )
    app = create_app(state)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
