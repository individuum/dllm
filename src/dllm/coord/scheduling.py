"""Scheduling mixin for the coordinator: FLOPs estimation, dynamic world_size
auto-scaling (Choice A), per-worker tier-aware retune, and AIMD straggler pacing.

Method bodies are moved verbatim from the original `server.py`; `self.*`
attributes resolve because the final `CoordinatorState` defines them in
`__init__`.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .server import CoordinatorState

log = logging.getLogger("dllm.coord")

# AIMD recovery: each round a worker keeps pace (submits), its pace_factor is
# nudged back toward 1.0 by this factor — so a worker that straggled once but can
# now keep up gradually reclaims its full inner_steps instead of being penalized
# forever. Paired with the multiplicative cut in _penalize_stragglers_locked.
_PACE_RECOVER = 1.25


class SchedulingMixin:
    """FLOPs accounting, world_size auto-scaling, and tier-aware/AIMD pacing."""

    def _estimate_round_flops(self: CoordinatorState, contributors: list[int] | None = None) -> float:
        """FLOPs estimate: 6 * N_params * sum(per-worker tokens this round).

        6 = 2 (forward) + 4 (backward) per param per token. Coarse but tracks
        the right order of magnitude — enough for AI-Act-threshold monitoring.

        With tier-aware scheduling, workers do different `inner_steps` so we
        sum each worker's actual contribution instead of multiplying by a
        single world_size×inner_steps figure. Workers without a tracked
        inner_steps fall back to the global config default.

        Caller passes the list of worker_ids that submitted in the current
        round (captured BEFORE deltas are cleared in _outer_step_locked).
        When None, falls back to the legacy world_size × default-steps
        estimate (used by tests / startup states).

        Caller holds self.lock.
        """
        n_params = float(self.model.num_params(non_embedding=False))
        # Default per-step token count from the coord's own config; only used
        # for workers that predate the worker-reported tokens_per_step field.
        default_toks_per_step = self.train_cfg.seq_len * self.train_cfg.micro_batch_size
        if contributors:
            toks_per_round = 0
            for wid in contributors:
                w = self.workers.get(wid, {})
                steps = int(w.get("inner_steps", self.train_cfg.inner_steps))
                # Use the worker's ACTUAL tokens/step when reported (#67): the
                # coord's --micro-batch-size can differ from the worker's, so
                # using it here mis-counts the AI Act FLOPs total by that ratio.
                tps_step = int(w.get("tokens_per_step") or default_toks_per_step)
                toks_per_round += tps_step * steps
        else:
            # Fall back to the legacy world_size × default-steps estimate.
            toks_per_round = (
                default_toks_per_step * self.train_cfg.inner_steps * self.world_size
            )
        return 6.0 * n_params * toks_per_round

    # -- dynamic world_size (Choice A: coord-only auto-scaling) ---------------

    def _recompute_world_size_locked(self: CoordinatorState) -> int:
        """Recompute self.world_size from the count of active registrations.

        Floor: self.min_world_size (set at startup). Eviction may briefly
        bring the active count below the floor; the floor wins so a freshly
        booted coord with zero workers still has world_size=1 (matches the
        legacy single-worker behaviour at `--world-size 1`).

        Also reassigns per-worker shard_index to a contiguous 0..N-1 mapping
        (Choice B). Workers learn the new assignment via DeltaAck on their
        next /delta and rebuild their ShardLoader accordingly. The
        reassignment ALWAYS runs (even when world_size is unchanged) because
        a worker leaving still leaves a hole in the index space that the
        survivors should compact away.

        Call ONLY at safe boundaries:
            - end of `_outer_step_locked` (between rounds)
            - end of `_evict_stale_workers` (shrinks are safe mid-round)

        NEVER call from `register()` — bumping the quorum target mid-round
        when a new worker has zero in-flight deltas would stall the round
        until that worker also finished an inner loop.

        Returns the new world_size (== self.world_size). Caller holds lock.
        """
        new_size = max(self.min_world_size, len(self.workers))
        size_changed = new_size != self.world_size
        if size_changed:
            old = self.world_size
            self.world_size = new_size
            log.info(
                "[WORLD_SIZE] %d -> %d (auto-scaled, active workers=%d, floor=%d)",
                old,
                new_size,
                len(self.workers),
                self.min_world_size,
            )
        # Compact shard_indices to 0..N-1 regardless of whether world_size
        # changed. Sorted by worker_id so the assignment is deterministic
        # across coord restarts (and across worker reads of /workers).
        active_wids = sorted(self.workers.keys())
        for idx, wid in enumerate(active_wids):
            self.workers[wid]["shard_index"] = idx
            self.workers[wid]["shard_world_size"] = self.world_size
        return new_size

    @property
    def _straggler_floor(self: CoordinatorState) -> int:
        """Lowest inner_steps an adaptive-paced straggler can be cut to. Below
        min_inner_steps (the normal sync-efficiency floor) on purpose: the whole
        point is to let a very slow device do *some* useful work per round rather
        than block the cohort. Derived from min_inner_steps so one knob scales both."""
        return max(1, self.min_inner_steps // 8)

    def _penalize_stragglers_locked(self: CoordinatorState) -> None:
        """AIMD multiplicative-decrease. Halve (×straggler_backoff) the
        inner_steps of every active worker that did NOT submit the round we're
        about to force-advance past, so next round it does less and can land
        inside the grace window. Floored at _straggler_floor. Caller holds lock
        and calls this immediately BEFORE _outer_step_locked on the force path."""
        if self.straggler_backoff >= 1.0:
            return
        submitted = set(self.deltas.get(self.round, {}).keys())
        floor = self._straggler_floor
        for wid, w in self.workers.items():
            if wid in submitted:
                continue
            # Cut the persistent pace_factor (so tier-aware's next retune sizes
            # this worker down and won't snap it back up) AND cut inner_steps now
            # (so the value rides back on the stale ack for an immediate effect).
            w["pace_factor"] = max(0.05, float(w.get("pace_factor", 1.0)) * self.straggler_backoff)
            cur = int(w.get("inner_steps", self.train_cfg.inner_steps))
            new = max(floor, int(cur * self.straggler_backoff))
            if new < cur:
                w["inner_steps"] = new
                log.info(
                    "[PACE] straggler worker=%d inner_steps %d -> %d pace_factor=%.2f "
                    "(force-advanced past round %d)",
                    wid, cur, new, w["pace_factor"], self.round,
                )

    def _maybe_retune_worker(
        self: CoordinatorState, worker_id: int, tokens_per_sec: float | None
    ) -> int | None:
        """Compute a new per-worker inner_steps if tier_aware is on AND the
        reported throughput suggests a materially different value than the
        worker is currently using. Mutates self.workers[worker_id]["inner_steps"]
        and returns the new value when a retune fired; None otherwise.

        Caller holds self.lock.
        """
        if not self.tier_aware or tokens_per_sec is None or tokens_per_sec <= 0:
            return None
        ws = self.workers[worker_id]
        # Size against the WORKER's real tokens-per-step (#67). The coord's
        # --micro-batch-size is set for its own CPU averaging (e.g. 2) and can
        # differ from a worker running --micro-batch-size 1 for VRAM; using the
        # coord's value here silently halved every worker's inner_steps so
        # rounds finished in ~half target_round_seconds. submit_delta stored
        # the worker's value just above; fall back to coord config for workers
        # that don't report it (backward compatible).
        toks_per_step = int(
            ws.get("tokens_per_step")
            or (self.train_cfg.seq_len * self.train_cfg.micro_batch_size)
        )
        target_tokens = tokens_per_sec * self.target_round_seconds
        # Scale the raw tok/s estimate by this worker's pace_factor (AIMD): 1.0
        # for a worker keeping pace (→ unchanged tier-aware behavior); cut on each
        # straggle and recovered gradually when it keeps up, so a chronically-slow
        # worker settles at the largest inner_steps it can finish per round
        # without holding up the cohort. Floor lets it drop below the normal
        # min_inner_steps sync-efficiency floor when it genuinely must.
        pace = float(ws.get("pace_factor", 1.0))
        ideal = int(round(target_tokens / max(1, toks_per_step) * pace))
        proposed = max(self._straggler_floor, min(self.max_inner_steps, ideal))
        current = int(ws.get("inner_steps", self.train_cfg.inner_steps))
        if current <= 0:
            current = self.train_cfg.inner_steps
        # Only retune when |Δ| / current > retune_threshold.
        rel_change = abs(proposed - current) / max(1, current)
        if rel_change < self.retune_threshold:
            return None
        ws["inner_steps"] = proposed
        log.info(
            "[TIER-AWARE] worker=%d retune inner_steps %d -> %d "
            "(tok/s=%.0f, target=%.0fs, |Δ|/cur=%.2f)",
            worker_id,
            current,
            proposed,
            tokens_per_sec,
            self.target_round_seconds,
            rel_change,
        )
        return proposed
