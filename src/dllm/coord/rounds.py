"""Round-lifecycle mixin for the coordinator: the background timeout thread,
stale-worker eviction, quorum/force-advance logic, and the Nesterov outer step.

Method bodies are moved verbatim from `server.py`. Shared module-level helpers
they depend on (`_PACE_RECOVER`, `save_checkpoint`, `average_deltas`) are
imported here; `_flat_norm` lives here because `_outer_step_locked` is its only
caller.
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import torch

from ..shared.serialize import average_deltas
from .persistence import save_checkpoint
from .scheduling import _PACE_RECOVER

if TYPE_CHECKING:
    from .server import CoordinatorState

log = logging.getLogger("dllm.coord")


class RoundsMixin:
    """Timeout thread, eviction, quorum checks, and the outer optimizer step."""

    # -- timeout-based round eviction ----------------------------------------

    def _timeout_loop(self: CoordinatorState) -> None:
        """Background thread: poll for timed-out rounds and stale workers."""
        while not self._timeout_stop.wait(timeout=2.0):
            try:
                self._check_and_force_advance()
            except Exception:  # noqa: BLE001
                log.exception("timeout-thread error")
            try:
                self._evict_stale_workers()
            except Exception:  # noqa: BLE001
                log.exception("evict-thread error")

    def _evict_stale_workers(self: CoordinatorState) -> int:
        """Drop registrations whose last_seen_ts is older than the inactive
        timeout. A worker that's never submitted a delta uses registered_at
        as its anchor — so an initial-pull worker has the full timeout to
        get its first delta in.

        Returns the count evicted (mostly for tests). Caller acquires `lock`
        below if needed; we acquire it here too so the call is safe from any
        thread.
        """
        if self.worker_inactive_timeout_seconds <= 0:
            return 0
        with self.lock:
            now = time.time()
            cutoff = now - self.worker_inactive_timeout_seconds
            to_evict: list[int] = []
            for wid, w in self.workers.items():
                last_seen = w.get("last_seen_ts")
                anchor = last_seen if last_seen is not None else w.get("registered_at", 0.0)
                if anchor < cutoff:
                    to_evict.append(wid)
            for wid in to_evict:
                last_seen = self.workers[wid].get("last_seen_ts")
                log.warning(
                    "[EVICT] worker_id=%d inactive >%.0fs (last_seen=%s); "
                    "dropping registration. If the worker comes back it'll "
                    "register fresh with a new id.",
                    wid,
                    self.worker_inactive_timeout_seconds,
                    "never" if last_seen is None else f"{now - last_seen:.0f}s ago",
                )
                del self.workers[wid]
                # Clear any unconsumed delta in the current round so the
                # outer step's quorum check reflects the surviving cohort.
                self.deltas.get(self.round, {}).pop(wid, None)
            if to_evict:
                # Dynamic world_size: shrinking mid-round is safe and
                # desired — lets the remaining workers close the round at
                # the lower quorum instead of waiting on a dead peer until
                # `round_timeout_seconds` fires.
                self._recompute_world_size_locked()
                # CRITICAL: if the shrink made the surviving deltas meet
                # quorum, fire the outer step NOW. The submit_delta path
                # already passed (it saw the larger world_size and didn't
                # trigger), and _check_and_force_advance bails when
                # submitted >= world_size — so without this, the round
                # deadlocks: a lone worker's delta sits forever while it
                # long-polls /status for an advance that never comes.
                self._maybe_close_round_locked()
            return len(to_evict)

    def _maybe_close_round_locked(self: CoordinatorState) -> bool:
        """Fire the outer step if the current round's delta count now meets
        the world_size quorum. Used after eviction/deregister shrinks
        world_size to <= the already-submitted count — the normal
        submit_delta trigger already passed at the larger world_size, so
        nothing else would ever close the round. Caller holds lock.
        Returns True if an outer step fired.
        """
        submitted = len(self.deltas.get(self.round, {}))
        if submitted >= 1 and submitted >= self.world_size:
            log.info(
                "[QUORUM] cohort shrank to world_size=%d with %d delta(s) in "
                "round %d — closing round now (would otherwise deadlock).",
                self.world_size,
                submitted,
                self.round,
            )
            self._outer_step_locked()
            return True
        return False

    def _check_and_force_advance(self: CoordinatorState) -> bool:
        """Idempotent. Force the outer step with whoever submitted when either:
          (1) straggler grace: quorum (>= min_workers) was met and we've since
              waited straggler_grace_seconds for the rest, OR
          (2) hard timeout: the round has been open > round_timeout_seconds.
        In both cases we require >= effective_min deltas and < world_size (the
        regular submit path closes a full round). Returns True on advance.
        """
        if self.round_timeout_seconds <= 0 and self.straggler_grace_seconds <= 0:
            return False
        with self.lock:
            now = time.time()
            elapsed = now - self.round_started_at
            submitted = len(self.deltas.get(self.round, {}))
            # min_workers is clamped DYNAMICALLY against current world_size.
            # Without this clamp, a cohort that shrank below the original
            # min_workers floor (e.g. configured for 5, now down to 2) would
            # never force-advance even though all surviving peers submitted.
            effective_min = max(1, min(self.min_workers, self.world_size))
            if submitted < effective_min:
                return False  # too few deltas — keep waiting (e.g. all workers offline)
            if submitted >= self.world_size:
                return False  # the regular path will handle this on the submitting thread

            # (1) straggler grace — caps how long the fast quorum waits for slow peers.
            if (
                self.straggler_grace_seconds > 0
                and self.quorum_met_at is not None
                and (now - self.quorum_met_at) >= self.straggler_grace_seconds
            ):
                log.warning(
                    "[STRAGGLER] forcing outer step at round=%d with %d/%d deltas "
                    "%.1fs after quorum (grace=%.1fs) — slow peers will resync",
                    self.round, submitted, self.world_size,
                    now - self.quorum_met_at, self.straggler_grace_seconds,
                )
                self._penalize_stragglers_locked()
                self._outer_step_locked()
                return True

            # (2) hard round timeout (legacy backstop).
            if self.round_timeout_seconds > 0 and elapsed >= self.round_timeout_seconds:
                log.warning(
                    "[TIMEOUT] forcing outer step at round=%d with %d/%d deltas after "
                    "%.1fs (timeout=%.1fs, effective_min=%d)",
                    self.round, submitted, self.world_size, elapsed,
                    self.round_timeout_seconds, effective_min,
                )
                self._penalize_stragglers_locked()
                self._outer_step_locked()
                return True
            return False

    # -- outer optimizer step (caller holds lock) -----------------------------

    def _outer_step_locked(self: CoordinatorState) -> None:
        # Free the cached serialized state blob (~600 MB at bf16 for 300M)
        # before we do the memory-hungry delta averaging. The next /state
        # call after this outer step would invalidate it anyway; doing it
        # eagerly buys us ~600 MB of headroom during the peak.
        self._invalidate_state_cache()
        # Memory note: at this point self.deltas[self.round] holds N×P fp32
        # tensors (~1.25 GB per delta for the 300M model). After we extract
        # the average, we don't need the per-worker copies — so we drop them
        # AFTER averaging but BEFORE the optimizer step to keep peak memory
        # off the OOM-killer's radar on small VPSes.
        round_deltas = list(self.deltas[self.round].values())
        n_deltas_in_round = len(round_deltas)  # captured before we free the list
        # Capture worker IDs BEFORE we clear the deltas dict — used by the
        # tier-aware FLOPs accounting below to sum each contributor's actual
        # inner_steps (which can differ across the cohort under tier-aware).
        contributors = list(self.deltas[self.round].keys())
        # AIMD recovery: every worker that submitted this round kept pace, so
        # nudge its pace_factor back toward 1.0 — a worker that straggled once
        # but can now keep up gradually reclaims its full inner_steps.
        for wid in contributors:
            w = self.workers.get(wid)
            if w is not None:
                w["pace_factor"] = min(1.0, float(w.get("pace_factor", 1.0)) * _PACE_RECOVER)
        avg = average_deltas(round_deltas)
        # Free the per-worker delta tensors immediately; the averaged copy
        # in `avg` is the only thing the outer step still needs.
        del round_deltas
        self.deltas[self.round].clear()

        self.outer_opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                if n not in avg:
                    raise KeyError(f"missing delta for parameter {n!r}")
                # If model dtype matches avg dtype, .to() returns the same
                # tensor (no copy). Otherwise it allocates once.
                p.grad = avg[n].to(p.device, dtype=p.dtype)
        # Capture the flat-norm summary before we free `avg` — the log line
        # below needs it for debugging.
        avg_norm = _flat_norm(avg)
        # avg is no longer needed — the gradients are attached to p.grad
        # which the optimizer will consume. Drop the dict to free the fp32
        # buffers before SGD's momentum-buffer update fires.
        avg.clear()
        del avg
        self.outer_opt.step()

        # bookkeeping. Headline val = the CONSENSUS-tracking worker's loss =
        # min across the round's per-worker reports. The MEAN is misleading in
        # a heterogeneous cohort: workers validate at DIFFERENT seq_len (a
        # shorter context reads structurally higher) and a fresh joiner reports
        # its drifted local θ — both inflate the mean far above the consensus
        # model's true val (e.g. 3.9 consensus + 6.8 fresh M5 → 5.3 mean, the
        # "loss=5!" false alarm). The worker validating at full seq_len and
        # tracking consensus has the lowest loss, so min is the truthful single
        # number. Per-worker values stay on /workers; the mean is kept as a
        # secondary field for transparency.
        # NOTE: self.round is still the round being closed here (the increment
        # happens below), so val_losses[self.round] == this round's reports.
        self._update_headline_val_locked(self.val_losses[self.round])
        self.flops_total += self._estimate_round_flops(contributors)

        # Cohort power + throughput aggregation. Both are SUMS because workers
        # train simultaneously — cohort instantaneous draw is the sum of each
        # worker's reported wattage, and cohort throughput is the sum of each
        # worker's tokens/sec. (The mean is also kept for the dashboard's
        # "per-worker avg" line.)
        round_powers = self.power_watts_per_round.get(self.round, [])
        round_tok_s = self.tokens_per_sec_per_round.get(self.round, [])
        n_reporting = len(round_powers)
        cohort_watts = sum(round_powers) if round_powers else None
        mean_power = (cohort_watts / n_reporting) if cohort_watts is not None else None
        total_tok_s = sum(round_tok_s) if round_tok_s else None

        # Energy = power * time. Cohort sum × actual round duration is the
        # right integral: all workers were drawing in parallel for that span.
        round_seconds = max(0.0, time.time() - self.round_started_at)
        if cohort_watts is not None and round_seconds > 0:
            self.energy_wh_total += cohort_watts * (round_seconds / 3600.0)

        # `last_power_watts` semantic = cohort sum (matches the dashboard's
        # "POWER DRAW (COHORT)" tile label). The per-worker mean is exposed
        # separately on /status so the dashboard can also show "x W avg per
        # worker" as a tooltip / sub-line.
        self.last_power_watts = cohort_watts
        self.last_power_watts_per_worker = mean_power
        self.last_tokens_per_sec = total_tok_s
        self.last_n_reporting_workers = n_reporting

        prev_round = self.round
        self.round += 1
        self.deltas[self.round] = {}
        self.val_losses[self.round] = []
        self.power_watts_per_round[self.round] = []
        self.tokens_per_sec_per_round[self.round] = []
        del self.deltas[prev_round]
        del self.val_losses[prev_round]
        self.power_watts_per_round.pop(prev_round, None)
        self.tokens_per_sec_per_round.pop(prev_round, None)
        self.round_started_at = time.time()
        self.quorum_met_at = None  # reset the straggler-grace timer for the new round
        self._invalidate_state_cache()
        # Dynamic world_size: pick up any registrations that arrived during
        # the round just closed (we deliberately didn't bump mid-round to
        # avoid stalling the quorum target).
        self._recompute_world_size_locked()

        # record for the dashboard's chart (persisted to disk)
        self._append_history(
            {
                "round": prev_round,
                "val_loss": self.last_val_loss,
                "flops_total": self.flops_total,
                "ts": time.time(),
                "round_seconds": round_seconds,
                # Cohort sum (matches the "POWER DRAW (COHORT)" tile).
                "power_watts": cohort_watts,
                "power_watts_per_worker": mean_power,
                "tokens_per_sec": total_tok_s,
                "energy_wh_total": self.energy_wh_total,
                "n_workers": n_deltas_in_round,
                "n_reporting_workers": n_reporting,
            }
        )

        log.info(
            "outer step %d -> %d (avg delta norm: %.4f, FLOPs ~%.2e, last_val_loss=%s%s%s)",
            prev_round,
            self.round,
            avg_norm,
            self.flops_total,
            f"{self.last_val_loss:.4f}" if self.last_val_loss is not None else "n/a",
            f", power={mean_power:.0f}W" if mean_power is not None else "",
            f", energy_total={self.energy_wh_total:.1f}Wh" if self.energy_wh_total > 0 else "",
        )

        if (
            self.checkpoint_dir is not None
            and self.checkpoint_every > 0
            and self.round % self.checkpoint_every == 0
        ):
            save_checkpoint(
                self.checkpoint_dir,
                round_no=self.round - 1,
                model=self.model,
                outer_opt=self.outer_opt,
                meta={
                    "preset_name": self.preset_name,
                    "world_size": self.world_size,
                    "flops_total": self.flops_total,
                    "last_val_loss": self.last_val_loss,
                    "energy_wh_total": self.energy_wh_total,
                },
            )

    def _update_headline_val_locked(self: CoordinatorState, round_vals: list[float]) -> None:
        """Set the dashboard headline `last_val_loss` (consensus-min) and the
        secondary `mean_val_loss` from a round's per-worker val reports.

        Headline = min(round_vals): the worker validating at full seq_len and
        tracking consensus reads lowest, so the min is the truthful single
        number for the consensus model's quality (the mean is skewed high in a
        heterogeneous cohort — see _outer_step_locked's note).

        Spike guard: when a round is SOLO-closed (< 2 reporters) AND the min
        jumps more than `val_spike_hold_factor` over the last headline, hold the
        previous headline instead — a lone worker reading structurally high (a
        short-seq_len M5; a fresh worker that resynced right after a coord
        restart before the full-seq peer rejoined) shouldn't register as a model
        regression, since the next multi-worker round drops straight back.
        Bounded by `val_spike_max_holds` consecutive holds so a *genuine*
        sustained regression is surfaced rather than masked forever. The held
        value also flows into history (the caller appends self.last_val_loss),
        so the chart is smoothed too. `mean_val_loss` and the per-worker
        /workers values are ALWAYS updated to the true numbers.

        Idempotent w.r.t. an empty list (no reports → leave everything as-is).
        Caller holds self.lock.
        """
        if not round_vals:
            return
        candidate = min(round_vals)
        self.mean_val_loss = sum(round_vals) / len(round_vals)
        prev = self.last_val_loss
        guard_on = self.val_spike_hold_factor > 1.0 and self.val_spike_max_holds > 0
        solo_spike = (
            guard_on
            and prev is not None
            and len(round_vals) < 2
            and candidate > prev * self.val_spike_hold_factor
        )
        if solo_spike and self._val_hold_count < self.val_spike_max_holds:
            self._val_hold_count += 1
            log.info(
                "[VAL] holding headline at %.4f (round %d solo-closed, candidate=%.4f "
                "= +%.0f%%; hold %d/%d) — mean_val_loss=%.4f carries the true value",
                prev, self.round, candidate, (candidate / prev - 1.0) * 100.0,
                self._val_hold_count, self.val_spike_max_holds, self.mean_val_loss,
            )
            # leave self.last_val_loss unchanged (held)
        else:
            self.last_val_loss = candidate
            self._val_hold_count = 0


def _flat_norm(state: dict[str, torch.Tensor]) -> float:
    total = 0.0
    for t in state.values():
        total += float(t.float().pow(2).sum().item())
    return total**0.5
