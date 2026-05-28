"""Tests for the cosine inner-LR schedule (cosine_lr_for_round).

The schedule is a pure function of the global coord round number, so the
whole cohort computes the same LR regardless of GPU speed or join time.
"""
from __future__ import annotations

import math

import pytest

from dllm.client.worker import cosine_lr_for_round

PEAK = 3e-4
MIN = 3e-5
WARMUP = 20
DECAY = 1000


def _lr(round_no: int) -> float:
    return cosine_lr_for_round(
        round_no, peak_lr=PEAK, min_lr=MIN, warmup_rounds=WARMUP, decay_rounds=DECAY
    )


def test_warmup_ramps_linearly_from_low_to_peak() -> None:
    # round 0 → 1/20 of peak; round 19 → 20/20 = peak.
    assert _lr(0) == pytest.approx(PEAK * 1 / 20)
    assert _lr(9) == pytest.approx(PEAK * 10 / 20)
    assert _lr(19) == pytest.approx(PEAK)


def test_peak_right_after_warmup() -> None:
    # At the warmup boundary the cosine starts at progress=0 → full peak.
    assert _lr(WARMUP) == pytest.approx(PEAK, rel=1e-6)


def test_midpoint_is_halfway_in_cosine() -> None:
    # Halfway through the decay span, cosine = 0.5 → lr = min + 0.5*(peak-min).
    mid_round = WARMUP + (DECAY - WARMUP) // 2
    expected = MIN + (PEAK - MIN) * 0.5
    assert _lr(mid_round) == pytest.approx(expected, rel=1e-2)


def test_floor_at_and_after_decay_horizon() -> None:
    assert _lr(DECAY) == pytest.approx(MIN)
    # Past the horizon stays flat at the floor (not 0) so DiLoCo keeps
    # training if contributors stay online.
    assert _lr(DECAY + 500) == pytest.approx(MIN)
    assert _lr(DECAY + 100000) == pytest.approx(MIN)


def test_monotonic_decrease_through_decay() -> None:
    # After warmup, LR must be non-increasing every round.
    prev = _lr(WARMUP)
    for r in range(WARMUP + 1, DECAY + 1, 10):
        cur = _lr(r)
        assert cur <= prev + 1e-12, f"LR increased at round {r}: {prev} -> {cur}"
        prev = cur


def test_lr_always_within_bounds() -> None:
    # During warmup the LR ramps from ~0 up to peak, so it can be BELOW the
    # cosine floor (min_lr) early — that's expected. The floor only bounds
    # the decay tail. Universal bound: 0 < lr <= peak.
    for r in [0, 1, 19, 20, 300, 500, 999, 1000, 5000]:
        lr = _lr(r)
        assert 0 < lr <= PEAK + 1e-9, f"lr {lr} out of bounds at round {r}"
    # Post-warmup, LR never drops below the floor.
    for r in [20, 300, 999, 1000, 5000]:
        assert _lr(r) >= MIN - 1e-9, f"lr below floor at round {r}"


def test_no_warmup_starts_at_peak() -> None:
    # warmup_rounds=0 → round 0 is already in the cosine at progress 0 = peak.
    lr0 = cosine_lr_for_round(
        0, peak_lr=PEAK, min_lr=MIN, warmup_rounds=0, decay_rounds=DECAY
    )
    assert lr0 == pytest.approx(PEAK, rel=1e-6)


def test_cohort_consistency_same_round_same_lr() -> None:
    """Two workers at the same coord round get identical LR — the whole
    point of keying on round number rather than per-worker step count."""
    for r in [5, 100, 450, 999]:
        a = _lr(r)
        b = _lr(r)
        assert a == b


def test_resuming_midrun_picks_correct_lr() -> None:
    """A worker joining at round 320 (our actual situation) gets the LR the
    schedule prescribes for 320 — partway down the cosine, NOT a fresh
    warmup."""
    lr320 = _lr(320)
    # 320 is well past warmup (20) and ~30% into the [20, 1000] decay span,
    # so LR should be high-but-below-peak.
    progress = (320 - WARMUP) / (DECAY - WARMUP)
    expected = MIN + (PEAK - MIN) * 0.5 * (1 + math.cos(math.pi * progress))
    assert lr320 == pytest.approx(expected, rel=1e-6)
    assert MIN < lr320 < PEAK
