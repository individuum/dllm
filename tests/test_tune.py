from __future__ import annotations

from dllm.client.worker import compute_inner_steps_for_target


def test_target_matches_simple_arithmetic() -> None:
    # 10k tok/s, 90s target, batch=4, seq=512 = 2048 toks/step
    # → 10000 * 90 / 2048 = 439.45 → 439
    n = compute_inner_steps_for_target(10_000.0, 90.0, batch=4, seq=512)
    assert n == 439


def test_fast_gpu_gets_more_steps_at_same_target() -> None:
    """The whole point: a 3060 (~16k tok/s) does ~1.5x the inner work an M5 does."""
    fast = compute_inner_steps_for_target(16_000.0, 90.0, batch=4, seq=512)
    slow = compute_inner_steps_for_target(10_000.0, 90.0, batch=4, seq=512)
    assert fast > slow
    # ratio should roughly match the speed ratio
    assert 1.4 < fast / slow < 1.7


def test_floor_clamps_tiny_gpus() -> None:
    """Below the floor, sync overhead dominates inner work — keep a minimum."""
    n = compute_inner_steps_for_target(100.0, 90.0, batch=4, seq=512, min_steps=50)
    assert n == 50


def test_ceiling_clamps_giant_targets() -> None:
    n = compute_inner_steps_for_target(
        1_000_000.0, 3600.0, batch=4, seq=512, max_steps=2000
    )
    assert n == 2000


def test_zero_throughput_returns_floor() -> None:
    n = compute_inner_steps_for_target(0.0, 90.0, batch=4, seq=512, min_steps=50)
    assert n == 50


def test_heterogeneous_fleet_synchronizes() -> None:
    """
    With auto-tune, slow and fast workers should both pick inner_steps such
    that their inner loop wall-clock ≈ target_seconds. Verify by inverting:
    given the chosen inner_steps, the expected wall clock is target ± rounding.
    """
    target = 90.0
    batch, seq = 4, 512
    for tok_per_s in (3_000.0, 8_000.0, 15_000.0, 80_000.0):
        n = compute_inner_steps_for_target(tok_per_s, target, batch, seq)
        # n is clamped to [50, 2000]; only check synchronization for un-clamped cases
        if 50 < n < 2000:
            expected_wall = n * batch * seq / tok_per_s
            assert abs(expected_wall - target) < 1.0, (
                f"tok/s={tok_per_s} → n={n} → wall={expected_wall:.1f}s (target {target})"
            )
