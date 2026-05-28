"""Round-timeout + min-quorum eviction.

When a worker stops contributing, the surviving worker(s) used to block forever.
With the timeout layer, the coord force-advances after `round_timeout_seconds`
with whoever submitted (at least `min_workers`). Late deltas hit the worker
resync path and rejoin cleanly.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from dllm.coord.server import CoordinatorState, create_app
from dllm.core.config import TrainConfig
from dllm.shared.identity import (
    load_or_create_identity,
    pubkey_hex,
    sign_delta,
)
from dllm.shared.protocol import RegisterRequest
from dllm.shared.version import PROTOCOL_VERSION
from dllm.shared.serialize import (
    compute_delta,
    snapshot,
    state_to_bytes,
)


@pytest.fixture
def cfg() -> TrainConfig:
    return TrainConfig(
        seq_len=32, micro_batch_size=4, inner_steps=3, max_outer_rounds=2, seed=0
    )


def _make_state(cfg: TrainConfig, **kwargs) -> CoordinatorState:
    """Coord with codecs that the tests can drive directly + timeout thread off."""
    return CoordinatorState(
        preset_name="smoke",
        world_size=kwargs.pop("world_size", 2),
        train_cfg=cfg,
        device="cpu",
        state_codec="fp32",
        delta_codec="fp32",
        checkpoint_dir=None,
        enable_timeout_thread=False,
        **kwargs,
    )


def _submit_one_delta(state: CoordinatorState, tmp_path: Path) -> None:
    """Register one worker and submit one valid delta for the current round."""
    client = TestClient(create_app(state))
    sk = load_or_create_identity(tmp_path / "id.key")
    client.post(
        "/register",
        json=RegisterRequest(pubkey=pubkey_hex(sk), preset="smoke", protocol_version=PROTOCOL_VERSION).model_dump(),
    )
    snap = snapshot(state.model)
    with torch.no_grad():
        for p in state.model.parameters():
            p.add_(torch.randn_like(p) * 0.001)
    delta = compute_delta(snap, state.model)
    with torch.no_grad():
        for n, p in state.model.named_parameters():
            p.copy_(snap[n])
    blob = state_to_bytes(delta)
    sig = sign_delta(sk, 0, state.round, blob)
    client.post(
        "/delta",
        params={"worker_id": 0, "round": state.round},
        content=blob,
        headers={"content-type": "application/octet-stream", "x-delta-signature": sig},
    )


def test_timeout_advances_with_partial_quorum(cfg: TrainConfig, tmp_path: Path) -> None:
    """ws=2 + only 1 delta + timeout exceeded + min_workers=1 → outer step fires."""
    state = _make_state(cfg, world_size=2, round_timeout_seconds=10.0, min_workers=1)
    _submit_one_delta(state, tmp_path)
    assert state.round == 0
    # simulate "round has been open for 100s" without waiting
    state.round_started_at = time.time() - 100.0
    advanced = state._check_and_force_advance()
    assert advanced is True
    assert state.round == 1


def test_timeout_does_not_advance_when_below_min_workers(
    cfg: TrainConfig, tmp_path: Path
) -> None:
    state = _make_state(cfg, world_size=2, round_timeout_seconds=10.0, min_workers=2)
    _submit_one_delta(state, tmp_path)  # only 1 delta
    state.round_started_at = time.time() - 100.0
    advanced = state._check_and_force_advance()
    assert advanced is False
    assert state.round == 0


def test_timeout_does_not_advance_before_timeout(
    cfg: TrainConfig, tmp_path: Path
) -> None:
    state = _make_state(cfg, world_size=2, round_timeout_seconds=300.0, min_workers=1)
    _submit_one_delta(state, tmp_path)
    # round_started_at still ~now → not yet expired
    advanced = state._check_and_force_advance()
    assert advanced is False
    assert state.round == 0


def test_timeout_disabled_when_seconds_is_zero(
    cfg: TrainConfig, tmp_path: Path
) -> None:
    state = _make_state(cfg, world_size=2, round_timeout_seconds=0.0, min_workers=1)
    _submit_one_delta(state, tmp_path)
    state.round_started_at = time.time() - 999999.0
    advanced = state._check_and_force_advance()
    assert advanced is False
    assert state.round == 0


def test_timeout_skips_when_full_quorum_already_present(
    cfg: TrainConfig, tmp_path: Path
) -> None:
    """If world_size deltas are in, the normal /delta path advances, not timeout."""
    state = _make_state(cfg, world_size=1, round_timeout_seconds=10.0, min_workers=1)
    _submit_one_delta(state, tmp_path)
    # ws=1 + 1 delta → the regular submit_delta path already advanced; timeout is no-op
    assert state.round == 1


def test_straggler_grace_advances_after_quorum(cfg: TrainConfig, tmp_path: Path) -> None:
    """ws=2, only 1 delta (quorum met, min_workers=1), grace exceeded → force-advance
    WITHOUT waiting the full round_timeout. The fast quorum doesn't idle for the
    slow peer."""
    state = _make_state(
        cfg, world_size=2, round_timeout_seconds=99999.0, min_workers=1,
        straggler_grace_seconds=10.0,
    )
    _submit_one_delta(state, tmp_path)
    assert state.round == 0
    assert state.quorum_met_at is not None  # quorum timer started
    # Simulate "grace elapsed since quorum" without the hard timeout firing.
    state.quorum_met_at = time.time() - 100.0
    assert state._check_and_force_advance() is True
    assert state.round == 1


def test_straggler_grace_waits_within_grace(cfg: TrainConfig, tmp_path: Path) -> None:
    """Within the grace window, the round stays open (gives the slow peer a chance)."""
    state = _make_state(
        cfg, world_size=2, round_timeout_seconds=99999.0, min_workers=1,
        straggler_grace_seconds=300.0,
    )
    _submit_one_delta(state, tmp_path)
    # quorum_met_at ~ now → only a moment elapsed, well under the 300s grace
    assert state._check_and_force_advance() is False
    assert state.round == 0


def test_straggler_grace_disabled_by_default(cfg: TrainConfig, tmp_path: Path) -> None:
    """grace=0 (default) → no straggler force-advance even long after quorum;
    only the hard round_timeout would advance."""
    state = _make_state(
        cfg, world_size=2, round_timeout_seconds=99999.0, min_workers=1,
        straggler_grace_seconds=0.0,
    )
    _submit_one_delta(state, tmp_path)
    state.quorum_met_at = time.time() - 100000.0  # ancient, but grace disabled
    assert state._check_and_force_advance() is False
    assert state.round == 0


def test_quorum_met_at_tracks_and_resets(cfg: TrainConfig, tmp_path: Path) -> None:
    """quorum_met_at is set when the round first hits min_workers, and reset to
    None when the round advances (so the next round's grace timer starts fresh)."""
    # ws=1: one delta meets quorum AND fills world_size → outer step fires →
    # quorum_met_at must be reset to None for round 1.
    state = _make_state(
        cfg, world_size=1, round_timeout_seconds=99999.0, min_workers=1,
        straggler_grace_seconds=10.0,
    )
    _submit_one_delta(state, tmp_path)
    assert state.round == 1                # advanced (ws=1)
    assert state.quorum_met_at is None     # timer reset for the new round


def _build_blob(state: CoordinatorState) -> bytes:
    snap = snapshot(state.model)
    with torch.no_grad():
        for p in state.model.parameters():
            p.add_(torch.randn_like(p) * 0.001)
    delta = compute_delta(snap, state.model)
    with torch.no_grad():
        for n, p in state.model.named_parameters():
            p.copy_(snap[n])
    return state_to_bytes(delta)


def test_straggler_backoff_cuts_non_submitter_and_delivers_on_stale(
    cfg: TrainConfig, tmp_path: Path
) -> None:
    """AIMD multiplicative-decrease: when the round force-advances past a slow
    worker, the coord halves its inner_steps + pace_factor, and the reduced
    inner_steps rides back on its stale /delta so its next loop is shorter."""
    state = _make_state(
        cfg, world_size=2, round_timeout_seconds=99999.0, min_workers=1,
        straggler_grace_seconds=5.0, straggler_backoff=0.5,
        tier_aware=True, min_inner_steps=50, max_inner_steps=2000,
    )
    client = TestClient(create_app(state))
    sk_a = load_or_create_identity(tmp_path / "a.key")
    sk_b = load_or_create_identity(tmp_path / "b.key")
    client.post("/register", json=RegisterRequest(pubkey=pubkey_hex(sk_a), preset="smoke", protocol_version=PROTOCOL_VERSION).model_dump())
    client.post("/register", json=RegisterRequest(pubkey=pubkey_hex(sk_b), preset="smoke", protocol_version=PROTOCOL_VERSION).model_dump())
    state.workers[1]["inner_steps"] = 100  # B is the straggler with a big loop

    blob = _build_blob(state)
    sig_a = sign_delta(sk_a, 0, 0, blob)
    client.post("/delta", params={"worker_id": 0, "round": 0}, content=blob,
                headers={"content-type": "application/octet-stream", "x-delta-signature": sig_a})
    assert state.round == 0

    state.quorum_met_at = time.time() - 100.0  # grace elapsed → force-advance past B
    assert state._check_and_force_advance() is True
    assert state.round == 1
    assert state.workers[1]["inner_steps"] == 50          # 100 × 0.5
    assert state.workers[1]["pace_factor"] == pytest.approx(0.5)
    assert state.workers[0]["pace_factor"] == pytest.approx(1.0)  # A kept pace

    # B's stale round-0 delta is rejected but carries the reduced inner_steps.
    sig_b = sign_delta(sk_b, 1, 0, blob)
    ack = client.post("/delta", params={"worker_id": 1, "round": 0}, content=blob,
                      headers={"content-type": "application/octet-stream", "x-delta-signature": sig_b}).json()
    assert ack["accepted"] is False and "stale" in ack["reason"]
    assert ack["inner_steps"] == 50  # delivered so B shrinks its next loop


def test_pace_factor_recovers_for_contributors(cfg: TrainConfig, tmp_path: Path) -> None:
    """AIMD additive-increase: a worker that previously straggled (pace_factor<1)
    nudges back toward 1.0 each round it keeps pace, reclaiming its full loops."""
    state = _make_state(
        cfg, world_size=1, round_timeout_seconds=99999.0, min_workers=1, tier_aware=True,
    )
    client = TestClient(create_app(state))
    sk = load_or_create_identity(tmp_path / "id.key")
    client.post("/register", json=RegisterRequest(pubkey=pubkey_hex(sk), preset="smoke", protocol_version=PROTOCOL_VERSION).model_dump())
    state.workers[0]["pace_factor"] = 0.4  # had straggled before
    blob = _build_blob(state)
    sig = sign_delta(sk, 0, 0, blob)
    client.post("/delta", params={"worker_id": 0, "round": 0}, content=blob,
                headers={"content-type": "application/octet-stream", "x-delta-signature": sig})
    assert state.round == 1  # ws=1 → submit closed the round
    assert state.workers[0]["pace_factor"] == pytest.approx(0.5)  # 0.4 × 1.25 recovery


def test_min_workers_clamped_to_world_size(cfg: TrainConfig) -> None:
    state = _make_state(cfg, world_size=2, round_timeout_seconds=10.0, min_workers=100)
    assert state.min_workers == 2  # clamped down
    state = _make_state(cfg, world_size=2, round_timeout_seconds=10.0, min_workers=0)
    assert state.min_workers == 1  # clamped up


def test_status_exposes_timeout_fields(cfg: TrainConfig) -> None:
    state = _make_state(cfg, world_size=2, round_timeout_seconds=42.0, min_workers=1)
    client = TestClient(create_app(state))
    body = client.get("/status").json()
    assert body["round_timeout_seconds"] == 42.0
    assert body["min_workers"] == 1
    assert "round_open_seconds" in body
    assert body["round_open_seconds"] >= 0.0


def test_late_delta_after_eviction_is_stale_rejected(
    cfg: TrainConfig, tmp_path: Path
) -> None:
    """The eviction story: worker A submits, B is too slow, coord force-advances on
    timeout, B finally submits its old round → resync path on the worker side.
    """
    state = _make_state(cfg, world_size=2, round_timeout_seconds=5.0, min_workers=1)
    client = TestClient(create_app(state))

    # A registers, submits round 0
    sk_a = load_or_create_identity(tmp_path / "a.key")
    sk_b = load_or_create_identity(tmp_path / "b.key")
    client.post("/register", json=RegisterRequest(pubkey=pubkey_hex(sk_a), preset="smoke", protocol_version=PROTOCOL_VERSION).model_dump())
    client.post("/register", json=RegisterRequest(pubkey=pubkey_hex(sk_b), preset="smoke", protocol_version=PROTOCOL_VERSION).model_dump())

    snap = snapshot(state.model)
    with torch.no_grad():
        for p in state.model.parameters():
            p.add_(torch.randn_like(p) * 0.001)
    delta = compute_delta(snap, state.model)
    with torch.no_grad():
        for n, p in state.model.named_parameters():
            p.copy_(snap[n])
    blob = state_to_bytes(delta)

    sig_a = sign_delta(sk_a, 0, 0, blob)
    client.post(
        "/delta",
        params={"worker_id": 0, "round": 0},
        content=blob,
        headers={"content-type": "application/octet-stream", "x-delta-signature": sig_a},
    )
    assert state.round == 0  # still waiting on B

    # B is too slow → force-advance with just A
    state.round_started_at = time.time() - 100.0
    state._check_and_force_advance()
    assert state.round == 1

    # B finally submits its stale round 0 delta
    sig_b = sign_delta(sk_b, 1, 0, blob)
    rej = client.post(
        "/delta",
        params={"worker_id": 1, "round": 0},
        content=blob,
        headers={"content-type": "application/octet-stream", "x-delta-signature": sig_b},
    ).json()
    assert rej["accepted"] is False
    assert "stale" in rej["reason"]
    assert rej["next_round"] == 1  # tells B where to resync to
