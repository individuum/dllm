"""FastAPI integration tests via TestClient — no network, no subprocess."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from dllm.coord.server import CoordinatorState, create_app
from dllm.core import PRESETS
from dllm.core.config import TrainConfig
from dllm.shared.protocol import RegisterRequest
from dllm.shared.serialize import (
    average_deltas,
    bytes_to_state,
    compute_delta,
    snapshot,
)


@pytest.fixture
def state(tmp_path: Path) -> CoordinatorState:
    cfg = TrainConfig(
        seq_len=32, micro_batch_size=4, inner_steps=3, max_outer_rounds=2, seed=0
    )
    # Use fp32 codecs so tests can use raw safetensors helpers; q8 path covered separately.
    return CoordinatorState(
        preset_name="smoke",
        world_size=2,
        train_cfg=cfg,
        device="cpu",
        state_codec="fp32",
        delta_codec="fp32",
        checkpoint_dir=None,
    )


@pytest.fixture
def q8_state(tmp_path: Path) -> CoordinatorState:
    """Coord configured with bf16 state + q8 delta — the production default."""
    cfg = TrainConfig(
        seq_len=32, micro_batch_size=4, inner_steps=3, max_outer_rounds=2, seed=0
    )
    return CoordinatorState(
        preset_name="smoke",
        world_size=2,
        train_cfg=cfg,
        device="cpu",
        state_codec="bf16",
        delta_codec="q8",
        checkpoint_dir=None,
    )


@pytest.fixture
def client(state: CoordinatorState) -> TestClient:
    return TestClient(create_app(state))


def test_status_initial(client: TestClient) -> None:
    r = client.get("/status")
    assert r.status_code == 200
    body = r.json()
    assert body["current_round"] == 0
    assert body["n_registered"] == 0
    assert body["n_submitted"] == 0
    assert body["waiting_for"] == 2  # world_size


def test_register_returns_worker_id_and_config(client: TestClient) -> None:
    req = RegisterRequest(pubkey="w0", preset="smoke")
    r = client.post("/register", json=req.model_dump())
    assert r.status_code == 200
    body = r.json()
    assert body["worker_id"] == 0
    assert body["current_round"] == 0
    assert body["world_size"] == 2
    assert "inner_steps" in body and body["inner_steps"] > 0


def test_register_rejects_preset_mismatch(client: TestClient) -> None:
    req = RegisterRequest(pubkey="w0", preset="124M")
    r = client.post("/register", json=req.model_dump())
    assert r.status_code == 400


def test_state_endpoint_returns_safetensors(client: TestClient) -> None:
    r = client.get("/state")
    assert r.status_code == 200
    assert r.headers.get("x-round") == "0"
    state = bytes_to_state(r.content)
    assert len(state) > 0
    # at least the embedding should be there
    assert any("tok_emb" in k for k in state)


def test_delta_rejects_unknown_worker(client: TestClient) -> None:
    r = client.post("/delta", params={"worker_id": 999, "round": 0}, content=b"")
    assert r.status_code == 404


def test_delta_rejects_stale_round(client: TestClient, state: CoordinatorState) -> None:
    # register one worker so the id is valid
    client.post("/register", json=RegisterRequest(pubkey="w0", preset="smoke").model_dump())
    # submit a fake (empty) blob at the wrong round — should be rejected without parsing
    r = client.post(
        "/delta",
        params={"worker_id": 0, "round": 999},
        content=b"",
        headers={"content-type": "application/octet-stream"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["accepted"] is False
    assert "stale" in body["reason"]


def test_full_round_advances_state(client: TestClient, state: CoordinatorState) -> None:
    """End-to-end: 2 registers, 2 valid deltas, round advances, state changes."""
    # both workers register
    for i in range(2):
        rr = client.post(
            "/register",
            json=RegisterRequest(pubkey=f"w{i}", preset="smoke").model_dump(),
        )
        assert rr.status_code == 200

    # snapshot pre-step state
    r0 = client.get("/state")
    assert int(r0.headers["x-round"]) == 0
    pre = bytes_to_state(r0.content)

    # build a valid pseudo-grad (any non-zero direction works)
    snap = snapshot(state.model)
    with torch.no_grad():
        for p in state.model.parameters():
            p.add_(torch.randn_like(p) * 0.01)
    delta = compute_delta(snap, state.model)
    # restore model so the coord's master θ matches what we just snapshotted
    with torch.no_grad():
        for n, p in state.model.named_parameters():
            p.copy_(snap[n])

    from dllm.shared.serialize import state_to_bytes

    blob = state_to_bytes(delta)

    # worker 0 submits
    ack0 = client.post(
        "/delta",
        params={"worker_id": 0, "round": 0},
        content=blob,
        headers={"content-type": "application/octet-stream"},
    )
    assert ack0.json()["accepted"] is True
    assert ack0.json()["next_round"] is None  # still waiting for w1

    # worker 1 submits, triggers outer step
    ack1 = client.post(
        "/delta",
        params={"worker_id": 1, "round": 0},
        content=blob,
        headers={"content-type": "application/octet-stream"},
    )
    assert ack1.json()["accepted"] is True
    assert ack1.json()["next_round"] == 1

    # state should now reflect the outer step
    r1 = client.get("/state")
    assert int(r1.headers["x-round"]) == 1
    post = bytes_to_state(r1.content)

    # at least one parameter should differ
    any_changed = any(not torch.equal(pre[k], post[k]) for k in pre)
    assert any_changed, "outer step did not move the model"


def test_q8_codec_advertised_in_register(q8_state: CoordinatorState) -> None:
    """A coord using production codecs advertises them on register."""
    client = TestClient(create_app(q8_state))
    r = client.post(
        "/register",
        json=RegisterRequest(pubkey="w0", preset="smoke").model_dump(),
    )
    assert r.status_code == 200
    body = r.json()
    assert body["state_codec"] == "bf16"
    assert body["delta_codec"] == "q8"


def test_q8_delta_flow_advances_round(q8_state: CoordinatorState) -> None:
    """Workers send q8-packed deltas; coord dequantizes and advances."""
    from dllm.shared.serialize import serialize_delta

    client = TestClient(create_app(q8_state))
    for i in range(2):
        client.post(
            "/register",
            json=RegisterRequest(pubkey=f"w{i}", preset="smoke").model_dump(),
        )

    snap = snapshot(q8_state.model)
    with torch.no_grad():
        for p in q8_state.model.parameters():
            p.add_(torch.randn_like(p) * 0.01)
    delta = compute_delta(snap, q8_state.model)
    with torch.no_grad():
        for n, p in q8_state.model.named_parameters():
            p.copy_(snap[n])

    blob = serialize_delta(delta, codec="q8")

    for wid in (0, 1):
        ack = client.post(
            "/delta",
            params={"worker_id": wid, "round": 0, "val_loss": 2.5},
            content=blob,
            headers={"content-type": "application/octet-stream"},
        )
        assert ack.json()["accepted"] is True

    s = client.get("/status").json()
    assert s["current_round"] == 1
    assert s["last_val_loss"] is not None
    assert s["flops_total"] > 0


def test_status_reports_flops_and_val(state: CoordinatorState) -> None:
    """Status fields exposed for AI Act + convergence monitoring."""
    client = TestClient(create_app(state))
    body = client.get("/status").json()
    assert "flops_total" in body
    assert "last_val_loss" in body
    assert body["flops_total"] == 0.0  # nothing trained yet


# ---------------------------------------------------------------------------
# signed-delta enforcement (Phase 1 Byzantine-prep)
# ---------------------------------------------------------------------------


@pytest.fixture
def signed_state(tmp_path: Path) -> CoordinatorState:
    cfg = TrainConfig(
        seq_len=32, micro_batch_size=4, inner_steps=3, max_outer_rounds=2, seed=0
    )
    return CoordinatorState(
        preset_name="smoke",
        world_size=1,
        train_cfg=cfg,
        device="cpu",
        state_codec="fp32",
        delta_codec="fp32",
        checkpoint_dir=None,
        require_signed_deltas=True,
    )


def _make_dummy_delta(state: CoordinatorState):
    """Build a small valid delta payload from the state's model."""
    snap = snapshot(state.model)
    with torch.no_grad():
        for p in state.model.parameters():
            p.add_(torch.randn_like(p) * 0.001)
    delta = compute_delta(snap, state.model)
    with torch.no_grad():
        for n, p in state.model.named_parameters():
            p.copy_(snap[n])
    from dllm.shared.serialize import state_to_bytes

    return state_to_bytes(delta)


def test_signed_required_register_rejects_bad_pubkey(signed_state: CoordinatorState) -> None:
    client = TestClient(create_app(signed_state))
    r = client.post(
        "/register",
        json=RegisterRequest(pubkey="not-hex!!", preset="smoke").model_dump(),
    )
    assert r.status_code == 400


def test_signed_required_rejects_unsigned_delta(signed_state: CoordinatorState) -> None:
    from dllm.shared.identity import load_or_create_identity, pubkey_hex

    client = TestClient(create_app(signed_state))
    sk = load_or_create_identity(Path("tests_tmp_key_a"))
    try:
        client.post(
            "/register",
            json=RegisterRequest(pubkey=pubkey_hex(sk), preset="smoke").model_dump(),
        )
        body = _make_dummy_delta(signed_state)
        ack = client.post(
            "/delta",
            params={"worker_id": 0, "round": 0},
            content=body,
            headers={"content-type": "application/octet-stream"},
        )
        assert ack.json()["accepted"] is False
        assert "signature" in ack.json()["reason"].lower()
    finally:
        Path("tests_tmp_key_a").unlink(missing_ok=True)


def test_signed_required_accepts_valid_signature(signed_state: CoordinatorState) -> None:
    from dllm.shared.identity import load_or_create_identity, pubkey_hex, sign_delta

    client = TestClient(create_app(signed_state))
    sk = load_or_create_identity(Path("tests_tmp_key_b"))
    try:
        client.post(
            "/register",
            json=RegisterRequest(pubkey=pubkey_hex(sk), preset="smoke").model_dump(),
        )
        body = _make_dummy_delta(signed_state)
        sig = sign_delta(sk, worker_id=0, round_no=0, body=body)
        ack = client.post(
            "/delta",
            params={"worker_id": 0, "round": 0},
            content=body,
            headers={
                "content-type": "application/octet-stream",
                "x-delta-signature": sig,
            },
        )
        assert ack.json()["accepted"] is True
        assert ack.json()["next_round"] == 1
    finally:
        Path("tests_tmp_key_b").unlink(missing_ok=True)


def test_worker_resync_on_stale_round_rejection(tmp_path: Path) -> None:
    """A slow worker whose delta arrives stale should resync to the latest
    consensus instead of bailing — the M5/3060 heterogeneous-fleet scenario.
    """
    from pathlib import Path as P

    from dllm.client.worker import Worker, pick_device
    from dllm.shared.identity import load_or_create_identity, pubkey_hex, sign_delta
    from dllm.shared.serialize import (
        compute_delta,
        serialize_delta,
        snapshot,
        state_to_bytes,
    )

    # ws=1 coord so a single test client can drive a round advance directly
    cfg = TrainConfig(
        seq_len=32, micro_batch_size=4, inner_steps=3, max_outer_rounds=2, seed=0
    )
    coord = CoordinatorState(
        preset_name="smoke",
        world_size=1,
        train_cfg=cfg,
        device="cpu",
        state_codec="fp32",
        delta_codec="fp32",
        checkpoint_dir=None,
    )
    client = TestClient(create_app(coord))

    sk = load_or_create_identity(tmp_path / "id.key")

    # register and pull initial state
    rr = client.post(
        "/register",
        json=RegisterRequest(pubkey=pubkey_hex(sk), preset="smoke").model_dump(),
    )
    assert rr.status_code == 200
    worker_id = rr.json()["worker_id"]
    initial_round = rr.json()["current_round"]
    assert initial_round == 0

    # advance the coord directly to simulate other (fast) workers having moved on
    for r in range(3):
        snap = snapshot(coord.model)
        with torch.no_grad():
            for p in coord.model.parameters():
                p.add_(torch.randn_like(p) * 0.001)
        delta = compute_delta(snap, coord.model)
        with torch.no_grad():
            for n, p in coord.model.named_parameters():
                p.copy_(snap[n])
        blob = state_to_bytes(delta)
        sig = sign_delta(sk, worker_id, r, blob)
        ack = client.post(
            "/delta",
            params={"worker_id": worker_id, "round": r},
            content=blob,
            headers={
                "content-type": "application/octet-stream",
                "x-delta-signature": sig,
            },
        ).json()
        assert ack["accepted"]

    assert coord.round == 3

    # Now simulate the slow worker submitting at the OLD round 0
    stale_blob = state_to_bytes({
        n: torch.zeros_like(p) for n, p in coord.model.named_parameters()
    })
    stale_sig = sign_delta(sk, worker_id, 0, stale_blob)
    rej = client.post(
        "/delta",
        params={"worker_id": worker_id, "round": 0},
        content=stale_blob,
        headers={
            "content-type": "application/octet-stream",
            "x-delta-signature": stale_sig,
        },
    ).json()
    assert rej["accepted"] is False
    assert rej["next_round"] == 3
    assert "stale" in rej["reason"]

    # GET /state should give us the round-3 state — the resync path
    sr = client.get("/state")
    assert sr.status_code == 200
    assert int(sr.headers["x-round"]) == 3


def test_signed_required_rejects_other_workers_signature(signed_state: CoordinatorState) -> None:
    """Worker A registers, worker B signs A's delta with B's key — must reject."""
    from dllm.shared.identity import load_or_create_identity, pubkey_hex, sign_delta

    client = TestClient(create_app(signed_state))
    sk_a = load_or_create_identity(Path("tests_tmp_key_c"))
    sk_b = load_or_create_identity(Path("tests_tmp_key_d"))
    try:
        # A registers as worker_id=0
        client.post(
            "/register",
            json=RegisterRequest(pubkey=pubkey_hex(sk_a), preset="smoke").model_dump(),
        )
        body = _make_dummy_delta(signed_state)
        # B signs claiming to be A — signature won't verify against A's pubkey
        sig_b = sign_delta(sk_b, worker_id=0, round_no=0, body=body)
        ack = client.post(
            "/delta",
            params={"worker_id": 0, "round": 0},
            content=body,
            headers={
                "content-type": "application/octet-stream",
                "x-delta-signature": sig_b,
            },
        )
        assert ack.json()["accepted"] is False
        assert "signature" in ack.json()["reason"].lower()
    finally:
        Path("tests_tmp_key_c").unlink(missing_ok=True)
        Path("tests_tmp_key_d").unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Tier-aware scheduling (per-worker inner_steps)
# ---------------------------------------------------------------------------


@pytest.fixture
def tier_state(tmp_path: Path) -> CoordinatorState:
    """Two-worker coord with tier-aware scheduling on."""
    cfg = TrainConfig(
        seq_len=32,
        micro_batch_size=4,
        inner_steps=100,
        max_outer_rounds=2,
        seed=0,
    )
    return CoordinatorState(
        preset_name="smoke",
        world_size=2,
        train_cfg=cfg,
        device="cpu",
        state_codec="fp32",
        delta_codec="fp32",
        checkpoint_dir=None,
        tier_aware=True,
        target_round_seconds=300.0,
        min_inner_steps=10,
        max_inner_steps=2000,
        retune_threshold=0.10,
    )


def _register_two_workers(client: TestClient) -> None:
    for i in range(2):
        rr = client.post(
            "/register",
            json=RegisterRequest(pubkey=f"w{i}", preset="smoke").model_dump(),
        )
        assert rr.status_code == 200


def _build_dummy_blob(state: CoordinatorState) -> bytes:
    """Tiny non-zero delta in the coord's expected codec (fp32 here)."""
    from dllm.shared.serialize import state_to_bytes

    snap = snapshot(state.model)
    with torch.no_grad():
        for p in state.model.parameters():
            p.add_(torch.randn_like(p) * 0.001)
    delta = compute_delta(snap, state.model)
    with torch.no_grad():
        for n, p in state.model.named_parameters():
            p.copy_(snap[n])
    return state_to_bytes(delta)


def test_tier_aware_assigns_per_worker_inner_steps(tier_state: CoordinatorState) -> None:
    """Two workers reporting very different tok/s should end up with very
    different inner_steps — fast does more, slow does less, both finish in
    ~target_round_seconds.
    """
    client = TestClient(create_app(tier_state))
    _register_two_workers(client)
    blob = _build_dummy_blob(tier_state)
    # Worker 0 reports 4000 tok/s (fast 3060 class).
    # Worker 1 reports 400 tok/s (slow M5 class). 10x ratio.
    ack0 = client.post(
        "/delta",
        params={
            "worker_id": 0,
            "round": 0,
            "val_loss": 5.0,
            "tokens_per_sec": 4000.0,
        },
        content=blob,
        headers={"content-type": "application/octet-stream"},
    ).json()
    ack1 = client.post(
        "/delta",
        params={
            "worker_id": 1,
            "round": 0,
            "val_loss": 5.0,
            "tokens_per_sec": 400.0,
        },
        content=blob,
        headers={"content-type": "application/octet-stream"},
    ).json()

    # Both acks should carry retuned inner_steps because they differ from
    # the default 100 by more than the 10% threshold.
    assert ack0["inner_steps"] is not None
    assert ack1["inner_steps"] is not None
    # Fast worker should do strictly MORE steps than slow worker.
    assert ack0["inner_steps"] > ack1["inner_steps"]

    # Sanity-check arithmetic: at target=300s, seq=32, batch=4,
    #   target_tokens(fast) = 4000 * 300 = 1.2M, /128 = 9375 steps clamped 2000
    #   target_tokens(slow) = 400  * 300 = 120k, /128 = 937.5 -> 938 (banker's)
    assert ack0["inner_steps"] == 2000  # hit max clamp
    assert ack1["inner_steps"] == 938

    # /workers should report the per-worker assignment too
    workers = client.get("/workers").json()["workers"]
    by_id = {w["worker_id"]: w for w in workers}
    assert by_id[0]["inner_steps"] == 2000
    assert by_id[1]["inner_steps"] == 938


def test_tier_aware_off_returns_no_inner_steps(state: CoordinatorState) -> None:
    """When tier_aware is off (default), ack carries no inner_steps update —
    workers keep using whatever they got at register time. Backward compat.
    """
    client = TestClient(create_app(state))
    _register_two_workers(client)
    blob = _build_dummy_blob(state)
    ack = client.post(
        "/delta",
        params={
            "worker_id": 0,
            "round": 0,
            "tokens_per_sec": 4000.0,
        },
        content=blob,
        headers={"content-type": "application/octet-stream"},
    ).json()
    assert ack["accepted"] is True
    assert ack["inner_steps"] is None


def test_tier_aware_retune_skips_small_changes(tier_state: CoordinatorState) -> None:
    """A second report with throughput within retune_threshold of the
    previous one shouldn't fire a new assignment — avoids dashboard churn
    from tok/s noise.
    """
    client = TestClient(create_app(tier_state))
    _register_two_workers(client)
    blob = _build_dummy_blob(tier_state)
    # First report at 2000 tok/s — definitely changes inner_steps from 100.
    ack0 = client.post(
        "/delta",
        params={"worker_id": 0, "round": 0, "tokens_per_sec": 2000.0},
        content=blob,
        headers={"content-type": "application/octet-stream"},
    ).json()
    assert ack0["inner_steps"] is not None
    first = ack0["inner_steps"]

    # Worker 1 submits to close round 0.
    client.post(
        "/delta",
        params={"worker_id": 1, "round": 0, "tokens_per_sec": 2000.0},
        content=blob,
        headers={"content-type": "application/octet-stream"},
    )
    assert tier_state.round == 1

    # Same worker reports 2050 tok/s in round 1 — well within 10% of first.
    # Coord should NOT re-assign inner_steps.
    ack1 = client.post(
        "/delta",
        params={"worker_id": 0, "round": 1, "tokens_per_sec": 2050.0},
        content=blob,
        headers={"content-type": "application/octet-stream"},
    ).json()
    assert ack1["inner_steps"] is None  # no change
    # Underlying per-worker value should still match the first assignment.
    workers = client.get("/workers").json()["workers"]
    by_id = {w["worker_id"]: w for w in workers}
    assert by_id[0]["inner_steps"] == first


def test_tier_aware_status_exposes_target(tier_state: CoordinatorState) -> None:
    """/status surfaces target_round_seconds + tier_aware flag for the
    dashboard's tier-aware indicator.
    """
    client = TestClient(create_app(tier_state))
    s = client.get("/status").json()
    assert s["tier_aware"] is True
    assert s["target_round_seconds"] == 300.0


def test_tier_aware_flops_account_per_worker(tier_state: CoordinatorState) -> None:
    """FLOPs accounting under tier-aware: cohort FLOPs sums each worker's
    actual inner_steps, not world_size × default. The fast worker's larger
    inner_steps should dominate the round's contribution.
    """
    client = TestClient(create_app(tier_state))
    _register_two_workers(client)
    blob = _build_dummy_blob(tier_state)
    # Fast + slow; close round to trigger FLOPs accounting.
    client.post(
        "/delta",
        params={"worker_id": 0, "round": 0, "tokens_per_sec": 4000.0},
        content=blob,
        headers={"content-type": "application/octet-stream"},
    )
    client.post(
        "/delta",
        params={"worker_id": 1, "round": 0, "tokens_per_sec": 400.0},
        content=blob,
        headers={"content-type": "application/octet-stream"},
    )
    assert tier_state.round == 1
    flops = client.get("/status").json()["flops_total"]
    # Per-worker steps fast=2000, slow=937 → total tokens = (2000 + 937) * 128.
    # Single-tier baseline would be 100 * 2 * 128 = 25600 tok. Per-worker
    # accounting gives (2000 + 937) * 128 = 375 936 tok — >10× higher.
    n_params = float(tier_state.model.num_params(non_embedding=False))
    expected_lo = 6.0 * n_params * (2000 + 900) * 128  # generous floor
    assert flops > expected_lo


# ---------------------------------------------------------------------------
# FLOPs alarm threshold (EU AI Act systemic-risk pre-warning)
# ---------------------------------------------------------------------------


def test_flops_alarm_threshold_default_5e24(state: CoordinatorState) -> None:
    """Default alarm is 5e24 (half of EU AI Act 10²⁵ systemic-risk line)."""
    client = TestClient(create_app(state))
    s = client.get("/status").json()
    assert s["flops_alarm_threshold"] == 5e24


def test_flops_alarm_threshold_configurable(tmp_path: Path) -> None:
    """Operator can lower the threshold (e.g. for a 7B Phase 2 run that's
    already approaching the line)."""
    cfg = TrainConfig(seq_len=32, micro_batch_size=4, inner_steps=3, seed=0)
    coord = CoordinatorState(
        preset_name="smoke",
        world_size=1,
        train_cfg=cfg,
        device="cpu",
        state_codec="fp32",
        delta_codec="fp32",
        checkpoint_dir=None,
        flops_alarm_threshold=1e23,
    )
    client = TestClient(create_app(coord))
    s = client.get("/status").json()
    assert s["flops_alarm_threshold"] == 1e23


# ---------------------------------------------------------------------------
# Worker auto-reregister on /delta 404
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Dynamic world_size (Choice A: coord-only, recompute at round boundaries)
# ---------------------------------------------------------------------------


def _coord_with_floor(min_world_size: int = 1) -> CoordinatorState:
    cfg = TrainConfig(
        seq_len=32, micro_batch_size=4, inner_steps=3, max_outer_rounds=2, seed=0
    )
    return CoordinatorState(
        preset_name="smoke",
        world_size=min_world_size,
        train_cfg=cfg,
        device="cpu",
        state_codec="fp32",
        delta_codec="fp32",
        checkpoint_dir=None,
        # Long inactivity timeout so registrations stick around for the test
        worker_inactive_timeout_seconds=3600.0,
        enable_timeout_thread=False,
    )


def test_world_size_initial_matches_floor() -> None:
    """A fresh coord with --world-size 1 (the floor) starts at world_size=1
    even before any worker registers, so a sole volunteer can close rounds.
    """
    coord = _coord_with_floor(min_world_size=1)
    client = TestClient(create_app(coord))
    s = client.get("/status").json()
    assert s["world_size"] == 1
    assert s["min_world_size"] == 1


def test_world_size_does_not_change_mid_round_on_register() -> None:
    """Registering a new worker mid-round must NOT bump the quorum target —
    otherwise an in-flight round would suddenly need a delta from the new
    worker (which has only just begun its inner loop), stalling forever.
    """
    coord = _coord_with_floor(min_world_size=1)
    client = TestClient(create_app(coord))
    # Round 0 opens with world_size=1.
    assert coord.world_size == 1
    # First worker registers.
    client.post("/register", json=RegisterRequest(pubkey="w0", preset="smoke").model_dump())
    # Second worker registers while round 0 is still open. world_size stays 1.
    client.post("/register", json=RegisterRequest(pubkey="w1", preset="smoke").model_dump())
    assert coord.world_size == 1, (
        "world_size must not auto-bump mid-round; the round would otherwise "
        "stall waiting on a worker that just joined."
    )


def test_world_size_grows_at_round_boundary() -> None:
    """When new workers register during round N and the outer step closes
    round N, world_size recomputes at the round-boundary so round N+1 opens
    with the correct quorum target.
    """
    coord = _coord_with_floor(min_world_size=1)
    client = TestClient(create_app(coord))
    # Two workers register during round 0.
    client.post("/register", json=RegisterRequest(pubkey="w0", preset="smoke").model_dump())
    client.post("/register", json=RegisterRequest(pubkey="w1", preset="smoke").model_dump())
    assert coord.world_size == 1  # not yet bumped (still mid-round)

    # Worker 0 submits → since world_size=1, the outer step fires immediately.
    blob = _build_dummy_blob(coord)
    ack0 = client.post(
        "/delta",
        params={"worker_id": 0, "round": 0, "val_loss": 5.0},
        content=blob,
        headers={"content-type": "application/octet-stream"},
    ).json()
    assert ack0["accepted"]
    assert ack0["next_round"] == 1
    # After the outer step bumps the round, world_size recomputes to 2.
    assert coord.world_size == 2
    s = client.get("/status").json()
    assert s["world_size"] == 2
    assert s["min_world_size"] == 1


def test_world_size_shrinks_on_eviction() -> None:
    """Evicting a stale worker mid-round drops world_size immediately so the
    remaining workers can close the round at the lower quorum.
    """
    import time as _t

    coord = _coord_with_floor(min_world_size=1)
    coord.world_size = 3  # pretend we're mid-3-worker run
    now = _t.time()
    coord.workers = {
        wid: {"registered_at": now - 60, "last_seen_ts": now}
        for wid in range(3)
    }
    # Worker 2 hasn't been seen for ages → eviction target.
    coord.workers[2]["last_seen_ts"] = now - 9999
    coord.worker_inactive_timeout_seconds = 60.0
    evicted = coord._evict_stale_workers()
    assert evicted == 1
    assert coord.world_size == 2, (
        "after a worker is evicted, world_size must drop so the remaining "
        "cohort can close the round at the new quorum target"
    )


def test_delta_ack_carries_shard_assignment() -> None:
    """Choice B: every DeltaAck includes the current (shard_index,
    shard_world_size). On the very first /delta this matches the worker's
    registered worker_id — but the worker still gets to see it as the
    authoritative source.
    """
    coord = _coord_with_floor(min_world_size=1)
    client = TestClient(create_app(coord))
    # Two workers register. Initial shard assignments (set at register)
    # are (worker_id, world_size_at_register).
    for i in range(2):
        client.post(
            "/register",
            json=RegisterRequest(pubkey=f"w{i}", preset="smoke").model_dump(),
        )
    blob = _build_dummy_blob(coord)
    ack0 = client.post(
        "/delta",
        params={"worker_id": 0, "round": 0},
        content=blob,
        headers={"content-type": "application/octet-stream"},
    ).json()
    assert ack0["accepted"]
    assert ack0["shard_index"] == 0
    # Coord recomputes contiguous indices at outer-step end. Worker 0
    # was the only submission in round 0, so round 0 closes immediately
    # with world_size=1 — the next round opens at world_size=2 (two
    # active workers), and Worker 0's shard becomes (0, 2).
    assert coord.world_size == 2
    assert coord.workers[0]["shard_index"] == 0
    assert coord.workers[0]["shard_world_size"] == 2
    assert coord.workers[1]["shard_index"] == 1
    assert coord.workers[1]["shard_world_size"] == 2


def test_shard_indices_compact_after_eviction() -> None:
    """When a worker is evicted from the middle of the active set, surviving
    workers get re-assigned to a contiguous 0..N-1 index space. Worker 2
    (id=2) moves from shard_index=2 to shard_index=1 after worker_id=1 is
    evicted; the train.bin slice it reads thus shifts.
    """
    import time as _t

    coord = _coord_with_floor(min_world_size=1)
    now = _t.time()
    coord.workers = {
        wid: {"registered_at": now - 60, "last_seen_ts": now, "shard_index": wid}
        for wid in range(3)
    }
    # Seed initial state as if round had just opened with 3 active workers.
    coord.world_size = 3
    coord._recompute_world_size_locked()
    assert coord.workers[0]["shard_index"] == 0
    assert coord.workers[1]["shard_index"] == 1
    assert coord.workers[2]["shard_index"] == 2

    # Worker 1 evicted.
    coord.workers[1]["last_seen_ts"] = now - 9999
    coord.worker_inactive_timeout_seconds = 60.0
    coord._evict_stale_workers()
    assert 1 not in coord.workers
    # Survivors compact to (0, 1).
    assert coord.workers[0]["shard_index"] == 0
    assert coord.workers[2]["shard_index"] == 1
    # world_size recomputes to 2 (down from 3).
    assert coord.world_size == 2
    # Their shard_world_size also reflects the new total.
    assert coord.workers[0]["shard_world_size"] == 2
    assert coord.workers[2]["shard_world_size"] == 2


def test_world_size_respects_min_floor_after_total_evict() -> None:
    """If every worker is evicted, world_size floors at min_world_size rather
    than collapsing to 0 (which would make even the next registration's
    sole-volunteer round impossible to close at quorum>=1).
    """
    coord = _coord_with_floor(min_world_size=2)
    assert coord.world_size == 2
    coord.workers = {
        wid: {"registered_at": 1.0, "last_seen_ts": 1.0}
        for wid in range(2)
    }
    coord.world_size = 2
    # All stale; should be evicted.
    import time as _t
    for w in coord.workers.values():
        w["last_seen_ts"] = _t.time() - 9999
    coord.worker_inactive_timeout_seconds = 60.0
    coord._evict_stale_workers()
    # world_size floors at min_world_size=2, even though active count is 0.
    assert coord.world_size == 2


def test_worker_reregister_resumes_with_fresh_id(tmp_path: Path) -> None:
    """When the coord evicts a worker (e.g. inactivity timeout), a fresh
    /delta returns 404. The worker's _reregister_and_resync method should
    re-register, pull state, and resume — assigning a NEW worker_id with
    the SAME pubkey. CLAUDE.md "Open: M5 deregistered before first delta".
    """
    import io
    from pathlib import Path as P

    from dllm.client.worker import Worker, pick_device
    from dllm.data.loader import ShardLoader
    from dllm.shared.identity import load_or_create_identity, pubkey_hex

    # Build a small fake corpus the loader can chew on. Tokens must be
    # uint16 for ShardLoader; ShardLoader needs at least seq_len*batch+1.
    train_bin = tmp_path / "train.bin"
    n_tokens = 32 * 4 * 8  # generous
    import numpy as np
    np.array(range(n_tokens), dtype=np.uint16).tofile(train_bin)

    cfg = TrainConfig(
        seq_len=32, micro_batch_size=4, inner_steps=3, max_outer_rounds=2, seed=0
    )
    coord = CoordinatorState(
        preset_name="smoke",
        world_size=1,
        train_cfg=cfg,
        device="cpu",
        state_codec="fp32",
        delta_codec="fp32",
        checkpoint_dir=None,
    )
    client = TestClient(create_app(coord))

    # Build a real Worker but point its http session at the TestClient. The
    # Worker class only uses .post / .get / .raise_for_status which
    # TestClient implements.
    device = pick_device("cpu")
    w = Worker(
        coord_url="http://testserver",
        preset="smoke",
        country="XX",
        device=device,
        train_data=train_bin,
        val_data=None,
        bf16=False,
        val_batches=1,
    )
    w.http = client  # type: ignore[assignment]

    # First registration cycle.
    w.register()
    first_id = w.worker_id
    assert first_id == 0
    w.pull_state()
    w._ensure_loader_and_opt()

    # Coord-side eviction: simulate the stale-registration sweep dropping
    # this worker (same mechanism as worker_inactive_timeout firing).
    with coord.lock:
        del coord.workers[first_id]

    # Re-register path runs: should get a NEW worker_id but preserve pubkey.
    pre_pubkey = w.pubkey_hex
    w._reregister_and_resync()
    assert w.worker_id != first_id  # got fresh id
    assert w.worker_id == 1  # next_worker_id was 1
    assert w.pubkey_hex == pre_pubkey  # same Ed25519 key, same identity
    # Coord now sees the new registration.
    assert w.worker_id in coord.workers
