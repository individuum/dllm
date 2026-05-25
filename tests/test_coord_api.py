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
