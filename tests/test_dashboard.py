"""Dashboard + /history endpoint."""
from __future__ import annotations

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
from dllm.shared.serialize import (
    compute_delta,
    snapshot,
    state_to_bytes,
)


@pytest.fixture
def state() -> CoordinatorState:
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
        enable_timeout_thread=False,
    )


def test_dashboard_returns_html(state: CoordinatorState) -> None:
    client = TestClient(create_app(state))
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    body = r.text
    assert "dllm coordinator" in body
    assert "/status" in body
    assert "/history" in body


def test_history_starts_empty(state: CoordinatorState) -> None:
    client = TestClient(create_app(state))
    body = client.get("/history").json()
    assert body == {"history": []}


def test_history_appends_after_each_outer_step(
    state: CoordinatorState, tmp_path: Path
) -> None:
    client = TestClient(create_app(state))
    sk = load_or_create_identity(tmp_path / "id.key")
    client.post(
        "/register",
        json=RegisterRequest(pubkey=pubkey_hex(sk), preset="smoke").model_dump(),
    )

    # do 3 outer rounds
    for r in range(3):
        snap = snapshot(state.model)
        with torch.no_grad():
            for p in state.model.parameters():
                p.add_(torch.randn_like(p) * 0.001)
        delta = compute_delta(snap, state.model)
        with torch.no_grad():
            for n, p in state.model.named_parameters():
                p.copy_(snap[n])
        blob = state_to_bytes(delta)
        sig = sign_delta(sk, 0, r, blob)
        client.post(
            "/delta",
            params={"worker_id": 0, "round": r, "val_loss": 7.5 - r * 0.5},
            content=blob,
            headers={"content-type": "application/octet-stream", "x-delta-signature": sig},
        )

    body = client.get("/history").json()
    assert len(body["history"]) == 3
    rows = body["history"]
    # Each row is (round, val_loss, flops_total, ts) — record happens AFTER outer
    # step, so first record is round=0 (the one we just completed)
    assert [r["round"] for r in rows] == [0, 1, 2]
    assert all("val_loss" in r and "flops_total" in r and "ts" in r for r in rows)
    # flops accumulates monotonically
    assert rows[0]["flops_total"] < rows[1]["flops_total"] < rows[2]["flops_total"]
    # val_loss got recorded from the query param
    assert rows[0]["val_loss"] == pytest.approx(7.5)
    assert rows[2]["val_loss"] == pytest.approx(6.5)
    # New energy/power fields exist (may be null if no power_watts was sent —
    # we'll cover the populated case in test_history_records_power_and_energy)
    for r in rows:
        assert "round_seconds" in r
        assert "power_watts" in r
        assert "tokens_per_sec" in r
        assert "energy_wh_total" in r
        assert "n_workers" in r


def test_history_records_power_and_energy(
    state: CoordinatorState, tmp_path: Path
) -> None:
    """When the worker passes power_watts + tokens_per_sec, the coord
    accumulates energy and surfaces it in history + status.
    """
    client = TestClient(create_app(state))
    sk = load_or_create_identity(tmp_path / "id.key")
    client.post(
        "/register",
        json=RegisterRequest(pubkey=pubkey_hex(sk), preset="smoke").model_dump(),
    )

    for r in range(3):
        snap = snapshot(state.model)
        with torch.no_grad():
            for p in state.model.parameters():
                p.add_(torch.randn_like(p) * 0.001)
        delta = compute_delta(snap, state.model)
        with torch.no_grad():
            for n, p in state.model.named_parameters():
                p.copy_(snap[n])
        blob = state_to_bytes(delta)
        sig = sign_delta(sk, 0, r, blob)
        client.post(
            "/delta",
            params={
                "worker_id": 0,
                "round": r,
                "val_loss": 5.0,
                "power_watts": 170.0,
                "tokens_per_sec": 4000.0,
            },
            content=blob,
            headers={"content-type": "application/octet-stream", "x-delta-signature": sig},
        )

    hist = client.get("/history").json()["history"]
    assert len(hist) == 3
    for r in hist:
        assert r["power_watts"] == pytest.approx(170.0)
        assert r["tokens_per_sec"] == pytest.approx(4000.0)
    # Energy is cumulative — monotonically non-decreasing
    energies = [r["energy_wh_total"] for r in hist]
    assert all(energies[i] >= energies[i-1] for i in range(1, len(energies)))
    # Status surfaces the latest values
    status = client.get("/status").json()
    assert status["last_power_watts"] == pytest.approx(170.0)
    assert status["last_tokens_per_sec"] == pytest.approx(4000.0)
    assert status["energy_wh_total"] >= 0  # >= 0 because TestClient rounds may be ~0s


def test_cohort_power_is_sum_not_mean_with_two_workers(tmp_path: Path) -> None:
    """The "POWER DRAW (COHORT)" tile should be the SUM across reporting workers
    (because workers train simultaneously). Was a bug: the coord recorded the
    MEAN, which made the tile drop when a low-power worker (M5 at ~35W) joined
    a high-power worker (3060 at ~140W). After the fix the tile climbs.
    """
    cfg = TrainConfig(
        seq_len=32, micro_batch_size=4, inner_steps=3, max_outer_rounds=2, seed=0
    )
    state = CoordinatorState(
        preset_name="smoke",
        world_size=2,  # 2 workers needed before the outer step fires
        train_cfg=cfg,
        device="cpu",
        state_codec="fp32",
        delta_codec="fp32",
        checkpoint_dir=None,
        enable_timeout_thread=False,
    )
    client = TestClient(create_app(state))
    sk_a = load_or_create_identity(tmp_path / "id_a.key")
    sk_b = load_or_create_identity(tmp_path / "id_b.key")
    client.post("/register", json=RegisterRequest(pubkey=pubkey_hex(sk_a), preset="smoke").model_dump())
    client.post("/register", json=RegisterRequest(pubkey=pubkey_hex(sk_b), preset="smoke").model_dump())

    snap = snapshot(state.model)
    with torch.no_grad():
        for p in state.model.parameters():
            p.add_(torch.randn_like(p) * 0.001)
    delta = compute_delta(snap, state.model)
    with torch.no_grad():
        for n, p in state.model.named_parameters():
            p.copy_(snap[n])
    blob = state_to_bytes(delta)

    # Worker A reports 140 W (3060-ish), worker B reports 35 W (M5-ish)
    for wid, sk, pw, tps in [(0, sk_a, 140.0, 15000.0), (1, sk_b, 35.0, 3500.0)]:
        sig = sign_delta(sk, wid, 0, blob)
        client.post(
            "/delta",
            params={
                "worker_id": wid, "round": 0,
                "val_loss": 5.0, "power_watts": pw, "tokens_per_sec": tps,
            },
            content=blob,
            headers={"content-type": "application/octet-stream", "x-delta-signature": sig},
        )

    status = client.get("/status").json()
    # COHORT sum, NOT mean: 140 + 35 = 175 W (the bug returned 87.5 W)
    assert status["last_power_watts"] == pytest.approx(175.0)
    # mean is still exposed for the dashboard's "avg per worker" sub-line
    assert status["last_power_watts_per_worker"] == pytest.approx(87.5)
    assert status["last_n_reporting_workers"] == 2
    # tok/s remains a sum
    assert status["last_tokens_per_sec"] == pytest.approx(18500.0)


def test_stale_workers_get_evicted(tmp_path: Path) -> None:
    """Worker registrations whose last_seen_ts is older than the inactive
    timeout get dropped automatically — prevents ghost workers from piling
    up in the dashboard (the M5 / "registered twice" scenario).
    """
    cfg = TrainConfig(
        seq_len=32, micro_batch_size=4, inner_steps=3, max_outer_rounds=2, seed=0
    )
    state = CoordinatorState(
        preset_name="smoke",
        world_size=1,
        train_cfg=cfg,
        device="cpu",
        state_codec="fp32",
        delta_codec="fp32",
        checkpoint_dir=None,
        worker_inactive_timeout_seconds=60.0,  # 60s for the test
        enable_timeout_thread=False,  # we drive evict manually
    )
    client = TestClient(create_app(state))
    sk_old = load_or_create_identity(tmp_path / "id_old.key")
    sk_new = load_or_create_identity(tmp_path / "id_new.key")

    # Ghost worker: registered, never submitted, registered_at is old
    client.post(
        "/register",
        json=RegisterRequest(pubkey=pubkey_hex(sk_old), preset="smoke").model_dump(),
    )
    state.workers[0]["registered_at"] = 0.0  # epoch — definitely older than 60s

    # Fresh worker: just registered
    client.post(
        "/register",
        json=RegisterRequest(pubkey=pubkey_hex(sk_new), preset="smoke").model_dump(),
    )

    # Before eviction: both registered
    assert len(state.workers) == 2

    n_evicted = state._evict_stale_workers()
    assert n_evicted == 1
    # Only the fresh worker survives — and its worker_id was re-mapped to whatever
    # registered second (worker_id=1 here)
    assert list(state.workers.keys()) == [1]


def test_eviction_respects_recently_contributed_worker(tmp_path: Path) -> None:
    """A worker that submitted recently shouldn't be evicted, even if it
    registered long ago."""
    cfg = TrainConfig(
        seq_len=32, micro_batch_size=4, inner_steps=3, max_outer_rounds=2, seed=0
    )
    state = CoordinatorState(
        preset_name="smoke",
        world_size=1,
        train_cfg=cfg,
        device="cpu",
        state_codec="fp32",
        delta_codec="fp32",
        checkpoint_dir=None,
        worker_inactive_timeout_seconds=60.0,
        enable_timeout_thread=False,
    )
    client = TestClient(create_app(state))
    sk = load_or_create_identity(tmp_path / "id.key")
    client.post(
        "/register",
        json=RegisterRequest(pubkey=pubkey_hex(sk), preset="smoke").model_dump(),
    )

    import time as _t
    # Old registration but recent activity
    state.workers[0]["registered_at"] = 0.0
    state.workers[0]["last_seen_ts"] = _t.time()  # now

    n_evicted = state._evict_stale_workers()
    assert n_evicted == 0
    assert len(state.workers) == 1


def test_workers_endpoint_lists_per_worker_stats(state: CoordinatorState, tmp_path: Path) -> None:
    """GET /workers returns each registered worker's contribution stats."""
    client = TestClient(create_app(state))
    sk = load_or_create_identity(tmp_path / "id.key")
    client.post(
        "/register",
        json=RegisterRequest(
            pubkey=pubkey_hex(sk), preset="smoke", country="DE", gpu="RTX 3060", vram_gb=11
        ).model_dump(),
    )

    # Before any delta: registered but never contributed
    workers = client.get("/workers").json()["workers"]
    assert len(workers) == 1
    w = workers[0]
    assert w["worker_id"] == 0
    assert w["country"] == "DE"
    assert w["gpu"] == "RTX 3060"
    assert w["vram_gb"] == 11
    assert w["rounds_contributed"] == 0
    assert w["last_seen_ts"] is None  # the "never contributed" sentinel
    assert w["last_round"] is None

    # Submit one delta; stats should update
    snap = snapshot(state.model)
    with torch.no_grad():
        for p in state.model.parameters():
            p.add_(torch.randn_like(p) * 0.001)
    delta = compute_delta(snap, state.model)
    with torch.no_grad():
        for n, p in state.model.named_parameters():
            p.copy_(snap[n])
    blob = state_to_bytes(delta)
    sig = sign_delta(sk, 0, 0, blob)
    client.post(
        "/delta",
        params={
            "worker_id": 0, "round": 0,
            "val_loss": 4.2, "power_watts": 142.0, "tokens_per_sec": 15500.0,
        },
        content=blob,
        headers={"content-type": "application/octet-stream", "x-delta-signature": sig},
    )

    workers = client.get("/workers").json()["workers"]
    w = workers[0]
    assert w["rounds_contributed"] == 1
    assert w["last_round"] == 0
    assert w["last_seen_ts"] is not None
    assert w["last_val_loss"] == pytest.approx(4.2)
    assert w["last_power_watts"] == pytest.approx(142.0)
    assert w["last_tokens_per_sec"] == pytest.approx(15500.0)


def test_history_omits_power_when_worker_does_not_report(
    state: CoordinatorState, tmp_path: Path
) -> None:
    """Old workers that don't send power_watts shouldn't break the record."""
    client = TestClient(create_app(state))
    sk = load_or_create_identity(tmp_path / "id.key")
    client.post(
        "/register",
        json=RegisterRequest(pubkey=pubkey_hex(sk), preset="smoke").model_dump(),
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
    sig = sign_delta(sk, 0, 0, blob)
    client.post(
        "/delta",
        params={"worker_id": 0, "round": 0, "val_loss": 5.0},
        content=blob,
        headers={"content-type": "application/octet-stream", "x-delta-signature": sig},
    )
    hist = client.get("/history").json()["history"]
    assert hist[0]["power_watts"] is None
    assert hist[0]["tokens_per_sec"] is None
    assert hist[0]["energy_wh_total"] == 0.0


def test_history_caps_at_maxlen(state: CoordinatorState) -> None:
    """Deque caps at 500 so we don't OOM after long runs."""
    from collections import deque
    assert isinstance(state.history, deque)
    assert state.history.maxlen == 500


def test_history_persists_to_disk_and_reloads(tmp_path: Path) -> None:
    """A round's history entry survives a CoordinatorState restart via
    history.jsonl in the checkpoint dir.
    """
    cfg = TrainConfig(
        seq_len=32, micro_batch_size=4, inner_steps=3, max_outer_rounds=2, seed=0
    )
    ckpt = tmp_path / "ckpts"
    ckpt.mkdir()

    s1 = CoordinatorState(
        preset_name="smoke",
        world_size=1,
        train_cfg=cfg,
        device="cpu",
        state_codec="fp32",
        delta_codec="fp32",
        checkpoint_dir=ckpt,
        enable_timeout_thread=False,
    )
    sk = load_or_create_identity(tmp_path / "id.key")
    client1 = TestClient(create_app(s1))
    client1.post(
        "/register",
        json=RegisterRequest(pubkey=pubkey_hex(sk), preset="smoke").model_dump(),
    )
    snap = snapshot(s1.model)
    with torch.no_grad():
        for p in s1.model.parameters():
            p.add_(torch.randn_like(p) * 0.001)
    delta = compute_delta(snap, s1.model)
    with torch.no_grad():
        for n, p in s1.model.named_parameters():
            p.copy_(snap[n])
    blob = state_to_bytes(delta)
    sig = sign_delta(sk, 0, 0, blob)
    client1.post(
        "/delta",
        params={"worker_id": 0, "round": 0, "val_loss": 4.2},
        content=blob,
        headers={"content-type": "application/octet-stream", "x-delta-signature": sig},
    )
    assert len(s1.history) == 1
    assert (ckpt / "history.jsonl").exists()

    # Restart simulation: new state, same checkpoint_dir
    s2 = CoordinatorState(
        preset_name="smoke",
        world_size=1,
        train_cfg=cfg,
        device="cpu",
        state_codec="fp32",
        delta_codec="fp32",
        checkpoint_dir=ckpt,
        enable_timeout_thread=False,
    )
    assert len(s2.history) == 1
    assert s2.history[0]["round"] == 0
    assert s2.history[0]["val_loss"] == pytest.approx(4.2)


def test_backfill_reads_checkpoint_metas(tmp_path: Path) -> None:
    """Old checkpoints without a history.jsonl get backfilled from their meta.json."""
    cfg = TrainConfig(
        seq_len=32, micro_batch_size=4, inner_steps=3, max_outer_rounds=2, seed=0
    )
    ckpt = tmp_path / "ckpts"
    ckpt.mkdir()
    # Fake some past checkpoints as if from before history existed
    import json as _json
    for r in (2, 5, 8):
        d = ckpt / f"ckpt_{r:06d}"
        d.mkdir()
        (d / "meta.json").write_text(
            _json.dumps({"round": r, "ts": 1000.0 + r, "flops_total": 1e15 * r, "last_val_loss": 7.0 - r * 0.3})
        )
        # bare-minimum dummy files so find_latest doesn't trip; full load
        # would need real safetensors but backfill doesn't care
    s = CoordinatorState(
        preset_name="smoke",
        world_size=1,
        train_cfg=cfg,
        device="cpu",
        state_codec="fp32",
        delta_codec="fp32",
        checkpoint_dir=ckpt,
        enable_timeout_thread=False,
    )
    rounds = [h["round"] for h in s.history]
    assert rounds == [2, 5, 8]
    assert s.history[1]["val_loss"] == pytest.approx(7.0 - 5 * 0.3)
    assert s.history[2]["flops_total"] == pytest.approx(1e15 * 8)
