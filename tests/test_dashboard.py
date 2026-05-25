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
