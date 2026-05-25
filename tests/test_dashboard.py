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


def test_history_caps_at_maxlen(state: CoordinatorState) -> None:
    """Deque caps at 500 so we don't OOM after long runs."""
    from collections import deque
    assert isinstance(state.history, deque)
    assert state.history.maxlen == 500
