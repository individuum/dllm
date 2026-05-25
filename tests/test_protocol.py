from __future__ import annotations

from dllm.shared.protocol import (
    DeltaAck,
    RegisterRequest,
    RegisterResponse,
    RoundStatus,
)


def test_register_request_defaults() -> None:
    r = RegisterRequest(pubkey="abc123")
    assert r.country == "XX"
    assert r.preset == "smoke"
    assert r.vram_gb == 0


def test_register_request_roundtrip() -> None:
    src = RegisterRequest(pubkey="k", country="DE", gpu="RTX 4090", vram_gb=24, preset="124M")
    payload = src.model_dump()
    rebuilt = RegisterRequest(**payload)
    assert rebuilt == src


def test_register_response_required_fields() -> None:
    r = RegisterResponse(
        worker_id=0,
        current_round=0,
        world_size=2,
        seed=0,
        inner_steps=50,
        seq_len=256,
        micro_batch_size=16,
    )
    payload = r.model_dump()
    assert payload["worker_id"] == 0
    assert payload["world_size"] == 2


def test_round_status_waiting_for_is_consistent() -> None:
    s = RoundStatus(current_round=3, n_registered=4, n_submitted=2, waiting_for=2)
    assert s.waiting_for == s.n_registered - s.n_submitted


def test_delta_ack_with_next_round() -> None:
    a = DeltaAck(accepted=True, next_round=5)
    assert a.accepted
    assert a.next_round == 5
    assert a.reason == ""
