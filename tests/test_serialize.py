from __future__ import annotations

import torch

from dllm.core import Transformer
from dllm.shared.serialize import (
    average_deltas,
    bytes_to_state,
    compute_delta,
    load_into_model,
    model_state,
    snapshot,
    state_to_bytes,
)


def test_state_safetensors_roundtrip(tiny_model: Transformer) -> None:
    state = model_state(tiny_model)
    blob = state_to_bytes(state)
    assert len(blob) > 0
    restored = bytes_to_state(blob)
    assert set(restored.keys()) == set(state.keys())
    for n in state:
        assert torch.equal(restored[n].cpu(), state[n].detach().cpu())


def test_load_into_model_rejects_shape_mismatch(tiny_model: Transformer) -> None:
    bad = {n: torch.zeros(1) for n in dict(tiny_model.named_parameters())}
    import pytest

    with pytest.raises(ValueError):
        load_into_model(tiny_model, bad)


def test_compute_delta_zero_when_unchanged(tiny_model: Transformer) -> None:
    snap = snapshot(tiny_model)
    delta = compute_delta(snap, tiny_model)
    for n, d in delta.items():
        assert torch.equal(d, torch.zeros_like(d))


def test_compute_delta_after_step(tiny_model: Transformer, tiny_cfg) -> None:
    snap = snapshot(tiny_model)
    opt = torch.optim.SGD(tiny_model.parameters(), lr=0.1)
    x = torch.randint(0, tiny_cfg.vocab_size, (2, 16))
    y = torch.randint(0, tiny_cfg.vocab_size, (2, 16))
    _, loss = tiny_model(x, y)
    loss.backward()
    opt.step()
    delta = compute_delta(snap, tiny_model)
    nonzero = any(d.abs().sum() > 0 for d in delta.values())
    assert nonzero


def test_average_deltas_simple() -> None:
    d1 = {"a": torch.ones(3), "b": torch.zeros(2)}
    d2 = {"a": -torch.ones(3), "b": torch.ones(2) * 4}
    avg = average_deltas([d1, d2])
    assert torch.equal(avg["a"], torch.zeros(3))
    assert torch.equal(avg["b"], torch.ones(2) * 2)


def test_average_deltas_rejects_empty() -> None:
    import pytest

    with pytest.raises(ValueError):
        average_deltas([])
