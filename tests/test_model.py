from __future__ import annotations

import torch

from dllm.core import PRESETS, Transformer


def test_forward_shape(tiny_model: Transformer, tiny_cfg) -> None:
    x = torch.randint(0, tiny_cfg.vocab_size, (2, 16))
    logits, loss = tiny_model(x)
    assert logits.shape == (2, 16, tiny_cfg.vocab_size)
    assert loss is None


def test_forward_with_targets(tiny_model: Transformer, tiny_cfg) -> None:
    x = torch.randint(0, tiny_cfg.vocab_size, (2, 16))
    y = torch.randint(0, tiny_cfg.vocab_size, (2, 16))
    logits, loss = tiny_model(x, y)
    assert loss is not None
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_backward_runs(tiny_model: Transformer, tiny_cfg) -> None:
    x = torch.randint(0, tiny_cfg.vocab_size, (2, 16))
    y = torch.randint(0, tiny_cfg.vocab_size, (2, 16))
    _, loss = tiny_model(x, y)
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in tiny_model.parameters())
    assert has_grad


def test_param_count_increases_with_size() -> None:
    for a, b in [("smoke", "124M"), ("124M", "1B"), ("1B", "7B")]:
        m_a = Transformer(PRESETS[a]).num_params()
        m_b = Transformer(PRESETS[b]).num_params()
        assert m_b > m_a, f"{b} should have more params than {a}"


def test_tied_embeddings_share_storage(tiny_model: Transformer) -> None:
    assert tiny_model.lm_head.weight.data_ptr() == tiny_model.tok_emb.weight.data_ptr()


def test_generate_appends_tokens(tiny_model: Transformer, tiny_cfg) -> None:
    x = torch.zeros((1, 4), dtype=torch.long)
    out = tiny_model.generate(x, max_new_tokens=8)
    assert out.shape == (1, 12)
    assert (out[:, :4] == x).all()
