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
    for a, b in [("smoke", "124M"), ("124M", "300M"), ("300M", "1B"), ("1B", "7B")]:
        m_a = Transformer(PRESETS[a]).num_params()
        m_b = Transformer(PRESETS[b]).num_params()
        assert m_b > m_a, f"{b} should have more params than {a}"


def test_300M_preset_has_long_context_and_grad_checkpoint() -> None:
    """The 300M preset is configured for long-context training (seq=8192,
    extrapolatable past) under gradient checkpointing (the only way it
    fits on 12 GB VRAM at that context length)."""
    cfg = PRESETS["300M"]
    assert cfg.max_seq_len == 8192
    assert cfg.use_grad_checkpoint is True
    assert cfg.n_kv_heads == 8, "300M uses 2:1 GQA (vs 4:1) for stronger attention"


def test_grad_checkpoint_backward_runs_and_produces_grads() -> None:
    """Gradient checkpointing must compute correct gradients end-to-end —
    the recomputed forward in backward shouldn't break autograd."""
    from dataclasses import replace

    from dllm.core import PRESETS
    cfg = replace(PRESETS["smoke"], use_grad_checkpoint=True)
    model = Transformer(cfg)
    model.train()
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = torch.randint(0, cfg.vocab_size, (2, 16))
    _, loss = model(x, y)
    loss.backward()
    n_with_grad = sum(
        1 for p in model.parameters()
        if p.grad is not None and p.grad.abs().sum() > 0
    )
    # Every parameter (or at least most) should have non-zero gradient
    n_params = sum(1 for _ in model.parameters())
    assert n_with_grad > n_params * 0.8, f"only {n_with_grad}/{n_params} params got grads"


def test_grad_checkpoint_equivalent_to_no_checkpoint() -> None:
    """Gradient checkpointing should produce numerically identical output
    (up to floating-point determinism) to the non-checkpointed forward."""
    from dataclasses import replace

    from dllm.core import PRESETS
    cfg_no_ckpt = replace(PRESETS["smoke"], use_grad_checkpoint=False)
    cfg_ckpt = replace(PRESETS["smoke"], use_grad_checkpoint=True)

    torch.manual_seed(0)
    m1 = Transformer(cfg_no_ckpt).eval()
    torch.manual_seed(0)
    m2 = Transformer(cfg_ckpt).eval()
    # Force identical weights regardless of checkpoint flag (init is deterministic
    # given the seed, so this is just paranoia; copying is the belt-and-braces)
    m2.load_state_dict(m1.state_dict())

    x = torch.randint(0, cfg_no_ckpt.vocab_size, (2, 16))
    with torch.no_grad():
        out1, _ = m1(x)
        out2, _ = m2(x)
    assert torch.allclose(out1, out2, atol=1e-5)


def test_tied_embeddings_share_storage(tiny_model: Transformer) -> None:
    assert tiny_model.lm_head.weight.data_ptr() == tiny_model.tok_emb.weight.data_ptr()


def test_generate_appends_tokens(tiny_model: Transformer, tiny_cfg) -> None:
    x = torch.zeros((1, 4), dtype=torch.long)
    out = tiny_model.generate(x, max_new_tokens=8)
    assert out.shape == (1, 12)
    assert (out[:, :4] == x).all()
