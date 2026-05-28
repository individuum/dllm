"""RoPE: the real-valued (MPS-native) rotation must be numerically identical
to the old complex64 (torch.polar / view_as_complex) implementation.

This equivalence is what guarantees existing checkpoints stay valid after the
switch — the pretrained weights were trained under the complex rotation, so
the replacement must apply the exact same per-position rotation. We inline the
OLD implementation here as ground truth and compare.
"""
from __future__ import annotations

import torch

from dllm.core.model import Transformer, apply_rotary_emb, precompute_freqs_cis
from dllm.core.config import ModelConfig


def _old_complex_rope(xq, xk, head_dim, seq_len, theta):
    """Verbatim copy of the pre-fix complex64 RoPE — the ground truth."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    fc = freqs_cis[: xq_.shape[1]].view(1, xq_.shape[1], 1, xq_.shape[-1])
    xq_out = torch.view_as_real(xq_ * fc).flatten(3)
    xk_out = torch.view_as_real(xk_ * fc).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def test_rope_real_matches_complex_fp32() -> None:
    torch.manual_seed(0)
    B, T, H, Dh, theta = 2, 32, 4, 16, 500000.0
    xq = torch.randn(B, T, H, Dh)
    xk = torch.randn(B, T, H, Dh)

    fc = precompute_freqs_cis(Dh, T, theta)
    assert fc.shape == (2, T, Dh // 2)  # real [cos, sin], no longer complex
    assert not fc.is_complex()

    new_q, new_k = apply_rotary_emb(xq, xk, fc)
    ref_q, ref_k = _old_complex_rope(xq, xk, Dh, T, theta)

    assert torch.allclose(new_q, ref_q, atol=1e-5, rtol=1e-5)
    assert torch.allclose(new_k, ref_k, atol=1e-5, rtol=1e-5)


def test_rope_real_matches_complex_bf16() -> None:
    # The worker runs under bf16 autocast; verify equivalence survives the cast.
    torch.manual_seed(1)
    B, T, H, Dh, theta = 1, 16, 8, 8, 10000.0
    xq = torch.randn(B, T, H, Dh, dtype=torch.bfloat16)
    xk = torch.randn(B, T, H, Dh, dtype=torch.bfloat16)
    fc = precompute_freqs_cis(Dh, T, theta)

    new_q, new_k = apply_rotary_emb(xq, xk, fc)
    ref_q, ref_k = _old_complex_rope(xq, xk, Dh, T, theta)
    assert new_q.dtype == torch.bfloat16 and new_k.dtype == torch.bfloat16
    # bf16 has ~3 decimal digits; both paths compute in fp32 then cast, so they
    # should match to within bf16 rounding.
    assert torch.allclose(new_q.float(), ref_q.float(), atol=2e-2, rtol=2e-2)
    assert torch.allclose(new_k.float(), ref_k.float(), atol=2e-2, rtol=2e-2)


def test_rope_table_long_enough_and_indexed_by_T() -> None:
    # Buffer is built for max_seq_len*2; applying at a shorter T must just slice.
    Dh, theta = 16, 500000.0
    fc = precompute_freqs_cis(Dh, 64, theta)
    xq = torch.randn(1, 8, 2, Dh)  # T=8 < table 64
    xk = torch.randn(1, 8, 2, Dh)
    out_q, out_k = apply_rotary_emb(xq, xk, fc)
    assert out_q.shape == xq.shape and out_k.shape == xk.shape


def test_position_zero_is_identity() -> None:
    # cos(0)=1, sin(0)=0 → the first position must be unrotated.
    Dh = 16
    fc = precompute_freqs_cis(Dh, 4, 500000.0)
    xq = torch.randn(1, 4, 1, Dh)
    out_q, _ = apply_rotary_emb(xq, xq.clone(), fc)
    assert torch.allclose(out_q[:, 0], xq[:, 0], atol=1e-5)


def test_model_forward_still_runs_after_rope_change() -> None:
    cfg = ModelConfig(
        vocab_size=128, n_layers=2, n_heads=4, n_kv_heads=2, dim=32,
        max_seq_len=32, tie_embeddings=True,
    )
    model = Transformer(cfg)
    x = torch.randint(0, 128, (2, 16))
    y = torch.randint(0, 128, (2, 16))
    logits, loss = model(x, y)
    assert logits.shape == (2, 16, 128)
    assert loss is not None and torch.isfinite(loss)
