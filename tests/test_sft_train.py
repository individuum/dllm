"""Tests for the single-GPU SFT trainer (dllm.sft_train)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from dllm.core.config import ModelConfig
from dllm.data.loader import SFTShardLoader
from dllm.sft_train import (
    build_model,
    cosine_lr,
    load_base_weights,
    run_sft,
    save_sft_checkpoint,
)


def _tiny_cfg(seq_len: int = 32) -> ModelConfig:
    return ModelConfig(
        vocab_size=128, n_layers=2, n_heads=2, n_kv_heads=2, dim=32,
        max_seq_len=seq_len, tie_embeddings=True,
    )


def _write_sft_bins(tmp_path: Path, n: int, mask_period: int) -> tuple[Path, Path]:
    ids = (np.arange(n) % 128).astype(np.uint16)
    mask = (np.arange(n) % mask_period == 0).astype(np.uint8) if mask_period > 0 \
        else np.zeros(n, dtype=np.uint8)
    ids_p = tmp_path / "sft_train.bin"
    mask_p = tmp_path / "sft_train_mask.bin"
    ids.tofile(ids_p)
    mask.tofile(mask_p)
    return ids_p, mask_p


# ---- LR schedule -----------------------------------------------------------


def test_cosine_lr_warmup_peak_floor() -> None:
    peak, mn, warm, total = 2e-5, 2e-6, 10, 100
    assert cosine_lr(0, total, peak, mn, warm) == pytest.approx(peak * 1 / 10)
    assert cosine_lr(9, total, peak, mn, warm) == pytest.approx(peak)        # end of warmup
    assert cosine_lr(10, total, peak, mn, warm) == pytest.approx(peak, rel=1e-6)  # cosine start
    assert cosine_lr(total, total, peak, mn, warm) == pytest.approx(mn)      # floor at horizon
    assert cosine_lr(total + 500, total, peak, mn, warm) == pytest.approx(mn)
    # monotonic non-increasing through the decay span
    prev = cosine_lr(warm, total, peak, mn, warm)
    for s in range(warm + 1, total + 1):
        cur = cosine_lr(s, total, peak, mn, warm)
        assert cur <= prev + 1e-15
        prev = cur


# ---- base-weight loading ---------------------------------------------------


def test_load_base_weights_roundtrip(tmp_path: Path) -> None:
    cfg = _tiny_cfg()
    src = build_model(cfg)
    # perturb so it's distinct from a fresh init
    with torch.no_grad():
        for p in src.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    ckpt = save_sft_checkpoint(src, tmp_path / "base", {"preset_name": "tiny", "round": 7})

    dst = build_model(cfg)
    meta = load_base_weights(dst, ckpt)
    assert meta.get("round") == 7
    for (n, a), (_, b) in zip(src.named_parameters(), dst.named_parameters()):
        assert torch.equal(a.detach(), b.detach()), f"param {n} did not round-trip"


def test_load_base_weights_rejects_wrong_shape(tmp_path: Path) -> None:
    src = build_model(_tiny_cfg())
    ckpt = save_sft_checkpoint(src, tmp_path / "base", {"preset_name": "tiny"})
    # A model with a different dim has mismatched param shapes/names.
    wrong = build_model(ModelConfig(
        vocab_size=128, n_layers=2, n_heads=2, n_kv_heads=2, dim=64,
        max_seq_len=32, tie_embeddings=True,
    ))
    with pytest.raises(ValueError):
        load_base_weights(wrong, ckpt)


# ---- training core ---------------------------------------------------------


def test_run_sft_updates_params_and_finite_loss(tmp_path: Path) -> None:
    T = 16
    cfg = _tiny_cfg(seq_len=T)
    model = build_model(cfg)
    ids_p, mask_p = _write_sft_bins(tmp_path, n=2048, mask_period=2)  # half the tokens are targets
    train = SFTShardLoader(ids_p, mask_p, seq_len=T, batch_size=4, seed=0)

    before = [p.detach().clone() for p in model.parameters()]
    hist = run_sft(
        model, train, None,
        steps=12, peak_lr=1e-2, min_lr=1e-3, warmup_steps=2,
        device_type="cpu",  # CPU, no autocast
    )
    # every logged train loss is finite
    assert hist["train_loss"], "expected at least one logged step"
    assert all(np.isfinite(loss) for _, loss in hist["train_loss"])
    # at least one parameter moved
    moved = any(
        not torch.equal(b, p.detach())
        for b, p in zip(before, model.parameters())
    )
    assert moved, "SFT should have updated parameters"


def test_run_sft_survives_all_ignored_batches(tmp_path: Path) -> None:
    """An all-zero mask means every target is -100 (NaN loss). The guard must
    skip those backward passes so training runs without crashing and leaves
    params untouched (no valid gradient anywhere)."""
    T = 16
    model = build_model(_tiny_cfg(seq_len=T))
    ids_p, mask_p = _write_sft_bins(tmp_path, n=512, mask_period=0)  # mask all zeros
    train = SFTShardLoader(ids_p, mask_p, seq_len=T, batch_size=2, seed=0)

    before = [p.detach().clone() for p in model.parameters()]
    hist = run_sft(model, train, None, steps=3, peak_lr=1e-2, warmup_steps=0, device_type="cpu")
    assert all(loss == 0.0 for _, loss in hist["train_loss"])  # nothing contributed
    for b, p in zip(before, model.parameters()):
        assert torch.equal(b, p.detach()), "no param should move when every batch is ignored"


def test_run_sft_eval_returns_finite(tmp_path: Path) -> None:
    T = 16
    model = build_model(_tiny_cfg(seq_len=T))
    ids_p, mask_p = _write_sft_bins(tmp_path, n=1024, mask_period=3)
    train = SFTShardLoader(ids_p, mask_p, seq_len=T, batch_size=2, seed=0)
    val = SFTShardLoader(ids_p, mask_p, seq_len=T, batch_size=2, seed=99)
    hist = run_sft(
        model, train, val,
        steps=6, peak_lr=5e-3, warmup_steps=1, val_every=3, val_batches=4, device_type="cpu",
    )
    assert hist["final_val"] is not None and np.isfinite(hist["final_val"])
    assert hist["val_loss"], "expected an interim val measurement"
