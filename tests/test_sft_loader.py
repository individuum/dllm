"""Tests for SFTShardLoader — the masked (ids, mask) → (x, y) loader.

The on-disk mask is parallel to ids and NOT pre-shifted: mask[j]==1 iff token
j is an assistant-generated target. The loader aligns it to the next-token
targets y (mask[i+1:i+1+T]) and sets non-target positions to -100 so the
model's F.cross_entropy(ignore_index=-100) only trains on assistant tokens.

Trick used throughout: write ids = arange(N) as uint16, so ids[idx] == idx.
Then x[r, 0] reveals the (otherwise hidden) random window start for row r,
letting us reconstruct the exact expected y/mask without the loader exposing
its sampled offsets.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from dllm.data.loader import SFTShardLoader

IGN = SFTShardLoader.IGNORE_INDEX


def _write_bins(tmp_path: Path, ids: np.ndarray, mask: np.ndarray) -> tuple[Path, Path]:
    ids_p = tmp_path / "sft_train.bin"
    mask_p = tmp_path / "sft_train_mask.bin"
    ids.astype(np.uint16).tofile(ids_p)
    mask.astype(np.uint8).tofile(mask_p)
    return ids_p, mask_p


def test_shapes_dtypes_and_full_shift_when_all_targets(tmp_path: Path) -> None:
    n, T, B = 512, 8, 4
    ids = np.arange(n, dtype=np.uint16)  # ids[idx] == idx (n < 65536)
    mask = np.ones(n, dtype=np.uint8)    # every token is a target
    ids_p, mask_p = _write_bins(tmp_path, ids, mask)
    ld = SFTShardLoader(ids_p, mask_p, seq_len=T, batch_size=B, seed=0)

    x, y = ld.next_batch()
    assert x.shape == (B, T) and y.shape == (B, T)
    assert x.dtype == torch.int64 and y.dtype == torch.int64
    # all targets present → nothing ignored
    assert bool((y != IGN).all())
    # y must be the next-token shift of x (ids == arange so this is exact)
    for r in range(B):
        start = int(x[r, 0].item())
        assert torch.equal(x[r], torch.arange(start, start + T))
        assert torch.equal(y[r], torch.arange(start + 1, start + 1 + T))


def test_mask_maps_to_ignore_index_exactly(tmp_path: Path) -> None:
    # Target every 4th token → each length-8 target window has ≥1 target
    # (so the resample guard always succeeds and no row is fully ignored),
    # while plenty of positions ARE ignored — exercises both branches.
    n, T, B = 512, 8, 6
    ids = np.arange(n, dtype=np.uint16)
    mask = (np.arange(n) % 4 == 0).astype(np.uint8)
    ids_p, mask_p = _write_bins(tmp_path, ids, mask)
    ld = SFTShardLoader(ids_p, mask_p, seq_len=T, batch_size=B, seed=1)

    x, y = ld.next_batch()
    for r in range(B):
        start = int(x[r, 0].item())
        assert (y[r] != IGN).any(), "guard should avoid an all-ignored row"
        for k in range(T):
            tgt_idx = start + 1 + k            # index of the target token in the stream
            if mask[tgt_idx] == 1:
                assert int(y[r, k].item()) == tgt_idx     # real next token kept
            else:
                assert int(y[r, k].item()) == IGN          # non-assistant → ignored
        # wherever a target survives, it equals the next-token of x
        keep = y[r] != IGN
        assert torch.equal(y[r][keep], x[r][keep] + 1)


def test_no_targets_in_window_does_not_corrupt_x(tmp_path: Path) -> None:
    # All-zero mask → every target position is ignored, but x (the input
    # context) must still be the raw token stream, untouched.
    n, T, B = 256, 8, 3
    ids = np.arange(n, dtype=np.uint16)
    mask = np.zeros(n, dtype=np.uint8)
    ids_p, mask_p = _write_bins(tmp_path, ids, mask)
    ld = SFTShardLoader(ids_p, mask_p, seq_len=T, batch_size=B, seed=2)
    x, y = ld.next_batch()
    assert bool((y == IGN).all())              # nothing to train on
    for r in range(B):
        start = int(x[r, 0].item())
        assert torch.equal(x[r], torch.arange(start, start + T))  # context intact


def test_length_mismatch_raises(tmp_path: Path) -> None:
    ids_p = tmp_path / "sft_train.bin"
    mask_p = tmp_path / "sft_train_mask.bin"
    np.arange(100, dtype=np.uint16).tofile(ids_p)
    np.ones(50, dtype=np.uint8).tofile(mask_p)
    with pytest.raises(RuntimeError, match="mismatch"):
        SFTShardLoader(ids_p, mask_p, seq_len=8, batch_size=2)


def test_shard_too_small_raises(tmp_path: Path) -> None:
    ids = np.arange(4, dtype=np.uint16)
    mask = np.ones(4, dtype=np.uint8)
    ids_p, mask_p = _write_bins(tmp_path, ids, mask)
    with pytest.raises(RuntimeError, match="too small"):
        SFTShardLoader(ids_p, mask_p, seq_len=8, batch_size=1)


def test_masked_loss_is_finite_and_ignores_nontargets(tmp_path: Path) -> None:
    """End-to-end: the -100 targets the loader emits flow straight into the
    model's F.cross_entropy(ignore_index=-100), yielding a finite loss
    computed over assistant tokens only."""
    from dllm.core.config import ModelConfig
    from dllm.core.model import Transformer

    n, T, B = 512, 16, 2
    ids = (np.arange(n) % 256).astype(np.uint16)  # keep ids < vocab_size
    mask = (np.arange(n) % 3 == 0).astype(np.uint8)
    ids_p, mask_p = _write_bins(tmp_path, ids, mask)
    ld = SFTShardLoader(ids_p, mask_p, seq_len=T, batch_size=B, seed=3)
    x, y = ld.next_batch()

    cfg = ModelConfig(
        vocab_size=256, n_layers=2, n_heads=2, n_kv_heads=2, dim=32,
        max_seq_len=T, tie_embeddings=True,
    )
    model = Transformer(cfg)
    _, loss = model(x, y)
    assert loss is not None
    assert torch.isfinite(loss), "masked loss must be finite when ≥1 target exists"
