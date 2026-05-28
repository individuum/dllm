from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class ShardLoader:
    """Memory-mapped uint16 token stream → fixed-length (x, y) batches.

    Each worker is assigned a (shard, offset, stride) tuple by the coordinator.
    For Phase 0, a single file is split deterministically by worker_id.
    """

    def __init__(
        self,
        path: str | Path,
        seq_len: int,
        batch_size: int,
        worker_id: int = 0,
        world_size: int = 1,
        device: str | torch.device = "cpu",
        seed: int = 0,
    ) -> None:
        self.path = Path(path)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.worker_id = worker_id
        self.world_size = world_size
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed + worker_id)

        self.tokens = np.memmap(self.path, dtype=np.uint16, mode="r")
        # split corpus into world_size contiguous slices; worker_id is a persistent
        # identity that can exceed world_size after many re-registrations, so mod it
        # back into the [0, world_size) shard space.
        self.shard_idx = worker_id % world_size
        per = len(self.tokens) // world_size
        self.start = self.shard_idx * per
        self.end = self.start + per
        if (self.end - self.start) < seq_len + 1:
            raise RuntimeError(
                f"shard {self.shard_idx} (worker {worker_id}) too small: "
                f"{self.end - self.start} tokens < seq_len+1"
            )

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        ix = self.rng.integers(self.start, self.end - self.seq_len - 1, size=self.batch_size)
        # numpy fancy-index into memmap is fine; cast to int64 for embedding lookup
        x = np.stack([self.tokens[i : i + self.seq_len] for i in ix]).astype(np.int64)
        y = np.stack([self.tokens[i + 1 : i + 1 + self.seq_len] for i in ix]).astype(np.int64)
        xt = torch.from_numpy(x)
        yt = torch.from_numpy(y)
        if self.device.type == "cuda":
            xt = xt.pin_memory().to(self.device, non_blocking=True)
            yt = yt.pin_memory().to(self.device, non_blocking=True)
        else:
            xt = xt.to(self.device)
            yt = yt.to(self.device)
        return xt, yt


class SFTShardLoader:
    """Memory-mapped (ids: uint16, mask: uint8) → (x, y) batches where y is
    masked to ignore_index (-100) on non-assistant tokens.

    Pairs with `dllm.data.sft_prepare`'s output (sft_*.bin + sft_*_mask.bin).
    The on-disk mask is NOT pre-shifted — `mask[j] == 1` iff token j is an
    assistant-generated target (incl. the closing <|im_end|>). We align it to
    the next-token targets `y` by taking `mask[i+1 : i+1+T]`, then set every
    y position with mask==0 to -100. The model's
    `F.cross_entropy(..., ignore_index=-100)` then trains ONLY on assistant
    tokens — role headers, user turns and tool results stay in context but
    contribute no gradient. No model change required.

    Mirrors `ShardLoader`'s API (random-window `next_batch`). A sampled window
    that happens to contain zero assistant targets is resampled, because an
    all -100 batch makes the mean cross-entropy NaN. SFT is normally single
    node, so `world_size` defaults to 1 (whole file); the sharding params are
    kept only for API symmetry with `ShardLoader`.
    """

    IGNORE_INDEX = -100

    def __init__(
        self,
        ids_path: str | Path,
        mask_path: str | Path,
        seq_len: int,
        batch_size: int,
        worker_id: int = 0,
        world_size: int = 1,
        device: str | torch.device = "cpu",
        seed: int = 0,
    ) -> None:
        self.ids_path = Path(ids_path)
        self.mask_path = Path(mask_path)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.worker_id = worker_id
        self.world_size = world_size
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed + worker_id)

        self.ids = np.memmap(self.ids_path, dtype=np.uint16, mode="r")
        self.mask = np.memmap(self.mask_path, dtype=np.uint8, mode="r")
        if len(self.ids) != len(self.mask):
            raise RuntimeError(
                f"ids/mask length mismatch: {len(self.ids)} ids vs "
                f"{len(self.mask)} mask entries (paths {self.ids_path}, {self.mask_path})"
            )

        self.shard_idx = worker_id % world_size
        per = len(self.ids) // world_size
        self.start = self.shard_idx * per
        self.end = self.start + per
        if (self.end - self.start) < seq_len + 1:
            raise RuntimeError(
                f"shard {self.shard_idx} (worker {worker_id}) too small: "
                f"{self.end - self.start} tokens < seq_len+1"
            )

    def _sample_start(self, max_tries: int = 8) -> int:
        """Pick a window start whose shifted target slice has ≥1 assistant
        token. Falls back to the last draw after max_tries (loss-side mean
        still works as long as the *batch* has any target)."""
        lo, hi = self.start, self.end - self.seq_len - 1
        i = lo
        for _ in range(max_tries):
            i = int(self.rng.integers(lo, hi))
            if self.mask[i + 1 : i + 1 + self.seq_len].any():
                return i
        return i

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        ix = [self._sample_start() for _ in range(self.batch_size)]
        x = np.stack([self.ids[i : i + self.seq_len] for i in ix]).astype(np.int64)
        y = np.stack([self.ids[i + 1 : i + 1 + self.seq_len] for i in ix]).astype(np.int64)
        m = np.stack([self.mask[i + 1 : i + 1 + self.seq_len] for i in ix])
        y[m == 0] = self.IGNORE_INDEX  # ignore_index — no gradient on non-assistant tokens
        xt = torch.from_numpy(x)
        yt = torch.from_numpy(y)
        if self.device.type == "cuda":
            xt = xt.pin_memory().to(self.device, non_blocking=True)
            yt = yt.pin_memory().to(self.device, non_blocking=True)
        else:
            xt = xt.to(self.device)
            yt = yt.to(self.device)
        return xt, yt
