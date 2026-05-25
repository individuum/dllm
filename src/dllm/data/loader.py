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
