from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dllm.data.loader import ShardLoader


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, 50000, size=10_000, dtype=np.uint16)
    p = tmp_path / "tokens.bin"
    p.write_bytes(tokens.tobytes())
    return p


def test_basic_batch_shape(corpus: Path) -> None:
    loader = ShardLoader(corpus, seq_len=64, batch_size=8)
    x, y = loader.next_batch()
    assert x.shape == (8, 64)
    assert y.shape == (8, 64)


def test_world_size_two_partitions(corpus: Path) -> None:
    a = ShardLoader(corpus, seq_len=32, batch_size=4, worker_id=0, world_size=2)
    b = ShardLoader(corpus, seq_len=32, batch_size=4, worker_id=1, world_size=2)
    assert a.start == 0
    assert b.start == a.end
    assert b.end == len(a.tokens)


def test_worker_id_exceeding_world_size_wraps(corpus: Path) -> None:
    """Regression: workers persist with incrementing ids across runs; loader must wrap."""
    a = ShardLoader(corpus, seq_len=32, batch_size=4, worker_id=5, world_size=2)
    b = ShardLoader(corpus, seq_len=32, batch_size=4, worker_id=1, world_size=2)
    # worker_id 5 mod 2 = 1, same shard as worker_id 1
    assert a.shard_idx == 1
    assert a.start == b.start
    assert a.end == b.end
    x, _ = a.next_batch()
    assert x.shape == (4, 32)
    assert x.numel() > 0


def test_single_worker_modulo_handles_any_id(corpus: Path) -> None:
    """world_size=1: every worker_id falls in shard 0 and produces non-empty batches."""
    for wid in (0, 1, 5, 100):
        loader = ShardLoader(corpus, seq_len=16, batch_size=2, worker_id=wid, world_size=1)
        assert loader.shard_idx == 0
        x, _ = loader.next_batch()
        assert x.shape == (2, 16)
        assert x.numel() == 32


def test_too_small_shard_raises(corpus: Path) -> None:
    with pytest.raises(RuntimeError, match="too small"):
        ShardLoader(corpus, seq_len=20_000, batch_size=1)
