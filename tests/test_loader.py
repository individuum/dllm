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


def test_worker_val_loader_spans_full_file_regardless_of_world_size(
    corpus: Path, tmp_path: Path
) -> None:
    """Regression: in a world_size > 1 cohort, each worker used to validate on
    its own partition of val.bin → per-worker val_losses landed on different
    EU-language distributions and weren't comparable. Worker now hard-codes
    val_loader to world_size=1, worker_id=0 so every worker measures on the
    same distribution.
    """
    from dllm.client.worker import Worker
    from dllm.core import PRESETS
    from dllm.shared.identity import load_or_create_identity

    # Manually wire a Worker (skip register/pull) — we only want to exercise
    # the _ensure_loader_and_opt method's val-loader construction.
    import torch
    load_or_create_identity(tmp_path / "id.key")  # populate identity
    val_corpus = corpus  # reuse the random fixture, big enough for split tests

    # Construct without going through __init__'s registration path
    w = object.__new__(Worker)
    w.preset = "smoke"
    w.cfg = PRESETS["smoke"]
    w.device = torch.device("cpu")
    w.train_data = corpus
    w.val_data = val_corpus
    w.bf16 = False
    w.val_batches = 2
    w.auto_tune_steps = False
    w.target_round_seconds = 90.0
    w.worker_id = 1     # would have given the "second half" shard pre-fix
    w.world_size = 4    # large world size: clearly partitioned territory
    w.current_round = 0
    w.inner_steps = 1
    w.seq_len = 32
    w.micro_batch_size = 4
    w.seed = 0
    w.train_loader = None
    w.val_loader = None
    w.opt = object()  # truthy sentinel so _ensure_loader_and_opt skips AdamW build
    # model not needed for loader construction; sidestep building it
    w.model = type("M", (), {"parameters": lambda self: iter([])})()

    w._ensure_loader_and_opt()
    assert w.val_loader is not None
    # The val_loader spans the WHOLE corpus regardless of worker_id/world_size
    assert w.val_loader.start == 0
    assert w.val_loader.end == len(w.val_loader.tokens)
    assert w.val_loader.shard_idx == 0
    # But the train_loader still partitions normally
    assert w.train_loader.shard_idx == w.worker_id % w.world_size
    assert w.train_loader.start != 0  # worker_id=1 is not the first shard
