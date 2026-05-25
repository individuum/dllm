from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from dllm.core import PRESETS, ModelConfig, Transformer
from dllm.core.config import TrainConfig


@pytest.fixture(scope="session")
def tiny_cfg() -> ModelConfig:
    """Smallest sensible model for tests — fast on CPU."""
    return ModelConfig(
        vocab_size=512,
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=64,
    )


@pytest.fixture
def tiny_model(tiny_cfg: ModelConfig) -> Transformer:
    torch.manual_seed(0)
    return Transformer(tiny_cfg)


@pytest.fixture(scope="session")
def tiny_train_cfg() -> TrainConfig:
    return TrainConfig(
        seq_len=32,
        micro_batch_size=4,
        inner_steps=3,
        max_outer_rounds=2,
        seed=0,
    )


@pytest.fixture
def tiny_token_file(tmp_path: Path, tiny_cfg: ModelConfig) -> Path:
    """Synthetic uint16 token file just big enough for the loader."""
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, tiny_cfg.vocab_size, size=4096, dtype=np.uint16)
    p = tmp_path / "tiny.bin"
    p.write_bytes(tokens.tobytes())
    return p


