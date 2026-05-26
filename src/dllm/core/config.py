from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 32768
    dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 4
    ffn_mult: float = 8 / 3
    ffn_multiple_of: int = 64
    max_seq_len: int = 1024
    rope_theta: float = 500000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TrainConfig:
    seq_len: int = 1024
    micro_batch_size: int = 8
    grad_accum_steps: int = 1
    inner_steps: int = 50  # per outer cycle; ~500 for production, smaller for smoke test
    inner_lr: float = 3e-4
    inner_lr_warmup: int = 20
    inner_weight_decay: float = 0.1
    inner_grad_clip: float = 1.0
    outer_lr: float = 0.7
    outer_momentum: float = 0.9
    max_outer_rounds: int = 50
    bf16: bool = True
    seed: int = 0


PRESETS: dict[str, ModelConfig] = {
    # Smoke-test sized: ~10M params, trains in minutes on CPU, seconds on GPU
    "smoke": ModelConfig(
        dim=128,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=512,
    ),
    # ~124M, GPT-2-small equivalent for Phase 0 validation
    "124M": ModelConfig(
        dim=768,
        n_layers=12,
        n_heads=12,
        n_kv_heads=4,
        max_seq_len=1024,
    ),
    # ~300M, GPT-2-medium class — sweet spot for an 11.7 GB consumer
    # GPU (3060 / 4060 Ti 16G / 4070): model state ~4.8 GB, activations
    # ~5 GB at batch=4 seq=512 bf16, total ~10 GB with comfortable
    # headroom. Roughly Chinchilla-optimal for our ~5 B-token corpus.
    "300M": ModelConfig(
        dim=1024,
        n_layers=24,
        n_heads=16,
        n_kv_heads=4,
        max_seq_len=1024,
    ),
    # ~1B, Phase 1 target
    "1B": ModelConfig(
        dim=2048,
        n_layers=24,
        n_heads=16,
        n_kv_heads=4,
        max_seq_len=2048,
    ),
    # ~7B, Phase 2 target
    "7B": ModelConfig(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        max_seq_len=4096,
    ),
}
