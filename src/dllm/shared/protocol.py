from __future__ import annotations

from pydantic import BaseModel, Field


class RegisterRequest(BaseModel):
    pubkey: str = Field(..., description="Ed25519 pubkey hex (placeholder in Phase 0)")
    country: str = Field("XX", description="ISO 3166-1 alpha-2; attestation only in Phase 0")
    gpu: str = Field("unknown")
    vram_gb: int = 0
    ram_gb: int = 0
    preset: str = Field("smoke", description="Model size preset; must match coordinator config")


class RegisterResponse(BaseModel):
    worker_id: int
    current_round: int
    world_size: int  # expected number of workers in the cohort
    seed: int
    inner_steps: int
    seq_len: int
    micro_batch_size: int
    state_codec: str = "bf16"  # "fp32" or "bf16"
    delta_codec: str = "q8"  # "fp32" or "q8"
    require_signed_deltas: bool = False  # if True, worker must sign /delta bodies


class RoundStatus(BaseModel):
    current_round: int
    n_registered: int
    n_submitted: int
    waiting_for: int  # world_size - n_submitted
    last_val_loss: float | None = None  # mean val loss across workers from the previous round
    flops_total: float = 0.0  # cumulative training FLOPs estimate
    round_open_seconds: float = 0.0  # wall-clock the current round has been open
    round_timeout_seconds: float = 0.0  # coord-configured eviction timeout (0 = disabled)
    min_workers: int = 1  # min deltas needed to force-advance on timeout


class DeltaAck(BaseModel):
    accepted: bool
    reason: str = ""
    next_round: int | None = None  # set when outer step happens immediately
