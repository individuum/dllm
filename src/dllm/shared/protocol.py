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
    # Energy / cohort throughput. Both populated as soon as workers start
    # reporting power_watts + tokens_per_sec on /delta.
    energy_wh_total: float = 0.0  # cumulative Wh used by the cohort
    last_power_watts: float | None = None  # COHORT sum of last round's draws
    last_power_watts_per_worker: float | None = None  # mean for reference
    last_n_reporting_workers: int = 0  # how many workers' power numbers fed the sum
    last_tokens_per_sec: float | None = None  # summed cohort throughput, previous round


class WorkerInfo(BaseModel):
    worker_id: int
    country: str
    gpu: str
    vram_gb: int = 0
    ram_gb: int = 0
    registered_at: float
    rounds_contributed: int = 0
    last_seen_ts: float | None = None
    last_round: int | None = None  # last round this worker submitted a delta for
    last_val_loss: float | None = None
    last_power_watts: float | None = None
    last_tokens_per_sec: float | None = None


class WorkersResponse(BaseModel):
    workers: list[WorkerInfo]


class DeltaAck(BaseModel):
    accepted: bool
    reason: str = ""
    next_round: int | None = None  # set when outer step happens immediately
