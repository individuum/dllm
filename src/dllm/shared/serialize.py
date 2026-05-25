from __future__ import annotations

from typing import Iterable, Literal

import torch
from safetensors.torch import load as st_load
from safetensors.torch import save as st_save

from .quantize import dequantize_q8, quantize_q8

StateCodec = Literal["fp32", "bf16"]
DeltaCodec = Literal["fp32", "q8"]


# ---------------------------------------------------------------------------
# raw safetensors I/O
# ---------------------------------------------------------------------------


def state_to_bytes(state: dict[str, torch.Tensor]) -> bytes:
    """Serialize a tensor dict to safetensors bytes (cpu, contiguous)."""
    flat = {k: v.detach().cpu().contiguous() for k, v in state.items()}
    return st_save(flat)


def bytes_to_state(blob: bytes) -> dict[str, torch.Tensor]:
    return st_load(blob)


# ---------------------------------------------------------------------------
# state transport: fp32 (lossless) or bf16 (half-size)
# ---------------------------------------------------------------------------


def serialize_state(state: dict[str, torch.Tensor], codec: StateCodec = "bf16") -> bytes:
    if codec == "fp32":
        return state_to_bytes(state)
    if codec == "bf16":
        bf = {k: v.detach().to(torch.bfloat16).cpu().contiguous() for k, v in state.items()}
        return st_save(bf)
    raise ValueError(f"unknown state codec {codec!r}")


def deserialize_state(blob: bytes, codec: StateCodec = "bf16") -> dict[str, torch.Tensor]:
    if codec not in ("fp32", "bf16"):
        raise ValueError(f"unknown state codec {codec!r}")
    return st_load(blob)


# ---------------------------------------------------------------------------
# delta transport: fp32 (lossless) or q8 (per-tensor symmetric int8)
# ---------------------------------------------------------------------------


def serialize_delta(delta: dict[str, torch.Tensor], codec: DeltaCodec = "q8") -> bytes:
    if codec == "fp32":
        return state_to_bytes(delta)
    if codec == "q8":
        packed = quantize_q8(delta)
        return st_save(packed)
    raise ValueError(f"unknown delta codec {codec!r}")


def deserialize_delta(blob: bytes, codec: DeltaCodec = "q8") -> dict[str, torch.Tensor]:
    if codec == "fp32":
        return st_load(blob)
    if codec == "q8":
        packed = st_load(blob)
        return dequantize_q8(packed)
    raise ValueError(f"unknown delta codec {codec!r}")


# ---------------------------------------------------------------------------
# model <-> tensor-dict helpers
# ---------------------------------------------------------------------------


def model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Trainable params only, in canonical name order. Buffers excluded."""
    return {n: p for n, p in model.named_parameters() if p.requires_grad}


def load_into_model(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    own = dict(model.named_parameters())
    with torch.no_grad():
        for n, p in state.items():
            if n not in own:
                raise KeyError(f"parameter {n} not in model")
            if own[n].shape != p.shape:
                raise ValueError(f"shape mismatch for {n}: model={own[n].shape} blob={p.shape}")
            own[n].copy_(p.to(own[n].dtype).to(own[n].device))


def snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Detached clone of trainable params, kept on the same device for fast diff."""
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def compute_delta(
    snapshot_state: dict[str, torch.Tensor],
    model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    """Pseudo-gradient Δθ = θ_snapshot - θ_local (applying -lr*Δ moves toward local)."""
    delta = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        delta[n] = (snapshot_state[n] - p.detach()).to(p.dtype)
    return delta


def average_deltas(deltas: Iterable[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Simple mean over deltas. Phase 0; production uses trimmed mean (see PLAN §3.2)."""
    deltas = list(deltas)
    if not deltas:
        raise ValueError("no deltas to average")
    out: dict[str, torch.Tensor] = {}
    for name in deltas[0]:
        stacked = torch.stack([d[name].float() for d in deltas], dim=0)
        out[name] = stacked.mean(dim=0)
    return out
