"""Per-tensor symmetric int8 quantization for outer-loop pseudo-gradient transport.

Trade-off: deltas are small magnitudes well-suited to int8; quantization error
on the delta is much less harmful than the same error on the model state.
We keep state in bf16/fp32 and only int8 the deltas.

Encoding for a tensor T:
  scale = max(|T|) / 127  (per-tensor)
  q     = clamp(round(T / scale), -127, 127).to(int8)

Decoding: T_hat = q.float() * scale.

In safetensors, each quantized tensor is stored as two entries:
  "<name>"        int8
  "<name>.scale"  float32 (0-d tensor)

Empty tensors / zero tensors are stored with scale=0 and skipped on dequantize.
"""
from __future__ import annotations

import torch


def quantize_q8(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Per-tensor symmetric int8 quantization. Returns dict with q + scale entries."""
    out: dict[str, torch.Tensor] = {}
    for name, t in state.items():
        t = t.detach().to(torch.float32).cpu()
        absmax = float(t.abs().max().item()) if t.numel() else 0.0
        if absmax == 0.0:
            out[name] = torch.zeros_like(t, dtype=torch.int8)
            out[f"{name}.scale"] = torch.tensor(0.0, dtype=torch.float32)
            continue
        scale = absmax / 127.0
        q = (t / scale).round().clamp(-127, 127).to(torch.int8)
        out[name] = q
        out[f"{name}.scale"] = torch.tensor(scale, dtype=torch.float32)
    return out


def dequantize_q8(packed: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Inverse of quantize_q8."""
    out: dict[str, torch.Tensor] = {}
    for name, t in packed.items():
        if name.endswith(".scale"):
            continue
        scale_name = f"{name}.scale"
        if scale_name not in packed:
            raise KeyError(f"missing scale for {name!r}")
        scale = float(packed[scale_name].item())
        if scale == 0.0:
            out[name] = torch.zeros_like(t, dtype=torch.float32)
            continue
        out[name] = t.to(torch.float32) * scale
    return out


def q8_size_bytes(packed: dict[str, torch.Tensor]) -> int:
    """Helper for telemetry: total bytes of an int8-packed state."""
    total = 0
    for t in packed.values():
        total += t.numel() * t.element_size()
    return total
