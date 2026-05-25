from __future__ import annotations

import torch

from dllm.shared.quantize import dequantize_q8, q8_size_bytes, quantize_q8


def test_q8_roundtrip_accuracy() -> None:
    torch.manual_seed(0)
    state = {
        "a": torch.randn(256, 64),
        "b": torch.randn(1000) * 0.01,  # small magnitudes
        "c": torch.randn(8, 8) * 100.0,  # large magnitudes
    }
    packed = quantize_q8(state)
    restored = dequantize_q8(packed)
    for name in state:
        # max relative error should be bounded by 1/127
        absmax = state[name].abs().max().item()
        err = (state[name] - restored[name]).abs().max().item()
        # symmetric int8: worst case ≈ absmax/127 due to round-to-nearest
        assert err <= absmax / 127 + 1e-5, f"{name}: err={err}, absmax={absmax}"


def test_q8_handles_zero_tensor() -> None:
    state = {"z": torch.zeros(10, 10)}
    packed = quantize_q8(state)
    restored = dequantize_q8(packed)
    assert torch.equal(restored["z"], torch.zeros(10, 10))


def test_q8_size_is_about_quarter_of_fp32() -> None:
    state = {"a": torch.randn(1000, 256)}
    packed = quantize_q8(state)
    fp32_bytes = 1000 * 256 * 4
    q8_bytes = q8_size_bytes(packed)
    # int8 payload + 4-byte scale per tensor
    assert q8_bytes < fp32_bytes / 3.5
    assert q8_bytes > fp32_bytes / 4.5


def test_q8_packed_keys_have_scale_entries() -> None:
    state = {"layer.weight": torch.randn(10, 10), "layer.bias": torch.randn(10)}
    packed = quantize_q8(state)
    assert "layer.weight" in packed and "layer.weight.scale" in packed
    assert "layer.bias" in packed and "layer.bias.scale" in packed
    assert packed["layer.weight"].dtype == torch.int8
    assert packed["layer.weight.scale"].dtype == torch.float32
