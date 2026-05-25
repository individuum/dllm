from __future__ import annotations

import pytest
import torch

from dllm.client.worker import pick_device


def test_pick_device_auto_returns_cpu_when_no_accel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert pick_device("auto").type == "cpu"


def test_pick_device_auto_with_require_gpu_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(SystemExit) as exc:
        pick_device("auto", require_gpu=True)
    assert "GPU CHECK" in str(exc.value)


def test_pick_device_explicit_cpu_with_require_gpu_raises() -> None:
    with pytest.raises(SystemExit) as exc:
        pick_device("cpu", require_gpu=True)
    assert "GPU CHECK" in str(exc.value)


def test_pick_device_cuda_request_fails_on_cpu_only_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """User asks for cuda but their torch is CPU-only — the failure mode that bit us early."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit) as exc:
        pick_device("cuda")
    assert "CPU-only build" in str(exc.value)


def test_pick_device_auto_picks_cuda_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    dev = pick_device("auto")
    assert dev.type == "cuda"
