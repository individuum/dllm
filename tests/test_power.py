"""Power meter behavior on each device type."""
from __future__ import annotations

import torch

from dllm.client.power import PowerMeter


def test_power_meter_cpu_returns_estimate() -> None:
    pm = PowerMeter(torch.device("cpu"))
    w = pm.sample()
    # CPU TDP estimate is hardcoded — should always be a positive float
    assert w is not None
    assert w > 0
    pm.close()  # idempotent / no-op when nvml never inited


def test_power_meter_override_wins() -> None:
    """--estimated-watts should pin the reading regardless of device."""
    pm = PowerMeter(torch.device("cpu"), override_watts=42.5)
    assert pm.sample() == 42.5
    pm.close()


def test_power_meter_close_is_idempotent() -> None:
    pm = PowerMeter(torch.device("cpu"))
    pm.close()
    pm.close()  # second call must not raise
