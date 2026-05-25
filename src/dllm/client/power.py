"""GPU/CPU power telemetry for workers.

Used to report per-round power draw so the coordinator can integrate it
into a cohort-wide energy estimate. The dashboard surfaces this as a
"watts/round" series and a cumulative Wh/kWh/MWh counter — useful both
for the operator (cost) and for the EU AI Act compute-disclosure obligation
that lives next to the 10²⁵ FLOPs threshold.

Telemetry sources (best-effort, never crash):
    - CUDA + pynvml installed: actual instantaneous draw via NVML.
    - CUDA without pynvml: VRAM-tier-based TDP estimate (3060/3090/4090/A100/H100).
    - MPS: Apple-Silicon TDP estimate (powermetrics needs sudo, not usable).
    - CPU: generic desktop CPU TDP estimate.
    - --estimated-watts override: forces a fixed value regardless of device.

Sampling overhead: NVML calls take ~ms — fine to sample every N inner steps.
"""
from __future__ import annotations

import logging

import torch

log = logging.getLogger("dllm.worker.power")

# Fallback whole-package TDP estimates when real telemetry is unavailable.
# Order-of-magnitude — good enough for the dashboard's energy column;
# the operator can override with --estimated-watts for accurate accounting
# on a known box.
_MPS_TDP_W = 35.0  # Apple Silicon (M1 ~20W ... M5 ~50W; midpoint guess)
_CPU_TDP_W = 65.0  # Mid-range desktop CPU


def _cuda_tdp_estimate(device: torch.device) -> float:
    """Crude VRAM-tier → TDP map for CUDA cards when NVML is unavailable."""
    try:
        props = torch.cuda.get_device_properties(device.index or 0)
        vram_gb = props.total_memory / (1024**3)
    except Exception:  # noqa: BLE001
        return 200.0
    # Datacenter
    if vram_gb >= 70:
        return 700.0  # H100 SXM / H200
    if vram_gb >= 35:
        return 400.0  # A100 80GB / 4090
    # Consumer / prosumer
    if vram_gb >= 20:
        return 320.0  # 3090 / 4080
    if vram_gb >= 10:
        return 170.0  # 3060 12GB / 3070 / 4070
    return 75.0  # entry-tier


class PowerMeter:
    """Sample instantaneous power draw on the worker's device.

    Typical use:

        meter = PowerMeter(device)
        ...
        watts = meter.sample()  # in inner loop, every N steps
        ...
        meter.close()  # at shutdown

    `sample()` returns None only if every source failed; normally it
    returns the real-or-estimated wattage so the coord always has data.
    """

    def __init__(
        self,
        device: torch.device,
        override_watts: float | None = None,
    ) -> None:
        self.device = device
        self.override_watts = override_watts
        self._nvml_handle = None
        self._nvml_init = False
        self._init_nvml()

    def _init_nvml(self) -> None:
        if self.device.type != "cuda" or self.override_watts is not None:
            return
        try:
            import pynvml  # type: ignore[import-not-found]
        except ImportError:
            log.info(
                "[POWER] pynvml not installed; power readings will use VRAM-tier "
                "TDP estimate. `pip install nvidia-ml-py` for actual draw."
            )
            return
        try:
            pynvml.nvmlInit()
            idx = self.device.index or 0
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            self._nvml_init = True
            # one read to validate before the first inner loop
            try:
                mw = pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle)
                log.warning(
                    "[POWER] NVML initialized for CUDA device %d (initial: %.1f W)",
                    idx,
                    mw / 1000.0,
                )
            except Exception as e:  # noqa: BLE001
                log.warning("[POWER] NVML init OK but read failed: %s", e)
        except Exception as e:  # noqa: BLE001
            log.warning("[POWER] NVML init failed: %s; falling back to estimate", e)
            self._nvml_handle = None
            self._nvml_init = False

    def sample(self) -> float | None:
        """Return current power draw in watts (or estimate); None if unavailable."""
        if self.override_watts is not None:
            return float(self.override_watts)
        if self._nvml_handle is not None:
            try:
                import pynvml  # type: ignore[import-not-found]

                mw = pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle)
                return mw / 1000.0
            except Exception as e:  # noqa: BLE001
                log.debug("[POWER] NVML sample failed: %s", e)
                # fall through to estimate so the cohort metric stays populated
        if self.device.type == "mps":
            return _MPS_TDP_W
        if self.device.type == "cpu":
            return _CPU_TDP_W
        if self.device.type == "cuda":
            return _cuda_tdp_estimate(self.device)
        return None

    def close(self) -> None:
        if self._nvml_init:
            try:
                import pynvml  # type: ignore[import-not-found]

                pynvml.nvmlShutdown()
            except Exception:  # noqa: BLE001
                pass
            self._nvml_init = False
            self._nvml_handle = None
