"""Tests for the slow-device viability gate + background heartbeat.

These target the two operational fixes for "Apple Silicon can't join 300M":
1. estimate_inner_loop_seconds + the viability gate that exits a too-slow
   device cleanly instead of letting it register and get round-timeout-evicted.
2. _start_heartbeat: keeps the coord registration alive during long no-/delta
   stretches (state pull, benchmark, slow first inner loop).
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest
import torch

from dllm.client.worker import (
    Worker,
    compute_inner_steps_for_target,
    estimate_inner_loop_seconds,
)


def _fake_register_response(seq_len: int = 4096, micro_batch: int = 2) -> MagicMock:
    r = MagicMock()
    r.status_code = 200
    r.raise_for_status = MagicMock()
    r.json.return_value = {
        "worker_id": 1, "world_size": 1, "current_round": 0, "inner_steps": 50,
        "seq_len": seq_len, "micro_batch_size": micro_batch, "seed": 0,
        "state_codec": "bf16", "delta_codec": "q8",
    }
    return r


def _bare_worker(seq_len_override=None, micro_override=None) -> Worker:
    w = object.__new__(Worker)
    w.device = torch.device("cpu")
    w.pubkey_hex = "deadbeef"
    w.country = "DE"
    w.preset = "300M"
    w.micro_batch_size_override = micro_override
    w.seq_len_override = seq_len_override
    w.http = MagicMock()
    return w


def test_estimate_inner_loop_seconds_basic() -> None:
    # 1000 tok/s, 50 steps, batch 2, seq 4096 → 50*8192/1000 = 409.6 s
    assert estimate_inner_loop_seconds(1000.0, 50, 2, 4096) == pytest.approx(409.6)
    # zero/negative tok/s clamps the divisor to 1.0 (no ZeroDivision)
    assert estimate_inner_loop_seconds(0.0, 10, 1, 16) == pytest.approx(160.0)


def test_viability_gate_separates_slow_from_viable_m5() -> None:
    """Reproduces the bug report's measured M5 numbers against the live 300M
    config (target=300s, batch=2, seq=4096, viability_factor=4 → ceiling 1200s).

    Pre-RoPE-fix 324 tok/s: forced to the 50-step floor, one loop ≈1264s > 1200
    → gated out (exits cleanly). A recovered ~970 tok/s: one 50-step loop ≈422s
    < 1200 → allowed to contribute. So the gate excludes the genuinely hopeless
    case without excluding a marginal-but-useful device.
    """
    target, batch, seq, factor = 300.0, 2, 4096, 4.0
    ceiling = factor * target

    slow_tps = 324.0
    slow_steps = compute_inner_steps_for_target(slow_tps, target, batch, seq)  # clamps to 50
    slow_loop = estimate_inner_loop_seconds(slow_tps, slow_steps, batch, seq)
    assert slow_steps == 50
    assert slow_loop > ceiling  # gated out

    fast_tps = 970.0
    fast_steps = compute_inner_steps_for_target(fast_tps, target, batch, seq)
    fast_loop = estimate_inner_loop_seconds(fast_tps, fast_steps, batch, seq)
    assert fast_loop <= ceiling  # allowed


def test_heartbeat_pings_status_then_stops() -> None:
    w = object.__new__(Worker)
    w.worker_id = 5
    w._hb_thread = None
    w._hb_stop = None
    w.http = MagicMock()

    w._start_heartbeat(interval=0.02)
    time.sleep(0.15)
    assert w.http.get.call_count >= 2, "heartbeat should ping repeatedly"
    # pings hit /status with our worker_id (same mechanism the coord treats as
    # a liveness refresh)
    args, kwargs = w.http.get.call_args
    assert args[0] == "/status"
    assert kwargs["params"] == {"worker_id": 5}

    w._stop_heartbeat()
    time.sleep(0.05)
    settled = w.http.get.call_count
    time.sleep(0.1)
    assert w.http.get.call_count == settled, "no pings after stop"


def test_heartbeat_noop_without_worker_id() -> None:
    w = object.__new__(Worker)
    w.worker_id = None
    w._hb_thread = None
    w._hb_stop = None
    w.http = MagicMock()
    w._start_heartbeat(interval=0.01)
    time.sleep(0.05)
    assert w.http.get.call_count == 0  # never starts without a worker_id
    assert w._hb_thread is None


# ---- seq_len override (the biggest compute lever for slow devices) ---------


def test_seq_len_override_caps_downward() -> None:
    w = _bare_worker(seq_len_override=1024)
    w.http.post.return_value = _fake_register_response(seq_len=4096)
    w.register()
    assert w.seq_len == 1024  # capped from the coord's 4096
    # tokens_per_step reported to the coord (#67) reflects the capped value
    assert w.micro_batch_size * w.seq_len == 2 * 1024


def test_seq_len_override_is_one_way_down_only() -> None:
    w = _bare_worker(seq_len_override=8192)  # larger than coord's → ignored
    w.http.post.return_value = _fake_register_response(seq_len=4096)
    w.register()
    assert w.seq_len == 4096


def test_no_seq_len_override_uses_coord_value() -> None:
    w = _bare_worker(seq_len_override=None)
    w.http.post.return_value = _fake_register_response(seq_len=4096)
    w.register()
    assert w.seq_len == 4096
