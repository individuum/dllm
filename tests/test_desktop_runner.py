"""Unit tests for the desktop WorkerRunner's log-line parsing.

These don't touch PySide6 internals — we just verify the regexes that turn
real worker stdout lines into high-level GUI signals. The patterns are the
contract between worker.py log format and the desktop client's UI.
"""
from __future__ import annotations

import pytest

# Optional: skip the entire module if PySide6 isn't installed (e.g. core
# install without the [desktop] extra). The runner imports PySide6 at
# top-level so we have to gate the import.
PySide6 = pytest.importorskip("PySide6")

from dllm.desktop.worker_runner import (  # noqa: E402
    _RE_INNER,
    _RE_POWER,
    _RE_REGISTERED,
    _RE_REREGISTER,
    _RE_RETUNE,
    _RE_SYNC,
    _RE_VAL,
)


def test_parse_registered_line() -> None:
    line = (
        "2026-05-27 18:11:32,473 dllm.worker INFO "
        "registered worker_id=0 round=78 world_size=1 inner=200 codecs=bf16/q8"
    )
    m = _RE_REGISTERED.search(line)
    assert m is not None
    assert int(m["wid"]) == 0
    assert int(m["round"]) == 78
    assert int(m["ws"]) == 1
    assert int(m["inner"]) == 200


def test_parse_inner_completed_line() -> None:
    line = (
        "2026-05-27 18:25:33,558 dllm.worker INFO "
        "inner round=65 avg_loss=4.0600 steps=200 3680 tok/s peak_vram=11216.2 MiB power=150W"
    )
    m = _RE_INNER.search(line)
    assert m is not None
    assert int(m["round"]) == 65
    assert float(m["loss"]) == 4.06
    assert int(m["steps"]) == 200
    assert float(m["tps"]) == 3680.0
    # Power on the same line — separately matchable.
    mp = _RE_POWER.search(line)
    assert mp is not None
    assert float(mp["watts"]) == 150.0


def test_parse_val_line() -> None:
    line = "2026-05-27 18:25:36,838 dllm.worker INFO val round=65 loss=4.1591"
    m = _RE_VAL.search(line)
    assert m is not None
    assert int(m["round"]) == 65
    assert float(m["loss"]) == pytest.approx(4.1591)


def test_parse_sync_applied_line() -> None:
    line = (
        "2026-05-27 18:25:37,111 dllm.worker INFO "
        "async sync applied round=66 (624024600 state bytes)"
    )
    m = _RE_SYNC.search(line)
    assert m is not None
    assert int(m["round"]) == 66


def test_parse_tier_aware_retune_line() -> None:
    line = (
        "2026-05-27 18:25:37,200 dllm.worker WARNING "
        "[TIER-AWARE] coord re-tuned inner_steps 200 -> 280 (applying next round)"
    )
    m = _RE_RETUNE.search(line)
    assert m is not None
    assert int(m["old"]) == 200
    assert int(m["new"]) == 280


def test_parse_reregister_line() -> None:
    line = (
        "2026-05-27 19:00:00,000 dllm.worker WARNING "
        "[REREGISTER] resumed as worker_id=4 at round=82 (was worker_id=2)"
    )
    m = _RE_REREGISTER.search(line)
    assert m is not None
    assert int(m["wid"]) == 4


def test_unmatched_line_returns_none() -> None:
    """Random output (e.g. nginx 502 noise) should not falsely match."""
    line = "2026-05-27 19:00:00,000 dllm.worker WARNING [HTTP] GET /state got 502 retry 1/4"
    assert _RE_REGISTERED.search(line) is None
    assert _RE_INNER.search(line) is None
    assert _RE_VAL.search(line) is None
    assert _RE_SYNC.search(line) is None
