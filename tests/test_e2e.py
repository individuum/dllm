"""Slow end-to-end test: real subprocess smoke test.

Run with: pytest -m slow tests/test_e2e.py
"""
from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.slow
def test_smoke_runs_and_passes() -> None:
    rc = subprocess.run(
        [
            sys.executable,
            "-m",
            "dllm.scripts.smoke_test",
            "--max-rounds", "2",
            "--inner-steps", "5",
            "--micro-batch-size", "4",
            "--seq-len", "64",
        ],
        check=False,
    ).returncode
    assert rc == 0
