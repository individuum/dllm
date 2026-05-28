"""Tests for the protocol-version 'magic hash' (dllm.shared.version)."""
from __future__ import annotations

from dllm.shared.version import (
    COMPAT_EPOCH,
    DLLM_VERSION,
    PROTOCOL_VERSION,
    compute_protocol_version,
)


def test_protocol_version_is_short_hex() -> None:
    assert len(PROTOCOL_VERSION) == 12
    int(PROTOCOL_VERSION, 16)  # must be valid hex


def test_protocol_version_is_deterministic() -> None:
    # Same inputs → same hash, every process. This is what lets coord and
    # worker agree without exchanging anything but the string.
    assert compute_protocol_version() == PROTOCOL_VERSION
    assert compute_protocol_version(DLLM_VERSION, COMPAT_EPOCH) == PROTOCOL_VERSION
    assert compute_protocol_version("0.0.1", 1) == compute_protocol_version("0.0.1", 1)


def test_epoch_bump_changes_hash() -> None:
    # Bumping the compatibility epoch (a breaking change) yields a different
    # hash, so older clients are rejected.
    assert compute_protocol_version("0.0.1", 1) != compute_protocol_version("0.0.1", 2)


def test_version_bump_changes_hash() -> None:
    assert compute_protocol_version("0.0.1", 1) != compute_protocol_version("0.0.2", 1)
