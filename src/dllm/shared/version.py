"""Protocol version handshake — the "magic hash" every client sends at
/register so the coordinator can reject clients running incompatible code.

`PROTOCOL_VERSION` is a short deterministic hash of (package version,
COMPAT_EPOCH). The worker sends it in RegisterRequest; the coord compares it to
its own and refuses registration on a mismatch (HTTP 426 Upgrade Required) with
an actionable "git pull && pip install -e ." message. Both sides import this
module, so identical code → identical hash; an out-of-date client computes a
different hash (or sends none) and is turned away before it can submit deltas
built against the wrong assumptions.

WHEN TO BUMP COMPAT_EPOCH
    Bump it whenever a change makes an OLDER client's deltas / state / wire
    messages incompatible with this coord — a wire-format change, a delta/state
    codec change, a NON-equivalent model-architecture or training-convention
    change — or any time you simply want to force the whole cohort onto the new
    build before it can contribute.

    Do NOT bump for numerically-equivalent refactors (e.g. the complex->real
    RoPE rewrite, proven identical by test_rope_real_matches_complex): those
    clients interoperate fine and a bump would needlessly lock them out.
"""
from __future__ import annotations

import hashlib

DLLM_VERSION = "0.0.1"

# Compatibility epoch — see the module docstring for the bump policy.
# 1: first versioned generation (post tier-aware tokens_per_step, real RoPE,
#    seq_len override, masked-SFT). Clients must be on this build or newer.
COMPAT_EPOCH = 1


def compute_protocol_version(version: str = DLLM_VERSION, epoch: int = COMPAT_EPOCH) -> str:
    """Deterministic 12-hex 'magic hash' of the compatibility identity."""
    return hashlib.sha256(f"dllm-proto/v{version}/epoch{epoch}".encode()).hexdigest()[:12]


PROTOCOL_VERSION = compute_protocol_version()
