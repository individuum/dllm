"""Ed25519 identity for workers — Phase 1 Byzantine-prep.

Each worker holds a persistent Ed25519 keypair. The pubkey is registered with
the coordinator; every delta submission carries a signature binding worker_id +
round + content hash. Coord rejects any delta whose signature does not verify
against the pubkey recorded at registration.

This rules out:
- Spoofing another worker's id (signature won't verify with attacker's key).
- Replaying a delta to a different round.
- Tampering with the delta body in transit (different sha256, sig mismatches).

Production hardening builds on this: trust scoring (PLAN §3.7), redundant
assignment, TOPLOC for inference rollouts. mTLS at nginx is a separate layer.
"""
from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

# canonical signed messages
DELTA_MESSAGE_PREFIX = "dllm-delta"  # "dllm-delta:{worker_id}:{round}:{sha256(body)}"
DEREGISTER_MESSAGE_PREFIX = "dllm-deregister"  # "dllm-deregister:{worker_id}:{ts_unix}"


def default_identity_path() -> Path:
    """Project-local identity (Phase 1). Production should move to per-user."""
    here = Path(__file__).resolve().parent
    repo_root = here.parent.parent.parent
    return repo_root / ".dllm" / "identity.key"


def load_or_create_identity(path: str | Path | None = None) -> Ed25519PrivateKey:
    p = Path(path) if path else default_identity_path()
    if p.exists():
        raw = p.read_bytes()
        return Ed25519PrivateKey.from_private_bytes(raw)
    p.parent.mkdir(parents=True, exist_ok=True)
    sk = Ed25519PrivateKey.generate()
    raw = sk.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    # tighten perms on POSIX; Windows ACLs are user's problem
    if os.name != "nt":
        p.write_bytes(raw)
        os.chmod(p, 0o600)
    else:
        p.write_bytes(raw)
    return sk


def pubkey_hex(sk: Ed25519PrivateKey) -> str:
    raw = sk.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return raw.hex()


def pubkey_from_hex(s: str) -> Ed25519PublicKey:
    return Ed25519PublicKey.from_public_bytes(bytes.fromhex(s))


def canonical_delta_message(worker_id: int, round_no: int, body: bytes) -> bytes:
    """The exact bytes a worker signs for a /delta submission."""
    digest = hashlib.sha256(body).hexdigest()
    return f"{DELTA_MESSAGE_PREFIX}:{worker_id}:{round_no}:{digest}".encode("utf-8")


def sign_delta(sk: Ed25519PrivateKey, worker_id: int, round_no: int, body: bytes) -> str:
    msg = canonical_delta_message(worker_id, round_no, body)
    sig = sk.sign(msg)
    return base64.b64encode(sig).decode("ascii")


def verify_delta(
    pubkey: Ed25519PublicKey,
    worker_id: int,
    round_no: int,
    body: bytes,
    signature_b64: str,
) -> bool:
    try:
        sig = base64.b64decode(signature_b64)
    except (ValueError, TypeError):
        return False
    msg = canonical_delta_message(worker_id, round_no, body)
    try:
        pubkey.verify(sig, msg)
        return True
    except InvalidSignature:
        return False


def canonical_deregister_message(worker_id: int, ts_unix: int) -> bytes:
    """Bytes signed for a voluntary /deregister request.

    Includes a unix timestamp so coord can reject replayed signatures
    older than a small window — defeats an attacker who sniffs a stale
    deregister and tries to re-use it later.
    """
    return f"{DEREGISTER_MESSAGE_PREFIX}:{worker_id}:{ts_unix}".encode("utf-8")


def sign_deregister(sk: Ed25519PrivateKey, worker_id: int, ts_unix: int) -> str:
    sig = sk.sign(canonical_deregister_message(worker_id, ts_unix))
    return base64.b64encode(sig).decode("ascii")


def verify_deregister(
    pubkey: Ed25519PublicKey,
    worker_id: int,
    ts_unix: int,
    signature_b64: str,
) -> bool:
    try:
        sig = base64.b64decode(signature_b64)
    except (ValueError, TypeError):
        return False
    try:
        pubkey.verify(sig, canonical_deregister_message(worker_id, ts_unix))
        return True
    except InvalidSignature:
        return False
