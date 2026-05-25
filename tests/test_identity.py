from __future__ import annotations

from pathlib import Path

import pytest

from dllm.shared.identity import (
    canonical_delta_message,
    load_or_create_identity,
    pubkey_from_hex,
    pubkey_hex,
    sign_delta,
    verify_delta,
)


def test_create_persists_key(tmp_path: Path) -> None:
    p = tmp_path / "id.key"
    sk1 = load_or_create_identity(p)
    sk2 = load_or_create_identity(p)
    assert pubkey_hex(sk1) == pubkey_hex(sk2)


def test_pubkey_hex_roundtrip(tmp_path: Path) -> None:
    sk = load_or_create_identity(tmp_path / "id.key")
    pk = sk.public_key()
    hex_str = pubkey_hex(sk)
    assert len(hex_str) == 64  # 32 raw bytes → 64 hex chars
    pk2 = pubkey_from_hex(hex_str)
    msg = b"hello"
    sig = sk.sign(msg)
    pk2.verify(sig, msg)  # would raise if mismatch


def test_sign_and_verify_delta_happy_path(tmp_path: Path) -> None:
    sk = load_or_create_identity(tmp_path / "id.key")
    pk = sk.public_key()
    body = b"some-safetensors-blob" * 100
    sig = sign_delta(sk, worker_id=3, round_no=7, body=body)
    assert verify_delta(pk, 3, 7, body, sig)


def test_verify_rejects_tampered_body(tmp_path: Path) -> None:
    sk = load_or_create_identity(tmp_path / "id.key")
    pk = sk.public_key()
    body = b"original blob"
    sig = sign_delta(sk, worker_id=0, round_no=0, body=body)
    assert not verify_delta(pk, 0, 0, body + b"tampered", sig)


def test_verify_rejects_replay_to_different_round(tmp_path: Path) -> None:
    sk = load_or_create_identity(tmp_path / "id.key")
    pk = sk.public_key()
    body = b"blob"
    sig_round5 = sign_delta(sk, worker_id=0, round_no=5, body=body)
    assert not verify_delta(pk, 0, 6, body, sig_round5)
    assert not verify_delta(pk, 1, 5, body, sig_round5)


def test_verify_rejects_signature_from_different_key(tmp_path: Path) -> None:
    sk_a = load_or_create_identity(tmp_path / "a.key")
    sk_b = load_or_create_identity(tmp_path / "b.key")
    body = b"data"
    sig = sign_delta(sk_a, 0, 0, body)
    assert not verify_delta(sk_b.public_key(), 0, 0, body, sig)


def test_verify_rejects_malformed_signature(tmp_path: Path) -> None:
    sk = load_or_create_identity(tmp_path / "id.key")
    pk = sk.public_key()
    assert not verify_delta(pk, 0, 0, b"x", "not-base64-at-all!!!")
    assert not verify_delta(pk, 0, 0, b"x", "")


def test_canonical_message_format() -> None:
    body = b"hello"
    msg = canonical_delta_message(2, 5, body)
    # binds prefix + worker_id + round + content hash
    text = msg.decode()
    assert text.startswith("dllm-delta:2:5:")
    assert len(text.split(":")[-1]) == 64  # sha256 hex
