"""Test the EU corpus prep pipeline with synthetic inputs (no network).

Network-dependent fetch is covered by the live `dllm.data.prepare` CLI.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("tokenizers", reason="data extras not installed; run `pip install -e .[data]`")


def _synthetic_corpus() -> dict[str, list[str]]:
    """Tiny multilingual corpus — enough to exercise BPE training in <1s."""
    return {
        "en": [
            "the quick brown fox jumps over the lazy dog. " * 50,
            "to be or not to be that is the question " * 50,
        ] * 8,
        "de": [
            "der schnelle braune fuchs springt über den faulen hund. " * 50,
            "sein oder nicht sein das ist hier die frage " * 50,
        ] * 8,
        "fr": [
            "le rapide renard brun saute par-dessus le chien paresseux. " * 50,
            "être ou ne pas être telle est la question " * 50,
        ] * 8,
    }


def test_bpe_training_produces_tokenizer() -> None:
    from dllm.data.prepare import EOT_TOKEN, train_bpe

    tok = train_bpe(_synthetic_corpus(), vocab_size=512)
    assert tok.get_vocab_size() <= 512
    assert tok.token_to_id(EOT_TOKEN) is not None


def test_bpe_roundtrip() -> None:
    from dllm.data.prepare import train_bpe

    tok = train_bpe(_synthetic_corpus(), vocab_size=512)
    text = "the quick brown fox"
    enc = tok.encode(text)
    dec = tok.decode(enc.ids)
    assert dec.strip() == text


def test_tokenize_and_shard_produces_bins(tmp_path: Path) -> None:
    from dllm.data.prepare import tokenize_and_shard, train_bpe

    corpus = _synthetic_corpus()
    tok = train_bpe(corpus, vocab_size=512)
    info = tokenize_and_shard(corpus, tok, tmp_path, val_frac=0.1)
    train_bin = tmp_path / "train.bin"
    val_bin = tmp_path / "val.bin"
    assert train_bin.exists() and val_bin.exists()
    assert info["train_tokens"] > 0
    assert info["val_tokens"] > 0
    assert set(info["tokens_per_lang"]) == set(corpus)


def test_tokenize_dtype_picked_by_vocab_size(tmp_path: Path) -> None:
    """vocab <= 65535 → uint16; otherwise uint32."""
    import numpy as np

    from dllm.data.prepare import tokenize_and_shard, train_bpe

    corpus = _synthetic_corpus()
    tok = train_bpe(corpus, vocab_size=512)
    info = tokenize_and_shard(corpus, tok, tmp_path, val_frac=0.1)
    assert info["dtype"] == np.uint16.__name__


def test_default_tokenizer_path_under_cache() -> None:
    from dllm.data.tokenizer import default_tokenizer_path

    p = default_tokenizer_path()
    assert p.name == "tokenizer.json"
    assert "cache" in p.parts


def test_interleave_round_robins_langs() -> None:
    """First N docs of the interleaved stream cover all langs (one each)."""
    from dllm.data.prepare import _interleave_docs

    corpus = {"de": ["de-1", "de-2", "de-3"], "fr": ["fr-1", "fr-2"], "en": ["en-1"]}
    out = _interleave_docs(corpus)
    first_three = [lang for lang, _ in out[:3]]
    assert sorted(first_three) == sorted(corpus)


def test_interleave_preserves_all_docs() -> None:
    from dllm.data.prepare import _interleave_docs

    corpus = {"a": ["a1", "a2"], "b": ["b1", "b2", "b3"], "c": ["c1"]}
    out = _interleave_docs(corpus)
    assert len(out) == sum(len(d) for d in corpus.values())
    assert {doc for _, doc in out} == {d for docs in corpus.values() for d in docs}


def test_interleave_balances_any_contiguous_slice() -> None:
    """A contiguous half of the interleaved stream should have every lang represented."""
    from dllm.data.prepare import _interleave_docs

    corpus = {f"l{i}": [f"l{i}-doc{j}" for j in range(20)] for i in range(5)}
    out = _interleave_docs(corpus)
    n = len(out)
    first_half = {lang for lang, _ in out[: n // 2]}
    second_half = {lang for lang, _ in out[n // 2 :]}
    assert first_half == set(corpus)
    assert second_half == set(corpus)
