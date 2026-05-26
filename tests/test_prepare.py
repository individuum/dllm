"""Test the multi-source EU corpus prep pipeline.

Network-dependent source fetches are NOT exercised here — they're covered
by the live `dllm.data.prepare --dry-run` CLI. These tests use synthetic
sources to verify the orchestration: BPE training, streaming
tokenization, train/val split, round-robin source interleaving, and
provenance manifest shape.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pytest

pytest.importorskip("tokenizers", reason="data extras not installed; run `pip install -e .[data]`")


def _synthetic_sample() -> list[str]:
    """Tiny multilingual sample — enough to exercise BPE training in <1s."""
    return [
        "the quick brown fox jumps over the lazy dog. " * 50,
        "to be or not to be that is the question " * 50,
        "der schnelle braune fuchs springt über den faulen hund. " * 50,
        "sein oder nicht sein das ist hier die frage " * 50,
        "le rapide renard brun saute par-dessus le chien paresseux. " * 50,
        "être ou ne pas être telle est la question " * 50,
    ] * 4


def test_bpe_training_produces_tokenizer() -> None:
    from dllm.data.prepare import EOT_TOKEN, train_bpe

    tok = train_bpe(_synthetic_sample(), vocab_size=512)
    assert tok.get_vocab_size() <= 512
    assert tok.token_to_id(EOT_TOKEN) is not None


def test_bpe_roundtrip() -> None:
    from dllm.data.prepare import train_bpe

    tok = train_bpe(_synthetic_sample(), vocab_size=512)
    text = "the quick brown fox"
    enc = tok.encode(text)
    dec = tok.decode(enc.ids)
    assert dec.strip() == text


def test_shard_writer_appends_and_flushes(tmp_path: Path) -> None:
    """ShardWriter accumulates uint16 tokens and flushes in chunks."""
    from dllm.data.prepare import ShardWriter

    path = tmp_path / "shard.bin"
    # Small flush size so we exercise the rollover path
    w = ShardWriter(path, flush_tokens=10)
    w.append([1, 2, 3])
    w.append([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])  # forces flush mid-call
    total = w.close()
    assert total == 15
    arr = np.fromfile(path, dtype=np.uint16)
    assert arr.tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]


def test_split_off_val_takes_last_n_tokens(tmp_path: Path) -> None:
    from dllm.data.prepare import split_off_val

    train = tmp_path / "train.bin"
    val = tmp_path / "val.bin"
    n_total = 1000
    np.arange(n_total, dtype=np.uint16).tofile(train)
    train_n, val_n = split_off_val(train, val, val_tokens=50)
    assert train_n == 950
    assert val_n == 50
    val_arr = np.fromfile(val, dtype=np.uint16)
    train_arr = np.fromfile(train, dtype=np.uint16)
    # Val is the LAST 50 tokens of the original stream
    assert val_arr.tolist() == list(range(950, 1000))
    # Train is the first 950 tokens
    assert train_arr.tolist() == list(range(950))


def test_split_off_val_caps_at_5pct(tmp_path: Path) -> None:
    """Even if val_tokens is huge, val is capped at 5 % of corpus."""
    from dllm.data.prepare import split_off_val

    train = tmp_path / "train.bin"
    val = tmp_path / "val.bin"
    n_total = 100
    np.arange(n_total, dtype=np.uint16).tofile(train)
    # Ask for 90 val tokens, but corpus is 100 → cap at 100/20 = 5
    train_n, val_n = split_off_val(train, val, val_tokens=90)
    assert val_n == 5
    assert train_n == 95


# ---------------------------------------------------------------------------
# Stream corpus orchestration (with fake sources)
# ---------------------------------------------------------------------------


class _FakeSource:
    """In-memory source for testing stream_corpus + tokenize_streaming."""

    def __init__(self, by_lang: dict[str, list[str]]) -> None:
        self.by_lang = by_lang

    def iter_docs(self, lang: str, char_budget: int) -> Iterator[str]:
        emitted = 0
        for doc in self.by_lang.get(lang, []):
            if emitted >= char_budget:
                return
            yield doc
            emitted += len(doc)

    def license_info(self) -> dict:
        return {"name": "fake", "license": "test"}

    def supported_langs(self) -> list[str]:
        return list(self.by_lang)


def _install_fake_sources(monkeypatch, fakes: dict[str, _FakeSource]) -> None:
    """Patch dllm.data.sources.load + .all_source_names to return fakes."""
    from dllm.data import sources as src_pkg

    def fake_load(name: str):
        if name in fakes:
            return fakes[name]
        raise ValueError(f"unknown source {name!r}")

    monkeypatch.setattr(src_pkg, "load", fake_load)
    monkeypatch.setattr(src_pkg, "all_source_names", lambda: tuple(fakes))


def test_stream_corpus_round_robins_sources_and_langs(monkeypatch) -> None:
    """Round-robin order: every (source, lang) pair gets one doc before any
    pair gets a second. Critical for ShardLoader partitioning."""
    from dllm.data.prepare import stream_corpus

    a = _FakeSource({"en": ["a-en-1", "a-en-2"], "de": ["a-de-1"]})
    b = _FakeSource({"en": ["b-en-1"], "de": ["b-de-1", "b-de-2"]})
    _install_fake_sources(monkeypatch, {"src_a": a, "src_b": b})

    alloc = {"src_a": {"en": 1_000_000, "de": 1_000_000},
             "src_b": {"en": 1_000_000, "de": 1_000_000}}
    out = list(stream_corpus(alloc, ["en", "de"]))
    # 4 (source, lang) pairs * starts at 1 each before any goes to 2nd
    first_four = [(s, l) for s, l, _ in out[:4]]
    assert sorted(first_four) == [
        ("src_a", "de"), ("src_a", "en"), ("src_b", "de"), ("src_b", "en"),
    ]
    # Every doc appears exactly once
    docs = [d for _, _, d in out]
    assert sorted(docs) == sorted([
        "a-en-1", "a-en-2", "a-de-1", "b-en-1", "b-de-1", "b-de-2",
    ])


def test_stream_corpus_respects_char_budget(monkeypatch) -> None:
    from dllm.data.prepare import stream_corpus

    # Each doc is 100 chars, budget is 250 → expect first 3 docs (300 chars > 250 → stops after 3)
    docs_5 = ["x" * 100 for _ in range(5)]
    a = _FakeSource({"en": docs_5})
    _install_fake_sources(monkeypatch, {"src_a": a})

    alloc = {"src_a": {"en": 250}}
    out = list(stream_corpus(alloc, ["en"]))
    # Fake source emits while emitted < budget; allows 3rd doc to push over
    assert len(out) == 3


def test_stream_corpus_skips_unsupported_lang(monkeypatch) -> None:
    """A source that doesn't support a requested lang is silently skipped."""
    from dllm.data.prepare import stream_corpus

    a = _FakeSource({"en": ["a-en-1"]})  # no DE
    _install_fake_sources(monkeypatch, {"src_a": a})

    alloc = {"src_a": {"en": 1_000_000, "de": 1_000_000}}
    out = list(stream_corpus(alloc, ["en", "de"]))
    langs_seen = {l for _, l, _ in out}
    assert langs_seen == {"en"}


def test_tokenize_streaming_end_to_end(tmp_path: Path, monkeypatch) -> None:
    """Stream from fake sources → tokenize → split_off_val produces
    correctly-sized .bin files."""
    from dllm.data.prepare import (
        split_off_val,
        stream_corpus,
        tokenize_streaming,
        train_bpe,
    )

    # Build a small fake corpus
    sample = _synthetic_sample()
    a = _FakeSource({"en": sample[:6], "de": sample[6:12]})
    _install_fake_sources(monkeypatch, {"src_a": a})

    tok = train_bpe(sample, vocab_size=512)
    alloc = {"src_a": {"en": 100_000, "de": 100_000}}
    docs = stream_corpus(alloc, ["en", "de"])
    out = tmp_path / "train.bin"
    total, per_pair = tokenize_streaming(docs, tok, out)
    assert total > 0
    assert out.exists()
    assert total * 2 == out.stat().st_size  # uint16 = 2 bytes per token
    # Per-(source, lang) counts populated
    assert ("src_a", "en") in per_pair
    assert ("src_a", "de") in per_pair

    val_out = tmp_path / "val.bin"
    train_n, val_n = split_off_val(out, val_out, val_tokens=10)
    assert train_n + val_n == total
    assert val_n == 10


def test_default_tokenizer_path_under_cache() -> None:
    from dllm.data.tokenizer import default_tokenizer_path

    p = default_tokenizer_path()
    assert p.name == "tokenizer.json"
    assert "cache" in p.parts


# ---------------------------------------------------------------------------
# Sources subpackage — verify each source advertises license + supported langs
# ---------------------------------------------------------------------------


def test_every_source_advertises_license_metadata() -> None:
    """Each source must expose name + license + license_url so the manifest
    can be assembled even without fetching."""
    from dllm.data import sources

    required = {"name", "license", "license_url", "url"}
    for name in sources.all_source_names():
        src = sources.load(name)
        info = src.license_info()
        assert required.issubset(info), f"{name} missing fields: {required - info.keys()}"
        assert isinstance(src.supported_langs(), list)
        assert len(src.supported_langs()) > 0


def test_gutenberg_supports_all_5_eu_target_langs() -> None:
    """Gutenberg must cover en/de/fr/it/es — English via sedthh/gutenberg_english,
    the other four via sedthh/gutenberg_multilang. Catches regressions where
    a non-English mirror is dropped and silently re-narrows the corpus to en-only.
    """
    from dllm.data import sources

    target = {"en", "de", "fr", "it", "es"}
    supported = set(sources.load("gutenberg").supported_langs())
    missing = target - supported
    assert not missing, f"gutenberg dropped literary register for {missing}"


def test_every_target_lang_has_at_least_one_literary_source() -> None:
    """The non-encyclopedic, non-legislative register (i.e. literary text) is
    served by gutenberg + pleias_books. Each EU target lang must be covered
    by at least one of them — pleias_books is non-EN only by design, gutenberg
    now covers all five."""
    from dllm.data import sources

    literary_sources = ("gutenberg", "pleias_books")
    target = {"en", "de", "fr", "it", "es"}
    covered: dict[str, list[str]] = {lang: [] for lang in target}
    for name in literary_sources:
        for lang in sources.load(name).supported_langs():
            if lang in covered:
                covered[lang].append(name)
    uncovered = [lang for lang, srcs in covered.items() if not srcs]
    assert not uncovered, f"no literary source for langs: {uncovered}"


def test_default_allocation_is_internally_consistent() -> None:
    """Every (source, lang) in DEFAULT_ALLOC_CHARS must be one the source supports.
    Partial-coverage sources (Gutenberg en-only) must NOT list de/fr/it/es."""
    from dllm.data import sources
    from dllm.data.prepare import DEFAULT_ALLOC_CHARS

    for src_name, lang_budgets in DEFAULT_ALLOC_CHARS.items():
        src = sources.load(src_name)
        supported = set(src.supported_langs())
        configured = set(lang_budgets)
        unsupported = configured - supported
        assert not unsupported, (
            f"DEFAULT_ALLOC_CHARS[{src_name!r}] requests {unsupported} but "
            f"{src_name} only supports {supported}"
        )
