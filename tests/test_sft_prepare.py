"""SFT orchestrator: fake sources → tokenize → train/val split with mask alignment."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pytest

pytest.importorskip("tokenizers", reason="data extras not installed; run `pip install -e .[data]`")

from dllm.data.sft_format import Conversation, Message, ToolCall


# ---------------------------------------------------------------------------
# Fake source plumbing
# ---------------------------------------------------------------------------


class _FakeSource:
    def __init__(self, by_lang: dict[str, list[Conversation]], task: str = "agentic") -> None:
        self.by_lang = by_lang
        self._task = task

    def iter_examples(self, lang: str, count_budget: int) -> Iterator[Conversation]:
        emitted = 0
        for conv in self.by_lang.get(lang, []):
            if emitted >= count_budget:
                return
            yield conv
            emitted += 1

    def license_info(self) -> dict:
        return {
            "name": "fake",
            "license": "test",
            "license_url": "https://example.invalid",
            "url": "https://example.invalid",
            "attribution": "fake",
        }

    def supported_langs(self) -> list[str]:
        return list(self.by_lang)

    def task_type(self) -> str:
        return self._task


def _install_fake_sources(monkeypatch, fakes: dict[str, _FakeSource]) -> None:
    from dllm.data import sft_sources as src_pkg

    def fake_load(name: str):
        if name in fakes:
            return fakes[name]
        raise ValueError(f"unknown source {name!r}")

    monkeypatch.setattr(src_pkg, "load", fake_load)
    monkeypatch.setattr(src_pkg, "all_source_names", lambda: tuple(fakes))


# ---------------------------------------------------------------------------
# A small tokenizer matching our pretraining schema (Byte-level BPE + EOT)
# ---------------------------------------------------------------------------


def _trivial_tokenizer():
    """Train a tiny byte-level BPE on synthetic ChatML-like text so the
    pipeline has a real Rust tokenizer to encode_batch through. EOT is the
    required special token."""
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDec
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPre
    from tokenizers.trainers import BpeTrainer

    sample = [
        "<|im_start|>user\nhi\n<|im_end|>\n",
        "<|im_start|>assistant\nhello\n<|im_end|>\n",
        "<|im_start|>assistant\n<tool_call>{\"name\":\"f\",\"arguments\":{}}</tool_call>\n<|im_end|>\n",
        "der schnelle braune fuchs " * 20,
        "the quick brown fox " * 20,
    ]
    tok = Tokenizer(BPE())
    tok.pre_tokenizer = ByteLevelPre(add_prefix_space=False)
    tok.decoder = ByteLevelDec()
    trainer = BpeTrainer(
        vocab_size=512,
        min_frequency=1,
        special_tokens=["<|endoftext|>"],
        initial_alphabet=ByteLevelPre.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(iter(sample), trainer)
    return tok


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_encode_conversation_batch_preserves_per_segment_mask() -> None:
    """Each token from an assistant segment must carry mask=1; each token
    from a user/system/tool segment must carry mask=0."""
    from dllm.data.sft_prepare import _encode_conversation_batch

    tok = _trivial_tokenizer()
    convs = [
        ("src_a", "en", Conversation(messages=[
            Message(role="user", content="hi"),
            Message(role="assistant", content="hello"),
        ])),
    ]
    out = _encode_conversation_batch(convs, tok)
    assert len(out) == 1
    src, lang, ids, mask = out[0]
    assert src == "src_a" and lang == "en"
    assert len(ids) == len(mask) > 0
    # Mask alignment: tokens belonging to assistant segments are 1, user/system are 0.
    # Sanity: at least some 0s (user header + content + close) and some 1s (asst content + close).
    assert 0 in mask and 1 in mask
    # The first few tokens are the user header (mask=0). The last few are the
    # assistant close (mask=1).
    assert mask[0] == 0, "first token (user header) should not be a loss target"
    assert mask[-1] == 1, "final token (assistant <|im_end|>) must be a loss target"


def test_stream_tokenize_writes_paired_ids_and_mask(tmp_path: Path, monkeypatch) -> None:
    """End-to-end: fake source → tokenize → both .bin files exist with
    identical length and the masks line up with their ids."""
    from dllm.data.sft_prepare import stream_tokenize_to_disk

    tok = _trivial_tokenizer()
    convs_en = [
        Conversation(messages=[
            Message(role="user", content="hi"),
            Message(role="assistant", content="hello"),
        ]),
        Conversation(messages=[
            Message(role="user", content="call f"),
            Message(role="assistant", content="", tool_calls=[ToolCall(name="f", arguments={"x": 1})]),
        ]),
    ]
    _install_fake_sources(monkeypatch, {
        "src_a": _FakeSource({"en": convs_en}, task="tool_call"),
    })

    ids_path = tmp_path / "sft_train.bin"
    mask_path = tmp_path / "sft_train_mask.bin"
    total, per_pair, boundaries = stream_tokenize_to_disk(
        alloc={"src_a": {"en": 10}},
        langs=["en"],
        tokenizer=tok,
        ids_path=ids_path,
        mask_path=mask_path,
        batch_size=4,
    )
    assert total > 0
    assert ids_path.exists() and mask_path.exists()
    # uint16 ids → 2 bytes/token; uint8 mask → 1 byte/token.
    assert ids_path.stat().st_size == total * 2
    assert mask_path.stat().st_size == total
    # Boundaries: [0, end_conv_1, end_conv_2 == total]
    assert boundaries[0] == 0 and boundaries[-1] == total
    assert len(boundaries) == 3
    # Per-(src, lang) counts populated.
    assert per_pair[("src_a", "en")] == total


def test_split_at_boundary_never_cuts_mid_conversation(tmp_path: Path) -> None:
    """split_at_boundary picks the nearest *conv* boundary, never mid-conv."""
    from dllm.data.sft_prepare import split_at_boundary

    # Build synthetic: 3 conversations of sizes 100, 200, 150 tokens → total 450.
    sizes = [100, 200, 150]
    total = sum(sizes)
    boundaries = [0, 100, 300, 450]

    ids = np.arange(total, dtype=np.uint16)
    mask = np.zeros(total, dtype=np.uint8)
    mask[50:60] = 1  # arbitrary loss targets
    train_ids = tmp_path / "train.bin"
    train_mask = tmp_path / "train_mask.bin"
    val_ids = tmp_path / "val.bin"
    val_mask = tmp_path / "val_mask.bin"
    train_ids.write_bytes(ids.tobytes())
    train_mask.write_bytes(mask.tobytes())

    # val_fraction=0.4 → target=270 → closest boundary is 300.
    train_n, val_n = split_at_boundary(
        train_ids, train_mask, val_ids, val_mask, boundaries, val_fraction=0.4,
    )
    assert train_n == 300
    assert val_n == 150
    # And: the val side starts at original position 300, so val_ids[0] == 300.
    val_arr = np.fromfile(val_ids, dtype=np.uint16)
    assert val_arr[0] == 300
    # Train was truncated correctly.
    train_arr = np.fromfile(train_ids, dtype=np.uint16)
    assert train_arr.shape[0] == 300
    # Masks parallel-truncated.
    assert np.fromfile(train_mask, dtype=np.uint8).shape[0] == 300
    assert np.fromfile(val_mask, dtype=np.uint8).shape[0] == 150


def test_dual_shard_writer_flushes_on_overflow(tmp_path: Path) -> None:
    """Small flush_tokens forces mid-write flushes; output must still be intact."""
    from dllm.data.sft_prepare import DualShardWriter

    ids_path = tmp_path / "ids.bin"
    mask_path = tmp_path / "mask.bin"
    w = DualShardWriter(ids_path, mask_path, flush_tokens=10)
    # Write two convs that together exceed the buffer.
    w.write_conv([1, 2, 3, 4, 5], [0, 0, 1, 1, 1])
    w.write_conv([6, 7, 8, 9, 10, 11, 12], [0, 0, 1, 1, 1, 1, 1])
    total, boundaries = w.close()
    assert total == 12
    assert boundaries == [0, 5, 12]
    arr = np.fromfile(ids_path, dtype=np.uint16)
    assert arr.tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    mask = np.fromfile(mask_path, dtype=np.uint8)
    assert mask.tolist() == [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1]


def test_xlam_only_yields_en(monkeypatch) -> None:
    """Real-source guard: sources that only support EN must not produce DE
    rows when DE is in the alloc. Catches a regression where adding a DE
    budget would silently fail in production."""
    from dllm.data import sft_sources

    src = sft_sources.load("xlam")
    assert "en" in src.supported_langs()
    assert "de" not in src.supported_langs()
    # Per-source coverage table, encoded:
    assert sft_sources.load("xlam").supported_langs() == ["en"]
    assert sft_sources.load("glaive_v2").supported_langs() == ["en"]
    assert sft_sources.load("hermes_fc").supported_langs() == ["en"]
    assert "de" in sft_sources.load("oasst2").supported_langs()
    assert "de" in sft_sources.load("aya").supported_langs()
    assert "de" in sft_sources.load("hand_written").supported_langs()


def test_full_pipeline_with_split(tmp_path: Path, monkeypatch) -> None:
    """End-to-end shoot-through with fake sources, then split + verify
    counts add up and on-disk files are coherent."""
    from dllm.data.sft_prepare import split_at_boundary, stream_tokenize_to_disk

    tok = _trivial_tokenizer()
    convs_en = [Conversation(messages=[
        Message(role="user", content=f"q{i}"),
        Message(role="assistant", content=f"a{i}"),
    ]) for i in range(10)]
    convs_de = [Conversation(messages=[
        Message(role="user", content=f"frage{i}"),
        Message(role="assistant", content=f"antwort{i}"),
    ]) for i in range(8)]
    _install_fake_sources(monkeypatch, {
        "src_a": _FakeSource({"en": convs_en, "de": convs_de}, task="agentic"),
    })

    ids_path = tmp_path / "sft_train.bin"
    mask_path = tmp_path / "sft_train_mask.bin"
    val_ids_path = tmp_path / "sft_val.bin"
    val_mask_path = tmp_path / "sft_val_mask.bin"

    total, per_pair, boundaries = stream_tokenize_to_disk(
        alloc={"src_a": {"en": 10, "de": 8}},
        langs=["en", "de"],
        tokenizer=tok,
        ids_path=ids_path,
        mask_path=mask_path,
        batch_size=4,
    )
    train_n, val_n = split_at_boundary(
        ids_path, mask_path, val_ids_path, val_mask_path, boundaries, val_fraction=0.2,
    )
    assert train_n + val_n == total
    # Both languages produced tokens.
    assert per_pair[("src_a", "en")] > 0
    assert per_pair[("src_a", "de")] > 0
    # All four files exist with correct sizes.
    assert ids_path.stat().st_size == train_n * 2
    assert mask_path.stat().st_size == train_n
    assert val_ids_path.stat().st_size == val_n * 2
    assert val_mask_path.stat().st_size == val_n
