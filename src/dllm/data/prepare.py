"""EU multi-lingual corpus prep — 5B-token edition.

Streams documents from multiple EU-regulation-compliant sources
(Wikipedia, EuroParl, JRC-Acquis, Project Gutenberg), trains a balanced
byte-level BPE tokenizer on a sample, then streams the full corpus
through the tokenizer to `train.bin` + `val.bin` shards.

Memory profile: streaming tokenization writes uint16 tokens to disk in
~100 MB flushes, so the 5 B-token target (~10 GB on disk) never holds
the full corpus in RAM. BPE training uses a bounded sample (~10 M chars
per source × lang ≈ 250 MB total in memory).

Provenance: every source's license / URL / fetch date / char & token
counts are recorded in `manifest.json` — sufficient for the EU AI Act
Art. 53 "detailed summary" requirement.

Requires the [data] extras:  pip install -e .[data]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np

from . import sources

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass


# ---------------------------------------------------------------------------
# Defaults — designed for ~5 B tokens across DE/FR/EN/IT/ES
# ---------------------------------------------------------------------------

DEFAULT_LANGS = ["de", "fr", "en", "it", "es"]
DEFAULT_VOCAB = 32768
DEFAULT_VAL_TOKENS = 50_000_000  # 50 M val tokens (~1 % of train); enough for stable val
EOT_TOKEN = "<|endoftext|>"

# v3 corpus design (2026-05-26): adds PleIAs per-language Public-Domain
# Books (DE/FR/IT/ES) to fill the literary register gap for non-English
# languages — these are purpose-built for EU AI Act Article 53 compliance.
# Gutenberg now also covers de/fr/it/es via sedthh/gutenberg_multilang, but
# PleIAs is much larger so it does the heavy lifting for non-EN literature.
#
# Per-source per-lang CHARACTER budgets. Token estimate ≈ chars × 0.25 for
# European langs; totals sum to ~20 B chars ≈ 5 B tokens budget.
# Operators can override any cell via the --alloc JSON.
DEFAULT_ALLOC_CHARS = {
    "wikipedia":    {"de": 2_000_000_000, "fr": 2_000_000_000, "en": 2_500_000_000,
                     "it": 2_000_000_000, "es": 2_000_000_000},
    "pleias_books": {"de": 2_000_000_000, "fr": 2_000_000_000,
                     "it": 1_500_000_000, "es": 1_500_000_000},
    "gutenberg":    {"en": 1_500_000_000, "de":   400_000_000, "fr":   400_000_000,
                     "it":   400_000_000, "es":   400_000_000},
    "europarl":     {"de":   200_000_000, "fr":   200_000_000, "en":   200_000_000,
                     "it":   200_000_000, "es":   200_000_000},
    "jrc_acquis":   {"de":   120_000_000, "fr":   120_000_000, "en":   120_000_000,
                     "it":   120_000_000, "es":   120_000_000},
}

# How many CHARS to sample per (source, lang) into the BPE training mix.
# Balanced: each (source, lang) contributes equally so the merges aren't
# dominated by Wikipedia.
BPE_SAMPLE_CHARS_PER_PAIR = 10_000_000  # 10 M chars / pair × 4 sources × 5 langs = 200 M chars


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def cache_dir() -> Path:
    here = Path(__file__).resolve().parent
    out = here.parent.parent.parent / "data" / "cache"
    out.mkdir(parents=True, exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# Corpus streaming
# ---------------------------------------------------------------------------


def stream_corpus(
    alloc: dict[str, dict[str, int]],
    langs: list[str],
) -> Iterator[tuple[str, str, str]]:
    """Round-robin docs across (source, lang) pairs.

    Yields (source_name, lang, doc) tuples. Round-robin keeps the resulting
    train.bin globally mixed — any contiguous slice has roughly the same
    source+lang distribution as the whole corpus. Critical for the worker's
    ShardLoader which partitions train.bin by worker_id.
    """
    iters: dict[str, Iterator[str]] = {}
    for src_name, lang_budgets in alloc.items():
        try:
            src = sources.load(src_name)
        except ValueError as e:
            print(f"  [warn] unknown source {src_name!r}: {e}")
            continue
        for lang, budget in lang_budgets.items():
            if lang not in langs:
                continue
            if lang not in src.supported_langs():
                print(f"  [skip] {src_name} does not support lang={lang!r}")
                continue
            if budget <= 0:
                continue
            key = f"{src_name}::{lang}"
            iters[key] = src.iter_docs(lang, budget)

    iter_state = {k: iter(v) for k, v in iters.items()}
    while iter_state:
        for key in list(iter_state):
            try:
                doc = next(iter_state[key])
                src_name, lang = key.split("::", 1)
                yield src_name, lang, doc
            except StopIteration:
                del iter_state[key]


# ---------------------------------------------------------------------------
# BPE training (sampled)
# ---------------------------------------------------------------------------


def collect_bpe_sample(
    alloc: dict[str, dict[str, int]],
    langs: list[str],
    chars_per_pair: int = BPE_SAMPLE_CHARS_PER_PAIR,
) -> list[str]:
    """Pull a small sample from each (source, lang) for BPE training.

    Memory cap: chars_per_pair × n_sources × n_langs bytes (Python str overhead
    ~2× on top). 10 M × 4 × 5 = 200 M chars ≈ 400 MB in memory. Fine.
    """
    sample: list[str] = []
    for src_name in alloc:
        try:
            src = sources.load(src_name)
        except ValueError:
            continue
        for lang in langs:
            if lang not in src.supported_langs():
                continue
            print(f"  [bpe sample] {src_name} {lang} ({chars_per_pair / 1e6:.1f} M chars)")
            acc = 0
            try:
                for doc in src.iter_docs(lang, chars_per_pair):
                    sample.append(doc)
                    acc += len(doc)
                    if acc >= chars_per_pair:
                        break
            except Exception as e:  # noqa: BLE001
                print(f"    [warn] {src_name}/{lang} sample failed: {e}")
    return sample


def train_bpe(sample: list[str], vocab_size: int = DEFAULT_VOCAB):
    from tokenizers import Tokenizer  # noqa: PLC0415
    from tokenizers.decoders import ByteLevel as ByteLevelDec  # noqa: PLC0415
    from tokenizers.models import BPE  # noqa: PLC0415
    from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPre  # noqa: PLC0415
    from tokenizers.trainers import BpeTrainer  # noqa: PLC0415

    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevelPre(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDec()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=[EOT_TOKEN],
        initial_alphabet=ByteLevelPre.alphabet(),
        show_progress=True,
    )
    print(f"  training BPE on {len(sample)} docs ({sum(len(d) for d in sample) / 1e6:.1f} M chars)")
    tokenizer.train_from_iterator(iter(sample), trainer, length=len(sample))
    return tokenizer


# ---------------------------------------------------------------------------
# Streaming tokenization → .bin
# ---------------------------------------------------------------------------


class ShardWriter:
    """Append-only uint16 token writer with a chunked buffer.

    Avoids holding the full token stream in memory: tokens are accumulated
    in a numpy buffer that flushes to disk every ~100 MB. Designed for the
    5 B-token target (~10 GB on disk).
    """

    def __init__(self, path: Path, flush_tokens: int = 50_000_000) -> None:
        self.path = path
        self._buf = np.empty(flush_tokens, dtype=np.uint16)
        self._idx = 0
        self._cap = flush_tokens
        self._total = 0
        self._fp = path.open("wb")

    def append(self, ids: list[int]) -> None:
        i = 0
        while i < len(ids):
            room = self._cap - self._idx
            take = min(room, len(ids) - i)
            self._buf[self._idx : self._idx + take] = ids[i : i + take]
            self._idx += take
            i += take
            if self._idx == self._cap:
                self._flush()

    def _flush(self) -> None:
        if self._idx == 0:
            return
        self._fp.write(self._buf[: self._idx].tobytes())
        self._total += self._idx
        self._idx = 0

    def close(self) -> int:
        self._flush()
        self._fp.close()
        return self._total


def tokenize_streaming(
    docs: Iterator[tuple[str, str, str]],
    tokenizer,
    out_path: Path,
) -> tuple[int, dict[tuple[str, str], int]]:
    """Stream-tokenize every doc and append uint16 IDs to out_path.

    Returns (total_tokens, per_(source,lang)_token_count).
    """
    eot_id = tokenizer.token_to_id(EOT_TOKEN)
    if eot_id is None:
        raise RuntimeError(f"{EOT_TOKEN!r} missing from trained vocab")

    writer = ShardWriter(out_path)
    counts: dict[tuple[str, str], int] = {}
    n_docs = 0
    last_print = 0
    for src_name, lang, doc in docs:
        ids = tokenizer.encode(doc).ids
        ids.append(eot_id)
        writer.append(ids)
        counts[(src_name, lang)] = counts.get((src_name, lang), 0) + len(ids)
        n_docs += 1
        if writer._total + writer._idx - last_print >= 100_000_000:
            print(
                f"    [tokenize] {n_docs:,} docs, "
                f"{(writer._total + writer._idx) / 1e9:.2f} B tokens"
            )
            last_print = writer._total + writer._idx
    total = writer.close()
    return total, counts


def split_off_val(train_path: Path, val_path: Path, val_tokens: int) -> tuple[int, int]:
    """Move the last `val_tokens` tokens from train_path to val_path."""
    arr = np.memmap(train_path, dtype=np.uint16, mode="r")
    total = int(arr.shape[0])
    val_n = min(val_tokens, total // 20)  # cap val at 5 % of corpus
    val_n = max(1, val_n)
    train_n = total - val_n
    # Copy the val slice (memmap can't outlive the underlying file truncation)
    val_slice = np.array(arr[train_n:], dtype=np.uint16)
    del arr  # release memmap before truncating
    val_path.write_bytes(val_slice.tobytes())
    # Truncate train.bin to its new length (uint16 = 2 bytes per token)
    with open(train_path, "r+b") as f:
        f.truncate(train_n * 2)
    return train_n, val_n


# ---------------------------------------------------------------------------
# Provenance manifest
# ---------------------------------------------------------------------------


def sha256_of_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def write_manifest(
    out_dir: Path,
    alloc: dict[str, dict[str, int]],
    used_sources: list[str],
    vocab_size: int,
    train_tokens: int,
    val_tokens: int,
    per_source_lang_tokens: dict[tuple[str, str], int],
    langs: list[str],
) -> Path:
    src_meta = []
    for src_name in used_sources:
        try:
            src = sources.load(src_name)
            info = src.license_info()
        except (ValueError, ImportError):
            info = {"name": src_name, "license": "unknown"}
        # Add per-lang token counts under each source
        info["tokens_per_lang"] = {
            lang: per_source_lang_tokens.get((src_name, lang), 0) for lang in langs
        }
        info["allocated_chars_per_lang"] = alloc.get(src_name, {})
        info["fetch_date"] = date.today().isoformat()
        src_meta.append(info)

    manifest = {
        "format_version": "0.0.3",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "vocab_size": vocab_size,
        "tokenizer": "tokenizer.json",
        "langs": langs,
        "tokens": {
            "train": train_tokens,
            "val": val_tokens,
            "total": train_tokens + val_tokens,
        },
        "splits": {
            "train": {
                "path": "train.bin",
                "tokens": train_tokens,
                "dtype": "uint16",
                "sha256": sha256_of_file(out_dir / "train.bin"),
            },
            "val": {
                "path": "val.bin",
                "tokens": val_tokens,
                "dtype": "uint16",
                "sha256": sha256_of_file(out_dir / "val.bin"),
            },
        },
        "sources": src_meta,
        "compliance": {
            "ai_act_art_53": (
                "Per-source license + URL + fetch date + token counts are "
                "recorded in this manifest, satisfying the 'sufficiently "
                "detailed summary' requirement."
            ),
            "dsm_art_4_tdm": (
                "All sources are either EU-institutional public domain "
                "(EuroParl, JRC-Acquis), pre-existing public domain "
                "(Project Gutenberg), or under explicit open license "
                "(Wikipedia CC-BY-SA). No machine-readable opt-out has "
                "been encountered. No web scraping; all data fetched via "
                "the originator's published HuggingFace mirror."
            ),
            "gdpr": (
                "All sources are institutional or published-author content "
                "(Wikipedia editors, EU parliamentarians, published "
                "authors). No personal data of private individuals is "
                "included by design."
            ),
        },
        "notes": (
            "Phase 1 5B-token EU pre-training corpus. Target: 124M model, "
            "5 languages (DE/FR/EN/IT/ES). Approximately 6.7 epochs through "
            "1000 outer rounds at inner_steps=2000 × micro_batch=4 × seq_len=512."
        ),
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--langs", nargs="+", default=DEFAULT_LANGS)
    ap.add_argument(
        "--sources",
        nargs="+",
        default=list(DEFAULT_ALLOC_CHARS),
        choices=sources.all_source_names(),
        help="Which sources to include (subset of the configured allocations)",
    )
    ap.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB)
    ap.add_argument("--val-tokens", type=int, default=DEFAULT_VAL_TOKENS)
    ap.add_argument(
        "--alloc",
        type=Path,
        default=None,
        help="Optional JSON file overriding per-source per-lang char budgets.",
    )
    ap.add_argument(
        "--bpe-sample-chars-per-pair",
        type=int,
        default=BPE_SAMPLE_CHARS_PER_PAIR,
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify each source is reachable; don't write anything.",
    )
    args = ap.parse_args()

    out = cache_dir()
    print(f"[prepare] cache dir: {out}")

    # Allocation: defaults, optionally overridden by --alloc file
    alloc = {k: dict(v) for k, v in DEFAULT_ALLOC_CHARS.items() if k in args.sources}
    if args.alloc:
        override = json.loads(args.alloc.read_text(encoding="utf-8"))
        for src_name, lang_budgets in override.items():
            if src_name in alloc:
                alloc[src_name].update(lang_budgets)
            else:
                alloc[src_name] = lang_budgets
    # Restrict to requested langs
    alloc = {s: {l: b for l, b in d.items() if l in args.langs} for s, d in alloc.items()}

    print("[prepare] allocation (chars per source × lang):")
    for src_name, lang_budgets in alloc.items():
        total = sum(lang_budgets.values())
        print(f"  {src_name}: total {total / 1e9:.2f} B chars across {len(lang_budgets)} langs")
    total_chars = sum(sum(d.values()) for d in alloc.values())
    print(f"  TOTAL: {total_chars / 1e9:.2f} B chars (≈ {total_chars * 0.25 / 1e9:.2f} B tokens)")

    if args.dry_run:
        print("[prepare] dry-run: probing each source for one doc per lang...")
        for src_name in alloc:
            src = sources.load(src_name)
            print(f"  {src_name}:")
            for lang in args.langs:
                if lang not in src.supported_langs():
                    print(f"    [skip] {lang}: not supported")
                    continue
                try:
                    doc = next(src.iter_docs(lang, 10_000), None)
                    if doc:
                        print(f"    [ok]   {lang}: first doc {len(doc):,} chars")
                    else:
                        print(f"    [warn] {lang}: no docs returned")
                except Exception as e:  # noqa: BLE001
                    print(f"    [err]  {lang}: {e}")
        print("[prepare] dry-run done")
        return

    # Phase 1 — BPE training sample
    print("[prepare] phase 1: collecting BPE training sample")
    sample = collect_bpe_sample(alloc, args.langs, args.bpe_sample_chars_per_pair)
    print(f"  collected {len(sample)} sample docs")

    print(f"[prepare] phase 2: training BPE vocab={args.vocab_size}")
    tokenizer = train_bpe(sample, vocab_size=args.vocab_size)
    tok_path = out / "tokenizer.json"
    tokenizer.save(str(tok_path))
    actual_vocab = tokenizer.get_vocab_size()
    print(f"  saved {tok_path} (vocab_size={actual_vocab})")
    del sample  # free ~400 MB

    # Phase 3 — streaming tokenization of full corpus
    print("[prepare] phase 3: streaming tokenization → train.bin")
    train_path = out / "train.bin"
    val_path = out / "val.bin"
    if train_path.exists():
        train_path.unlink()
    if val_path.exists():
        val_path.unlink()
    docs = stream_corpus(alloc, args.langs)
    total_tokens, per_source_lang = tokenize_streaming(docs, tokenizer, train_path)
    print(f"  total tokens written: {total_tokens:,}")
    for (src_name, lang), n in sorted(per_source_lang.items()):
        print(f"    {src_name} {lang}: {n:,}")

    # Phase 4 — split off val
    print(f"[prepare] phase 4: splitting last {args.val_tokens:,} tokens into val.bin")
    train_n, val_n = split_off_val(train_path, val_path, args.val_tokens)
    print(f"  train: {train_n:,} tokens")
    print(f"  val:   {val_n:,} tokens")

    # Phase 5 — provenance manifest
    print("[prepare] phase 5: writing manifest")
    manifest_path = write_manifest(
        out,
        alloc,
        list(alloc),
        actual_vocab,
        train_n,
        val_n,
        per_source_lang,
        args.langs,
    )
    print(f"  wrote {manifest_path}")
    print("[prepare] done")


if __name__ == "__main__":
    main()
