"""EU multilingual corpus prep — Phase 1 prototype.

Downloads Wikipedia samples for major EU languages, trains a balanced
byte-level BPE tokenizer, tokenizes the corpus, and writes uint16 .bin shards
plus a provenance manifest.

Production target (PLAN §4): CulturaX + OSCAR + Europeana + EuroParl + EUR-Lex,
all 24 EU official languages, ~3T tokens, 128k vocab, with full DSM Art. 4
opt-out compliance + PII redaction + provenance hash chain.

This prototype validates the pipeline shape end-to-end with a small,
fully-balanced 5–7 language Wikipedia sample.

Requires the [data] extras:  pip install -e .[data]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass


DEFAULT_LANGS = ["de", "fr", "es", "it", "en", "nl", "pl"]
DEFAULT_VOCAB = 32768
DEFAULT_ARTICLES_PER_LANG = 2000
DEFAULT_VAL_FRAC = 0.05
WIKI_SNAPSHOT = "20231101"
EOT_TOKEN = "<|endoftext|>"


def cache_dir() -> Path:
    here = Path(__file__).resolve().parent
    out = here.parent.parent.parent / "data" / "cache"
    out.mkdir(parents=True, exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------


def fetch_wikipedia_sample(
    lang: str,
    n_articles: int,
    snapshot: str = WIKI_SNAPSHOT,
    min_chars: int = 200,
) -> list[str]:
    """Stream wikimedia/wikipedia for `lang`; return up to n_articles non-stub texts."""
    from datasets import load_dataset  # noqa: PLC0415

    config = f"{snapshot}.{lang}"
    print(f"  streaming wikimedia/wikipedia config={config} target={n_articles} articles")
    ds = load_dataset("wikimedia/wikipedia", config, split="train", streaming=True)
    texts: list[str] = []
    for row in ds:
        if len(texts) >= n_articles:
            break
        t = row.get("text") or ""
        if len(t) >= min_chars:
            texts.append(t)
    return texts


# ---------------------------------------------------------------------------
# BPE training
# ---------------------------------------------------------------------------


def train_bpe(
    corpus_per_lang: dict[str, list[str]],
    vocab_size: int = DEFAULT_VOCAB,
):
    """Train byte-level BPE on a balanced (equal-per-lang) sample."""
    from tokenizers import Tokenizer  # noqa: PLC0415
    from tokenizers.decoders import ByteLevel as ByteLevelDec  # noqa: PLC0415
    from tokenizers.models import BPE  # noqa: PLC0415
    from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPre  # noqa: PLC0415
    from tokenizers.trainers import BpeTrainer  # noqa: PLC0415

    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevelPre(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDec()

    # balanced sample — same number of articles per lang
    per = min(len(t) for t in corpus_per_lang.values()) if corpus_per_lang else 0
    balanced: list[str] = []
    for texts in corpus_per_lang.values():
        balanced.extend(texts[:per])
    print(f"  BPE training corpus: {len(balanced)} docs ({per}/lang × {len(corpus_per_lang)} langs)")

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=[EOT_TOKEN],
        initial_alphabet=ByteLevelPre.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(iter(balanced), trainer, length=len(balanced))
    return tokenizer


# ---------------------------------------------------------------------------
# tokenize + shard
# ---------------------------------------------------------------------------


def _encode_stream(tokenizer, docs: Iterable[str], eot_id: int) -> Iterable[int]:
    for d in docs:
        yield from tokenizer.encode(d).ids
        yield eot_id


def _interleave_docs(corpus_per_lang: dict[str, list[str]]) -> list[tuple[str, str]]:
    """Round-robin docs across languages so any contiguous slice has every lang.

    Without this, ShardLoader(world_size=2) gives worker 0 a different lang mix
    than worker 1, and val (last 5%) is dominated by whichever lang sorted last.
    """
    iters = {lang: iter(docs) for lang, docs in corpus_per_lang.items()}
    out: list[tuple[str, str]] = []
    exhausted: set[str] = set()
    while len(exhausted) < len(iters):
        for lang, it in iters.items():
            if lang in exhausted:
                continue
            try:
                out.append((lang, next(it)))
            except StopIteration:
                exhausted.add(lang)
    return out


def tokenize_and_shard(
    corpus_per_lang: dict[str, list[str]],
    tokenizer,
    out_dir: Path,
    val_frac: float = DEFAULT_VAL_FRAC,
) -> dict:
    eot_id = tokenizer.token_to_id(EOT_TOKEN)
    if eot_id is None:
        raise RuntimeError(f"{EOT_TOKEN!r} not in trained vocab")

    interleaved = _interleave_docs(corpus_per_lang)
    print(f"  interleaved {len(interleaved)} docs across {len(corpus_per_lang)} langs")

    all_ids: list[int] = []
    tokens_per_lang: dict[str, int] = {lang: 0 for lang in corpus_per_lang}
    for lang, doc in interleaved:
        before = len(all_ids)
        for tok in _encode_stream(tokenizer, [doc], eot_id):
            all_ids.append(tok)
        tokens_per_lang[lang] += len(all_ids) - before
    for lang, count in tokens_per_lang.items():
        print(f"  tokenized {lang}: {count:,} tokens")

    if not all_ids:
        raise RuntimeError("empty corpus after tokenization")

    vocab_size = tokenizer.get_vocab_size()
    dtype = np.uint16 if vocab_size <= 65535 else np.uint32
    arr = np.array(all_ids, dtype=dtype)
    n_val = max(1, int(len(arr) * val_frac))

    train = arr[:-n_val]
    val = arr[-n_val:]
    (out_dir / "train.bin").write_bytes(train.tobytes())
    (out_dir / "val.bin").write_bytes(val.tobytes())

    return {
        "tokens_per_lang": tokens_per_lang,
        "train_tokens": int(len(train)),
        "val_tokens": int(len(val)),
        "dtype": dtype.__name__,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--langs", nargs="+", default=DEFAULT_LANGS)
    ap.add_argument("--articles-per-lang", type=int, default=DEFAULT_ARTICLES_PER_LANG)
    ap.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB)
    ap.add_argument("--val-frac", type=float, default=DEFAULT_VAL_FRAC)
    ap.add_argument("--snapshot", default=WIKI_SNAPSHOT)
    args = ap.parse_args()

    out = cache_dir()
    print(f"[prepare] cache dir: {out}")
    print(
        f"[prepare] langs={args.langs} articles_per_lang={args.articles_per_lang} "
        f"vocab={args.vocab_size}"
    )

    corpus_per_lang: dict[str, list[str]] = {}
    for lang in args.langs:
        print(f"[prepare] fetching {lang}")
        texts = fetch_wikipedia_sample(lang, args.articles_per_lang, snapshot=args.snapshot)
        chars = sum(len(t) for t in texts)
        print(f"  -> {len(texts)} articles, {chars:,} chars")
        corpus_per_lang[lang] = texts

    print(f"[prepare] training BPE vocab={args.vocab_size}")
    tokenizer = train_bpe(corpus_per_lang, vocab_size=args.vocab_size)
    tok_path = out / "tokenizer.json"
    tokenizer.save(str(tok_path))
    actual_vocab = tokenizer.get_vocab_size()
    print(f"  -> saved {tok_path} (vocab_size={actual_vocab})")

    print("[prepare] tokenizing + sharding")
    info = tokenize_and_shard(corpus_per_lang, tokenizer, out, val_frac=args.val_frac)
    print(f"  train: {info['train_tokens']:,} tokens")
    print(f"  val:   {info['val_tokens']:,} tokens")

    manifest = {
        "format_version": "0.0.2",
        "created": date.today().isoformat(),
        "vocab_size": actual_vocab,
        "tokenizer": "tokenizer.json",
        "langs": args.langs,
        "articles_per_lang": args.articles_per_lang,
        "tokens_per_lang": info["tokens_per_lang"],
        "splits": {
            "train": {"path": "train.bin", "tokens": info["train_tokens"], "dtype": info["dtype"]},
            "val": {"path": "val.bin", "tokens": info["val_tokens"], "dtype": info["dtype"]},
        },
        "sources": [
            {
                "name": "wikimedia/wikipedia",
                "config": f"{args.snapshot}.{lang}",
                "license": "CC BY-SA 4.0",
                "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
                "url": "https://huggingface.co/datasets/wikimedia/wikipedia",
                "fetch_date": date.today().isoformat(),
            }
            for lang in args.langs
        ],
        "notes": (
            "Phase 1 prototype EU corpus. Production target per PLAN.md §4: "
            "CulturaX + OSCAR + Europeana + EuroParl + EUR-Lex, all 24 EU langs, "
            "128k vocab, full DSM Art. 4 opt-out + PII pipeline."
        ),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"  wrote {out / 'manifest.json'}")
    print("[prepare] done")


if __name__ == "__main__":
    main()
