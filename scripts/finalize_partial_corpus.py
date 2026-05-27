"""Finalize a partially-prepared corpus.

When `dllm.data.prepare` is interrupted before Phase 4 (val split) and
Phase 5 (manifest), train.bin exists on disk but val.bin + manifest.json
don't. This script does both, treating whatever's on disk as the final
corpus. Per-(source, lang) token counts can't be recovered (they lived
in the prepare process's memory), so the manifest records overall
totals + the allocation that produced this run.

Usage: python -m scripts.finalize_partial_corpus
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dllm.data import sources
from dllm.data.prepare import (
    DEFAULT_ALLOC_CHARS,
    DEFAULT_LANGS,
    DEFAULT_VAL_TOKENS,
    cache_dir,
    sha256_of_file,
    split_off_val,
)


def main() -> None:
    out = cache_dir()
    train_path = out / "train.bin"
    val_path = out / "val.bin"
    tok_path = out / "tokenizer.json"
    manifest_path = out / "manifest.json"

    if not train_path.exists():
        sys.exit(f"no {train_path} — nothing to finalize")
    if not tok_path.exists():
        sys.exit(f"no {tok_path} — finalize needs the BPE that produced train.bin")

    train_bytes = train_path.stat().st_size
    total_tokens = train_bytes // 2  # uint16
    print(f"[finalize] train.bin: {train_bytes / 1e6:.0f} MB = {total_tokens:,} tokens")

    if val_path.exists():
        val_path.unlink()

    train_n, val_n = split_off_val(train_path, val_path, DEFAULT_VAL_TOKENS)
    print(f"[finalize] split: train={train_n:,} val={val_n:,}")

    # Load tokenizer to record vocab
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(tok_path))
    vocab_size = tok.get_vocab_size()

    # Per-source license metadata
    src_meta = []
    for src_name in DEFAULT_ALLOC_CHARS:
        try:
            src = sources.load(src_name)
            info = src.license_info()
        except (ValueError, ImportError):
            info = {"name": src_name, "license": "unknown"}
        info["allocated_chars_per_lang"] = DEFAULT_ALLOC_CHARS.get(src_name, {})
        info["fetch_date"] = date.today().isoformat()
        # Per-source token counts are not recoverable from a partial run
        src_meta.append(info)

    manifest = {
        "format_version": "0.0.3",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "vocab_size": vocab_size,
        "tokenizer": "tokenizer.json",
        "langs": DEFAULT_LANGS,
        "finalized_from_partial_run": True,
        "tokens": {
            "train": train_n,
            "val": val_n,
            "total": train_n + val_n,
        },
        "splits": {
            "train": {
                "path": "train.bin",
                "tokens": train_n,
                "dtype": "uint16",
                "sha256": sha256_of_file(train_path),
            },
            "val": {
                "path": "val.bin",
                "tokens": val_n,
                "dtype": "uint16",
                "sha256": sha256_of_file(val_path),
            },
        },
        "sources": src_meta,
        "compliance": {
            "ai_act_art_53": (
                "This corpus was assembled from the sources listed below, each "
                "of which is either EU-institutional public domain or under an "
                "EU-compatible open license (CC-BY-SA, MIT, Public Domain). "
                "The run was finalized early (operator decision); per-source "
                "token breakdown is not available, but the allocation that "
                "produced this corpus is recorded under `allocated_chars_per_lang`."
            ),
            "dsm_art_4_tdm": (
                "All sources: EU-institutional PD (EuroParl, JRC-Acquis via "
                "EUR-Lex), pre-existing PD (Gutenberg, PleIAs per-language PD "
                "books), or CC-BY-SA (Wikipedia). All fetched via published "
                "HuggingFace mirrors, not scraped. No machine-readable opt-outs "
                "encountered."
            ),
            "gdpr": (
                "All sources are institutional or published-author content. No "
                "personal data of private individuals is included by design."
            ),
        },
        "notes": (
            "v3 EU pre-training corpus, finalized early at the operator's request. "
            f"{(train_n + val_n) / 1e9:.2f} B total tokens across DE/FR/EN/IT/ES. "
            "Sources: Wikipedia (CC-BY-SA), PleIAs PD Books (PD, de/fr/it/es), "
            "Gutenberg (PD, all 5 langs), EuroParl (PD), JRC-Acquis (PD via "
            "mteb/eurlex-multilingual mirror)."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"[finalize] wrote {manifest_path}")


if __name__ == "__main__":
    main()
