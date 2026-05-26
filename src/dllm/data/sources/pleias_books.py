"""PleIAs per-language Public-Domain books — multilingual literary corpus.

The PleIAs collective curates large public-domain corpora *specifically*
for AI-Act-compliant training. Each per-language dataset is:

  - 100% works whose copyright has expired (EU >70-year rule)
  - de-duplicated and OCR-corrected
  - distributed in clean parquet (no script-based loaders)
  - paired with explicit provenance documentation aimed at EU AI Act
    Article 53's "sufficiently detailed summary" requirement

This source dispatches by language to four flagship PleIAs datasets:

  lang : dataset                  : approx. words
  ─────────────────────────────────────────────────
  de   : PleIAs/German-PD         : 37.6 B
  fr   : PleIAs/French-PD-Books   : 16.4 B
  it   : PleIAs/Italian-PD        : 12.9 B
  es   : PleIAs/Spanish-PD-Books  : 13.9 B

English is intentionally NOT here — `gutenberg.py` already provides
public-domain EN literature via `sedthh/gutenberg_english`. Adding a
second EN source would just over-weight EN literary register.

License: Public Domain — EU >70-year copyright expiry. PleIAs's
position on PD reuse (per their model cards) explicitly aligns with
EU Copyright Directive 2019 Art. 14 (works in the public domain remain
in the public domain after digitization).
"""
from __future__ import annotations

from typing import Iterator

# Language → (hf_dataset, approx_billion_words)
_LANG_TO_DATASET = {
    "de": ("PleIAs/German-PD", 37.6),
    "fr": ("PleIAs/French-PD-Books", 16.4),
    "it": ("PleIAs/Italian-PD", 12.9),
    "es": ("PleIAs/Spanish-PD-Books", 13.9),
}

# Books are LONG. We slice each book into doc-sized chunks so the
# tokenizer sees something comparable to a Wikipedia article length,
# not 500k tokens at once.
DOC_TARGET_CHARS = 8000  # ~2 k tokens / chunk
MIN_CHARS = 500


def supported_langs() -> list[str]:
    return list(_LANG_TO_DATASET)


def license_info() -> dict:
    return {
        "name": "PleIAs Public-Domain Books (per-language)",
        "hf_datasets": {lang: ds for lang, (ds, _) in _LANG_TO_DATASET.items()},
        "license": "Public Domain (EU Copyright Directive 2019 Art. 14)",
        "license_url": (
            "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32019L0790"
        ),
        "url": "https://huggingface.co/PleIAs",
        "attribution": (
            "Public-domain literature from PleIAs (de/fr/it/es). Each text's "
            "copyright has expired under EU law (>70 years post-mortem). "
            "Curated specifically to satisfy EU AI Act Article 53 disclosure."
        ),
    }


def _extract_text(row) -> str | None:
    """PleIAs datasets share a common schema but field names drift slightly
    across the collection. Try the common ones in order."""
    if isinstance(row, str):
        return row
    if not isinstance(row, dict):
        return None
    for key in ("text", "complete_text", "content", "body", "TEXT"):
        v = row.get(key)
        if isinstance(v, str) and v:
            return v
    return None


def _split_into_chunks(text: str) -> Iterator[str]:
    """Paragraph-aligned ~DOC_TARGET_CHARS chunks. Preserves natural breaks."""
    paragraphs = text.split("\n\n")
    buf: list[str] = []
    buf_len = 0
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        buf.append(p)
        buf_len += len(p) + 2
        if buf_len >= DOC_TARGET_CHARS:
            yield "\n\n".join(buf)
            buf, buf_len = [], 0
    if buf_len >= MIN_CHARS:
        yield "\n\n".join(buf)


def iter_docs(lang: str, char_budget: int) -> Iterator[str]:
    from datasets import load_dataset  # noqa: PLC0415

    if lang not in _LANG_TO_DATASET:
        return
    dataset_name, _ = _LANG_TO_DATASET[lang]

    ds = load_dataset(dataset_name, split="train", streaming=True)
    emitted = 0
    for row in ds:
        if emitted >= char_budget:
            return
        raw = _extract_text(row)
        if not raw or len(raw) < MIN_CHARS:
            continue
        for chunk in _split_into_chunks(raw):
            if emitted >= char_budget:
                return
            remaining = char_budget - emitted
            if len(chunk) > remaining * 1.5:
                chunk = chunk[:remaining]
            yield chunk
            emitted += len(chunk)
