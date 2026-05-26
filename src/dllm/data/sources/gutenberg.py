"""Project Gutenberg — public-domain literature in 5 EU languages.

Two HF mirrors are used, both schema-stable and parquet-native (no
script loaders), both MIT-licensed by the same maintainer:

  - `sedthh/gutenberg_english` for English (~48k books).
  - `sedthh/gutenberg_multilang` for de/fr/it/es (book counts:
    DE 1735, FR 2863, IT 692, ES 717). Language lives inside a
    JSON-encoded METADATA column, so we filter row-by-row.

License: MIT on the compilation; underlying texts are Public Domain per
Project Gutenberg eligibility. The Gutenberg trademark header/footer is
stripped from each book so we ship only the public-domain body.
"""
from __future__ import annotations

import json
from typing import Iterator

_LANGS = ["en", "de", "fr", "it", "es"]
_EN_DATASET = "sedthh/gutenberg_english"
_MULTILANG_DATASET = "sedthh/gutenberg_multilang"

DOC_TARGET_CHARS = 8000  # ~2 k tokens per chunk
MIN_CHARS = 500


def supported_langs() -> list[str]:
    return list(_LANGS)


def license_info() -> dict:
    return {
        "name": "Project Gutenberg",
        "hf_dataset": f"{_EN_DATASET} (en) + {_MULTILANG_DATASET} (de/fr/it/es)",
        "license": "MIT (dataset compilation) + Public Domain (underlying texts)",
        "license_url": "https://www.gutenberg.org/policy/permission.html",
        "url": "https://www.gutenberg.org/",
        "attribution": (
            "Public-domain literature from Project Gutenberg. Trademark "
            "header/footer stripped — only the public-domain body is included. "
            "Schema-stable parquet mirrors maintained by sedthh, MIT-licensed."
        ),
    }


_HEADER_END_MARKERS = (
    "*** START OF THE PROJECT GUTENBERG EBOOK",
    "*** START OF THIS PROJECT GUTENBERG EBOOK",
    "***START OF THE PROJECT GUTENBERG EBOOK",
)
_FOOTER_START_MARKERS = (
    "*** END OF THE PROJECT GUTENBERG EBOOK",
    "*** END OF THIS PROJECT GUTENBERG EBOOK",
    "***END OF THE PROJECT GUTENBERG EBOOK",
)


def _strip_gutenberg_envelope(text: str) -> str:
    """Remove Gutenberg's header/footer — those carry a trademark license,
    only the body in between is public domain."""
    for marker in _HEADER_END_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            text = text[idx:]
            nl = text.find("\n")
            if nl != -1:
                text = text[nl + 1 :]
            break
    for marker in _FOOTER_START_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
            break
    return text.strip()


def _split_into_chunks(text: str) -> Iterator[str]:
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


def _extract_text(row) -> str | None:
    """Defensive row→text extractor. HF schemas drift; absorb gracefully."""
    if isinstance(row, str):
        return row
    if not isinstance(row, dict):
        return None
    for key in ("TEXT", "text", "content", "body"):
        v = row.get(key)
        if isinstance(v, str) and v:
            return v
    return None


def _row_lang(row) -> str | None:
    """Extract the ISO 639-1 language code from a row's METADATA JSON.
    Returns None if the field is missing or unparseable."""
    if not isinstance(row, dict):
        return None
    meta = row.get("METADATA") or row.get("metadata")
    if isinstance(meta, dict):
        lang = meta.get("language")
    elif isinstance(meta, str):
        try:
            lang = json.loads(meta).get("language")
        except (json.JSONDecodeError, AttributeError):
            return None
    else:
        return None
    if not isinstance(lang, str):
        return None
    return lang.strip().lower()[:2] or None


def iter_docs(lang: str, char_budget: int) -> Iterator[str]:
    from datasets import load_dataset  # noqa: PLC0415

    if lang not in _LANGS:
        return

    if lang == "en":
        ds = load_dataset(_EN_DATASET, split="train", streaming=True)
        lang_filter = None  # the English mirror is already monolingual
    else:
        ds = load_dataset(_MULTILANG_DATASET, split="train", streaming=True)
        lang_filter = lang

    emitted = 0
    for row in ds:
        if emitted >= char_budget:
            return
        if lang_filter is not None and _row_lang(row) != lang_filter:
            continue
        raw = _extract_text(row)
        if not raw:
            continue
        body = _strip_gutenberg_envelope(raw)
        if len(body) < MIN_CHARS:
            continue
        for chunk in _split_into_chunks(body):
            if emitted >= char_budget:
                return
            remaining = char_budget - emitted
            if len(chunk) > remaining * 1.5:
                chunk = chunk[:remaining]
            yield chunk
            emitted += len(chunk)
