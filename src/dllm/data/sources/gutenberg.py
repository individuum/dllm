"""Project Gutenberg — public-domain literature.

Status: English-only via `sedthh/gutenberg_english`, which is the
best-maintained, schema-stable mirror on HF. The multilingual mirrors
(`manu/project_gutenberg`, etc.) either don't exist, return string-only
rows, or are English-only despite their name. For non-English literary
register we currently rely on Wikipedia + EuroParl + (future) OPUS Books.

License: Public Domain — works whose copyright has expired or never
applied. We strip Gutenberg's trademark header/footer so we ship only
the public-domain body.
"""
from __future__ import annotations

from typing import Iterator

_LANGS = ["en"]  # multilingual Gutenberg via HF is brittle; en only for now
DOC_TARGET_CHARS = 8000  # ~2 k tokens per chunk
MIN_CHARS = 500


def supported_langs() -> list[str]:
    return list(_LANGS)


def license_info() -> dict:
    return {
        "name": "Project Gutenberg (English subset)",
        "hf_dataset": "sedthh/gutenberg_english",
        "license": "Public Domain (per Gutenberg eligibility criteria)",
        "license_url": "https://www.gutenberg.org/policy/permission.html",
        "url": "https://www.gutenberg.org/",
        "attribution": (
            "Public-domain literature from Project Gutenberg. Trademark "
            "header/footer stripped — only the public-domain body is included. "
            "English-only at present; non-English Gutenberg integration is "
            "blocked on schema-stable HF mirrors."
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
    # Common field names across Gutenberg mirrors
    for key in ("text", "TEXT", "content", "body"):
        v = row.get(key)
        if isinstance(v, str) and v:
            return v
    return None


def iter_docs(lang: str, char_budget: int) -> Iterator[str]:
    from datasets import load_dataset  # noqa: PLC0415

    if lang not in _LANGS:
        return

    ds = load_dataset("sedthh/gutenberg_english", split="train", streaming=True)
    emitted = 0
    for row in ds:
        if emitted >= char_budget:
            return
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
