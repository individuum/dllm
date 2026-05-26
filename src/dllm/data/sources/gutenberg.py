"""Project Gutenberg — public-domain literature, all 5 langs.

Gutenberg's catalog of ~70k public-domain books is the canonical "free
literature" corpus. Heavy on 19th/early-20th-century European authors,
which is exactly the cultural/literary depth we want for an EU-grounded
model. Counterbalances Wikipedia's encyclopedic register.

License: Public domain (US, where Gutenberg is hosted) — works whose
copyright has expired or never applied. Some texts have an additional
"Gutenberg Trademark License" wrapping the PD body; we strip headers/
footers so we ship only the PD body.

HF dataset: `manu/project_gutenberg` (multilingual snapshot of the index)
or fallback to `sedthh/gutenberg_english` for the EN subset. We pick by
language and yield book chunks (a whole book is often > 200k tokens —
we split at paragraph boundaries into ~8 KB chunks so per-doc lengths
stay reasonable).
"""
from __future__ import annotations

from typing import Iterator

_LANGS = ["de", "fr", "en", "it", "es"]
DOC_TARGET_CHARS = 8000  # ~2k tokens per chunk
MIN_CHARS = 500


def supported_langs() -> list[str]:
    return list(_LANGS)


def license_info() -> dict:
    return {
        "name": "Project Gutenberg",
        "hf_dataset": "manu/project_gutenberg",
        "license": "Public Domain (per Gutenberg eligibility criteria)",
        "license_url": "https://www.gutenberg.org/policy/permission.html",
        "url": "https://www.gutenberg.org/",
        "attribution": (
            "Public-domain literature from Project Gutenberg. Trademark "
            "header/footer stripped — only the public-domain body is included."
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
            # skip to the line after the marker
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
    """Split a book into paragraph-aligned ~DOC_TARGET_CHARS chunks."""
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

    if lang not in _LANGS:
        return

    # Try the multilingual snapshot first; fall back to sedthh for EN
    ds = None
    try:
        ds = load_dataset("manu/project_gutenberg", split="train", streaming=True)
    except Exception:  # noqa: BLE001
        if lang == "en":
            ds = load_dataset("sedthh/gutenberg_english", split="train", streaming=True)
        else:
            return  # no fallback for non-en yet — skip cleanly

    emitted = 0
    for row in ds:
        if emitted >= char_budget:
            return
        # manu/project_gutenberg schema: {"text": ..., "metadata": {"language": ...}}
        # sedthh/gutenberg_english schema: {"TEXT": ..., "METADATA": {...}}
        meta = row.get("metadata") or row.get("METADATA") or {}
        row_lang = (meta.get("language") or meta.get("LANGUAGE") or "").lower()
        # Accept exact match or "en, fr" style multi-lang labels
        if lang not in row_lang.split(",") and row_lang != lang:
            continue
        raw = row.get("text") or row.get("TEXT") or ""
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
