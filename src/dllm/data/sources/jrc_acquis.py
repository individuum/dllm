"""JRC-Acquis — EU law in 22 official languages, parallel corpus.

The Acquis Communautaire is the body of EU law (treaties, regulations,
directives, decisions) compiled by the JRC. Excellent for teaching the
model EU legal/administrative register, which generalizes well to policy,
contracts, and formal writing.

License: Public domain (EU institutional content under Council Decision
2011/833/EU).

HF dataset: `multi_eurlex` (a superset that re-hosts JRC-Acquis style
documents from EUR-Lex) is the most reliable mirror. Each document is a
full legal act with a CELEX identifier; we yield the article-body text.
"""
from __future__ import annotations

from typing import Iterator

_LANGS = ["de", "fr", "en", "it", "es"]
MIN_CHARS = 500  # skip tiny entries


def supported_langs() -> list[str]:
    return list(_LANGS)


def license_info() -> dict:
    return {
        "name": "JRC-Acquis (via multi_eurlex)",
        "hf_dataset": "multi_eurlex",
        "license": "Public Domain (Council Decision 2011/833/EU)",
        "license_url": "https://eur-lex.europa.eu/eli/dec/2011/833/oj",
        "url": "https://eur-lex.europa.eu/",
        "attribution": (
            "EU legal acts from EUR-Lex / JRC-Acquis. EU institutional content; "
            "free reuse permitted by EU open data policy."
        ),
    }


def iter_docs(lang: str, char_budget: int) -> Iterator[str]:
    from datasets import load_dataset  # noqa: PLC0415

    if lang not in _LANGS:
        return
    # multi_eurlex uses ISO 639-1 lang codes as config names
    try:
        ds = load_dataset("multi_eurlex", lang, split="train", streaming=True)
    except (ValueError, KeyError):
        # Older revisions name the config "all_languages" with per-row lang
        # filtering; this is the fallback path.
        ds = load_dataset("multi_eurlex", "all_languages", split="train", streaming=True)

    emitted = 0
    for row in ds:
        if emitted >= char_budget:
            return
        # Row schema (lang-specific cfg): {"text": "...", "labels": [...]}
        # Row schema (all_languages cfg): {"text": {"de": "...", ...}}
        text_field = row.get("text")
        if isinstance(text_field, dict):
            text = (text_field.get(lang) or "").strip()
        else:
            text = (text_field or "").strip()
        if len(text) < MIN_CHARS:
            continue
        remaining = char_budget - emitted
        if len(text) > remaining * 1.5:
            text = text[:remaining]
        yield text
        emitted += len(text)
