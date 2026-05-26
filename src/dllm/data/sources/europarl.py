"""European Parliament proceedings — formal political register, all 5 langs.

EuroParl is the canonical sentence-aligned parallel corpus of EP debates,
extracted by Philipp Koehn (Edinburgh) from the public proceedings record.
We only consume the MONOLINGUAL side per language — the parallel alignment
is irrelevant for pre-training and the deduplication of identical
translations across alignment pairs is a separate concern.

License: Public domain (EU institutional content under Council Decision
2011/833/EU — the EU's open data policy).

HF dataset: `Helsinki-NLP/europarl` (re-hosted by University of Helsinki).
Each row is a sentence; we concatenate adjacent sentences into ~4 KB
"documents" before yielding so the tokenizer sees real context windows.
"""
from __future__ import annotations

from typing import Iterator

# EuroParl ships only a subset of EU languages — these 5 are all in scope
_LANGS = ["de", "fr", "en", "it", "es"]
DOC_TARGET_CHARS = 4000  # ~1k tokens per doc


def supported_langs() -> list[str]:
    return list(_LANGS)


def license_info() -> dict:
    return {
        "name": "EuroParl v10",
        "hf_dataset": "Helsinki-NLP/europarl",
        "license": "Public Domain (Council Decision 2011/833/EU)",
        "license_url": "https://eur-lex.europa.eu/eli/dec/2011/833/oj",
        "url": "https://www.statmt.org/europarl/",
        "attribution": (
            "European Parliament proceedings 1996–2011. EU institutional "
            "content under the EU's open data policy; no copyright restrictions."
        ),
    }


def iter_docs(lang: str, char_budget: int) -> Iterator[str]:
    from datasets import load_dataset  # noqa: PLC0415

    if lang not in _LANGS:
        return
    # The Helsinki-NLP/europarl set is published as language pairs. Pick any
    # pair containing our target language and harvest the requested side.
    # `en` is always one side of every other pair, so we use `lang-en` (or
    # `en-lang` reversed) as the universal selector.
    if lang == "en":
        cfg = "de-en"  # arbitrary partner; we keep only the EN side
        key = "en"
    else:
        cfg = f"{lang}-en"  # e.g. de-en, fr-en, it-en, es-en
        key = lang

    try:
        ds = load_dataset("Helsinki-NLP/europarl", cfg, split="train", streaming=True)
    except (ValueError, KeyError):
        # Some configs use reversed naming; fall back.
        cfg_rev = "-".join(reversed(cfg.split("-")))
        ds = load_dataset("Helsinki-NLP/europarl", cfg_rev, split="train", streaming=True)

    buf: list[str] = []
    buf_len = 0
    emitted = 0
    for row in ds:
        if emitted >= char_budget:
            return
        # Row schema: {"translation": {"en": "...", "de": "..."}}
        tx = row.get("translation") or {}
        sent = (tx.get(key) or "").strip()
        if not sent:
            continue
        buf.append(sent)
        buf_len += len(sent) + 1
        if buf_len >= DOC_TARGET_CHARS:
            doc = " ".join(buf)
            yield doc
            emitted += len(doc)
            buf, buf_len = [], 0
    if buf and emitted < char_budget:
        yield " ".join(buf)
