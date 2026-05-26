"""Wikipedia source — bulk of the multi-lingual corpus.

Streams the Wikimedia Foundation's curated `wikimedia/wikipedia` HF dataset,
which is the cleanest open mirror (per-snapshot, per-language, full-text,
already stripped of templates / nav / infoboxes).

License: CC BY-SA 4.0 — attribution + share-alike. Required action: keep
the source URL + revision in the manifest; downstream model release must
also be CC-BY-SA or compatible.
"""
from __future__ import annotations

from typing import Iterator

SNAPSHOT = "20231101"
MIN_CHARS = 200  # drop stubs

_LANGS = ["de", "fr", "en", "it", "es", "nl", "pl"]


def supported_langs() -> list[str]:
    return list(_LANGS)


def license_info() -> dict:
    return {
        "name": "Wikipedia",
        "hf_dataset": "wikimedia/wikipedia",
        "snapshot": SNAPSHOT,
        "license": "CC BY-SA 4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "url": "https://huggingface.co/datasets/wikimedia/wikipedia",
        "attribution": (
            "Text from Wikipedia, available under CC BY-SA 4.0. Article-level "
            "edit history is upstream in the snapshot manifest."
        ),
    }


def iter_docs(lang: str, char_budget: int) -> Iterator[str]:
    """Stream Wikipedia articles in `lang` until char_budget characters emitted."""
    from datasets import load_dataset  # noqa: PLC0415

    if lang not in _LANGS:
        return
    config = f"{SNAPSHOT}.{lang}"
    ds = load_dataset("wikimedia/wikipedia", config, split="train", streaming=True)
    emitted = 0
    for row in ds:
        if emitted >= char_budget:
            return
        text = row.get("text") or ""
        if len(text) < MIN_CHARS:
            continue
        # Defensive truncation if a single article would blow the remaining
        # budget by 10×. Keeps the budget tight without losing the whole doc.
        remaining = char_budget - emitted
        if len(text) > remaining * 1.5:
            text = text[:remaining]
        yield text
        emitted += len(text)
