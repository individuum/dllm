"""JRC-Acquis / EUR-Lex — EU legislative text in all 5 target languages.

Provides the body of EU legal acts (commission decisions, regulations,
directives, etc.) compiled from EUR-Lex. Excellent for teaching the
model EU legal/administrative register, which generalizes well to policy,
contracts, and formal writing.

HF dataset: `mteb/eurlex-multilingual` — a parquet-converted mirror
maintained by the MTEB benchmark. We use this instead of the original
`multi_eurlex` (and its forks: coastalcph, Muennighoff, joelniklaus)
because every one of those still ships a `multi_eurlex.py` loading
script, which the modern `datasets` library (>=2.20) refuses to execute
even with `trust_remote_code=True`. The mteb mirror has the same
content (~65k EU laws, train/val/test split, CELEX IDs) in flat parquet.

License: The underlying EU legislative text is **public domain** under
Council Decision 2011/833/EU (the EU's open data policy). The mteb
mirror itself is **CC-BY-SA-4.0**, matching our Wikipedia source —
attribution to mteb + URL is preserved in `license_info()` and lands in
the corpus manifest.

Row schema: `{"id": str, "text": str, "label": list[int]}`. We only
consume `text` — labels are EUROVOC classification IDs, irrelevant for
pre-training.
"""
from __future__ import annotations

from typing import Iterator

_LANGS = ["de", "fr", "en", "it", "es"]
MIN_CHARS = 500  # skip tiny entries (boilerplate-only acts)


def supported_langs() -> list[str]:
    return list(_LANGS)


def license_info() -> dict:
    return {
        "name": "JRC-Acquis / EUR-Lex (via mteb/eurlex-multilingual)",
        "hf_dataset": "mteb/eurlex-multilingual",
        "license": (
            "CC-BY-SA-4.0 (mteb mirror); underlying EU legislative content "
            "is public domain per Council Decision 2011/833/EU"
        ),
        "license_url": "https://eur-lex.europa.eu/eli/dec/2011/833/oj",
        "url": "https://huggingface.co/datasets/mteb/eurlex-multilingual",
        "attribution": (
            "EU legal acts from EUR-Lex, mirrored by the MTEB benchmark as a "
            "parquet-converted variant of `multi_eurlex`. EU institutional "
            "content; free reuse permitted by EU open data policy."
        ),
    }


def iter_docs(lang: str, char_budget: int) -> Iterator[str]:
    from datasets import load_dataset  # noqa: PLC0415

    if lang not in _LANGS:
        return
    ds = load_dataset("mteb/eurlex-multilingual", lang, split="train", streaming=True)

    emitted = 0
    for row in ds:
        if emitted >= char_budget:
            return
        text = (row.get("text") or "").strip()
        if len(text) < MIN_CHARS:
            continue
        remaining = char_budget - emitted
        if len(text) > remaining * 1.5:
            text = text[:remaining]
        yield text
        emitted += len(text)
