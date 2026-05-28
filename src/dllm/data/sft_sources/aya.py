"""Aya dataset — Cohere For AI's human-annotated multilingual instructions.

Schema (HF dataset row):
    {
        "inputs":         str,   # user instruction / question
        "targets":        str,   # human-written response
        "language":       str,   # full language name, e.g. "German", "English"
        "language_code":  str,   # ISO 639-1, e.g. "deu", "eng"
        ...
    }

Aya is *human-written* across 65 languages — distinct from translated or
synthetic instruction sets. The DE/EN slices give us native-quality SFT
content without translation artifacts, especially valuable for German
where high-quality open-license native instruction data is scarce.

License: Apache 2.0. Cite C4AI Cohere Aya in the model card per their request.
"""
from __future__ import annotations

from typing import Iterator

from dllm.data.sft_format import Conversation, Message

HF_DATASET = "CohereForAI/aya_dataset"

# Aya uses ISO 639-3-ish names; map to our 2-letter codes.
_LANG_NAME_TO_CODE = {
    "English": "en",
    "German": "de",
    "French": "fr",
    "Spanish": "es",
    "Italian": "it",
}

_SUPPORTED = ["en", "de", "fr", "es", "it"]


def supported_langs() -> list[str]:
    return list(_SUPPORTED)


def task_type() -> str:
    return "general"


def license_info() -> dict:
    return {
        "name": "Aya dataset (CohereForAI)",
        "hf_dataset": HF_DATASET,
        "license": "Apache-2.0",
        "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
        "url": f"https://huggingface.co/datasets/{HF_DATASET}",
        "attribution": (
            "Singh et al., 'Aya Dataset: An Open-Access Collection for "
            "Multilingual Instruction Tuning' (Cohere For AI, 2024). Released "
            "under Apache 2.0. Used for human-written DE/EN instruction SFT."
        ),
    }


def iter_examples(lang: str, count_budget: int) -> Iterator[Conversation]:
    if lang not in _SUPPORTED or count_budget <= 0:
        return
    from datasets import load_dataset  # noqa: PLC0415

    ds = load_dataset(HF_DATASET, split="train", streaming=True)
    emitted = 0
    for row in ds:
        if emitted >= count_budget:
            return
        # Filter by Aya's language column. The full-name field is more stable
        # across dataset revisions than the iso-639-3 code field.
        lang_name = row.get("language") or ""
        if _LANG_NAME_TO_CODE.get(lang_name) != lang:
            continue
        inputs = (row.get("inputs") or "").strip()
        targets = (row.get("targets") or "").strip()
        if not inputs or not targets:
            continue
        yield Conversation(messages=[
            Message(role="user", content=inputs),
            Message(role="assistant", content=targets),
        ])
        emitted += 1
