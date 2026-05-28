"""Glaive function-calling v2, machine-translated to German.

This source reads a pre-translated JSONL produced offline by
`scripts/translate_glaive_to_de.py`. Translation does NOT happen at
sft_prepare time — that script does a one-shot opus-mt-en-de pass over
the Glaive v2 stream and writes the result to:

    data/cache/sft/glaive_v2_de.jsonl

Each line is one DE Conversation in the same schema hand_written.py uses:

    {"lang": "de",
     "messages": [{"role": ..., "content": "...", "tool_calls": [...]?, ...}]}

This file just streams that JSONL — same as hand_written.py reads its
local files. No transformers/CUDA imports here, so the prepare pipeline
stays lean.

If the JSONL doesn't exist, this source yields nothing and the orchestrator
silently skips it (operator gets a friendly nudge to run the offline script).

License: Apache 2.0 (Glaive v2) + Apache 2.0 (opus-mt-en-de). Both are
recorded in `license_info()` and propagated to the SFT manifest.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from dllm.data.sft_format import Conversation, Message, ToolCall

HF_DATASET = "glaiveai/glaive-function-calling-v2"
MT_MODEL = "Helsinki-NLP/opus-mt-en-de"


def _jsonl_path() -> Path:
    here = Path(__file__).resolve()
    # src/dllm/data/sft_sources/glaive_v2_de.py → ../../../../data/cache/sft/glaive_v2_de.jsonl
    return here.parents[4] / "data" / "cache" / "sft" / "glaive_v2_de.jsonl"


def supported_langs() -> list[str]:
    return ["de"]


def task_type() -> str:
    return "tool_call"


def license_info() -> dict:
    return {
        "name": "Glaive function-calling v2 (DE, machine-translated)",
        "hf_dataset": HF_DATASET,
        "license": "Apache-2.0 (Glaive) + Apache-2.0 (opus-mt-en-de)",
        "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
        "url": f"https://huggingface.co/datasets/{HF_DATASET}",
        "mt_model": MT_MODEL,
        "attribution": (
            "EN→DE translation of Glaive function-calling v2 (Apache 2.0) via "
            "Helsinki-NLP/opus-mt-en-de (Apache 2.0). Tool names, tool-call "
            "arguments, and JSON tool responses are preserved verbatim; only "
            "natural-language prose is translated. Translation produced offline "
            "by scripts/translate_glaive_to_de.py and stored at "
            "data/cache/sft/glaive_v2_de.jsonl."
        ),
    }


def _parse_message(d: dict) -> Message | None:
    role = d.get("role")
    if role not in ("system", "user", "assistant", "tool"):
        return None
    tool_calls: list[ToolCall] = []
    for tc in d.get("tool_calls") or []:
        if isinstance(tc, dict) and isinstance(tc.get("name"), str) and isinstance(tc.get("arguments"), dict):
            tool_calls.append(ToolCall(name=tc["name"], arguments=tc["arguments"], id=tc.get("id")))
    return Message(
        role=role,
        content=d.get("content") or "",
        tool_calls=tool_calls,
        tool_call_id=d.get("tool_call_id"),
    )


def iter_examples(lang: str, count_budget: int) -> Iterator[Conversation]:
    if lang != "de" or count_budget <= 0:
        return
    path = _jsonl_path()
    if not path.exists():
        print(
            f"  [glaive_v2_de] {path} not found — run "
            "`python -m scripts.translate_glaive_to_de` first. Yielding 0 examples."
        )
        return
    emitted = 0
    with path.open(encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, 1):
            if emitted >= count_budget:
                return
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [glaive_v2_de] {path}:{line_no} invalid JSON: {e}")
                continue
            if obj.get("lang") != "de":
                continue
            messages: list[Message] = []
            for m in obj.get("messages") or []:
                msg = _parse_message(m)
                if msg is not None:
                    messages.append(msg)
            if any(m.role == "assistant" for m in messages):
                yield Conversation(messages=messages)
                emitted += 1
