"""xLAM function-calling 60k — Salesforce's open tool-calling SFT set.

**Gated on HuggingFace** (verified 2026-05-27): visit the dataset page and
request access before this source will resolve. The orchestrator does NOT
include xLAM in the default allocation for this reason — opt in explicitly:

    python -m dllm.data.sft_prepare --sources xlam glaive_v2 hermes_fc oasst2 aya hand_written

The remaining EN tool-call sources (Glaive v2 = 113k, Hermes-FC = variable)
already provide more total examples than xLAM, so the default mix doesn't
strictly need it.

Schema (HF dataset row):
    {
        "id": int,
        "query": str,           # user request
        "answers": str (JSON),  # [{"name": str, "arguments": {...}}, ...]
        "tools": str (JSON),    # available tools — list of OpenAPI-ish schemas
    }

Single-turn: user query → one or more parallel tool_calls. No tool_response
follow-up, so the resulting Conversation is just [system, user, assistant].
System message embeds the tool catalog so the model learns to ground its calls.

License: CC-BY-4.0. Attribution required — embedded in license_info(),
propagated to the SFT manifest.
"""
from __future__ import annotations

import json
from typing import Iterator

from dllm.data.sft_format import Conversation, Message, ToolCall

HF_DATASET = "Salesforce/xlam-function-calling-60k"


def supported_langs() -> list[str]:
    return ["en"]


def task_type() -> str:
    return "tool_call"


def license_info() -> dict:
    return {
        "name": "xLAM function-calling 60k",
        "hf_dataset": HF_DATASET,
        "license": "CC-BY-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "url": f"https://huggingface.co/datasets/{HF_DATASET}",
        "attribution": (
            "Liu et al., 'xLAM: A Family of Large Action Models to Empower AI "
            "Agent Systems' (Salesforce AI Research, 2024). Used under CC-BY-4.0; "
            "attribution preserved in the SFT manifest and downstream model card."
        ),
    }


def _system_message_for_tools(tools_json: str) -> str:
    """Turn the dataset's tool catalog into a system prompt that primes the
    assistant to emit `<tool_call>{...}</tool_call>` in our schema."""
    return (
        "You are an assistant with access to the following tools. "
        "When a user request requires one, respond by emitting a "
        "<tool_call>{\"name\": ..., \"arguments\": {...}}</tool_call> block. "
        "If no tool is needed, answer in plain text.\n\n"
        f"Tools:\n{tools_json}"
    )


def _parse_answers(answers_raw) -> list[ToolCall]:
    """xLAM's `answers` field is a JSON-stringified list of {name, arguments}.
    Returns ToolCall objects; silently drops malformed entries."""
    if isinstance(answers_raw, str):
        try:
            parsed = json.loads(answers_raw)
        except json.JSONDecodeError:
            return []
    else:
        parsed = answers_raw
    out: list[ToolCall] = []
    for a in parsed if isinstance(parsed, list) else []:
        if not isinstance(a, dict):
            continue
        name = a.get("name")
        args = a.get("arguments", {})
        if isinstance(name, str) and isinstance(args, dict):
            out.append(ToolCall(name=name, arguments=args))
    return out


def iter_examples(lang: str, count_budget: int) -> Iterator[Conversation]:
    if lang != "en" or count_budget <= 0:
        return
    from datasets import load_dataset  # noqa: PLC0415

    ds = load_dataset(HF_DATASET, split="train", streaming=True)
    emitted = 0
    for row in ds:
        if emitted >= count_budget:
            return
        query = (row.get("query") or "").strip()
        tools_raw = row.get("tools") or "[]"
        tool_calls = _parse_answers(row.get("answers"))
        if not query or not tool_calls:
            continue
        tools_json = tools_raw if isinstance(tools_raw, str) else json.dumps(tools_raw)
        yield Conversation(messages=[
            Message(role="system", content=_system_message_for_tools(tools_json)),
            Message(role="user", content=query),
            Message(role="assistant", content="", tool_calls=tool_calls),
        ])
        emitted += 1
