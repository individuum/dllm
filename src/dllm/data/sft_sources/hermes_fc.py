"""Hermes function-calling v1 — Nous Research's ChatML-native tool dataset.

Schema (HF dataset row):
    {
        "conversations": [
            {"from": "system" | "human" | "gpt" | "tool", "value": str},
            ...
        ]
    }

"value" content may embed:
  - `<tools>[{...},{...}]</tools>` in the system turn — the tool catalog
  - `<tool_call>{"name":..., "arguments":...}</tool_call>` in assistant turns
  - `<tool_response>{...}</tool_response>` in tool turns

Closest of all our sources to the target schema — minimal transformation:
just relabel ShareGPT-style roles and lift each tool_call block out into a
structured ToolCall so the canonical renderer can produce its own
`<tool_call>...</tool_call>` (matching our exact special-token form).

License: Apache 2.0.
"""
from __future__ import annotations

import json
import re
from typing import Iterator

from dllm.data.sft_format import Conversation, Message, ToolCall

HF_DATASET = "NousResearch/hermes-function-calling-v1"

_ROLE_MAP = {
    "system": "system",
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "tool": "tool",
}

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", flags=re.DOTALL)


def supported_langs() -> list[str]:
    return ["en"]


def task_type() -> str:
    return "tool_call"


def license_info() -> dict:
    return {
        "name": "Hermes function-calling v1",
        "hf_dataset": HF_DATASET,
        "license": "Apache-2.0",
        "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
        "url": f"https://huggingface.co/datasets/{HF_DATASET}",
        "attribution": (
            "NousResearch Hermes function-calling v1 dataset, released under "
            "Apache 2.0. Provides ChatML-native function-calling traces; used "
            "as the structural reference for the EuroAgent tool-call schema."
        ),
    }


def _extract_tool_calls(value: str) -> tuple[list[ToolCall], str]:
    """Lift every `<tool_call>{...}</tool_call>` out of `value`; return them
    as ToolCall objects plus the remaining text content (with the blocks removed).
    """
    calls: list[ToolCall] = []
    counter = 0
    for m in _TOOL_CALL_RE.finditer(value):
        raw = m.group(1).strip()
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("name")
        args = obj.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"_raw": args}
        if isinstance(name, str) and isinstance(args, dict):
            counter += 1
            calls.append(ToolCall(name=name, arguments=args, id=f"call_{counter}"))
    stripped = _TOOL_CALL_RE.sub("", value).strip()
    return calls, stripped


def _iter_configs():
    """Hermes-FC is sharded across several configs; iterate them all.

    The dataset has multiple parquet configs (e.g. 'func_calling_singleturn',
    'glaive_func_calling', 'json_mode_*'). Listing them explicitly avoids
    a remote-config probe and keeps streaming deterministic.
    """
    return [
        "func_calling_singleturn",
        "func_calling",
        "glaive_func_calling",
    ]


def iter_examples(lang: str, count_budget: int) -> Iterator[Conversation]:
    if lang != "en" or count_budget <= 0:
        return
    from datasets import load_dataset  # noqa: PLC0415

    emitted = 0
    for config in _iter_configs():
        if emitted >= count_budget:
            return
        try:
            ds = load_dataset(HF_DATASET, config, split="train", streaming=True)
        except Exception:  # noqa: BLE001 — config may have been renamed upstream
            continue
        for row in ds:
            if emitted >= count_budget:
                return
            convs = row.get("conversations") or []
            if not isinstance(convs, list) or not convs:
                continue
            msgs: list[Message] = []
            ok = True
            for turn in convs:
                if not isinstance(turn, dict):
                    ok = False
                    break
                raw_role = (turn.get("from") or "").lower()
                value = turn.get("value") or ""
                role = _ROLE_MAP.get(raw_role)
                if role is None:
                    ok = False
                    break
                if role == "assistant":
                    calls, text = _extract_tool_calls(value)
                    msgs.append(Message(role="assistant", content=text, tool_calls=calls))
                else:
                    msgs.append(Message(role=role, content=value))
            if not ok or not any(m.role == "assistant" for m in msgs):
                continue
            yield Conversation(messages=msgs)
            emitted += 1
