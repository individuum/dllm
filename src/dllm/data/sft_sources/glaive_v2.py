"""Glaive function-calling v2 — multi-turn tool dialogues.

Schema (HF dataset row, verified 2026-05-27):
    {
        "system": str,   # starts with "SYSTEM: " prefix; contains tool catalog
        "chat":  str,    # turns separated by "\\n\\n\\n", each prefixed by
                         # "USER: ", "ASSISTANT: ", or "FUNCTION RESPONSE: ".
                         # Tool calls are EMBEDDED in ASSISTANT turns as:
                         #   "<functioncall> {\"name\":..., \"arguments\": '{...}'}"
                         # — note the Python-string-quoted inner arguments,
                         # not pure JSON. "<|endoftext|>" terminates each turn.
    }

Distinct from xLAM in two ways:
  - **Multi-turn**: user → tool_call → tool_response → assistant follow-up,
    sometimes several rounds. This is the shape that fine-tunes the model to
    reason about tool output, not just to emit a call.
  - **Free-form tool catalog**: the system prompt embeds tools as plain text
    rather than xLAM's structured field.

License: Apache 2.0.
"""
from __future__ import annotations

import json
import re
from typing import Iterator

from dllm.data.sft_format import Conversation, Message, ToolCall

HF_DATASET = "glaiveai/glaive-function-calling-v2"

# Each chat turn begins with one of these markers. Glaive separates turns
# by blank lines and ends each with "<|endoftext|>".
_TURN_RE = re.compile(
    r"^(USER|ASSISTANT|FUNCTION RESPONSE):\s*",
    flags=re.MULTILINE,
)

# Inside an ASSISTANT turn, function calls appear as:
#   <functioncall> {"name": "...", "arguments": '{...}'}
# The arguments value is single-quoted Python-string-style around inner JSON.
_FN_CALL_RE = re.compile(
    r"<functioncall>\s*(\{.*?\})\s*(?:<\|endoftext\|>)?\s*$",
    flags=re.DOTALL,
)

_EOT_RE = re.compile(r"\s*<\|endoftext\|>\s*$")


def supported_langs() -> list[str]:
    return ["en"]


def task_type() -> str:
    return "tool_call"


def license_info() -> dict:
    return {
        "name": "Glaive function-calling v2",
        "hf_dataset": HF_DATASET,
        "license": "Apache-2.0",
        "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
        "url": f"https://huggingface.co/datasets/{HF_DATASET}",
        "attribution": (
            "Glaive function-calling v2 dataset by Glaive AI, released under "
            "Apache 2.0. Used for multi-turn tool-use SFT in the EuroAgent mix."
        ),
    }


def _split_turns(chat: str) -> list[tuple[str, str]]:
    """Split Glaive's flat `chat` string into [(role_marker, body), ...]."""
    parts = _TURN_RE.split(chat)
    # _TURN_RE.split with one capture group yields: [pre, marker, body, marker, body, ...]
    turns: list[tuple[str, str]] = []
    for i in range(1, len(parts), 2):
        marker = parts[i]
        body = parts[i + 1] if i + 1 < len(parts) else ""
        # Strip trailing <|endoftext|> and surrounding whitespace.
        body = _EOT_RE.sub("", body).strip()
        turns.append((marker, body))
    return turns


def _parse_glaive_call_object(blob: str) -> ToolCall | None:
    """Parse a Glaive `<functioncall> {...}` payload.

    Glaive's payload is JSON-shaped except that ``arguments`` is wrapped in
    *single quotes* containing an inner JSON object. ``json.loads`` rejects
    single-quoted strings, so we extract `name` and `arguments` by regex,
    then decode the inner JSON.
    """
    # name: extract value of the "name" key.
    name_m = re.search(r'"name"\s*:\s*"([^"]+)"', blob)
    if not name_m:
        return None
    name = name_m.group(1)
    # arguments: value is either a single-quoted JSON string '{...}' or a
    # double-quoted JSON string "{...}", or rarely a nested object {...}.
    args_obj: dict = {}
    sq = re.search(r"\"arguments\"\s*:\s*'(\{.*?\})'", blob, flags=re.DOTALL)
    dq = re.search(r"\"arguments\"\s*:\s*\"(\{.*?\})\"", blob, flags=re.DOTALL)
    nested = re.search(r"\"arguments\"\s*:\s*(\{.*\})\s*\}", blob, flags=re.DOTALL)
    inner: str | None = None
    if sq:
        inner = sq.group(1)
    elif dq:
        # Double-quoted form: escapes are JSON's, so a json-string-decode is needed.
        inner = json.loads(f'"{dq.group(1)}"')
    elif nested:
        inner = nested.group(1)
    if inner:
        try:
            parsed = json.loads(inner)
            if isinstance(parsed, dict):
                args_obj = parsed
        except json.JSONDecodeError:
            args_obj = {"_raw": inner}
    return ToolCall(name=name, arguments=args_obj)


def _split_assistant_turn(body: str) -> tuple[str, list[ToolCall]]:
    """Lift any `<functioncall> {...}` from an assistant turn, return
    (remaining_text, [tool_calls])."""
    calls: list[ToolCall] = []
    m = _FN_CALL_RE.search(body)
    text = body
    while m:
        tc = _parse_glaive_call_object(m.group(1))
        if tc is not None:
            calls.append(tc)
        # Remove the matched block and keep scanning (rarely multiple in one turn).
        text = (text[: m.start()] + text[m.end():]).strip()
        m = _FN_CALL_RE.search(text)
    return text, calls


def _strip_system_prefix(system: str) -> str:
    return re.sub(r"^SYSTEM:\s*", "", system or "", count=1).strip()


def _conversation_from_row(system: str, chat: str) -> Conversation | None:
    turns = _split_turns(chat)
    if not turns:
        return None
    msgs: list[Message] = []
    sys_clean = _strip_system_prefix(system)
    if sys_clean:
        msgs.append(Message(role="system", content=sys_clean))
    last_tool_id_counter = 0
    for marker, body in turns:
        if marker == "USER":
            if body:
                msgs.append(Message(role="user", content=body))
        elif marker == "ASSISTANT":
            text, calls = _split_assistant_turn(body)
            if calls:
                last_tool_id_counter += 1
                calls_with_ids = [
                    ToolCall(name=tc.name, arguments=tc.arguments,
                             id=f"call_{last_tool_id_counter}_{i}")
                    for i, tc in enumerate(calls)
                ]
                msgs.append(Message(role="assistant", content=text, tool_calls=calls_with_ids))
            elif text:
                msgs.append(Message(role="assistant", content=text))
        elif marker == "FUNCTION RESPONSE":
            tool_id = (
                f"call_{last_tool_id_counter}_0" if last_tool_id_counter else None
            )
            msgs.append(Message(role="tool", content=body, tool_call_id=tool_id))
    if not any(m.role == "assistant" for m in msgs):
        return None
    return Conversation(messages=msgs)


def iter_examples(lang: str, count_budget: int) -> Iterator[Conversation]:
    if lang != "en" or count_budget <= 0:
        return
    from datasets import load_dataset  # noqa: PLC0415

    ds = load_dataset(HF_DATASET, split="train", streaming=True)
    emitted = 0
    for row in ds:
        if emitted >= count_budget:
            return
        chat = row.get("chat") or ""
        system = row.get("system") or ""
        if not chat:
            continue
        conv = _conversation_from_row(system, chat)
        if conv is None:
            continue
        yield conv
        emitted += 1
