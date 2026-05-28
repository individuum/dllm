"""Hand-written EU-specific SFT examples — local JSONL files.

This is the long-tail hook for the EuroAgent mix: tool definitions and
multi-turn dialogues we write ourselves to cover EU-specific corner cases
that no open dataset covers cleanly. Examples in scope:

  - EUR-Lex / national gazette lookup tools (search_eur_lex, get_kvk_entry).
  - Multilingual code-switching turns (user asks in DE, tool result is EN).
  - Refusals grounded in GDPR / DSA reasoning (decline to dox an EU resident).
  - EU-specific units, formats (IBAN, BSN, NIE, eIDAS-attested attributes).

File layout (under repo root):

    data/sft_handwritten/
        de_tool_calls.jsonl
        en_tool_calls.jsonl
        de_agentic.jsonl
        en_agentic.jsonl

Each line is one Conversation, serialized as:

    {
      "lang": "de" | "en",
      "messages": [
        {"role": "system" | "user" | "assistant" | "tool",
         "content": str,
         "tool_calls": [{"name": str, "arguments": {...}, "id": str?}],
         "tool_call_id": str?}
      ]
    }

Missing fields default to absent. Missing directory or empty files yield
nothing — this source is silently no-op until we populate it. That makes
the prepare pipeline safe to run from day one and lets us add curated
examples without code changes.

License: declared by us in license_info() — examples we write are released
under Apache 2.0 with the rest of the repo.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from dllm.data.sft_format import Conversation, Message, ToolCall

_SUPPORTED = ["en", "de"]


def _root() -> Path:
    """Repo root / data / sft_handwritten."""
    here = Path(__file__).resolve()
    # src/dllm/data/sft_sources/hand_written.py → ../../../../data/sft_handwritten
    return here.parents[4] / "data" / "sft_handwritten"


def supported_langs() -> list[str]:
    return list(_SUPPORTED)


def task_type() -> str:
    # Primary axis is tool_call: this source is the ONLY DE tool-call coverage
    # in the v0.1 mix (xlam/glaive/hermes are EN-only and not translation-safe).
    # Files like *_agentic.jsonl are accepted but the source registers as
    # tool_call because that's its load-bearing role in the mix.
    return "tool_call"


def license_info() -> dict:
    return {
        "name": "EuroAgent hand-written examples",
        "hf_dataset": None,
        "license": "Apache-2.0",
        "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
        "url": "local: data/sft_handwritten/",
        "attribution": (
            "Curated by the EuroDLLM team to cover EU-specific tool calls "
            "(EUR-Lex search, GDPR-aware refusals, multilingual code-switch). "
            "Released alongside the EuroAgent model under Apache 2.0."
        ),
    }


def _parse_message(d: dict) -> Message | None:
    role = d.get("role")
    if role not in ("system", "user", "assistant", "tool"):
        return None
    content = d.get("content") or ""
    tool_calls: list[ToolCall] = []
    for tc in d.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        name = tc.get("name")
        args = tc.get("arguments", {})
        if isinstance(name, str) and isinstance(args, dict):
            tool_calls.append(ToolCall(name=name, arguments=args, id=tc.get("id")))
    return Message(
        role=role,
        content=content,
        tool_calls=tool_calls,
        tool_call_id=d.get("tool_call_id"),
    )


def _iter_jsonl(path: Path, lang: str) -> Iterator[Conversation]:
    with path.open(encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                # Skip silently rather than fail the whole prep run; surface in dry-run.
                print(f"  [warn] {path}:{line_no} invalid JSON: {e}")
                continue
            if obj.get("lang") != lang:
                continue
            messages_raw = obj.get("messages") or []
            messages: list[Message] = []
            for m in messages_raw:
                msg = _parse_message(m)
                if msg is not None:
                    messages.append(msg)
            if any(m.role == "assistant" for m in messages):
                yield Conversation(messages=messages)


def iter_examples(lang: str, count_budget: int) -> Iterator[Conversation]:
    if lang not in _SUPPORTED or count_budget <= 0:
        return
    root = _root()
    if not root.exists():
        return
    emitted = 0
    # Iterate in sorted order so the corpus is reproducible across machines.
    for path in sorted(root.glob("*.jsonl")):
        if emitted >= count_budget:
            return
        for conv in _iter_jsonl(path, lang):
            yield conv
            emitted += 1
            if emitted >= count_budget:
                return
