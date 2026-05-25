"""Canonical chat + tool-call schema for SFT.

We render to a ChatML-like format with a structured `<tool_call>` block.
The same renderer also emits a per-segment loss mask: only assistant turns
contribute to the cross-entropy. Role headers and user/tool content are seen
during forward pass (they're in context) but not trained on.

The "EuroAgent" mix in particular ships through this format regardless of
upstream source — OASST2 trees, Glaive function-calling traces, xLAM JSON
calls, hand-written EU-specific examples — so the SFT trainer only ever
sees one schema.

Special-token strategy (Phase 2 SFT v0.1):
    For now we use plain ASCII markers (`<|im_start|>` / `<|im_end|>` /
    `<tool_call>` / `</tool_call>`) tokenized by the existing 32k BPE.
    Subword fragmentation is suboptimal but keeps the pretrained 124M's
    embedding table untouched. Later: add 4–6 atomic special tokens and
    resize the model's embedding + lm_head.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal

Role = Literal["system", "user", "assistant", "tool"]

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"


@dataclass(frozen=True)
class ToolCall:
    """A function/tool invocation emitted by the assistant."""

    name: str
    arguments: dict
    id: str | None = None

    def to_json(self) -> str:
        # canonical, sorted keys, no whitespace — stable across runs
        return json.dumps(
            {"name": self.name, "arguments": self.arguments},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )


@dataclass
class Message:
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None  # only used when role == "tool"


@dataclass
class Conversation:
    messages: list[Message] = field(default_factory=list)


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def render_chatml(conv: Conversation) -> str:
    """Flatten a Conversation to a single ChatML-like string (no mask).

    Mostly useful for inference templating and round-trip tests.
    """
    return "".join(text for text, _ in _render_segments(conv))


def render_chatml_with_mask(
    conv: Conversation,
) -> list[tuple[str, bool]]:
    """Return a list of (text_chunk, is_loss_target) segments.

    Tokenizing each segment independently sidesteps any subword-alignment
    edge cases between the model's BPE and our character-level loss intent.
    """
    return list(_render_segments(conv))


def _render_segments(conv: Conversation):
    for msg in conv.messages:
        is_assistant = msg.role == "assistant"
        # Header: <|im_start|>{role}\n  — never a loss target (in context only).
        yield (f"{IM_START}{msg.role}\n", False)

        # Body parts: tool calls (assistant only), then content.
        if msg.tool_calls:
            for tc in msg.tool_calls:
                # Wrap each tool call in <tool_call>...</tool_call>\n.
                yield (f"{TOOL_CALL_OPEN}{tc.to_json()}{TOOL_CALL_CLOSE}\n", is_assistant)
        if msg.content:
            yield (msg.content + "\n", is_assistant)

        # Footer: <|im_end|>\n — assistant must learn to stop, so this is a
        # loss target on assistant turns; otherwise just context.
        yield (f"{IM_END}\n", is_assistant)
