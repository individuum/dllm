"""ChatML rendering + loss-mask correctness for the SFT canonical schema."""
from __future__ import annotations

from dllm.data.sft_format import (
    IM_END,
    IM_START,
    TOOL_CALL_CLOSE,
    TOOL_CALL_OPEN,
    Conversation,
    Message,
    ToolCall,
    render_chatml,
    render_chatml_with_mask,
)


def test_basic_user_assistant_renders_chatml_pair() -> None:
    conv = Conversation(messages=[
        Message(role="user", content="hi"),
        Message(role="assistant", content="hello"),
    ])
    text = render_chatml(conv)
    assert text == (
        f"{IM_START}user\nhi\n{IM_END}\n"
        f"{IM_START}assistant\nhello\n{IM_END}\n"
    )


def test_loss_mask_only_targets_assistant_segments() -> None:
    """User segments are context (mask=False); assistant segments are
    training targets (mask=True) — including the closing <|im_end|>."""
    conv = Conversation(messages=[
        Message(role="user", content="hi"),
        Message(role="assistant", content="hello"),
    ])
    segs = render_chatml_with_mask(conv)
    # User: header(F), content(F), end(F) = 3 segs all False.
    # Assistant: header(F), content(T), end(T) = 3 segs F,T,T.
    user_segs = [(s, m) for s, m in segs if "user" in s or s == "hi\n"]
    assistant_segs = [(s, m) for s, m in segs if "assistant" in s or s == "hello\n"]
    # Concrete check: collect all (text, mask) and verify role-vs-mask
    mask_by_text = dict(segs)
    assert mask_by_text[f"{IM_START}user\n"] is False
    assert mask_by_text["hi\n"] is False
    # The trailing <|im_end|>\n appears after BOTH user and assistant; collect both.
    # Use the order in the segment list: 3 user segs, then 3 assistant segs.
    user, asst = segs[:3], segs[3:]
    assert [m for _, m in user] == [False, False, False]
    assert [m for _, m in asst] == [False, True, True]


def test_tool_call_renders_with_canonical_json() -> None:
    """ToolCall.to_json must be deterministic (sorted keys, no whitespace) so
    identical calls tokenize to identical token sequences across runs."""
    tc = ToolCall(name="get_weather", arguments={"city": "Berlin", "unit": "C"})
    j = tc.to_json()
    # Sorted keys: arguments before name; arguments dict also sorted.
    assert j == '{"arguments":{"city":"Berlin","unit":"C"},"name":"get_weather"}'


def test_assistant_turn_with_tool_call_has_mask_on_call_block() -> None:
    """Tool-call block in an assistant turn must be a loss target — the model
    has to learn to emit it. If we accidentally masked it out, training would
    leave tool-calling capability untrained."""
    tc = ToolCall(name="f", arguments={"x": 1})
    conv = Conversation(messages=[
        Message(role="user", content="call f"),
        Message(role="assistant", content="", tool_calls=[tc]),
    ])
    segs = render_chatml_with_mask(conv)
    # Assistant segments: header(F), tool_call(T), end(T) — no content seg since "".
    asst_segs = [s for s in segs if s[0].startswith(f"{IM_START}assistant")
                 or s[0].startswith(TOOL_CALL_OPEN)
                 or (s[0] == f"{IM_END}\n" and segs.index(s) > 2)]
    # More direct: last 3 segs are the assistant turn.
    assistant_three = segs[-3:]
    header, tool_call, end = assistant_three
    assert header[0] == f"{IM_START}assistant\n" and header[1] is False
    assert tool_call[0].startswith(TOOL_CALL_OPEN) and tool_call[0].endswith(TOOL_CALL_CLOSE + "\n")
    assert tool_call[1] is True  # critical
    assert end[0] == f"{IM_END}\n" and end[1] is True


def test_system_turn_is_context_not_target() -> None:
    """System turns (e.g. tool catalog) should never be loss targets."""
    conv = Conversation(messages=[
        Message(role="system", content="you can call f"),
        Message(role="user", content="call f"),
        Message(role="assistant", content="ok"),
    ])
    segs = render_chatml_with_mask(conv)
    system_segs = segs[:3]
    assert all(m is False for _, m in system_segs)


def test_tool_response_turn_is_context_not_target() -> None:
    """The tool's response is part of the input context; only the assistant's
    subsequent reasoning about it should be a loss target."""
    tc = ToolCall(name="f", arguments={"x": 1})
    conv = Conversation(messages=[
        Message(role="user", content="call f"),
        Message(role="assistant", content="", tool_calls=[tc]),
        Message(role="tool", content='{"result": 42}', tool_call_id=None),
        Message(role="assistant", content="result is 42"),
    ])
    segs = render_chatml_with_mask(conv)
    # Tool segments: header(F), content(F), end(F)
    tool_start_idx = next(i for i, (s, _) in enumerate(segs) if s == f"{IM_START}tool\n")
    tool_three = segs[tool_start_idx : tool_start_idx + 3]
    assert all(m is False for _, m in tool_three)


def test_roundtrip_render_chatml_text_matches_concatenated_segments() -> None:
    """render_chatml must equal the concatenation of segment texts from
    render_chatml_with_mask — they share an implementation."""
    conv = Conversation(messages=[
        Message(role="system", content="sys"),
        Message(role="user", content="u"),
        Message(role="assistant", content="a"),
    ])
    text = render_chatml(conv)
    seg_text = "".join(t for t, _ in render_chatml_with_mask(conv))
    assert text == seg_text
