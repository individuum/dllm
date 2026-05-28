"""Avemio German-RAG-SFT — native-DE RAG + agentic + function-selection.

Hessian.AI + Avemio's TRAIN-split SFT mix for German RAG and agentic tasks.
Schema (per row, verified 2026-05-27):

    {
        "conversations": [{"from": "human"|"gpt", "value": str}, ...],
        "system":        str,   # task-specific instruction + sometimes tool catalog
        "tools":         str,   # JSON list of tool definitions (empty for non-FC tasks)
    }

Round-robins across the German-native task configs. Specifically EXCLUDES:
  - `extended_function-calling-xlam-en` — English, redundant with our existing
    Glaive/Hermes-FC sources
  - the EASY-BENCHMARK / HARD-BENCHMARK companion datasets — those are TEST
    splits used for evaluation; using them as training would contaminate
    later evals of any model

Function-calling note: `select-function-calls-de` is *function selection*
(the assistant emits `{"functions_to_use": [name]}`), not full tool calling
with arguments. Complementary to our opus-mt-translated Glaive — it teaches
the model to PICK a function from a German catalog, while Glaive-DE teaches
it to GENERATE arguments.

License: MIT.
"""
from __future__ import annotations

from typing import Iterator

from dllm.data.sft_format import Conversation, Message

HF_DATASET = "avemio/German-RAG-SFT-ShareGPT-HESSIAN-AI"

# Per-config descriptions for the manifest:
_CONFIGS = (
    "select-function-calls-de",       # DE function-name selection
    "reasoning",                      # German chain-of-thought reasoning
    "classification-json",            # structured JSON classification
    "qa-without-timedifference",      # context-grounded QA
    "qa-with-timedifference",         # time-aware context-grounded QA
    "qa-with-multiple-references",    # multi-doc RAG QA
    "relevant-context",               # context selection
    "summarizations",                 # German summarization
    "extraction-recall",              # information extraction
    "questions",                      # generate questions from text
    "ocr-correction",                 # OCR error correction in DE
)

_ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
}


def supported_langs() -> list[str]:
    return ["de"]


def task_type() -> str:
    # Mix of agentic + tool-selection; "agentic" is the better general label.
    # `select-function-calls-de` alone is tool_call-like but it's one of 11 configs.
    return "agentic"


def license_info() -> dict:
    return {
        "name": "Avemio German-RAG-SFT (Hessian.AI)",
        "hf_dataset": HF_DATASET,
        "license": "MIT",
        "license_url": "https://opensource.org/licenses/MIT",
        "url": f"https://huggingface.co/datasets/{HF_DATASET}",
        "configs": list(_CONFIGS),
        "attribution": (
            "Avemio + Hessian.AI German-RAG-SFT training data (MIT licensed). "
            "Native-German RAG, multi-turn agentic reasoning, structured output, "
            "and function-name selection. Translated/regenerated from xLAM + "
            "Wikipedia via open-source LLMs (not OpenAI-derived)."
        ),
    }


def _conv_from_row(row: dict) -> Conversation | None:
    """ShareGPT-format row -> Conversation. Embeds `system` + `tools` as the
    initial system turn if non-empty."""
    msgs: list[Message] = []
    system = (row.get("system") or "").strip()
    if system:
        msgs.append(Message(role="system", content=system))
    convs_raw = row.get("conversations") or []
    if not isinstance(convs_raw, list):
        return None
    for turn in convs_raw:
        if not isinstance(turn, dict):
            return None
        raw_role = (turn.get("from") or "").lower()
        role = _ROLE_MAP.get(raw_role)
        if role is None:
            return None
        value = (turn.get("value") or "").strip()
        if value:
            msgs.append(Message(role=role, content=value))
    if not any(m.role == "assistant" for m in msgs):
        return None
    return Conversation(messages=msgs)


def iter_examples(lang: str, count_budget: int) -> Iterator[Conversation]:
    """Round-robin across the German-native task configs.

    Round-robin (vs. config-by-config) ensures the resulting mix has all
    task types represented even if `count_budget` is hit early. Critical
    because the ShardLoader partitions train.bin by worker_id — we want
    every shard to see every task type.
    """
    if lang != "de" or count_budget <= 0:
        return
    from datasets import load_dataset  # noqa: PLC0415

    iters: dict[str, Iterator[dict]] = {}
    for cfg in _CONFIGS:
        try:
            ds = load_dataset(HF_DATASET, cfg, split="train", streaming=True)
            iters[cfg] = iter(ds)
        except Exception as e:  # noqa: BLE001 — a missing config shouldn't kill the rest
            print(f"  [avemio_german_rag] skip config {cfg}: {e}")

    emitted = 0
    while iters and emitted < count_budget:
        # One pass through every still-live config = one round of the round-robin
        for cfg in list(iters):
            if emitted >= count_budget:
                return
            try:
                row = next(iters[cfg])
            except StopIteration:
                del iters[cfg]
                continue
            conv = _conv_from_row(row)
            if conv is None:
                continue
            yield conv
            emitted += 1
