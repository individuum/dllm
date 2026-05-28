"""DiscoResearch/germanrag — native German RAG conversations.

Schema (HF dataset row, verified 2026-05-27):
    {
        "contexts":         [str, ...],  # candidate passages
        "question":         str,         # German question
        "answer":           str,         # German answer (or refusal)
        "positive_ctx_idx": int,         # index of relevant context;
                                         # -1 means "not in context" — these
                                         # are grounding-refusal counterfactuals.
    }

We render as a 2-turn conversation:

    user:       Beantworte die Frage anhand des folgenden Kontexts.
                Kontext:
                {positive_ctx}
                Frage: {question}
    assistant:  {answer}

For positive_ctx_idx == -1 (no grounded answer), we pass through the entire
contexts list and the model learns to refuse — these are valuable training
data for grounding behavior.

Filling a real gap: native-DE agentic SFT data is scarce. germanrag is small
(~3k rows) but it's human-curated and Apache-2.0-licensed.

License: Apache 2.0.
"""
from __future__ import annotations

from typing import Iterator

from dllm.data.sft_format import Conversation, Message

HF_DATASET = "DiscoResearch/germanrag"


def supported_langs() -> list[str]:
    return ["de"]


def task_type() -> str:
    return "agentic"


def license_info() -> dict:
    return {
        "name": "DiscoResearch germanrag",
        "hf_dataset": HF_DATASET,
        "license": "Apache-2.0",
        "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
        "url": f"https://huggingface.co/datasets/{HF_DATASET}",
        "attribution": (
            "DiscoResearch germanrag — human-curated native-German RAG "
            "examples including grounding-refusal counterfactuals. Apache 2.0. "
            "Primary source for native-DE agentic SFT in the EuroAgent mix."
        ),
    }


def _render_user_turn(question: str, contexts: list[str], positive_idx: int) -> str:
    """RAG-style user turn: instruction + context block + question.

    For positive_idx == -1 (counterfactual / refusal-grounding), include the
    full context list so the model can verify nothing answers the question.
    Otherwise show only the positive context, which is what we want the model
    to actually train on (no distractor noise).
    """
    if positive_idx is None or positive_idx < 0:
        # All-contexts variant — train model to recognise no-answer cases.
        ctx_block = "\n\n---\n\n".join(c.strip() for c in contexts if isinstance(c, str))
    else:
        ctx = contexts[positive_idx] if 0 <= positive_idx < len(contexts) else ""
        ctx_block = ctx.strip()
    return (
        "Beantworte die Frage anhand des folgenden Kontexts. "
        "Wenn die Antwort nicht im Kontext steht, sage das ehrlich.\n\n"
        f"Kontext:\n{ctx_block}\n\n"
        f"Frage: {question.strip()}"
    )


def iter_examples(lang: str, count_budget: int) -> Iterator[Conversation]:
    if lang != "de" or count_budget <= 0:
        return
    from datasets import load_dataset  # noqa: PLC0415

    ds = load_dataset(HF_DATASET, split="train", streaming=True)
    emitted = 0
    for row in ds:
        if emitted >= count_budget:
            return
        question = (row.get("question") or "").strip()
        answer = (row.get("answer") or "").strip()
        contexts = row.get("contexts") or []
        if not question or not answer or not isinstance(contexts, list):
            continue
        positive_idx = row.get("positive_ctx_idx")
        user_text = _render_user_turn(question, contexts, positive_idx)
        yield Conversation(messages=[
            Message(role="user", content=user_text),
            Message(role="assistant", content=answer),
        ])
        emitted += 1
