"""OpenAssistant Conversations v2 (OASST2) — human-curated multi-turn agentic.

Schema: rows are individual messages forming a conversation forest. Each row
has parent_id (None for root user prompts), role, text, lang, message_tree_id,
rank (siblings ranked by humans). We pick the highest-ranked path from each
tree to extract one linear conversation per tree.

This is our primary source for:
  - **German (de)**: native multi-turn dialogues, not translations
  - **English (en)**: top-quality human-curated agentic conversations
Plus a long tail of fr/es/it which we surface if requested but do not use
by default in the DE+EN-only mix.

License: Apache 2.0. Attribution: OpenAssistant project / LAION.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterator

from dllm.data.sft_format import Conversation, Message

HF_DATASET = "OpenAssistant/oasst2"

# OASST role tags → our Role literals. "prompter" is the user.
_ROLE_MAP = {
    "prompter": "user",
    "assistant": "assistant",
}

_SUPPORTED = ["en", "de", "fr", "es", "it"]


def supported_langs() -> list[str]:
    return list(_SUPPORTED)


def task_type() -> str:
    return "agentic"


def license_info() -> dict:
    return {
        "name": "OpenAssistant Conversations v2 (OASST2)",
        "hf_dataset": HF_DATASET,
        "license": "Apache-2.0",
        "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
        "url": f"https://huggingface.co/datasets/{HF_DATASET}",
        "attribution": (
            "OASST2 by the OpenAssistant project / LAION, released under Apache "
            "2.0. Human-curated multi-turn conversations; primary source for "
            "native-German agentic SFT data in the EuroAgent mix."
        ),
    }


def _best_path(by_parent: dict, root_id: str) -> list[dict]:
    """Walk root→leaf picking the highest-ranked child at each step.

    `rank` is human-judged sibling ordering (0 = best, None = unranked).
    Returns the list of message rows on that path, root first.
    """
    path: list[dict] = []
    cur_id: str | None = root_id
    while cur_id is not None:
        children = by_parent.get(cur_id, [])
        if not children:
            break
        # Sort: ranked-with-rank-0 first, then by ascending rank, then unranked.
        children_sorted = sorted(
            children,
            key=lambda r: (r.get("rank") is None, r.get("rank") if r.get("rank") is not None else 0),
        )
        best = children_sorted[0]
        path.append(best)
        cur_id = best.get("message_id")
    return path


def _trees_to_conversations(rows: list[dict], lang: str) -> Iterator[Conversation]:
    """Build conversations from a buffered set of OASST rows.

    Yields every valid Conversation in the batch; the outer caller is
    responsible for enforcing count_budget.
    """
    by_parent: dict[str, list[dict]] = defaultdict(list)
    roots: list[dict] = []
    for r in rows:
        if r.get("lang") != lang:
            continue
        parent = r.get("parent_id")
        if parent is None:
            roots.append(r)
        else:
            by_parent[parent].append(r)

    for root in roots:
        if (root.get("role") or "").lower() != "prompter":
            continue
        path = [root] + _best_path(by_parent, root.get("message_id"))
        msgs: list[Message] = []
        for r in path:
            role = _ROLE_MAP.get((r.get("role") or "").lower())
            text = (r.get("text") or "").strip()
            if role is None or not text:
                continue
            msgs.append(Message(role=role, content=text))
        # Must end on an assistant turn — otherwise nothing is loss-targeted.
        while msgs and msgs[-1].role != "assistant":
            msgs.pop()
        if len(msgs) < 2:
            continue
        yield Conversation(messages=msgs)


def iter_examples(lang: str, count_budget: int) -> Iterator[Conversation]:
    if lang not in _SUPPORTED or count_budget <= 0:
        return
    from datasets import load_dataset  # noqa: PLC0415

    ds = load_dataset(HF_DATASET, split="train", streaming=True)
    rows: list[dict] = []
    emitted = 0
    # Buffer until we have a reasonable chunk; OASST is sorted by tree so
    # buffering 5k messages still keeps tree integrity per flush.
    for r in ds:
        rows.append(dict(r))
        if len(rows) >= 5000:
            for conv in _trees_to_conversations(rows, lang):
                yield conv
                emitted += 1
                if emitted >= count_budget:
                    return
            rows = []
    if rows:
        for conv in _trees_to_conversations(rows, lang):
            yield conv
            emitted += 1
            if emitted >= count_budget:
                return
