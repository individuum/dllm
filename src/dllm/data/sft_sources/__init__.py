"""Pluggable post-training (SFT) data sources for the EuroAgent mix.

Each source module exposes:

    iter_examples(lang: str, count_budget: int) -> Iterator[Conversation]
        Yield Conversation objects (see dllm.data.sft_format) in the requested
        language. Stop once ``count_budget`` examples have been emitted.
        Downstream tokenization converts each Conversation into a stream of
        (token_id, loss_mask_bit) pairs.

    license_info() -> dict
        name, license, license_url, url, attribution, hf_dataset. Lands verbatim
        in the SFT manifest so the EuroAgent release card can quote per-source
        license terms (EU AI Act Art. 53 + each source's attribution clause).

    supported_langs() -> list[str]
        ISO 639-1 codes this source can serve.

    task_type() -> "tool_call" | "agentic" | "general"
        Coarse classification surfaced in the manifest so we can tell which
        proportion of the SFT mix targets which capability — important when
        rebalancing for tool-calling regressions.

Compliance posture (matches the pretraining sources): every source listed here
is under an EU-compatible open license (Apache 2.0, CC-BY-4.0, CC-BY-SA-4.0).
No OpenAI-output-derived datasets (Alpaca, ShareGPT, OpenOrca-derivatives) —
their terms-of-use restrict downstream model training and would be a problem
under the GPAI copyright-compliance policy.
"""
from __future__ import annotations

from importlib import import_module

_SOURCE_NAMES = (
    "xlam",
    "glaive_v2",
    "glaive_v2_de",
    "hermes_fc",
    "oasst2",
    "aya",
    "germanrag",
    "avemio_german_rag",
    "hand_written",
)


def load(name: str):
    """Lazy-load a source module. ImportError-resilient: a missing optional
    library makes the source unavailable but doesn't break the orchestrator."""
    if name not in _SOURCE_NAMES:
        raise ValueError(f"unknown SFT source {name!r}; have {_SOURCE_NAMES}")
    return import_module(f".{name}", package=__name__)


def all_source_names() -> tuple[str, ...]:
    return _SOURCE_NAMES
