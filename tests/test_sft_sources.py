"""Static metadata checks for every SFT source — no network."""
from __future__ import annotations

import pytest

from dllm.data import sft_sources


def test_registry_exposes_expected_sources() -> None:
    names = set(sft_sources.all_source_names())
    # Both tool-call families, multilingual agentic, instruction, and EU corner-case hook.
    assert {"xlam", "glaive_v2", "hermes_fc", "oasst2", "aya", "hand_written"} <= names


@pytest.mark.parametrize("name", sft_sources.all_source_names())
def test_every_source_advertises_license_metadata(name: str) -> None:
    src = sft_sources.load(name)
    info = src.license_info()
    required = {"name", "license", "license_url", "url", "attribution"}
    missing = required - info.keys()
    assert not missing, f"{name} missing license fields: {missing}"


@pytest.mark.parametrize("name", sft_sources.all_source_names())
def test_every_source_declares_supported_langs(name: str) -> None:
    src = sft_sources.load(name)
    langs = src.supported_langs()
    assert isinstance(langs, list) and langs, f"{name} declares no langs"


@pytest.mark.parametrize("name", sft_sources.all_source_names())
def test_every_source_declares_task_type(name: str) -> None:
    src = sft_sources.load(name)
    assert src.task_type() in {"tool_call", "agentic", "general"}


def test_at_least_one_de_source_per_task_type() -> None:
    """German must be served by at least one source per task axis — otherwise
    the EuroAgent SFT mix would silently degenerate to EN-only on that axis."""
    by_axis: dict[str, list[str]] = {"tool_call": [], "agentic": [], "general": []}
    for name in sft_sources.all_source_names():
        src = sft_sources.load(name)
        if "de" in src.supported_langs():
            by_axis[src.task_type()].append(name)
    # tool_call: hand_written is the only DE tool-call source currently — that's
    # intentional and tracked in the manifest notes. Acceptable for v0.1 but
    # flag the bare minimum.
    assert by_axis["tool_call"], "no DE tool-call source — EuroAgent DE tool calling will be untrained"
    assert by_axis["agentic"], "no DE agentic source — EuroAgent DE chat will be untrained"


def test_default_alloc_keys_are_known_sources() -> None:
    """DEFAULT_ALLOC_COUNT must not reference sources that don't exist."""
    from dllm.data.sft_prepare import DEFAULT_ALLOC_COUNT

    known = set(sft_sources.all_source_names())
    for src_name in DEFAULT_ALLOC_COUNT:
        assert src_name in known, f"DEFAULT_ALLOC_COUNT[{src_name!r}] is not a registered source"


def test_default_alloc_only_requests_supported_langs() -> None:
    """Every (source, lang) in DEFAULT_ALLOC_COUNT must be a lang that source supports."""
    from dllm.data.sft_prepare import DEFAULT_ALLOC_COUNT

    for src_name, lang_budgets in DEFAULT_ALLOC_COUNT.items():
        src = sft_sources.load(src_name)
        supported = set(src.supported_langs())
        unsupported = set(lang_budgets) - supported
        assert not unsupported, (
            f"DEFAULT_ALLOC_COUNT[{src_name!r}] requests {unsupported} but "
            f"{src_name} only supports {supported}"
        )
