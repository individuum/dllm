"""Pluggable data sources for the EU multi-lingual corpus.

Each source module exposes:

    iter_docs(lang: str, char_budget: int) -> Iterator[str]
        Yield raw text documents (one element per "doc" — article, book chapter,
        legal article, etc.) in the requested language. Stop when ``char_budget``
        characters have been emitted (≈ char_budget / 4 tokens for European
        languages).

    license_info() -> dict
        Static metadata: name, license, license_url, url, attribution. Lands
        verbatim in the prepared corpus's provenance manifest so we can satisfy
        EU AI Act Art. 53's "sufficiently detailed summary" requirement.

    supported_langs() -> list[str]
        ISO 639-1 codes this source can serve. Lets the orchestrator skip
        gracefully when a source can't speak a requested language.

Compliance posture: every source listed here is either public domain (EU
institutional, expired copyright) or under an EU-compatible open license
(CC-BY-SA, CC-0, Apache 2.0). No scraped web text, no shadow libraries,
no copyrighted books. See PLAN.md §4.
"""
from __future__ import annotations

from importlib import import_module
from typing import Iterator, Protocol


class Source(Protocol):
    """Structural type for source modules."""

    def iter_docs(self, lang: str, char_budget: int) -> Iterator[str]: ...
    def license_info(self) -> dict: ...
    def supported_langs(self) -> list[str]: ...


_SOURCE_NAMES = (
    "wikipedia",
    "europarl",
    "jrc_acquis",
    "gutenberg",
)


def load(name: str):
    """Lazy-load a source module by short name. ImportError-resilient: a
    missing optional dataset library makes that source unavailable but
    doesn't break the orchestrator.
    """
    if name not in _SOURCE_NAMES:
        raise ValueError(f"unknown source {name!r}; have {_SOURCE_NAMES}")
    return import_module(f".{name}", package=__name__)


def all_source_names() -> tuple[str, ...]:
    return _SOURCE_NAMES
