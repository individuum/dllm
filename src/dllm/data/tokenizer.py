"""Helpers for loading the EU-balanced BPE tokenizer trained by `dllm.data.prepare`."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tokenizers import Tokenizer


EOT_TOKEN = "<|endoftext|>"


def default_tokenizer_path() -> Path:
    """data/cache/tokenizer.json next to data/cache/train.bin."""
    here = Path(__file__).resolve().parent
    return here.parent.parent.parent / "data" / "cache" / "tokenizer.json"


def load_tokenizer(path: str | Path | None = None) -> "Tokenizer":
    """Lazy import so workers/coord don't need to depend on `tokenizers` at runtime."""
    from tokenizers import Tokenizer  # noqa: PLC0415

    path = Path(path) if path else default_tokenizer_path()
    if not path.exists():
        raise FileNotFoundError(
            f"tokenizer not found at {path}; run `python -m dllm.data.prepare` first"
        )
    return Tokenizer.from_file(str(path))
