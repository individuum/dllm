"""Phase 0 corpus prep: TinyShakespeare via tiktoken GPT-2 BPE.

Splits 90/10 train/val and writes uint16 .bin files under data/cache/.
Production uses CulturaX / OSCAR / EU sources tokenized with a custom 128k BPE
(see PLAN.md §4); for the smoke test, TinyShakespeare is enough to verify the
training loop converges.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import requests
import tiktoken

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass


TINYSHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def cache_dir() -> Path:
    here = Path(__file__).resolve().parent
    out = here.parent.parent.parent / "data" / "cache"
    out.mkdir(parents=True, exist_ok=True)
    return out


def download_tinyshakespeare(dest: Path) -> str:
    if dest.exists():
        return dest.read_text(encoding="utf-8")
    print(f"downloading TinyShakespeare → {dest}")
    text = requests.get(TINYSHAKESPEARE_URL, timeout=30).text
    dest.write_text(text, encoding="utf-8")
    return text


def encode_and_split(text: str, val_frac: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode_ordinary(text)
    arr = np.array(ids, dtype=np.uint16)
    n_val = int(len(arr) * val_frac)
    return arr[:-n_val], arr[-n_val:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-frac", type=float, default=0.1)
    args = ap.parse_args()

    out = cache_dir()
    raw = out / "tinyshakespeare.txt"
    text = download_tinyshakespeare(raw)

    train, val = encode_and_split(text, args.val_frac)
    (out / "train.bin").write_bytes(train.tobytes())
    (out / "val.bin").write_bytes(val.tobytes())

    print(f"train tokens: {len(train):,}")
    print(f"val   tokens: {len(val):,}")
    print(f"wrote {out / 'train.bin'} and {out / 'val.bin'}")


if __name__ == "__main__":
    main()
