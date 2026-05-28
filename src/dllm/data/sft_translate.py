"""opus-mt-en-de wrapper for the EuroAgent SFT pipeline.

Used by `scripts/translate_glaive_to_de.py` to produce a one-shot DE-translated
copy of Glaive function-calling v2 ahead of time. The result is written to
`data/cache/sft/glaive_v2_de.jsonl`, which the `glaive_v2_de` source module
then streams from.

Two reasons translation is offline rather than inside the source:
  1. Batching efficiency. opus-mt's GPU throughput plateaus around batch
     size 64–128. Translating one Conversation at a time (3–9 prose strings)
     leaves the GPU idle between calls. Batching across many conversations
     keeps it saturated.
  2. The orchestrator (`sft_prepare`) stays free of `transformers` and CUDA
     dependencies — it only reads the pre-translated JSONL.

Design notes:
  - SQLite cache (`translation_cache.sqlite`) keyed by sha1(en_text). Re-runs
    pay only for newly seen text, so iterating on a smoke set is cheap.
  - opus-mt-en-de is generally robust to snake_case identifiers (`get_weather`,
    `validate_iban`) — they survive unchanged because they don't lex as any
    known German word. The smoke test in scripts/translate_glaive_to_de.py
    verifies this on a 100-conv sample before committing the full 25k pass.
  - Apache 2.0 license on both the source dataset (Glaive v2) and the
    translation model (Helsinki-NLP/opus-mt-en-de). Both attributions land in
    the manifest verbatim.
"""
from __future__ import annotations

import hashlib
import re
import sqlite3
import sys
import time
from pathlib import Path

import torch

DEFAULT_MODEL = "Helsinki-NLP/opus-mt-en-de"
# opus-mt's max input is 512 tokens (~2000 chars for EN). Stay well under to
# avoid silent truncation on long assistant turns.
MAX_CHARS_PER_CHUNK = 1500


def cache_path() -> Path:
    here = Path(__file__).resolve().parent
    p = here.parent.parent.parent / "data" / "cache" / "sft" / "translation_cache.sqlite"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


class TranslationCache:
    """sha1-keyed cache so smoke tests + restarts don't pay the full cost twice."""

    def __init__(self, path: Path) -> None:
        self._conn = sqlite3.connect(str(path), isolation_level=None)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS t (h TEXT PRIMARY KEY, en TEXT, de TEXT)"
        )
        self._conn.execute("PRAGMA journal_mode=WAL")

    @staticmethod
    def _hash(en: str) -> str:
        return hashlib.sha1(en.encode("utf-8")).hexdigest()

    def get_many(self, ens: list[str]) -> dict[str, str]:
        """Bulk-lookup; returns {en: de} for the ones we already have."""
        if not ens:
            return {}
        hashes = [self._hash(e) for e in ens]
        rows = self._conn.execute(
            f"SELECT h, de FROM t WHERE h IN ({','.join(['?'] * len(hashes))})",
            hashes,
        ).fetchall()
        h2de = dict(rows)
        return {en: h2de[self._hash(en)] for en in ens if self._hash(en) in h2de}

    def put_many(self, pairs: list[tuple[str, str]]) -> None:
        if not pairs:
            return
        rows = [(self._hash(en), en, de) for en, de in pairs]
        self._conn.executemany("INSERT OR REPLACE INTO t VALUES (?, ?, ?)", rows)


def split_chunks(text: str) -> list[str]:
    """Greedy sentence-aware chunking under MAX_CHARS_PER_CHUNK.

    Short messages pass through unchanged (one chunk). Long messages are split
    on sentence boundaries and re-joined into chunks under the per-call limit.
    Reassembly happens at the caller; this function just yields chunks.
    """
    if len(text) <= MAX_CHARS_PER_CHUNK:
        return [text]
    sents = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    cur = ""
    for s in sents:
        if cur and len(cur) + len(s) + 1 > MAX_CHARS_PER_CHUNK:
            chunks.append(cur)
            cur = s
        else:
            cur = (cur + " " + s).strip() if cur else s
    if cur:
        chunks.append(cur)
    return chunks


class Translator:
    """opus-mt-en-de wrapper. One per process; cached on-disk between runs."""

    def __init__(
        self,
        device: str | None = None,
        batch_size: int = 16,
        model_name: str = DEFAULT_MODEL,
        num_beams: int = 2,
        max_length: int = 256,
    ) -> None:
        from transformers import MarianMTModel, MarianTokenizer  # noqa: PLC0415

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        # Make CUDA usage VISIBLE — never trust silent fallback. (See feedback
        # memory: always verify GPU is actually used.)
        print(
            f"[translate] model={model_name} device={self.device} "
            f"cuda_available={torch.cuda.is_available()}",
            file=sys.stderr, flush=True,
        )
        t0 = time.time()
        self.tok = MarianTokenizer.from_pretrained(model_name)
        self.mdl = MarianMTModel.from_pretrained(model_name).to(self.device).eval()
        params_m = sum(p.numel() for p in self.mdl.parameters()) / 1e6
        print(
            f"[translate] loaded in {time.time() - t0:.1f}s params={params_m:.1f}M",
            file=sys.stderr, flush=True,
        )
        if self.device.type == "cuda":
            alloc_mb = torch.cuda.memory_allocated(self.device) / 1e6
            free_mb, total_mb = (m / 1e6 for m in torch.cuda.mem_get_info(self.device))
            print(
                f"[translate] CUDA mem: model={alloc_mb:.0f} MB, "
                f"free={free_mb:.0f} MB, total={total_mb:.0f} MB",
                file=sys.stderr, flush=True,
            )

        self.batch_size = batch_size
        self.num_beams = num_beams
        self.max_length = max_length
        self.cache = TranslationCache(cache_path())
        self.stats = {"translated": 0, "cached": 0, "chars_in": 0, "chars_out": 0}

    @torch.inference_mode()
    def _translate_batch(self, texts: list[str]) -> list[str]:
        inputs = self.tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        out_ids = self.mdl.generate(
            **inputs,
            max_length=self.max_length,
            num_beams=self.num_beams,
        )
        decoded = self.tok.batch_decode(out_ids, skip_special_tokens=True)
        # Release the per-batch KV cache + intermediate tensors so the
        # caching allocator doesn't retain them across batches. Keeps the
        # high-water mark predictable for long runs.
        del inputs, out_ids
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return decoded

    def translate_many(self, en_texts: list[str]) -> list[str]:
        """Translate a list of strings. Returns DE strings in input order.

        Cache-aware: pre-checks the sqlite cache for each input, only sends
        cache-misses through the GPU. Long inputs are chunk-split and
        re-joined; chunks are batched across inputs for maximum GPU
        utilization.
        """
        if not en_texts:
            return []
        # Step 1: cache lookup for whole-string hits
        cached = self.cache.get_many(en_texts)
        for en in en_texts:
            if en in cached:
                self.stats["cached"] += 1

        # Step 2: chunk-split the misses, build a flat translation queue
        miss_inputs: list[str] = []   # the original miss texts, in order
        miss_chunks: list[list[str]] = []  # parallel: chunks per miss
        for en in en_texts:
            if en in cached:
                continue
            miss_inputs.append(en)
            miss_chunks.append(split_chunks(en) if en.strip() else [""])

        # Step 3: flatten + dedupe chunks for the GPU call
        flat: list[str] = []
        positions: list[tuple[int, int]] = []  # (miss_idx, chunk_idx)
        for i, chunks in enumerate(miss_chunks):
            for j, c in enumerate(chunks):
                flat.append(c)
                positions.append((i, j))

        # Step 4: batched translation of unique non-empty chunks
        unique_to_translate = [c for c in flat if c.strip()]
        unique_dedup = list({c: None for c in unique_to_translate}.keys())  # preserve order
        translations: dict[str, str] = {}
        for s in range(0, len(unique_dedup), self.batch_size):
            batch = unique_dedup[s : s + self.batch_size]
            outs = self._translate_batch(batch)
            translations.update(zip(batch, outs))

        # Step 5: reassemble
        results_per_miss: list[list[str]] = [[""] * len(c) for c in miss_chunks]
        for (i, j), chunk in zip(positions, flat):
            if not chunk.strip():
                results_per_miss[i][j] = ""
            else:
                results_per_miss[i][j] = translations.get(chunk, "")

        de_per_miss: list[str] = [" ".join(r).strip() for r in results_per_miss]

        # Step 6: persist new pairs + assemble final output in input order
        self.cache.put_many(list(zip(miss_inputs, de_per_miss)))

        miss_iter = iter(zip(miss_inputs, de_per_miss))
        miss_map = dict(miss_iter)
        out: list[str] = []
        for en in en_texts:
            if en in cached:
                out.append(cached[en])
            else:
                de = miss_map.get(en, "")
                self.stats["translated"] += 1
                self.stats["chars_in"] += len(en)
                self.stats["chars_out"] += len(de)
                out.append(de)
        return out
