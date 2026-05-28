"""Offline EN→DE translation of Glaive function-calling v2 → JSONL.

Produces data/cache/sft/glaive_v2_de.jsonl in the same Conversation-per-line
format that hand_written.py and germanrag.py use. The sft_sources module
`glaive_v2_de` reads this JSONL — translation does NOT happen during
sft_prepare.

Architecture rationale: opus-mt's GPU throughput plateaus around batch 64–128.
Translating one conversation at a time inside a producer thread leaves the
GPU idle between calls. Doing it offline, batching all prose across many
conversations at once, keeps it saturated.

What we translate:
  - user.content
  - assistant.content (the prose part — tool_call blocks are structured)
  - system.content (prose with embedded tool-name identifiers — opus-mt is
    robust to snake_case which falls through verbatim)

What we preserve (NEVER translated):
  - tool_calls[].name           — tool names stay English (identifier)
  - tool_calls[].arguments      — JSON structure + technical values
  - tool.content                — JSON tool responses (structured data)

Smoke mode (--smoke N): translate only N conversations and print 3 of them
side-by-side so the operator can eyeball quality before committing to a
full 25k pass.

Usage:
    python -m scripts.translate_glaive_to_de --smoke 100
    python -m scripts.translate_glaive_to_de --n 25000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from dllm.data import sft_sources
from dllm.data.sft_format import Conversation, Message, ToolCall
from dllm.data.sft_translate import Translator

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass


def out_path() -> Path:
    here = Path(__file__).resolve().parent
    p = here.parent / "data" / "cache" / "sft" / "glaive_v2_de.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def collect_prose(conv: Conversation) -> tuple[list[str], list[tuple[int, str]]]:
    """Collect prose strings + per-string (msg_index, kind) so we can
    splice translations back.

    Translates ONLY user + assistant prose. System messages are skipped
    because Glaive's system content is mostly raw JSON tool catalog -
    opus-mt mangles JSON keys ("name" -> "Name", "type" -> "Typ"), breaking
    the catalog structure. Leaving system content English is fine: the
    model just learns that tool catalogs are documented in EN regardless
    of conversation language.

    Tool turns (JSON responses) and tool_call blocks (structured) are
    untouched by design.
    """
    prose: list[str] = []
    locs: list[tuple[int, str]] = []
    for i, msg in enumerate(conv.messages):
        if msg.role in ("user", "assistant") and msg.content:
            prose.append(msg.content)
            locs.append((i, "content"))
    return prose, locs


def splice_translations(
    conv: Conversation,
    locs: list[tuple[int, str]],
    de_texts: list[str],
) -> Conversation:
    """Build a new Conversation with DE prose; preserve tool_calls + tool turns."""
    new_msgs: list[Message] = []
    de_by_loc = dict(zip(locs, de_texts))
    for i, msg in enumerate(conv.messages):
        if (i, "content") in de_by_loc:
            new_msgs.append(Message(
                role=msg.role,
                content=de_by_loc[(i, "content")],
                tool_calls=msg.tool_calls,
                tool_call_id=msg.tool_call_id,
            ))
        else:
            new_msgs.append(msg)
    return Conversation(messages=new_msgs)


def conversation_to_dict(conv: Conversation) -> dict:
    """Serialize Conversation to the JSONL schema that hand_written.py reads."""
    return {
        "lang": "de",
        "messages": [
            {
                "role": m.role,
                "content": m.content,
                **({"tool_calls": [
                    {"name": tc.name, "arguments": tc.arguments,
                     **({"id": tc.id} if tc.id else {})}
                    for tc in m.tool_calls
                ]} if m.tool_calls else {}),
                **({"tool_call_id": m.tool_call_id} if m.tool_call_id else {}),
            }
            for m in conv.messages
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=25_000,
                    help="Number of Glaive conversations to translate.")
    ap.add_argument("--smoke", type=int, default=0,
                    help="If >0, only translate N conversations and print 3 samples for sniff.")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="opus-mt batch size. 32 keeps activations modest with "
                         "num_beams=2 + max_length=256; raise if memory headroom allows.")
    ap.add_argument("--prose-buffer", type=int, default=200,
                    help="Convs collected per translation wave. Higher = better "
                         "batching efficiency, but each wave's prose all sits in RAM.")
    ap.add_argument("--device", default=None, choices=["cuda", "cpu"],
                    help="Override device. Default: cuda if available.")
    args = ap.parse_args()

    n_total = args.smoke if args.smoke > 0 else args.n
    print(f"[translate-glaive] target: {n_total} conversations -> {out_path()}", file=sys.stderr)

    translator = Translator(device=args.device, batch_size=args.batch_size)

    glaive = sft_sources.load("glaive_v2")
    src_iter = glaive.iter_examples("en", n_total)

    t0 = time.time()
    out_p = out_path()
    if args.smoke > 0:
        out_p = out_p.with_name(out_p.stem + "_smoke.jsonl")

    n_written = 0
    samples_for_smoke: list[tuple[Conversation, Conversation]] = []

    with out_p.open("w", encoding="utf-8") as fp:
        buf: list[Conversation] = []

        def flush_buffer() -> None:
            nonlocal n_written
            if not buf:
                return
            # Collect all prose across the buffered conversations.
            all_prose: list[str] = []
            per_conv_locs: list[list[tuple[int, str]]] = []
            per_conv_offsets: list[tuple[int, int]] = []
            for c in buf:
                prose, locs = collect_prose(c)
                per_conv_offsets.append((len(all_prose), len(all_prose) + len(prose)))
                all_prose.extend(prose)
                per_conv_locs.append(locs)
            # One big batched call into the GPU.
            de_texts = translator.translate_many(all_prose)
            # Splice back, write JSONL.
            for c, (s, e), locs in zip(buf, per_conv_offsets, per_conv_locs):
                de_conv = splice_translations(c, locs, de_texts[s:e])
                fp.write(json.dumps(conversation_to_dict(de_conv), ensure_ascii=False) + "\n")
                if args.smoke > 0 and len(samples_for_smoke) < 3:
                    samples_for_smoke.append((c, de_conv))
                n_written += 1
            buf.clear()

        for conv in src_iter:
            buf.append(conv)
            if len(buf) >= args.prose_buffer:
                flush_buffer()
                elapsed = time.time() - t0
                stats = translator.stats
                rate = n_written / max(elapsed, 1e-3)
                eta = (n_total - n_written) / max(rate, 1e-3)
                print(
                    f"  [translate] {n_written:,}/{n_total:,} convs "
                    f"({rate:.1f} convs/s, ETA {eta/60:.1f} min) "
                    f"cache_hits={stats['cached']} new={stats['translated']}",
                    file=sys.stderr, flush=True,
                )
            if n_written + len(buf) >= n_total:
                break
        flush_buffer()

    elapsed = time.time() - t0
    print(f"[translate-glaive] wrote {n_written} convs to {out_p} in {elapsed/60:.1f} min",
          file=sys.stderr)
    stats = translator.stats
    print(f"[translate-glaive] stats: {stats}", file=sys.stderr)

    if args.smoke > 0:
        print("\n========== SMOKE SAMPLES ==========\n", file=sys.stderr)
        for k, (en_conv, de_conv) in enumerate(samples_for_smoke):
            print(f"--- Sample {k + 1} ---", file=sys.stderr)
            for en_msg, de_msg in zip(en_conv.messages, de_conv.messages):
                role = en_msg.role
                if en_msg.tool_calls:
                    tc = en_msg.tool_calls[0]
                    print(f"  [{role}] tool_call: {tc.name}({tc.arguments})",
                          file=sys.stderr)
                if en_msg.content and en_msg.content != de_msg.content:
                    en_short = en_msg.content[:240].replace("\n", " ")
                    de_short = de_msg.content[:240].replace("\n", " ")
                    print(f"  [{role}] EN: {en_short}", file=sys.stderr)
                    print(f"  [{role}] DE: {de_short}", file=sys.stderr)
                elif en_msg.content:
                    print(f"  [{role}] (unchanged) {en_msg.content[:120]}",
                          file=sys.stderr)
            print("", file=sys.stderr)


if __name__ == "__main__":
    main()
