"""Sample text from a coord checkpoint (or any model.safetensors in coord layout).

Loads the weights into a Transformer and autoregressively samples continuations
for a set of prompts, so you can eyeball the base model's quality between/after
training runs. The repo had no inference entry point before this.

Usage:
    python -m dllm.scripts.generate --ckpt ckpt_latest.safetensors --device cpu
    python -m dllm.scripts.generate --ckpt coord/state/latest/model.safetensors \
        --device cuda --max-new-tokens 80 --temperature 0.8 --top-k 40

Runs on CPU by default so it doesn't contend with a live training worker for VRAM.
"""
from __future__ import annotations

import argparse

import torch
from safetensors.torch import load_file

from ..core import PRESETS
from ..core.model import Transformer
from ..data.tokenizer import load_tokenizer

# A few short multilingual prompts (DE/EN/IT/ES/FR) — the corpus is 5-language,
# so this surfaces whether the base learned each.
DEFAULT_PROMPTS = [
    "Die Hauptstadt von Deutschland ist",
    "The European Union was founded to",
    "La capitale dell'Italia è",
    "La inteligencia artificial es",
    "Il était une fois",
]


def load_model(ckpt_path: str, preset: str, device: torch.device) -> Transformer:
    cfg = PRESETS[preset]
    model = Transformer(cfg)
    state = load_file(ckpt_path)
    own = dict(model.named_parameters())
    missing = [n for n in own if n not in state]
    with torch.no_grad():
        for n, p in state.items():
            if n in own:
                own[n].copy_(p.to(own[n].dtype))
    # tie_embeddings: lm_head.weight aliases tok_emb.weight, so it won't appear
    # in the checkpoint separately — that's expected, not a real "missing" param.
    real_missing = [n for n in missing if not n.endswith("lm_head.weight")]
    if real_missing:
        raise SystemExit(f"checkpoint missing params: {real_missing[:5]}...")
    return model.to(device).eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt_latest.safetensors")
    ap.add_argument("--preset", default="300M", choices=list(PRESETS))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt", action="append", help="override prompts (repeatable)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    tok = load_tokenizer()
    model = load_model(args.ckpt, args.preset, device)
    print(
        f"loaded {args.ckpt}: {model.num_params() / 1e6:.0f}M params, "
        f"preset={args.preset}, device={device}"
    )

    prompts = args.prompt or DEFAULT_PROMPTS
    for pr in prompts:
        ids = tok.encode(pr).ids
        x = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(
            x,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        text = tok.decode(out[0].tolist())
        print("\n" + "=" * 70)
        print(f"PROMPT: {pr!r}")
        print("-" * 70)
        print(text)


if __name__ == "__main__":
    main()
