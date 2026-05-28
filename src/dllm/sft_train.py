"""Single-GPU supervised fine-tuning (SFT) on the EuroAgent corpus.

Loads a pretrained base (a coord checkpoint's `model.safetensors`), fine-tunes
on `data/cache/sft/{sft_train,sft_val}.bin` + `*_mask.bin` with **assistant-only
loss** (`SFTShardLoader` emits -100 on non-assistant targets, which the model's
`F.cross_entropy(ignore_index=-100)` skips), using AdamW + a cosine LR schedule,
then writes a fine-tuned `model.safetensors`.

Why single-GPU and not the DiLoCo coord/worker path: SFT is small (~50–80 M
tokens, a couple of epochs) and benefits from one clean run. The distributed
machinery only pays off for the expensive pretraining. The output checkpoint is
written in the SAME layout as a coord checkpoint (model.safetensors + meta.json)
so it can seed a serve/RLHF step or be uploaded back to the coord later.

Run:
    python -m dllm.sft_train --base /path/to/ckpt_000474 \\
        --steps 2000 --batch-size 8 --seq-len 2048 --device cuda --require-gpu
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import torch
from safetensors.torch import load_file as st_load_file
from safetensors.torch import save_file as st_save_file

from .core.config import PRESETS, ModelConfig
from .core.model import Transformer
from .data.loader import SFTShardLoader

log = logging.getLogger("dllm.sft_train")


# ---------------------------------------------------------------------------
# model + base weights
# ---------------------------------------------------------------------------


def build_model(cfg: ModelConfig) -> Transformer:
    return Transformer(cfg)


def load_base_weights(model: torch.nn.Module, base_path: Path) -> dict:
    """Copy pretrained θ into `model`. `base_path` may be a coord checkpoint
    dir (contains model.safetensors + meta.json) or a .safetensors file.

    Returns the base meta dict ({} if a bare file). Mirrors the param-copy in
    `dllm.coord.persistence.load_checkpoint` but loads ONLY the model (SFT
    brings its own optimizer, so outer_opt.pt is ignored).
    """
    base_path = Path(base_path)
    if base_path.is_dir():
        st_path = base_path / "model.safetensors"
        meta_path = base_path / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    else:
        st_path = base_path
        meta = {}
    if not st_path.exists():
        raise FileNotFoundError(f"base weights not found: {st_path}")

    state = st_load_file(str(st_path))
    own = dict(model.named_parameters())
    missing = set(own) - set(state)
    unexpected = set(state) - set(own)
    if missing or unexpected:
        raise ValueError(
            f"base/model param mismatch — missing={sorted(missing)[:4]}... "
            f"unexpected={sorted(unexpected)[:4]}... (wrong --preset for this base?)"
        )
    with torch.no_grad():
        for n, p in state.items():
            if own[n].shape != p.shape:
                raise ValueError(f"shape mismatch for {n}: model={own[n].shape} base={p.shape}")
            own[n].copy_(p.to(own[n].dtype).to(own[n].device))
    return meta


def resolve_preset(base_path: Path, preset_arg: str | None) -> tuple[str, ModelConfig]:
    """Pick the preset: explicit --preset wins, else read meta.json from the
    base checkpoint dir, else error."""
    if preset_arg:
        return preset_arg, PRESETS[preset_arg]
    base_path = Path(base_path)
    if base_path.is_dir() and (base_path / "meta.json").exists():
        meta = json.loads((base_path / "meta.json").read_text())
        name = meta.get("preset_name")
        if name and name in PRESETS:
            return name, PRESETS[name]
    raise SystemExit(
        "could not infer --preset from base (no meta.json preset_name); pass --preset explicitly"
    )


# ---------------------------------------------------------------------------
# schedule + eval
# ---------------------------------------------------------------------------


def cosine_lr(step: int, total_steps: int, peak_lr: float, min_lr: float, warmup_steps: int) -> float:
    """Linear warmup → cosine decay → flat floor. Step-based sibling of the
    worker's round-based cosine_lr_for_round."""
    if warmup_steps > 0 and step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    val_loader: SFTShardLoader,
    n_batches: int,
    *,
    autocast_dtype: torch.dtype | None,
    device_type: str,
) -> float:
    """Mean assistant-only val loss over n_batches (skips non-finite batches —
    a rare all-ignored window contributes no signal)."""
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    for _ in range(n_batches):
        x, y = val_loader.next_batch()
        if autocast_dtype is not None:
            with torch.autocast(device_type=device_type, dtype=autocast_dtype):
                _, loss = model(x, y)
        else:
            _, loss = model(x, y)
        if loss is not None and torch.isfinite(loss):
            total += float(loss.item())
            count += 1
    if was_training:
        model.train()
    return total / max(1, count)


# ---------------------------------------------------------------------------
# training core (testable; main() is a thin CLI around it)
# ---------------------------------------------------------------------------


def run_sft(
    model: torch.nn.Module,
    train_loader: SFTShardLoader,
    val_loader: SFTShardLoader | None,
    *,
    steps: int,
    peak_lr: float = 2e-5,
    min_lr: float = 2e-6,
    warmup_steps: int = 50,
    weight_decay: float = 0.0,
    grad_clip: float = 1.0,
    grad_accum: int = 1,
    betas: tuple[float, float] = (0.9, 0.95),
    autocast_dtype: torch.dtype | None = None,
    device_type: str = "cpu",
    log_every: int = 25,
    val_every: int = 0,
    val_batches: int = 20,
) -> dict:
    """Run `steps` optimizer steps of masked-loss SFT. Returns a history dict
    {train_loss: [(step, loss)], val_loss: [(step, loss)], final_val}."""
    opt = torch.optim.AdamW(
        model.parameters(), lr=peak_lr, betas=betas, weight_decay=weight_decay
    )
    model.train()
    history: dict = {"train_loss": [], "val_loss": [], "final_val": None}

    for step in range(steps):
        lr = cosine_lr(step, steps, peak_lr, min_lr, warmup_steps)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(grad_accum):
            x, y = train_loader.next_batch()
            if autocast_dtype is not None:
                with torch.autocast(device_type=device_type, dtype=autocast_dtype):
                    _, loss = model(x, y)
            else:
                _, loss = model(x, y)
            # Guard the (rare) all-ignored micro-batch: its loss is NaN and
            # would poison the grads. Skip its backward contribution.
            if loss is None or not torch.isfinite(loss):
                continue
            (loss / grad_accum).backward()
            step_loss += float(loss.item()) / grad_accum

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        opt.step()

        if step % log_every == 0 or step == steps - 1:
            history["train_loss"].append((step, step_loss))
            log.info("sft step %d/%d loss=%.4f lr=%.2e", step, steps, step_loss, lr)
        if val_loader is not None and val_every > 0 and step > 0 and step % val_every == 0:
            vl = evaluate(
                model, val_loader, val_batches,
                autocast_dtype=autocast_dtype, device_type=device_type,
            )
            history["val_loss"].append((step, vl))
            log.info("sft step %d val_loss=%.4f", step, vl)

    if val_loader is not None:
        history["final_val"] = evaluate(
            model, val_loader, val_batches,
            autocast_dtype=autocast_dtype, device_type=device_type,
        )
        log.info("sft done; final val_loss=%.4f", history["final_val"])
    return history


def save_sft_checkpoint(model: torch.nn.Module, out_dir: Path, meta: dict) -> Path:
    """Write fine-tuned θ in coord-checkpoint layout (model.safetensors +
    meta.json) so it can be loaded by load_base_weights / seeded to the coord."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {n: p.detach().cpu().contiguous() for n, p in model.named_parameters()}
    st_save_file(state, str(out_dir / "model.safetensors"))
    (out_dir / "meta.json").write_text(json.dumps({**meta, "ts": time.time()}, indent=2))
    return out_dir


# ---------------------------------------------------------------------------
# device + CLI
# ---------------------------------------------------------------------------


def pick_device(arg: str) -> torch.device:
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, type=Path,
                    help="Pretrained base: a coord checkpoint dir (ckpt_NNNNNN) or a model.safetensors file.")
    ap.add_argument("--preset", default=None, choices=list(PRESETS),
                    help="Model preset. Inferred from the base checkpoint's meta.json if omitted.")
    ap.add_argument("--sft-dir", type=Path, default=Path("data/cache/sft"),
                    help="Directory holding sft_{train,val}.bin + *_mask.bin.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output checkpoint dir (default: <sft-dir>/sft_model).")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=2048,
                    help="Clamped to the preset's max_seq_len.")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--min-lr", type=float, default=2e-6)
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--val-every", type=int, default=200)
    ap.add_argument("--val-batches", type=int, default=20)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--require-gpu", action="store_true",
                    help="Abort if the resolved device is CPU (guards against silent slow runs).")
    ap.add_argument("--no-bf16", action="store_true", help="Disable bf16 autocast on CUDA.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    preset_name, cfg = resolve_preset(args.base, args.preset)
    seq_len = min(args.seq_len, cfg.max_seq_len)
    device = pick_device(args.device)
    if args.require_gpu and device.type == "cpu":
        raise SystemExit("--require-gpu set but resolved device is CPU; aborting.")
    torch.manual_seed(args.seed)

    log.info("SFT: preset=%s device=%s seq_len=%d", preset_name, device, seq_len)
    model = build_model(cfg).to(device)
    base_meta = load_base_weights(model, args.base)
    log.info("loaded base weights from %s (base round=%s)", args.base, base_meta.get("round"))

    sft = args.sft_dir
    train_loader = SFTShardLoader(
        sft / "sft_train.bin", sft / "sft_train_mask.bin",
        seq_len=seq_len, batch_size=args.batch_size, device=device, seed=args.seed,
    )
    val_path = sft / "sft_val.bin"
    val_loader = None
    if val_path.exists():
        val_loader = SFTShardLoader(
            val_path, sft / "sft_val_mask.bin",
            seq_len=seq_len, batch_size=args.batch_size, device=device, seed=args.seed,
        )

    use_bf16 = (device.type == "cuda") and not args.no_bf16
    autocast_dtype = torch.bfloat16 if use_bf16 else None

    t0 = time.time()
    history = run_sft(
        model, train_loader, val_loader,
        steps=args.steps, peak_lr=args.lr, min_lr=args.min_lr,
        warmup_steps=args.warmup_steps, weight_decay=args.weight_decay,
        grad_accum=args.grad_accum, autocast_dtype=autocast_dtype,
        device_type=device.type, val_every=args.val_every, val_batches=args.val_batches,
    )

    out_dir = args.out or (sft / "sft_model")
    meta = {
        "kind": "sft",
        "preset_name": preset_name,
        "base": str(args.base),
        "base_round": base_meta.get("round"),
        "steps": args.steps,
        "seq_len": seq_len,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "peak_lr": args.lr,
        "final_val_loss": history.get("final_val"),
        "wall_seconds": round(time.time() - t0, 1),
    }
    save_sft_checkpoint(model, out_dir, meta)
    log.info("wrote SFT checkpoint to %s (final_val=%s)", out_dir, history.get("final_val"))


if __name__ == "__main__":
    main()
