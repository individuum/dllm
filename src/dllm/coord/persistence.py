"""Coord checkpointing — atomic on-disk save of θ + outer optimizer state + meta.

Layout under <root>:
    ckpt_<round:06d>/
        model.safetensors    (fp32 master θ)
        outer_opt.pt         (torch.save of optimizer.state_dict())
        meta.json            ({round, preset_name, world_size, ts})
    latest -> ckpt_<round:06d>/     (symlink on POSIX; junction/copy on Windows)
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path

import torch
from safetensors.torch import load_file as st_load_file
from safetensors.torch import save_file as st_save_file

log = logging.getLogger("dllm.coord.persistence")


def _atomic_dir_swap(target: Path, src: Path) -> None:
    """Rename src -> target atomically, replacing target if present.

    On POSIX, os.replace can swap directories if target is empty; we delete target
    first. On Windows, we delete target then rename.
    """
    if target.exists():
        if target.is_symlink() or target.is_file():
            target.unlink()
        else:
            shutil.rmtree(target)
    os.replace(src, target)


def save_checkpoint(
    root: Path,
    round_no: int,
    model: torch.nn.Module,
    outer_opt: torch.optim.Optimizer,
    meta: dict,
    keep_last: int = 3,
) -> Path:
    """Atomically write a checkpoint dir; prune older ones beyond keep_last."""
    root.mkdir(parents=True, exist_ok=True)
    final = root / f"ckpt_{round_no:06d}"
    staging = root / f".staging_{round_no:06d}_{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    state = {n: p.detach().cpu().contiguous() for n, p in model.named_parameters()}
    st_save_file(state, str(staging / "model.safetensors"))

    torch.save(outer_opt.state_dict(), staging / "outer_opt.pt")

    meta_full = {**meta, "round": round_no, "ts": time.time()}
    (staging / "meta.json").write_text(json.dumps(meta_full, indent=2))

    _atomic_dir_swap(final, staging)
    _update_latest_pointer(root, final)
    _prune_old(root, keep_last)
    log.info("saved checkpoint %s", final)
    return final


def _update_latest_pointer(root: Path, target: Path) -> None:
    """Best-effort 'latest' pointer. Symlink where supported, plain file fallback."""
    latest = root / "latest"
    if latest.exists() or latest.is_symlink():
        try:
            if latest.is_symlink() or latest.is_file():
                latest.unlink()
            else:
                shutil.rmtree(latest)
        except OSError:
            pass
    try:
        os.symlink(target.name, latest, target_is_directory=True)
    except (OSError, NotImplementedError):
        # Windows non-admin: write a plain text file pointing to the dir name
        latest.write_text(target.name)


def _prune_old(root: Path, keep_last: int) -> None:
    ckpts = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("ckpt_")])
    if len(ckpts) <= keep_last:
        return
    for d in ckpts[:-keep_last]:
        shutil.rmtree(d, ignore_errors=True)


def find_latest(root: Path) -> Path | None:
    """Find most recent checkpoint dir. Returns None if root has none."""
    if not root.exists():
        return None
    latest = root / "latest"
    if latest.is_symlink() or latest.is_dir():
        resolved = latest.resolve() if latest.is_symlink() else latest
        if resolved.exists() and (resolved / "meta.json").exists():
            return resolved
    if latest.is_file():
        # plain-text fallback
        target = root / latest.read_text().strip()
        if target.exists():
            return target
    candidates = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("ckpt_")])
    return candidates[-1] if candidates else None


def load_checkpoint(
    ckpt_dir: Path,
    model: torch.nn.Module,
    outer_opt: torch.optim.Optimizer,
) -> dict:
    """Load θ + optimizer state from disk into in-memory model + optimizer. Returns meta."""
    state = st_load_file(str(ckpt_dir / "model.safetensors"))
    own = dict(model.named_parameters())
    with torch.no_grad():
        for n, p in state.items():
            if n not in own:
                raise KeyError(f"checkpoint param {n!r} not in model")
            if own[n].shape != p.shape:
                raise ValueError(f"shape mismatch for {n}: model={own[n].shape} ckpt={p.shape}")
            own[n].copy_(p.to(own[n].dtype).to(own[n].device))
    outer_state = torch.load(ckpt_dir / "outer_opt.pt", weights_only=False, map_location="cpu")
    outer_opt.load_state_dict(outer_state)
    meta = json.loads((ckpt_dir / "meta.json").read_text())
    log.info("loaded checkpoint %s round=%d", ckpt_dir, meta.get("round"))
    return meta
