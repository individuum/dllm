from __future__ import annotations

from pathlib import Path

import torch

from dllm.coord.persistence import find_latest, load_checkpoint, save_checkpoint
from dllm.core import Transformer


def _opt(model: torch.nn.Module) -> torch.optim.Optimizer:
    return torch.optim.SGD(model.parameters(), lr=0.7, momentum=0.9, nesterov=True)


def test_save_and_load_roundtrip(tmp_path: Path, tiny_model: Transformer) -> None:
    opt = _opt(tiny_model)
    # take an optimizer step so its state is non-empty
    for p in tiny_model.parameters():
        p.grad = torch.ones_like(p) * 0.1
    opt.step()

    save_checkpoint(tmp_path, round_no=7, model=tiny_model, outer_opt=opt, meta={"preset_name": "tiny"})
    ckpt = find_latest(tmp_path)
    assert ckpt is not None
    assert ckpt.name == "ckpt_000007"

    # nuke params, then reload
    with torch.no_grad():
        for p in tiny_model.parameters():
            p.zero_()
    fresh_opt = _opt(tiny_model)
    meta = load_checkpoint(ckpt, tiny_model, fresh_opt)
    assert meta["round"] == 7
    assert meta["preset_name"] == "tiny"
    # params now non-zero again
    assert any(p.abs().sum() > 0 for p in tiny_model.parameters())


def test_keep_last_prunes_old(tmp_path: Path, tiny_model: Transformer) -> None:
    opt = _opt(tiny_model)
    for r in range(5):
        save_checkpoint(tmp_path, round_no=r, model=tiny_model, outer_opt=opt, meta={}, keep_last=2)
    ckpts = sorted([d for d in tmp_path.iterdir() if d.is_dir() and d.name.startswith("ckpt_")])
    assert len(ckpts) == 2
    assert ckpts[-1].name == "ckpt_000004"
    assert ckpts[0].name == "ckpt_000003"


def test_find_latest_on_empty_dir(tmp_path: Path) -> None:
    assert find_latest(tmp_path) is None
    nonexistent = tmp_path / "nope"
    assert find_latest(nonexistent) is None
