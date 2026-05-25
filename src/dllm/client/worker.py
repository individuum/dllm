"""Phase 0.5 worker.

DiLoCo inner loop with bf16 state download, q8 delta upload, optional val-loss
reporting back to coord.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import httpx
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

from ..core import PRESETS, Transformer
from ..data.loader import ShardLoader
from ..shared.protocol import RegisterRequest
from ..shared.serialize import (
    compute_delta,
    deserialize_state,
    load_into_model,
    serialize_delta,
    snapshot,
)

log = logging.getLogger("dllm.worker")


def pick_device(requested: str, require_gpu: bool = False) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if require_gpu:
            raise SystemExit(
                "[GPU CHECK] FAIL: --require-gpu set but no CUDA/MPS available. "
                f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}"
            )
        return torch.device("cpu")
    dev = torch.device(requested)
    if require_gpu and dev.type == "cpu":
        raise SystemExit(f"[GPU CHECK] FAIL: --require-gpu but device={dev}")
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            f"[GPU CHECK] FAIL: requested {dev} but torch.cuda.is_available()=False. "
            f"You have torch={torch.__version__} — likely a CPU-only build."
        )
    return dev


def gpu_info(device: torch.device) -> tuple[str, int]:
    if device.type == "cuda":
        idx = device.index or 0
        props = torch.cuda.get_device_properties(idx)
        return props.name, props.total_memory // (1024**3)
    return device.type, 0


def log_device_banner(device: torch.device) -> None:
    """Prominent, hard-to-miss log line stating which device this process will use."""
    if device.type == "cuda":
        idx = device.index or 0
        props = torch.cuda.get_device_properties(idx)
        log.warning(
            "[GPU CHECK] OK: device=%s name=%s vram=%d GB cuda=%s torch=%s",
            device,
            props.name,
            props.total_memory // (1024**3),
            torch.version.cuda,
            torch.__version__,
        )
    elif device.type == "mps":
        log.warning("[GPU CHECK] OK: device=mps (Apple Metal) torch=%s", torch.__version__)
    else:
        log.warning(
            "[GPU CHECK] WARNING: running on CPU. torch=%s cuda_available=%s. "
            "If you expected GPU, pass --require-gpu to fail fast next time.",
            torch.__version__,
            torch.cuda.is_available(),
        )


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


class Worker:
    def __init__(
        self,
        coord_url: str,
        preset: str,
        country: str,
        device: torch.device,
        train_data: Path,
        val_data: Path | None,
        bf16: bool,
        val_batches: int = 8,
    ) -> None:
        self.coord_url = coord_url.rstrip("/")
        self.preset = preset
        self.country = country
        self.device = device
        self.train_data = train_data
        self.val_data = val_data
        self.bf16 = bf16 and device.type == "cuda"
        self.val_batches = val_batches

        log_device_banner(device)

        cfg = PRESETS[preset]
        self.cfg = cfg
        self.model = Transformer(cfg).to(device)
        log.info("model params: %d (preset=%s)", self.model.num_params(), preset)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            alloc = torch.cuda.memory_allocated(device) / (1024**2)
            log.info("VRAM after model load: %.1f MiB allocated on %s", alloc, device)

        self.http = httpx.Client(base_url=self.coord_url, timeout=httpx.Timeout(120.0))

        # filled after register()
        self.worker_id: int | None = None
        self.world_size: int = 1
        self.current_round: int = 0
        self.inner_steps: int = 0
        self.seq_len: int = 0
        self.micro_batch_size: int = 0
        self.seed: int = 0
        self.state_codec: str = "bf16"
        self.delta_codec: str = "q8"

        self.train_loader: ShardLoader | None = None
        self.val_loader: ShardLoader | None = None
        self.opt: torch.optim.Optimizer | None = None

    # -- registration & state sync -------------------------------------------

    def register(self) -> None:
        name, vram = gpu_info(self.device)
        req = RegisterRequest(
            pubkey=f"phase0-{os.getpid()}",
            country=self.country,
            gpu=name,
            vram_gb=vram,
            ram_gb=0,
            preset=self.preset,
        )
        r = self.http.post("/register", json=req.model_dump())
        r.raise_for_status()
        data = r.json()
        self.worker_id = data["worker_id"]
        self.world_size = data["world_size"]
        self.current_round = data["current_round"]
        self.inner_steps = data["inner_steps"]
        self.seq_len = data["seq_len"]
        self.micro_batch_size = data["micro_batch_size"]
        self.seed = data["seed"]
        self.state_codec = data.get("state_codec", "bf16")
        self.delta_codec = data.get("delta_codec", "q8")
        log.info(
            "registered worker_id=%d round=%d world_size=%d inner=%d codecs=%s/%s",
            self.worker_id,
            self.current_round,
            self.world_size,
            self.inner_steps,
            self.state_codec,
            self.delta_codec,
        )

    def _ensure_loader_and_opt(self) -> None:
        if self.train_loader is None:
            assert self.worker_id is not None
            self.train_loader = ShardLoader(
                self.train_data,
                seq_len=self.seq_len,
                batch_size=self.micro_batch_size,
                worker_id=self.worker_id,
                world_size=self.world_size,
                device=self.device,
                seed=self.seed,
            )
        if self.val_loader is None and self.val_data is not None and self.val_data.exists():
            try:
                self.val_loader = ShardLoader(
                    self.val_data,
                    seq_len=self.seq_len,
                    batch_size=self.micro_batch_size,
                    worker_id=self.worker_id,
                    world_size=self.world_size,
                    device=self.device,
                    seed=self.seed + 10_000,  # different stream from train
                )
            except RuntimeError as e:
                log.warning("val loader unavailable: %s", e)
                self.val_loader = None
        if self.opt is None:
            self.opt = torch.optim.AdamW(
                self.model.parameters(),
                lr=3e-4,
                betas=(0.9, 0.95),
                weight_decay=0.1,
                fused=(self.device.type == "cuda"),
            )

    def pull_state(self) -> int:
        r = self.http.get("/state")
        r.raise_for_status()
        round_no = int(r.headers["x-round"])
        codec = r.headers.get("x-codec", self.state_codec)
        state = deserialize_state(r.content, codec=codec)  # type: ignore[arg-type]
        load_into_model(self.model, state)
        self.current_round = round_no
        log.info("pulled state at round=%d (%d bytes, codec=%s)", round_no, len(r.content), codec)
        return round_no

    # -- inner loop -----------------------------------------------------------

    def run_inner(self) -> dict[str, torch.Tensor]:
        assert self.opt is not None and self.train_loader is not None
        self.model.train()
        snap = snapshot(self.model)

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.bf16
            else _nullcontext()
        )

        loss_sum = 0.0
        t0 = time.time()
        for _ in range(self.inner_steps):
            x, y = self.train_loader.next_batch()
            with autocast_ctx:
                _, loss = self.model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
            loss_sum += float(loss.detach().item())

        avg_loss = loss_sum / max(1, self.inner_steps)
        dt = time.time() - t0
        tok_per_s = self.inner_steps * self.micro_batch_size * self.seq_len / dt
        if self.device.type == "cuda":
            peak_mib = torch.cuda.max_memory_allocated(self.device) / (1024**2)
            log.info(
                "inner round=%d avg_loss=%.4f steps=%d %.0f tok/s peak_vram=%.1f MiB",
                self.current_round,
                avg_loss,
                self.inner_steps,
                tok_per_s,
                peak_mib,
            )
            torch.cuda.reset_peak_memory_stats(self.device)
        else:
            log.info(
                "inner round=%d avg_loss=%.4f steps=%d %.0f tok/s (CPU)",
                self.current_round,
                avg_loss,
                self.inner_steps,
                tok_per_s,
            )

        return compute_delta(snap, self.model)

    @torch.no_grad()
    def run_val(self) -> float | None:
        if self.val_loader is None:
            return None
        self.model.eval()
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.bf16
            else _nullcontext()
        )
        losses: list[float] = []
        for _ in range(self.val_batches):
            x, y = self.val_loader.next_batch()
            with autocast_ctx:
                _, loss = self.model(x, y)
            losses.append(float(loss.item()))
        self.model.train()
        return sum(losses) / len(losses)

    # -- delta submission & barrier ------------------------------------------

    def submit_delta(
        self, delta: dict[str, torch.Tensor], val_loss: float | None = None
    ) -> dict:
        blob = serialize_delta(delta, codec=self.delta_codec)  # type: ignore[arg-type]
        params: dict = {"worker_id": self.worker_id, "round": self.current_round}
        if val_loss is not None:
            params["val_loss"] = val_loss
        r = self.http.post(
            "/delta",
            params=params,
            content=blob,
            headers={"content-type": "application/octet-stream"},
        )
        r.raise_for_status()
        return r.json()

    def wait_for_next_round(self) -> int:
        while True:
            r = self.http.get("/status")
            r.raise_for_status()
            s = r.json()
            if s["current_round"] > self.current_round:
                return s["current_round"]
            time.sleep(0.5)

    # -- top-level loop -------------------------------------------------------

    def run(self, max_rounds: int) -> None:
        self.register()
        self.pull_state()
        self._ensure_loader_and_opt()

        for _ in range(max_rounds):
            delta = self.run_inner()
            val_loss = self.run_val()
            if val_loss is not None:
                log.info("val round=%d loss=%.4f", self.current_round, val_loss)
            ack = self.submit_delta(delta, val_loss=val_loss)
            if not ack["accepted"]:
                log.warning("delta rejected: %s", ack.get("reason"))
                if ack.get("next_round") is not None and ack["next_round"] > self.current_round:
                    self.pull_state()
                continue
            if ack.get("next_round") is None:
                _ = self.wait_for_next_round()
            self.pull_state()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coord", default="http://127.0.0.1:8000")
    ap.add_argument("--preset", default="smoke", choices=list(PRESETS))
    ap.add_argument("--country", default="XX")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--data", default="data/cache/train.bin")
    ap.add_argument("--val-data", default=None, help="Optional val tokens; defaults to val.bin alongside --data")
    ap.add_argument("--val-batches", type=int, default=8)
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--no-bf16", dest="bf16", action="store_false")
    ap.add_argument("--max-rounds", type=int, default=50)
    ap.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail fast if no CUDA/MPS available instead of silently falling back to CPU",
    )
    ap.add_argument("--log-level", default="info")
    args = ap.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    device = pick_device(args.device, require_gpu=args.require_gpu)
    train_data = Path(args.data).resolve()
    if not train_data.exists():
        raise SystemExit(f"data file not found: {train_data}\nrun `python -m dllm.data.prepare` first")
    val_data: Path | None
    if args.val_data:
        val_data = Path(args.val_data).resolve()
    else:
        val_default = train_data.with_name("val.bin")
        val_data = val_default if val_default.exists() else None

    w = Worker(
        coord_url=args.coord,
        preset=args.preset,
        country=args.country,
        device=device,
        train_data=train_data,
        val_data=val_data,
        bf16=args.bf16,
        val_batches=args.val_batches,
    )
    w.run(max_rounds=args.max_rounds)


if __name__ == "__main__":
    main()
