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
from concurrent.futures import Future, ThreadPoolExecutor
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
from ..shared.identity import (
    load_or_create_identity,
    pubkey_hex,
    sign_delta,
)
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
                f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} "
                f"mps_available={torch.backends.mps.is_available()}"
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
    if dev.type == "mps" and not torch.backends.mps.is_available():
        raise SystemExit(
            f"[GPU CHECK] FAIL: requested {dev} but torch.backends.mps.is_available()=False. "
            f"Needs Apple Silicon + macOS 12.3+ + a recent PyTorch."
        )
    return dev


def _apple_chip_info() -> tuple[str, int]:
    """Return (chip name, unified memory GB) on macOS. ('Apple Silicon', 0) on failure."""
    import platform
    import subprocess

    if platform.system() != "Darwin":
        return "Apple Silicon", 0
    try:
        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except Exception:  # noqa: BLE001
        chip = "Apple Silicon"
    try:
        mem_bytes = int(
            subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        )
        mem_gb = mem_bytes // (1024**3)
    except Exception:  # noqa: BLE001
        mem_gb = 0
    return chip, mem_gb


def gpu_info(device: torch.device) -> tuple[str, int]:
    if device.type == "cuda":
        idx = device.index or 0
        props = torch.cuda.get_device_properties(idx)
        return props.name, props.total_memory // (1024**3)
    if device.type == "mps":
        return _apple_chip_info()
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
        chip, unified_gb = _apple_chip_info()
        log.warning(
            "[GPU CHECK] OK: device=mps name=%s unified_memory=%d GB torch=%s",
            chip,
            unified_gb,
            torch.__version__,
        )
    else:
        log.warning(
            "[GPU CHECK] WARNING: running on CPU. torch=%s cuda_available=%s mps_available=%s. "
            "If you expected GPU, pass --require-gpu to fail fast next time.",
            torch.__version__,
            torch.cuda.is_available(),
            torch.backends.mps.is_available(),
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
        # bf16 autocast on CUDA (Ampere+) and MPS (PyTorch 2.5+ on Apple Silicon).
        # On older MPS where bf16 is rejected, the run_inner autocast will surface
        # an error and the user can rerun with --no-bf16.
        self.bf16 = bf16 and device.type in ("cuda", "mps")
        self.val_batches = val_batches

        # Ed25519 identity — persisted in .dllm/identity.key
        self.sk = load_or_create_identity()
        self.pubkey_hex = pubkey_hex(self.sk)
        log.info("identity pubkey: %s", self.pubkey_hex[:16] + "...")

        log_device_banner(device)

        cfg = PRESETS[preset]
        self.cfg = cfg
        self.model = Transformer(cfg).to(device)
        log.info("model params: %d (preset=%s)", self.model.num_params(), preset)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            alloc = torch.cuda.memory_allocated(device) / (1024**2)
            log.info("VRAM after model load: %.1f MiB allocated on %s", alloc, device)
        elif device.type == "mps":
            alloc = torch.mps.current_allocated_memory() / (1024**2)
            log.info("unified mem after model load: %.1f MiB allocated on %s", alloc, device)

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

        # async sync: a single background thread submits Δ + fetches the next θ
        # while the GPU runs the next inner loop. `last_ref` is the most recently
        # known coord state; new deltas are computed as (last_ref - local).
        # Lazy-async DiLoCo: local is never reset, only last_ref tracks the
        # consensus. See arXiv:2401.09135.
        self.last_ref: dict[str, torch.Tensor] = {}
        self._exec = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dllm-sync")

    # -- registration & state sync -------------------------------------------

    def register(self) -> None:
        name, vram = gpu_info(self.device)
        req = RegisterRequest(
            pubkey=self.pubkey_hex,
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
        self._capture_ref()
        log.info("pulled state at round=%d (%d bytes, codec=%s)", round_no, len(r.content), codec)
        return round_no

    def _capture_ref(self) -> None:
        """Snapshot current model params as the reference for the next delta."""
        with torch.no_grad():
            self.last_ref = {
                n: p.detach().clone() for n, p in self.model.named_parameters()
            }

    # -- inner loop -----------------------------------------------------------

    def run_inner(self) -> dict[str, torch.Tensor]:
        assert self.opt is not None and self.train_loader is not None
        self.model.train()
        snap = snapshot(self.model)

        autocast_ctx = (
            torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
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
        elif self.device.type == "mps":
            # MPS reports current rather than peak; close enough at end-of-loop
            mem_mib = torch.mps.current_allocated_memory() / (1024**2)
            log.info(
                "inner round=%d avg_loss=%.4f steps=%d %.0f tok/s mem=%.1f MiB (MPS)",
                self.current_round,
                avg_loss,
                self.inner_steps,
                tok_per_s,
                mem_mib,
            )
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
            torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
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

    # -- async sync (background thread) --------------------------------------

    def _async_sync_io(
        self, blob: bytes, val_loss: float | None, round_no: int
    ) -> dict:
        """Background: sign, POST /delta, wait for round advance, GET /state."""
        sig = sign_delta(self.sk, self.worker_id, round_no, blob)
        params: dict = {"worker_id": self.worker_id, "round": round_no}
        if val_loss is not None:
            params["val_loss"] = val_loss
        r = self.http.post(
            "/delta",
            params=params,
            content=blob,
            headers={
                "content-type": "application/octet-stream",
                "x-delta-signature": sig,
            },
        )
        r.raise_for_status()
        ack = r.json()
        if not ack.get("accepted"):
            # Stale-round rejection: in heterogeneous fleets, fast workers can
            # advance the coord past where we submitted. Don't bail — fetch the
            # current consensus state and re-align. The caller (_apply_sync_result)
            # will reload the local model from this state, dropping the delta
            # we just wasted but keeping the worker contributing.
            next_r = ack.get("next_round")
            if next_r is not None and next_r > round_no:
                r2 = self.http.get("/state")
                r2.raise_for_status()
                codec = r2.headers.get("x-codec", self.state_codec)
                new_state = deserialize_state(r2.content, codec=codec)  # type: ignore[arg-type]
                return {
                    "ack": ack,
                    "state": new_state,
                    "round": int(r2.headers["x-round"]),
                    "state_bytes": len(r2.content),
                    "resync": True,
                }
            return {"ack": ack, "state": None, "round": round_no, "state_bytes": 0}

        target = ack.get("next_round")
        if target is None:
            # other workers haven't submitted yet — long-poll status
            while True:
                rs = self.http.get("/status")
                rs.raise_for_status()
                cur = rs.json()["current_round"]
                if cur > round_no:
                    break
                time.sleep(0.5)

        r2 = self.http.get("/state")
        r2.raise_for_status()
        codec = r2.headers.get("x-codec", self.state_codec)
        new_state = deserialize_state(r2.content, codec=codec)  # type: ignore[arg-type]
        return {
            "ack": ack,
            "state": new_state,
            "round": int(r2.headers["x-round"]),
            "state_bytes": len(r2.content),
        }

    def _apply_sync_result(self, result: dict) -> bool:
        """Update last_ref (always); on resync, also reload the local model to
        avoid unbounded drift from old consensus. Returns True if applied.
        """
        state = result.get("state")
        if state is None:
            log.warning("sync rejected without recoverable state: %s", result.get("ack"))
            return False
        with torch.no_grad():
            for n, ref_t in self.last_ref.items():
                ref_t.copy_(state[n].to(ref_t.dtype).to(ref_t.device))
        if result.get("resync"):
            # We were stale; the next delta we send would otherwise encode
            # several rounds of accumulated local drift from a no-longer-current
            # consensus. Reload local θ from the current consensus so the next
            # inner loop starts fresh.
            load_into_model(self.model, state)
            log.warning(
                "RESYNC to round=%d (caught up from stale-round rejection); "
                "local model reloaded from consensus",
                result["round"],
            )
        self.current_round = result["round"]
        log.info(
            "async sync applied round=%d (%d state bytes)%s",
            self.current_round,
            result.get("state_bytes", 0),
            " [resync]" if result.get("resync") else "",
        )
        return True

    # -- top-level loop -------------------------------------------------------

    def run(self, max_rounds: int) -> None:
        """Lazy-async DiLoCo: GPU runs continuously; network exchanges overlap."""
        self.register()
        self.pull_state()  # captures last_ref
        self._ensure_loader_and_opt()

        rounds_committed = 0
        pending: Future | None = None

        try:
            while rounds_committed < max_rounds:
                # GPU: inner steps then val on the current local θ
                self.run_inner()
                val_loss = self.run_val()
                if val_loss is not None:
                    log.info("val round=%d loss=%.4f", self.current_round, val_loss)

                # finalize previous round's sync (blocks if network slower than compute)
                if pending is not None:
                    result = pending.result()
                    pending = None
                    if not self._apply_sync_result(result):
                        log.error("sync failed; stopping early")
                        break
                    rounds_committed += 1
                    if rounds_committed >= max_rounds:
                        break

                # delta = last known global − current local; sent for the next outer step
                delta = compute_delta(self.last_ref, self.model)
                blob = serialize_delta(delta, codec=self.delta_codec)  # type: ignore[arg-type]

                # kick off async sync — next inner loop runs in parallel
                pending = self._exec.submit(
                    self._async_sync_io, blob, val_loss, self.current_round
                )

            # drain any in-flight sync
            if pending is not None:
                log.info("draining final sync...")
                if self._apply_sync_result(pending.result()):
                    rounds_committed += 1
        finally:
            self._exec.shutdown(wait=False)


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
