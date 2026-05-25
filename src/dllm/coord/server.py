"""Phase 0.5 coordinator.

Holds the global model state; collects pseudo-gradient deltas from N workers;
applies a Nesterov outer step when all workers in the current round have submitted.
Supports bf16 state transport + q8 delta transport, periodic disk checkpoints,
val-loss aggregation, and cumulative FLOPs accounting.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse

_DASHBOARD_PATH = Path(__file__).parent / "dashboard.html"


def _load_dashboard_html() -> str:
    """Re-read on every request — UI tweaks no longer require a coord restart."""
    return _DASHBOARD_PATH.read_text(encoding="utf-8")

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

from ..core import PRESETS, ModelConfig, Transformer
from ..core.config import TrainConfig
from ..shared.identity import pubkey_from_hex, verify_delta
from ..shared.protocol import DeltaAck, RegisterRequest, RegisterResponse, RoundStatus
from ..shared.serialize import (
    DeltaCodec,
    StateCodec,
    average_deltas,
    deserialize_delta,
    model_state,
    serialize_state,
)
from .persistence import find_latest, load_checkpoint, save_checkpoint

log = logging.getLogger("dllm.coord")


class CoordinatorState:
    """All mutable global state; guarded by `lock`."""

    def __init__(
        self,
        preset_name: str,
        world_size: int,
        train_cfg: TrainConfig,
        device: str = "cpu",
        state_codec: StateCodec = "bf16",
        delta_codec: DeltaCodec = "q8",
        checkpoint_dir: Path | None = None,
        checkpoint_every: int = 10,
        resume: bool = True,
        require_signed_deltas: bool = False,
        round_timeout_seconds: float = 900.0,
        min_workers: int = 1,
        enable_timeout_thread: bool = True,
    ) -> None:
        if preset_name not in PRESETS:
            raise ValueError(f"unknown preset {preset_name!r}; have {list(PRESETS)}")
        self.preset_name = preset_name
        self.cfg: ModelConfig = PRESETS[preset_name]
        self.train_cfg = train_cfg
        self.world_size = world_size
        self.device = torch.device(device)
        self.state_codec: StateCodec = state_codec
        self.delta_codec: DeltaCodec = delta_codec
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_every = checkpoint_every
        self.require_signed_deltas = require_signed_deltas
        # Round-level timeout / min-quorum: if a round has been open >
        # round_timeout_seconds and we have >= min_workers deltas, force the
        # outer step with whoever submitted. Lets the cohort survive worker
        # dropouts without operator intervention. Late deltas hit the
        # resync path on the worker side.
        self.round_timeout_seconds = max(0.0, round_timeout_seconds)
        self.min_workers = max(1, min(min_workers, world_size))
        # Ring buffer of (round, val_loss, flops_total, ts) appended at each
        # outer step. Powers the dashboard chart at GET /.
        # Persisted to <checkpoint_dir>/history.jsonl (append-only NDJSON)
        # so the chart survives coord restarts.
        self.history: deque[dict] = deque(maxlen=500)
        self._history_path: Path | None = (
            self.checkpoint_dir / "history.jsonl" if self.checkpoint_dir else None
        )

        torch.manual_seed(train_cfg.seed)
        self.model = Transformer(self.cfg).to(self.device)
        self.outer_opt = torch.optim.SGD(
            self.model.parameters(),
            lr=train_cfg.outer_lr,
            momentum=train_cfg.outer_momentum,
            nesterov=True,
        )

        self.lock = threading.Lock()
        self.round = 0
        self.next_worker_id = 0
        self.workers: dict[int, dict[str, Any]] = {}
        self.deltas: dict[int, dict[int, dict[str, torch.Tensor]]] = {0: {}}
        self.val_losses: dict[int, list[float]] = {0: []}
        self.last_val_loss: float | None = None
        self.flops_total: float = 0.0  # cumulative estimate
        self.round_started_at = time.time()
        self._state_bytes: bytes | None = None

        self._timeout_stop = threading.Event()
        self._timeout_thread: threading.Thread | None = None
        if enable_timeout_thread and self.round_timeout_seconds > 0:
            self._timeout_thread = threading.Thread(
                target=self._timeout_loop,
                daemon=True,
                name="dllm-coord-timeout",
            )
            self._timeout_thread.start()

        if resume and self.checkpoint_dir is not None:
            latest = find_latest(self.checkpoint_dir)
            if latest is not None:
                try:
                    meta = load_checkpoint(latest, self.model, self.outer_opt)
                    self.round = int(meta.get("round", 0)) + 1
                    self.flops_total = float(meta.get("flops_total", 0.0))
                    self.deltas = {self.round: {}}
                    self.val_losses = {self.round: []}
                    log.info("resumed from %s -> starting at round %d", latest, self.round)
                except (KeyError, ValueError, RuntimeError, OSError) as e:
                    log.warning(
                        "checkpoint %s incompatible or unreadable (%s); "
                        "starting fresh at round 0",
                        latest,
                        e,
                    )
            # Always try to populate history, even if no compatible checkpoint
            self._load_history_from_disk()
            self._backfill_history_from_checkpoint_metas()

    # -- helpers --------------------------------------------------------------

    def _serialize_state(self) -> bytes:
        if self._state_bytes is None:
            self._state_bytes = serialize_state(model_state(self.model), codec=self.state_codec)
        return self._state_bytes

    def _invalidate_state_cache(self) -> None:
        self._state_bytes = None

    # -- history persistence ---------------------------------------------------

    def _append_history(self, entry: dict) -> None:
        """Append to the in-memory deque AND the on-disk NDJSON log."""
        self.history.append(entry)
        if self._history_path is not None:
            try:
                self._history_path.parent.mkdir(parents=True, exist_ok=True)
                with self._history_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")
            except OSError as e:
                log.warning("could not persist history entry: %s", e)

    def _load_history_from_disk(self) -> None:
        """Load history.jsonl into the deque on startup (caps at maxlen)."""
        if self._history_path is None or not self._history_path.exists():
            return
        entries: list[dict] = []
        try:
            with self._history_path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue  # tolerate partial last write
        except OSError as e:
            log.warning("could not load history.jsonl: %s", e)
            return
        # Deque maxlen drops oldest automatically
        for entry in entries[-self.history.maxlen :]:
            self.history.append(entry)
        log.info("loaded %d history entries from %s", len(self.history), self._history_path)

    def _backfill_history_from_checkpoint_metas(self) -> None:
        """Reconstruct sparse history from each ckpt_*/meta.json (one per
        checkpoint-every rounds). First-time enable: fills in everything that
        happened before history persistence existed.
        """
        if self.checkpoint_dir is None or not self.checkpoint_dir.exists():
            return
        existing_rounds = {h.get("round") for h in self.history}
        added = 0
        for ckpt_dir in sorted(self.checkpoint_dir.iterdir()):
            if not ckpt_dir.is_dir() or not ckpt_dir.name.startswith("ckpt_"):
                continue
            meta_file = ckpt_dir / "meta.json"
            if not meta_file.exists():
                continue
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            round_no = meta.get("round")
            if round_no in existing_rounds:
                continue
            self.history.append(
                {
                    "round": round_no,
                    "val_loss": meta.get("last_val_loss"),
                    "flops_total": float(meta.get("flops_total", 0.0)),
                    "ts": float(meta.get("ts", 0.0)),
                }
            )
            existing_rounds.add(round_no)
            added += 1
        if added:
            # Re-sort by round so the chart draws cleanly
            sorted_hist = sorted(self.history, key=lambda h: (h.get("round") or 0))
            self.history.clear()
            self.history.extend(sorted_hist)
            log.info(
                "backfilled %d history entries from %d checkpoint meta files",
                added,
                added,
            )

    # -- timeout-based round eviction ----------------------------------------

    def _timeout_loop(self) -> None:
        """Background thread: poll for timed-out rounds and force-advance them."""
        while not self._timeout_stop.wait(timeout=2.0):
            try:
                self._check_and_force_advance()
            except Exception:  # noqa: BLE001
                log.exception("timeout-thread error")

    def _check_and_force_advance(self) -> bool:
        """Idempotent. If round has been open > timeout AND >= min_workers have
        submitted, run the outer step with what we have. Returns True on advance.
        """
        if self.round_timeout_seconds <= 0:
            return False
        with self.lock:
            elapsed = time.time() - self.round_started_at
            submitted = len(self.deltas.get(self.round, {}))
            if elapsed < self.round_timeout_seconds:
                return False
            if submitted < self.min_workers:
                return False  # too few deltas — keep waiting (e.g. all workers offline)
            if submitted >= self.world_size:
                return False  # the regular path will handle this on the submitting thread
            log.warning(
                "[TIMEOUT] forcing outer step at round=%d with %d/%d deltas after %.1fs (timeout=%.1fs, min_workers=%d)",
                self.round,
                submitted,
                self.world_size,
                elapsed,
                self.round_timeout_seconds,
                self.min_workers,
            )
            self._outer_step_locked()
            return True

    def stop(self) -> None:
        """Signal the timeout thread to exit. Idempotent; thread is daemon anyway."""
        self._timeout_stop.set()

    def _estimate_round_flops(self) -> float:
        """Crude FLOPs estimate: 6 * N_params * tokens_per_round * world_size.

        6 = 2 (forward) + 4 (backward) per param per token. Coarse but tracks
        the right order of magnitude — enough for AI-Act-threshold monitoring.
        """
        n_params = float(self.model.num_params(non_embedding=False))
        toks_per_step = self.train_cfg.seq_len * self.train_cfg.micro_batch_size
        toks_per_round = toks_per_step * self.train_cfg.inner_steps
        return 6.0 * n_params * toks_per_round * float(self.world_size)

    # -- API used by FastAPI handlers ----------------------------------------

    def register(self, req: RegisterRequest) -> RegisterResponse:
        with self.lock:
            if req.preset != self.preset_name:
                raise HTTPException(
                    status_code=400,
                    detail=f"preset mismatch: coord={self.preset_name}, worker={req.preset}",
                )
            # parse pubkey eagerly so bad keys fail at registration, not at /delta
            parsed_pubkey = None
            if req.pubkey:
                try:
                    parsed_pubkey = pubkey_from_hex(req.pubkey)
                except (ValueError, TypeError) as e:
                    if self.require_signed_deltas:
                        raise HTTPException(400, f"invalid pubkey: {e}") from None
            elif self.require_signed_deltas:
                raise HTTPException(400, "pubkey required when signed deltas enforced")

            wid = self.next_worker_id
            self.next_worker_id += 1
            self.workers[wid] = {
                "pubkey_hex": req.pubkey,
                "pubkey": parsed_pubkey,
                "country": req.country,
                "gpu": req.gpu,
                "vram_gb": req.vram_gb,
                "ram_gb": req.ram_gb,
                "registered_at": time.time(),
            }
            log.info(
                "register worker=%d country=%s gpu=%s preset=%s",
                wid,
                req.country,
                req.gpu,
                req.preset,
            )
            return RegisterResponse(
                worker_id=wid,
                current_round=self.round,
                world_size=self.world_size,
                seed=self.train_cfg.seed,
                inner_steps=self.train_cfg.inner_steps,
                seq_len=self.train_cfg.seq_len,
                micro_batch_size=self.train_cfg.micro_batch_size,
                state_codec=self.state_codec,
                delta_codec=self.delta_codec,
                require_signed_deltas=self.require_signed_deltas,
            )

    def status(self) -> RoundStatus:
        with self.lock:
            n_sub = len(self.deltas.get(self.round, {}))
            return RoundStatus(
                current_round=self.round,
                n_registered=len(self.workers),
                n_submitted=n_sub,
                waiting_for=max(0, self.world_size - n_sub),
                last_val_loss=self.last_val_loss,
                flops_total=self.flops_total,
                round_open_seconds=time.time() - self.round_started_at,
                round_timeout_seconds=self.round_timeout_seconds,
                min_workers=self.min_workers,
            )

    def state_blob(self) -> tuple[bytes, int]:
        with self.lock:
            return self._serialize_state(), self.round

    def submit_delta(
        self,
        worker_id: int,
        claimed_round: int,
        blob: bytes,
        val_loss: float | None = None,
        signature_b64: str | None = None,
    ) -> DeltaAck:
        with self.lock:
            if worker_id not in self.workers:
                raise HTTPException(404, f"unknown worker_id {worker_id}")
            if claimed_round != self.round:
                return DeltaAck(
                    accepted=False,
                    reason=f"stale: coord round={self.round}, worker={claimed_round}",
                    next_round=self.round,
                )
            # signature check (binds worker_id + round + sha256(body) to the pubkey
            # registered at /register time, defeating spoof + replay + tamper)
            pubkey = self.workers[worker_id].get("pubkey")
            if self.require_signed_deltas or signature_b64 is not None:
                if signature_b64 is None:
                    return DeltaAck(accepted=False, reason="missing signature")
                if pubkey is None:
                    return DeltaAck(accepted=False, reason="worker registered without pubkey")
                if not verify_delta(pubkey, worker_id, claimed_round, blob, signature_b64):
                    log.warning(
                        "REJECTED bad signature from worker_id=%d round=%d", worker_id, claimed_round
                    )
                    return DeltaAck(accepted=False, reason="signature verification failed")
            delta = deserialize_delta(blob, codec=self.delta_codec)
            self.deltas[self.round][worker_id] = delta
            if val_loss is not None:
                self.val_losses[self.round].append(val_loss)
            log.info(
                "delta round=%d worker=%d (%d/%d)%s",
                self.round,
                worker_id,
                len(self.deltas[self.round]),
                self.world_size,
                f" val_loss={val_loss:.4f}" if val_loss is not None else "",
            )
            ready = len(self.deltas[self.round]) >= self.world_size
            if ready:
                self._outer_step_locked()
                return DeltaAck(accepted=True, next_round=self.round)
            return DeltaAck(accepted=True)

    # -- outer optimizer step (caller holds lock) -----------------------------

    def _outer_step_locked(self) -> None:
        round_deltas = list(self.deltas[self.round].values())
        avg = average_deltas(round_deltas)

        self.outer_opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                if n not in avg:
                    raise KeyError(f"missing delta for parameter {n!r}")
                g = avg[n].to(p.device, dtype=p.dtype)
                p.grad = g
        self.outer_opt.step()

        # bookkeeping
        if self.val_losses[self.round]:
            self.last_val_loss = sum(self.val_losses[self.round]) / len(
                self.val_losses[self.round]
            )
        self.flops_total += self._estimate_round_flops()

        prev_round = self.round
        self.round += 1
        self.deltas[self.round] = {}
        self.val_losses[self.round] = []
        del self.deltas[prev_round]
        del self.val_losses[prev_round]
        self.round_started_at = time.time()
        self._invalidate_state_cache()

        # record for the dashboard's loss-curve chart (persisted to disk)
        self._append_history(
            {
                "round": prev_round,
                "val_loss": self.last_val_loss,
                "flops_total": self.flops_total,
                "ts": time.time(),
            }
        )

        log.info(
            "outer step %d -> %d (avg delta norm: %.4f, FLOPs ~%.2e, last_val_loss=%s)",
            prev_round,
            self.round,
            _flat_norm(avg),
            self.flops_total,
            f"{self.last_val_loss:.4f}" if self.last_val_loss is not None else "n/a",
        )

        if (
            self.checkpoint_dir is not None
            and self.checkpoint_every > 0
            and self.round % self.checkpoint_every == 0
        ):
            save_checkpoint(
                self.checkpoint_dir,
                round_no=self.round - 1,
                model=self.model,
                outer_opt=self.outer_opt,
                meta={
                    "preset_name": self.preset_name,
                    "world_size": self.world_size,
                    "flops_total": self.flops_total,
                    "last_val_loss": self.last_val_loss,
                },
            )


def _flat_norm(state: dict[str, torch.Tensor]) -> float:
    total = 0.0
    for t in state.values():
        total += float(t.float().pow(2).sum().item())
    return total**0.5


# -- FastAPI wiring -----------------------------------------------------------


def create_app(state: CoordinatorState) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        log.info(
            "coord up: preset=%s world_size=%d params=%d state_codec=%s delta_codec=%s",
            state.preset_name,
            state.world_size,
            state.model.num_params(),
            state.state_codec,
            state.delta_codec,
        )
        yield

    app = FastAPI(title="dllm-coordinator", version="0.0.2", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> HTMLResponse:
        return HTMLResponse(_load_dashboard_html())

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "round": state.round}

    @app.get("/history")
    def get_history() -> dict:
        with state.lock:
            return {"history": list(state.history)}

    @app.post("/register", response_model=RegisterResponse)
    def register(req: RegisterRequest) -> RegisterResponse:
        return state.register(req)

    @app.get("/status", response_model=RoundStatus)
    def status() -> RoundStatus:
        return state.status()

    @app.get("/state")
    def get_state():
        blob, round_no = state.state_blob()
        return Response(
            content=blob,
            media_type="application/octet-stream",
            headers={"x-round": str(round_no), "x-codec": state.state_codec},
        )

    @app.post("/delta")
    async def post_delta(
        request: Request,
        worker_id: int,
        round: int,
        val_loss: float | None = None,
    ):
        body = await request.body()
        sig = request.headers.get("x-delta-signature")
        ack = state.submit_delta(
            worker_id, round, body, val_loss=val_loss, signature_b64=sig
        )
        return JSONResponse(ack.model_dump())

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="smoke", choices=list(PRESETS))
    ap.add_argument("--world-size", type=int, default=2)
    ap.add_argument("--inner-steps", type=int, default=50)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--micro-batch-size", type=int, default=16)
    ap.add_argument("--inner-lr", type=float, default=3e-4)
    ap.add_argument("--outer-lr", type=float, default=0.7)
    ap.add_argument("--outer-momentum", type=float, default=0.9)
    ap.add_argument("--max-rounds", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--state-codec", default="bf16", choices=["fp32", "bf16"])
    ap.add_argument("--delta-codec", default="q8", choices=["fp32", "q8"])
    ap.add_argument("--checkpoint-dir", default="coord/state")
    ap.add_argument("--checkpoint-every", type=int, default=10)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument(
        "--require-signed-deltas",
        action="store_true",
        help="Reject /delta submissions without a valid Ed25519 signature (recommended)",
    )
    ap.add_argument(
        "--round-timeout-seconds",
        type=float,
        default=900.0,
        help="Max wall-clock for a round; if exceeded with >= min-workers deltas, force-advance.",
    )
    ap.add_argument(
        "--min-workers",
        type=int,
        default=1,
        help="Minimum deltas needed for a timed-out round to advance (clamped to [1, world_size]).",
    )
    ap.add_argument("--log-level", default="info")
    args = ap.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    train_cfg = TrainConfig(
        seq_len=args.seq_len,
        micro_batch_size=args.micro_batch_size,
        inner_steps=args.inner_steps,
        inner_lr=args.inner_lr,
        outer_lr=args.outer_lr,
        outer_momentum=args.outer_momentum,
        max_outer_rounds=args.max_rounds,
        seed=args.seed,
    )

    ckpt_dir = Path(args.checkpoint_dir).resolve() if args.checkpoint_dir else None

    state = CoordinatorState(
        preset_name=args.preset,
        world_size=args.world_size,
        train_cfg=train_cfg,
        device=args.device,
        state_codec=args.state_codec,
        delta_codec=args.delta_codec,
        checkpoint_dir=ckpt_dir,
        checkpoint_every=args.checkpoint_every,
        resume=not args.no_resume,
        require_signed_deltas=args.require_signed_deltas,
        round_timeout_seconds=args.round_timeout_seconds,
        min_workers=args.min_workers,
    )
    app = create_app(state)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
