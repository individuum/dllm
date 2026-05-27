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
from collections.abc import Callable
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
from .power import PowerMeter

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


def compute_inner_steps_for_target(
    tok_per_s: float,
    target_seconds: float,
    batch: int,
    seq: int,
    min_steps: int = 50,
    max_steps: int = 2000,
) -> int:
    """Choose `inner_steps` so one inner loop runs for ~`target_seconds` wall clock.

    Heterogeneous-fleet sizing per PLAN §3: each worker hits the same per-round
    wall clock regardless of GPU speed, so no one idles waiting for the slow
    node. (Anchor pods on H100s and individuals on consumer GPUs land on the
    same outer-step cadence.)

    Bounds reflect DiLoCo paper guidance: <50 steps gives too little local work
    to amortize the sync cost; >2000 risks worker drift exceeding what the
    outer optimizer can pull back.
    """
    if tok_per_s <= 0:
        return min_steps
    target_tokens = tok_per_s * target_seconds
    steps = int(target_tokens / max(1, batch * seq))
    return max(min_steps, min(max_steps, steps))


def _sync_device(device: torch.device) -> None:
    """Wait for in-flight kernels on the given device before timing."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


# Transient network/transport failures that should be retried. Truncated bodies
# (RemoteProtocolError) are common when pulling the ~200 MB /state blob over a
# long-haul link — nginx, podman or the TLS termination can drop mid-stream
# and the worker used to crash with no recovery.
_RETRY_HTTPX_EXC: tuple[type[BaseException], ...] = (
    httpx.RemoteProtocolError,
    httpx.ReadError,
    httpx.ReadTimeout,
    httpx.WriteError,
    httpx.WriteTimeout,
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
)


def parallel_state_get(
    client: httpx.Client,
    url: str,
    *,
    n_chunks: int = 4,
    log_label: str = "GET /state",
) -> httpx.Response:
    """Fetch a /state-like URL via N parallel HTTP Range requests.

    Why: ISPs like Vodafone Kabel DE throttle individual TCP flows
    aggressively (~8 Mbit/s observed) while allowing the user's full
    aggregate bandwidth (~600 Mbit/s) across many parallel streams.
    Workers on those networks were taking >10 minutes to pull a 624 MB
    state per round — half the wall clock was sync.

    Algorithm:
      1. HEAD the URL to learn content-length + check Range support
      2. If server says Accept-Ranges: bytes → split into N chunks
      3. Fetch each chunk in a thread (each on its own TCP connection)
      4. Concatenate, synthesize a final Response object with the
         full content + first chunk's headers (x-round, x-codec).
      5. If server refuses Range OR HEAD fails → fall back to single
         GET. Lets older coords and quirky middleboxes still work.

    Returns an httpx.Response-shaped object. Caller uses .content,
    .headers, .raise_for_status as usual.
    """
    # 1. Probe.
    try:
        head = client.head(url)
    except _RETRY_HTTPX_EXC as e:
        log.warning("[%s] HEAD probe failed (%s); falling back to single GET", log_label, e)
        return client.get(url)
    if head.status_code != 200 or head.headers.get("accept-ranges", "").lower() != "bytes":
        log.info(
            "[%s] no Range support (status=%d, accept-ranges=%r); single GET",
            log_label,
            head.status_code,
            head.headers.get("accept-ranges"),
        )
        return client.get(url)
    try:
        total = int(head.headers["content-length"])
    except (KeyError, ValueError):
        log.info("[%s] no content-length on HEAD; single GET", log_label)
        return client.get(url)

    # 2. Plan chunks.
    n = max(1, n_chunks)
    if total < 4 * 1024 * 1024:  # <4 MiB: not worth parallelizing
        return client.get(url)
    chunk_size = (total + n - 1) // n
    ranges = []
    for i in range(n):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total) - 1
        if start <= end:
            ranges.append((i, start, end))

    log.info(
        "[%s] parallel-Range fetch: %d chunks × %s bytes each, total %s",
        log_label,
        len(ranges),
        f"{chunk_size:,}",
        f"{total:,}",
    )

    # 3. Fetch in parallel.
    import time as _time
    t0 = _time.time()
    chunks: list[bytes | None] = [None] * len(ranges)
    common_headers: dict[str, str] = {}
    sample_request: httpx.Request | None = None

    def _fetch(idx: int, start: int, end: int):
        r = client.get(url, headers={"Range": f"bytes={start}-{end}"})
        r.raise_for_status()
        if r.status_code not in (206, 200):
            raise httpx.HTTPError(f"unexpected status {r.status_code} on Range fetch")
        return idx, r.content, dict(r.headers), r.request

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=n, thread_name_prefix="dllm-state-pull") as pool:
        futures = [pool.submit(_fetch, i, s, e) for i, s, e in ranges]
        for fut in as_completed(futures):
            idx, body, hdrs, req = fut.result()
            chunks[idx] = body
            if not common_headers:
                common_headers = hdrs  # use any chunk's headers as canonical
            if sample_request is None:
                sample_request = req
    dt = _time.time() - t0
    full_body = b"".join(c for c in chunks if c is not None)
    if len(full_body) != total:
        raise httpx.HTTPError(
            f"parallel fetch length mismatch: got {len(full_body)}, expected {total}"
        )
    speed_mb_per_s = total / dt / 1_000_000 if dt > 0 else 0
    log.info(
        "[%s] %d-stream parallel pull: %.1f MB in %.1fs (%.1f MB/s)",
        log_label,
        len(ranges),
        total / 1_000_000,
        dt,
        speed_mb_per_s,
    )

    # 4. Synthesize a Response-like object so callers don't care about
    # parallel vs single. httpx.Response constructor accepts content +
    # headers; we strip Range-specific ones the caller doesn't expect.
    # `request=` is required for the caller's r.raise_for_status() to
    # work — httpx refuses to format errors without a request context.
    common_headers.pop("content-range", None)
    common_headers["content-length"] = str(len(full_body))
    return httpx.Response(
        status_code=200,
        headers=common_headers,
        content=full_body,
        request=sample_request,
    )


def retry_http(
    fn: Callable[[], httpx.Response],
    *,
    label: str,
    max_attempts: int = 4,
    base_delay: float = 1.0,
) -> httpx.Response:
    """Call ``fn`` with retry + exponential backoff on transient errors and 5xx.

    Retries: connection drops, truncated bodies, timeouts, and HTTP 5xx.
    Passes through (no retry): 4xx responses — the coord is telling us
    something the client must handle (e.g. 404 = deregistered, 409 = stale).
    """
    last_exc: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            r = fn()
        except _RETRY_HTTPX_EXC as e:
            last_exc = e
            if attempt == max_attempts:
                log.error(
                    "[HTTP] %s failed after %d attempts: %s", label, attempt, e
                )
                raise
            delay = base_delay * (2 ** (attempt - 1))
            log.warning(
                "[HTTP] %s transient error (attempt %d/%d): %s; retrying in %.1fs",
                label, attempt, max_attempts, e, delay,
            )
            time.sleep(delay)
            continue
        if 500 <= r.status_code < 600 and attempt < max_attempts:
            delay = base_delay * (2 ** (attempt - 1))
            log.warning(
                "[HTTP] %s got %d (attempt %d/%d); retrying in %.1fs",
                label, r.status_code, attempt, max_attempts, delay,
            )
            time.sleep(delay)
            continue
        return r
    raise RuntimeError(f"{label} retry loop exhausted") from last_exc


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
        auto_tune_steps: bool = False,
        target_round_seconds: float = 90.0,
        estimated_watts: float | None = None,
        micro_batch_size_override: int | None = None,
    ) -> None:
        self.coord_url = coord_url.rstrip("/")
        self.preset = preset
        # VRAM-constrained GPUs (Colab T4, smaller cards) can override the
        # coord's micro_batch_size downward. Used by register() to clamp
        # the value coming back from /register. None = use coord's value.
        self.micro_batch_size_override = micro_batch_size_override
        self.country = country
        self.device = device
        self.train_data = train_data
        self.val_data = val_data
        # bf16 autocast on CUDA (Ampere+) and MPS (PyTorch 2.5+ on Apple Silicon).
        # On older MPS where bf16 is rejected, the run_inner autocast will surface
        # an error and the user can rerun with --no-bf16.
        self.bf16 = bf16 and device.type in ("cuda", "mps")
        self.val_batches = val_batches
        self.auto_tune_steps = auto_tune_steps
        self.target_round_seconds = target_round_seconds
        self.power_meter = PowerMeter(device, override_watts=estimated_watts)

        # Ed25519 identity. Default: <repo_root>/.dllm/identity.key (legacy,
        # for `dllm-worker` runs from the repo). Desktop client sets
        # DLLM_IDENTITY_KEY to a per-user path so the same volunteer's
        # identity persists across upgrades / cwd changes / reinstalls.
        identity_override = os.environ.get("DLLM_IDENTITY_KEY")
        self.sk = load_or_create_identity(identity_override)
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
        # Dynamic sharding (Choice B). On register these default to
        # (worker_id, world_size) so legacy single-worker behaviour is
        # unchanged. When the coord re-balances the cohort it sends new
        # values in DeltaAck; we rebuild the train_loader between rounds.
        self.shard_index: int = 0
        self.shard_world_size: int = 1

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

        # Per-inner-loop telemetry, set inside run_inner() and shipped to the
        # coord on the next /delta. Used by the dashboard's energy + throughput
        # series.
        self.last_inner_power_watts: float | None = None
        self.last_inner_tok_per_s: float | None = None

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
        # HTTP 429 = the coord's cohort cap is full. Surface a clear,
        # actionable message instead of the generic HTTPStatusError that
        # raise_for_status would print. Desktop client greps for "[CAP]"
        # to show a friendly retry-later banner; CLI users see it on stderr.
        if r.status_code == 429:
            try:
                reason = r.json().get("detail", "cohort full")
            except Exception:  # noqa: BLE001
                reason = "cohort full"
            log.error(
                "[CAP] %s. Try again later — workers are auto-evicted after "
                "inactivity, freeing slots. Stopping cleanly.",
                reason,
            )
            raise SystemExit(2)
        r.raise_for_status()
        data = r.json()
        self.worker_id = data["worker_id"]
        self.world_size = data["world_size"]
        self.current_round = data["current_round"]
        self.inner_steps = data["inner_steps"]
        self.seq_len = data["seq_len"]
        coord_batch = int(data["micro_batch_size"])
        if self.micro_batch_size_override is not None and self.micro_batch_size_override < coord_batch:
            self.micro_batch_size = int(self.micro_batch_size_override)
            log.warning(
                "[BATCH-OVERRIDE] coord said micro_batch=%d; "
                "this worker uses %d (VRAM-constrained). FLOPs accounting "
                "may slightly under-count this worker's contribution.",
                coord_batch,
                self.micro_batch_size,
            )
        else:
            self.micro_batch_size = coord_batch
        self.seed = data["seed"]
        self.state_codec = data.get("state_codec", "bf16")
        self.delta_codec = data.get("delta_codec", "q8")
        # Dynamic sharding: default to (worker_id, world_size) so the first
        # round reads a sensible slice; the coord can re-balance via
        # DeltaAck when the cohort changes.
        self.shard_index = self.worker_id
        self.shard_world_size = self.world_size
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
            # ShardLoader partitions by (shard_index, shard_world_size).
            # These default to (worker_id, world_size) on register but are
            # mutated when the coord re-balances the cohort — _build_train_loader
            # is the one place that needs to read them, so we can rebuild
            # cleanly when they change. `getattr` fallbacks keep older test
            # paths working (some tests instantiate Worker via `object.__new__`
            # and skip `__init__`).
            shard_idx = getattr(self, "shard_index", self.worker_id)
            shard_ws = getattr(self, "shard_world_size", self.world_size)
            self.train_loader = ShardLoader(
                self.train_data,
                seq_len=self.seq_len,
                batch_size=self.micro_batch_size,
                worker_id=shard_idx,
                world_size=shard_ws,
                device=self.device,
                seed=self.seed,
            )
        if self.val_loader is None and self.val_data is not None and self.val_data.exists():
            try:
                # val is NOT partitioned by worker_id — every worker validates
                # against the full val.bin. Was a bug: with world_size > 1 each
                # worker got a different EU-language slice from val.bin so
                # per-worker val_losses lived on different distributions
                # (Polish perplexity ≠ German perplexity even at consensus θ).
                # The averaged last_val_loss then jumped when a new worker
                # joined even if the consensus model was unchanged.
                # See CLAUDE.md "val_loss skew" section.
                self.val_loader = ShardLoader(
                    self.val_data,
                    seq_len=self.seq_len,
                    batch_size=self.micro_batch_size,
                    worker_id=0,
                    world_size=1,
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
        # Pass the round we believe the coord is at (from /register response)
        # as a query param so Cloudflare can cache per-round state under a
        # distinct URL — `/state?round=89` and `/state?round=90` get their
        # own cache entries, immutable for the round's lifetime. Coord
        # returns 409 if we're out of sync; we recover by retrying with
        # the coord-reported current_round.
        url = f"/state?round={self.current_round}" if self.current_round else "/state"
        r = retry_http(
            lambda: parallel_state_get(self.http, url, log_label="GET /state (initial)"),
            label="GET /state (initial)",
        )
        # 409 = round mismatch; coord tells us its current round, retry once.
        if r.status_code == 409:
            try:
                their_round = int(r.json().get("current_round", -1))
            except Exception:  # noqa: BLE001
                their_round = -1
            if their_round >= 0:
                log.info(
                    "state-pull round mismatch (we said %d, coord at %d); retrying",
                    self.current_round,
                    their_round,
                )
                self.current_round = their_round
                retry_url = f"/state?round={their_round}"
                r = retry_http(
                    lambda: parallel_state_get(
                        self.http, retry_url, log_label="GET /state (retry 409)"
                    ),
                    label="GET /state (retry after 409)",
                )
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

        # Power samples. Read every ~max(1, inner_steps // 32) steps so we get
        # at least one reading on tiny inner loops and don't burn CPU on big ones.
        power_samples: list[float] = []
        sample_every = max(1, self.inner_steps // 32)

        loss_sum = 0.0
        t0 = time.time()
        for step_i in range(self.inner_steps):
            x, y = self.train_loader.next_batch()
            with autocast_ctx:
                _, loss = self.model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
            loss_sum += float(loss.detach().item())
            if step_i % sample_every == 0:
                w = self.power_meter.sample()
                if w is not None:
                    power_samples.append(w)

        avg_loss = loss_sum / max(1, self.inner_steps)
        dt = time.time() - t0
        tok_per_s = self.inner_steps * self.micro_batch_size * self.seq_len / dt
        avg_power = (sum(power_samples) / len(power_samples)) if power_samples else None

        # Side-channel: read by run() right before /delta is submitted.
        self.last_inner_tok_per_s = tok_per_s
        self.last_inner_power_watts = avg_power

        power_msg = f" power={avg_power:.0f}W" if avg_power is not None else ""

        if self.device.type == "cuda":
            peak_mib = torch.cuda.max_memory_allocated(self.device) / (1024**2)
            log.info(
                "inner round=%d avg_loss=%.4f steps=%d %.0f tok/s peak_vram=%.1f MiB%s",
                self.current_round,
                avg_loss,
                self.inner_steps,
                tok_per_s,
                peak_mib,
                power_msg,
            )
            torch.cuda.reset_peak_memory_stats(self.device)
        elif self.device.type == "mps":
            # MPS reports current rather than peak; close enough at end-of-loop
            mem_mib = torch.mps.current_allocated_memory() / (1024**2)
            log.info(
                "inner round=%d avg_loss=%.4f steps=%d %.0f tok/s mem=%.1f MiB (MPS)%s",
                self.current_round,
                avg_loss,
                self.inner_steps,
                tok_per_s,
                mem_mib,
                power_msg,
            )
        else:
            log.info(
                "inner round=%d avg_loss=%.4f steps=%d %.0f tok/s (CPU)%s",
                self.current_round,
                avg_loss,
                self.inner_steps,
                tok_per_s,
                power_msg,
            )

        return compute_delta(snap, self.model)

    def _benchmark_throughput(
        self, warmup_steps: int = 5, measure_steps: int = 30
    ) -> float:
        """Warmup + timed inner-loop probe. Returns measured tokens/sec on this device.

        Runs real forward+backward+step on the actual model/data; the optimizer
        state is left warmed up (not reset), so this counts as a small head-start
        on the first round rather than wasted compute.
        """
        assert self.opt is not None and self.train_loader is not None
        log.info(
            "[AUTO-TUNE] benchmarking throughput: %d warmup + %d measured steps...",
            warmup_steps,
            measure_steps,
        )
        self.model.train()
        autocast_ctx = (
            torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
            if self.bf16
            else _nullcontext()
        )

        for _ in range(warmup_steps):
            x, y = self.train_loader.next_batch()
            with autocast_ctx:
                _, loss = self.model(x, y)
            loss.backward()
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
        _sync_device(self.device)

        t0 = time.time()
        for _ in range(measure_steps):
            x, y = self.train_loader.next_batch()
            with autocast_ctx:
                _, loss = self.model(x, y)
            loss.backward()
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
        _sync_device(self.device)
        dt = time.time() - t0

        tok_per_s = measure_steps * self.micro_batch_size * self.seq_len / dt
        log.warning(
            "[AUTO-TUNE] measured %.0f tok/s (%d steps in %.1fs)",
            tok_per_s,
            measure_steps,
            dt,
        )
        return tok_per_s

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
        self,
        blob: bytes,
        val_loss: float | None,
        round_no: int,
        power_watts: float | None = None,
        tokens_per_sec: float | None = None,
    ) -> dict:
        """Background: sign, POST /delta, wait for round advance, GET /state."""
        sig = sign_delta(self.sk, self.worker_id, round_no, blob)
        params: dict = {"worker_id": self.worker_id, "round": round_no}
        if val_loss is not None:
            params["val_loss"] = val_loss
        if power_watts is not None:
            params["power_watts"] = power_watts
        if tokens_per_sec is not None:
            params["tokens_per_sec"] = tokens_per_sec
        r = retry_http(
            lambda: self.http.post(
                "/delta",
                params=params,
                content=blob,
                headers={
                    "content-type": "application/octet-stream",
                    "x-delta-signature": sig,
                },
            ),
            label="POST /delta",
        )
        if r.status_code == 404:
            # Coord no longer recognizes this worker — it timed us out for
            # inactivity. Signal so caller can re-register and resume.
            log.error(
                "POST /delta -> 404; coord deregistered worker_id=%s.",
                self.worker_id,
            )
            return {"ack": None, "state": None, "round": round_no, "state_bytes": 0, "dropped": True}
        r.raise_for_status()
        ack = r.json()
        # Tier-aware: coord may have re-tuned this worker's inner_steps based
        # on the throughput we just reported. Stash it on the result so
        # _apply_sync_result can hand it to the main loop (which applies it
        # AT THE START of the next round — never mid-inner-loop).
        new_inner = ack.get("inner_steps")
        # Dynamic sharding: coord may have re-assigned this worker's shard
        # partition (e.g. another worker joined or got evicted). Same
        # contract as inner_steps — applied between rounds, never mid-loop.
        new_shard_index = ack.get("shard_index")
        new_shard_world_size = ack.get("shard_world_size")
        if not ack.get("accepted"):
            reason = ack.get("reason", "")
            # Signature mismatch = our worker_id is now bound to someone else's
            # pubkey on the coord (typical after coord restart + another worker
            # grabbed the freed-up id before we did). Treat as "we're stale —
            # re-register". Same contract as the 404 path; the dropped flag
            # tells run() to fire _reregister_and_resync.
            if "signature" in reason.lower():
                log.error(
                    "POST /delta -> accepted=False, reason=%r — worker_id=%s "
                    "is no longer ours on the coord (likely restart race). "
                    "Re-registering.",
                    reason,
                    self.worker_id,
                )
                return {
                    "ack": ack,
                    "state": None,
                    "round": round_no,
                    "state_bytes": 0,
                    "dropped": True,
                }
            # Stale-round rejection: in heterogeneous fleets, fast workers can
            # advance the coord past where we submitted. Don't bail — fetch the
            # current consensus state and re-align. The caller (_apply_sync_result)
            # will reload the local model from this state, dropping the delta
            # we just wasted but keeping the worker contributing.
            next_r = ack.get("next_round")
            if next_r is not None and next_r > round_no:
                # Per-round URL = Cloudflare-cacheable. The coord just told
                # us its current round via `next_round`; we ask for that
                # specific one, immutable forever once written. Parallel
                # range fetch defeats per-flow ISP throttling.
                resync_url = f"/state?round={next_r}"
                r2 = retry_http(
                    lambda: parallel_state_get(
                        self.http, resync_url, log_label="GET /state (resync)"
                    ),
                    label="GET /state (resync)",
                )
                r2.raise_for_status()
                codec = r2.headers.get("x-codec", self.state_codec)
                new_state = deserialize_state(r2.content, codec=codec)  # type: ignore[arg-type]
                return {
                    "ack": ack,
                    "state": new_state,
                    "round": int(r2.headers["x-round"]),
                    "state_bytes": len(r2.content),
                    "resync": True,
                    "inner_steps": new_inner,
                    "shard_index": new_shard_index,
                    "shard_world_size": new_shard_world_size,
                }
            return {
                "ack": ack,
                "state": None,
                "round": round_no,
                "state_bytes": 0,
                "inner_steps": new_inner,
                "shard_index": new_shard_index,
                "shard_world_size": new_shard_world_size,
            }

        target = ack.get("next_round")
        if target is None:
            # other workers haven't submitted yet — long-poll status
            while True:
                rs = retry_http(
                    lambda: self.http.get("/status"), label="GET /status (poll)"
                )
                rs.raise_for_status()
                cur = rs.json()["current_round"]
                if cur > round_no:
                    target = cur
                    break
                time.sleep(0.5)

        # By here `target` is the new round we want state for. Per-round URL
        # so Cloudflare caches each round's state distinctly + forever.
        # Parallel-range fetch defeats single-flow ISP throttles.
        post_url = f"/state?round={target}"
        r2 = retry_http(
            lambda: parallel_state_get(
                self.http, post_url, log_label="GET /state (post-accept)"
            ),
            label="GET /state (post-accept)",
        )
        r2.raise_for_status()
        codec = r2.headers.get("x-codec", self.state_codec)
        new_state = deserialize_state(r2.content, codec=codec)  # type: ignore[arg-type]
        return {
            "ack": ack,
            "state": new_state,
            "round": int(r2.headers["x-round"]),
            "state_bytes": len(r2.content),
            "inner_steps": new_inner,
            "shard_index": new_shard_index,
            "shard_world_size": new_shard_world_size,
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
        # Tier-aware: coord-assigned inner_steps takes effect from the next
        # inner loop onward. Safe to mutate self.inner_steps here because we
        # only call _apply_sync_result between inner loops (run() drains the
        # previous round's sync BEFORE starting the next run_inner()).
        new_inner = result.get("inner_steps")
        if new_inner is not None and new_inner != self.inner_steps:
            log.warning(
                "[TIER-AWARE] coord re-tuned inner_steps %d -> %d (applying next round)",
                self.inner_steps,
                new_inner,
            )
            self.inner_steps = int(new_inner)
        # Dynamic sharding: same between-rounds contract as inner_steps. If
        # the coord reassigned our shard (cohort grew/shrank), throw away
        # the current train_loader and rebuild on the new partition. The
        # next run_inner() will hit `if self.train_loader is None:` in
        # _ensure_loader_and_opt and rebuild with self.shard_index /
        # self.shard_world_size.
        new_si = result.get("shard_index")
        new_sws = result.get("shard_world_size")
        if (new_si is not None and new_sws is not None
                and (new_si != self.shard_index or new_sws != self.shard_world_size)):
            log.warning(
                "[RESHARD] coord reassigned shard (%d/%d -> %d/%d); "
                "rebuilding train_loader",
                self.shard_index,
                self.shard_world_size,
                new_si,
                new_sws,
            )
            self.shard_index = int(new_si)
            self.shard_world_size = int(new_sws)
            self.train_loader = None  # _ensure_loader_and_opt rebuilds on next call
            self._ensure_loader_and_opt()
        log.info(
            "async sync applied round=%d (%d state bytes)%s",
            self.current_round,
            result.get("state_bytes", 0),
            " [resync]" if result.get("resync") else "",
        )
        return True

    # -- re-registration after coord-side eviction ----------------------------

    def _reregister_and_resync(self) -> None:
        """Recover from POST /delta -> 404 by re-registering and pulling the
        current consensus state. The coord has already dropped our old
        worker_id (M5-class case from CLAUDE.md "Open: deregistered before
        first delta"). Without this, the worker bails on a single eviction
        even though the cohort is healthy and the local model is still useful.

        Optimizer state is preserved (matches the resync path's contract; the
        momentum buffer is still useful gradient signal). Train loader is
        rebuilt because the new worker_id changes which shard slice we read.
        """
        old_wid = self.worker_id
        log.warning(
            "[REREGISTER] coord evicted worker_id=%s; re-registering and resyncing...",
            old_wid,
        )
        # Re-register: new worker_id, fresh inner_steps from coord. world_size
        # and seed should not change (coord-side immutable across the run).
        self.register()
        # Pull current consensus state — same path as initial startup.
        self.pull_state()  # also refreshes self.last_ref
        # Rebuild the train_loader with the new worker_id so the shard
        # partition (worker_id % world_size) is correct. val_loader is
        # already hard-coded to (worker_id=0, world_size=1) so it doesn't
        # need rebuilding.
        self.train_loader = None
        self._ensure_loader_and_opt()
        log.warning(
            "[REREGISTER] resumed as worker_id=%d at round=%d (was worker_id=%s)",
            self.worker_id,
            self.current_round,
            old_wid,
        )

    # -- top-level loop -------------------------------------------------------

    def run(self, max_rounds: int) -> None:
        """Lazy-async DiLoCo: GPU runs continuously; network exchanges overlap."""
        self.register()
        self.pull_state()  # captures last_ref
        self._ensure_loader_and_opt()

        # Auto-tune: pick inner_steps so this worker's inner loop ≈ target wall
        # clock, regardless of GPU speed. Lets a slow M5 and a fast 3060 finish
        # in the same window and avoids the fast one idling on the slow one.
        if self.auto_tune_steps:
            tok_per_s = self._benchmark_throughput()
            new_steps = compute_inner_steps_for_target(
                tok_per_s,
                self.target_round_seconds,
                self.micro_batch_size,
                self.seq_len,
            )
            log.warning(
                "[AUTO-TUNE] target=%.0fs/round -> inner_steps=%d (coord said %d)",
                self.target_round_seconds,
                new_steps,
                self.inner_steps,
            )
            self.inner_steps = new_steps

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
                    # 404 from /delta = coord evicted us. Re-register and
                    # resync instead of bailing — this is the M5-tier "first
                    # delta arrives after dereg" recovery path from CLAUDE.md.
                    # The delta we just computed is wasted (the local θ was
                    # built on a state the coord no longer recognizes), but
                    # the worker stays alive and contributes from the next
                    # round onward.
                    if result.get("dropped"):
                        self._reregister_and_resync()
                        # Skip submitting this round's stale delta — start
                        # fresh from the next inner loop.
                        continue
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
                    self._async_sync_io,
                    blob,
                    val_loss,
                    self.current_round,
                    self.last_inner_power_watts,
                    self.last_inner_tok_per_s,
                )

            # drain any in-flight sync
            if pending is not None:
                log.info("draining final sync...")
                if self._apply_sync_result(pending.result()):
                    rounds_committed += 1
        finally:
            self._exec.shutdown(wait=False)
            self.power_meter.close()


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
        "--auto-tune-steps",
        action="store_true",
        help="Benchmark GPU at startup, pick inner_steps to hit --target-round-seconds",
    )
    ap.add_argument(
        "--target-round-seconds",
        type=float,
        default=90.0,
        help="Target wall-clock per inner loop when --auto-tune-steps is on",
    )
    ap.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail fast if no CUDA/MPS available instead of silently falling back to CPU",
    )
    ap.add_argument(
        "--estimated-watts",
        type=float,
        default=None,
        help=(
            "Override the auto-detected power reading with this constant (W). "
            "Use when NVML isn't available or to model a known device exactly. "
            "Default: NVML on CUDA if installed; VRAM-tier / TDP estimate otherwise."
        ),
    )
    ap.add_argument(
        "--micro-batch-size",
        type=int,
        default=None,
        help=(
            "Override the coord's micro_batch_size for THIS worker (VRAM-"
            "constrained GPUs like Colab T4 may need 1 even if the cohort "
            "default is 2). Override is one-way: can only DECREASE the "
            "value. FLOPs accounting will slightly under-count this worker."
        ),
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
        auto_tune_steps=args.auto_tune_steps,
        target_round_seconds=args.target_round_seconds,
        estimated_watts=args.estimated_watts,
        micro_batch_size_override=args.micro_batch_size,
    )
    w.run(max_rounds=args.max_rounds)


if __name__ == "__main__":
    main()
