FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch \
 && pip install --no-cache-dir .

# coord listens here inside the container; nginx on host proxies to it
EXPOSE 8000

# checkpoints land on a mounted volume
ENV DLLM_CHECKPOINT_DIR=/data/checkpoints

ENTRYPOINT ["dllm-coord"]
CMD [ \
    "--host", "0.0.0.0", \
    "--port", "8000", \
    "--preset", "300M", \
    "--world-size", "1", \
    "--inner-steps", "50", \
    "--seq-len", "4096", \
    "--micro-batch-size", "2", \
    "--checkpoint-dir", "/data/checkpoints", \
    "--checkpoint-every", "3", \
    "--require-signed-deltas", \
    "--round-timeout-seconds", "1800", \
    "--min-workers", "1", \
    "--straggler-grace-seconds", "180", \
    "--straggler-backoff", "0.5", \
    "--worker-inactive-timeout-seconds", "600", \
    "--max-active-workers", "4", \
    "--tier-aware", \
    "--target-round-seconds", "420", \
    "--device", "cpu" \
]
# --target-round-seconds 420: tier-aware sizes each worker's inner_steps so a
# round takes ~7 min of compute. Longer than the old 300 (DiLoCo favors more
# local steps per sync = less comms, better convergence) and gives slow
# contributors more budget. Well under --round-timeout-seconds 1800.
#
# --inner-steps 50 is now only the BOOTSTRAP value a worker uses for its FIRST
# round (tier-aware can't size a worker until it has reported tok/s once). It
# was 200, but a cold worker — especially MPS, which pays a big JIT-warmup tax
# on its first loop — could blow past BOTH the 600s inactivity timeout (→
# evicted mid-loop) and the 1800s round timeout (→ delta arrives stale → tok/s
# never recorded → tier-aware never sizes it → stuck at 200 forever). That's
# exactly why the M5 never landed a delta. 50 keeps the first loop short enough
# to land, report tok/s, and let tier-aware ramp it to ~target within 2-3
# rounds. Fast GPUs just do one quick 50-step round before ramping up.
# --max-active-workers 4: hard cap so the 8 GB VPS doesn't OOM. Each
# active fp32 delta for the 300M model is ~1.25 GB; baseline RSS ~3.6 GB
# leaves room for ~3 deltas + averaging overhead before the kernel OOM-
# killer fires near 7 GB. 4 is the lived ceiling. Bump only after the
# coord moves to a larger VPS or we ship streaming delta averaging.
