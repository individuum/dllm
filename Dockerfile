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
    "--inner-steps", "200", \
    "--seq-len", "4096", \
    "--micro-batch-size", "2", \
    "--checkpoint-dir", "/data/checkpoints", \
    "--checkpoint-every", "3", \
    "--require-signed-deltas", \
    "--round-timeout-seconds", "1800", \
    "--min-workers", "1", \
    "--worker-inactive-timeout-seconds", "3600", \
    "--max-active-workers", "4", \
    "--tier-aware", \
    "--target-round-seconds", "600", \
    "--device", "cpu" \
]
# --max-active-workers 4: hard cap so the 8 GB VPS doesn't OOM. Each
# active fp32 delta for the 300M model is ~1.25 GB; baseline RSS ~3.6 GB
# leaves room for ~3 deltas + averaging overhead before the kernel OOM-
# killer fires near 7 GB. 4 is the lived ceiling. Bump only after the
# coord moves to a larger VPS or we ship streaming delta averaging.
