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
    "--preset", "smoke", \
    "--world-size", "1", \
    "--inner-steps", "30", \
    "--seq-len", "128", \
    "--micro-batch-size", "16", \
    "--checkpoint-dir", "/data/checkpoints", \
    "--checkpoint-every", "5", \
    "--device", "cpu" \
]
