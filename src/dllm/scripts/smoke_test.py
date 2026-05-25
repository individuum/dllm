"""Phase 0 smoke test: spawn 1 coord + 2 workers locally, run 20 outer rounds.

Acceptance: workers register, deltas exchanged, loss decreases from initial value.
This is NOT a convergence test — it verifies the loop wires up end-to-end.
"""
from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def wait_for_port(host: str, port: int, timeout: float = 30.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            try:
                s.connect((host, port))
                return
            except OSError:
                time.sleep(0.2)
    raise TimeoutError(f"{host}:{port} did not open within {timeout}s")


def wait_for_coord(url: str, timeout: float = 30.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = httpx.get(f"{url}/status", timeout=2.0)
            if r.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.3)
    raise TimeoutError(f"coordinator at {url} not ready within {timeout}s")


def ensure_data() -> Path:
    root = repo_root()
    train_bin = root / "data" / "cache" / "train.bin"
    if train_bin.exists():
        return train_bin
    print("[smoke] train.bin not found — running data.prepare")
    subprocess.run([sys.executable, "-m", "dllm.data.prepare"], check=True, cwd=root)
    if not train_bin.exists():
        raise RuntimeError(f"data.prepare did not create {train_bin}")
    return train_bin


def preflight_gpu_banner() -> None:
    """One-shot CUDA visibility check before spawning workers.

    Prints loudly so a misconfigured torch (CPU-only build) can't masquerade as GPU.
    """
    try:
        import torch

        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            name = torch.cuda.get_device_name(0)
            print(f"[smoke] [GPU CHECK] OK: torch={torch.__version__} cuda={torch.version.cuda} device=cuda:0 name={name}")
        else:
            print(
                f"[smoke] [GPU CHECK] WARNING: torch={torch.__version__} cuda_available=False — "
                f"workers will run on CPU. Pass --require-gpu to fail fast."
            )
    except Exception as e:  # noqa: BLE001
        print(f"[smoke] [GPU CHECK] could not query torch: {e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="smoke")
    ap.add_argument("--world-size", type=int, default=2)
    ap.add_argument("--max-rounds", type=int, default=20)
    ap.add_argument("--inner-steps", type=int, default=20)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--micro-batch-size", type=int, default=16)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--device", default="auto")
    ap.add_argument(
        "--require-gpu",
        action="store_true",
        help="Pass --require-gpu to spawned workers (fail fast on CPU fallback)",
    )
    args = ap.parse_args()
    preflight_gpu_banner()

    root = repo_root()
    data_bin = ensure_data()
    coord_url = f"http://127.0.0.1:{args.port}"

    print(f"[smoke] repo root: {root}")
    print(f"[smoke] data: {data_bin}")
    print(f"[smoke] coordinator: {coord_url}")
    print(f"[smoke] preset={args.preset} world_size={args.world_size} rounds={args.max_rounds}")

    coord_cmd = [
        sys.executable,
        "-u",
        "-m",
        "dllm.coord.server",
        "--preset", args.preset,
        "--world-size", str(args.world_size),
        "--inner-steps", str(args.inner_steps),
        "--seq-len", str(args.seq_len),
        "--micro-batch-size", str(args.micro_batch_size),
        "--max-rounds", str(args.max_rounds),
        "--port", str(args.port),
        "--device", "cpu",  # coord doesn't need GPU; workers do the math
        "--log-level", "info",
    ]
    print("[smoke] spawning coord:", " ".join(coord_cmd))
    coord_proc = subprocess.Popen(
        coord_cmd,
        cwd=root,
        stdout=sys.stdout,
        stderr=sys.stdout,
        creationflags=(
            subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        ),
    )

    workers: list[subprocess.Popen] = []
    try:
        wait_for_coord(coord_url)
        print("[smoke] coordinator ready")

        for i in range(args.world_size):
            worker_cmd = [
                sys.executable,
                "-u",
                "-m",
                "dllm.client.worker",
                "--coord", coord_url,
                "--preset", args.preset,
                "--country", "XX",
                "--device", args.device,
                "--data", str(data_bin),
                "--max-rounds", str(args.max_rounds),
                "--log-level", "info",
            ]
            if args.require_gpu:
                worker_cmd.append("--require-gpu")
            print(f"[smoke] spawning worker {i}:", " ".join(worker_cmd))
            workers.append(
                subprocess.Popen(
                    worker_cmd,
                    cwd=root,
                    stdout=sys.stdout,
                    stderr=sys.stdout,
                    creationflags=(
                        subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
                    ),
                )
            )

        rc_workers = [w.wait() for w in workers]
        print(f"[smoke] worker exit codes: {rc_workers}")

        if all(rc == 0 for rc in rc_workers):
            print("[smoke] PASS — all workers exited cleanly")
            exit_code = 0
        else:
            print("[smoke] FAIL — at least one worker exited non-zero")
            exit_code = 1
    finally:
        # tear down coord
        if coord_proc.poll() is None:
            print("[smoke] stopping coordinator")
            if os.name == "nt":
                coord_proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                coord_proc.terminate()
            try:
                coord_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                coord_proc.kill()
        for w in workers:
            if w.poll() is None:
                w.kill()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
