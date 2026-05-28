"""FastAPI route layer for the coordinator.

`create_app(state)` wires the HTTP surface onto a `CoordinatorState`. Moved
verbatim from `server.py`; `from __future__ import annotations` + `TYPE_CHECKING`
keep the `CoordinatorState` type hint from creating an import cycle (server.py
imports `create_app` from here).
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse

from ..shared.protocol import RegisterRequest, RegisterResponse, RoundStatus

if TYPE_CHECKING:
    from .server import CoordinatorState

log = logging.getLogger("dllm.coord")

_DASHBOARD_PATH = Path(__file__).parent / "dashboard.html"


def _load_dashboard_html() -> str:
    """Re-read on every request — UI tweaks no longer require a coord restart."""
    return _DASHBOARD_PATH.read_text(encoding="utf-8")


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

    @app.get("/workers")
    def get_workers() -> dict:
        return {"workers": state.list_workers()}

    @app.post("/register", response_model=RegisterResponse)
    def register(req: RegisterRequest) -> RegisterResponse:
        return state.register(req)

    @app.post("/deregister")
    def deregister(worker_id: int, ts: int, signature: str = "") -> dict:
        """Worker voluntarily leaves. Signed; idempotent."""
        return state.deregister(worker_id, ts, signature)

    @app.get("/status", response_model=RoundStatus)
    def status(worker_id: int | None = None) -> RoundStatus:
        # `worker_id` is optional: dashboards omit it; long-polling workers
        # pass it so their /status poll doubles as an inactivity heartbeat.
        return state.status(heartbeat_worker_id=worker_id)

    # api_route w/ GET+HEAD: FastAPI doesn't auto-handle HEAD on @app.get,
    # so plain HEAD returns 405. Workers probe with HEAD before deciding
    # to parallel-fetch via Range — we MUST answer it. FastAPI strips the
    # body on HEAD automatically when the route is registered for HEAD.
    @app.api_route("/state", methods=["GET", "HEAD"])
    def get_state(request: Request, round: int | None = None):
        # Cache-friendly state endpoint (task #61). When a worker passes
        # `?round=N`, Cloudflare/CDN treats each round's state as a distinct
        # URL — `/state?round=89` and `/state?round=90` get separate cache
        # entries, both immutable for their lifetime. Without the param,
        # behaviour falls back to legacy "serve whatever the coord has now"
        # (uncacheable on the CDN side).
        blob, round_no = state.state_blob()
        headers = {
            "x-round": str(round_no),
            "x-codec": state.state_codec,
            # Always advertise Range support so clients can probe with HEAD.
            "Accept-Ranges": "bytes",
        }
        if round is not None:
            if round != round_no:
                # Worker asked for state @ round N but coord moved on (or
                # hasn't reached N yet). Tell them what round we're at; they
                # should refresh via /status before pulling again.
                return JSONResponse(
                    status_code=409,
                    content={"current_round": round_no, "requested_round": round},
                    headers={"x-round": str(round_no)},
                )
            # Per-round URL is content-addressable. Safe to cache forever
            # — the bytes for round N never change. Cloudflare's free tier
            # caches at this header.
            headers["Cache-Control"] = (
                "public, max-age=86400, s-maxage=86400, immutable"
            )
        else:
            # Legacy `/state` with no round param. Don't cache — content
            # depends on coord's current round which moves over time.
            headers["Cache-Control"] = "no-store"

        # Range support (task #62). Lets workers parallel-fetch chunks over
        # separate TCP connections — defeats per-TCP-flow QoS throttling
        # common on Vodafone Kabel DE and similar ISPs. Clients on
        # well-behaved networks can still single-stream; this is opt-in.
        range_hdr = request.headers.get("range", "").strip()
        total = len(blob)
        if range_hdr.startswith("bytes="):
            spec = range_hdr[len("bytes=") :].strip()
            try:
                start_s, end_s = spec.split("-", 1)
                start = int(start_s) if start_s else 0
                end = int(end_s) if end_s else total - 1
            except (ValueError, AttributeError):
                return JSONResponse(
                    status_code=416,
                    content={"detail": f"malformed Range header: {range_hdr!r}"},
                )
            if start < 0 or end >= total or start > end:
                return JSONResponse(
                    status_code=416,
                    content={
                        "detail": f"range {start}-{end} outside body 0-{total - 1}"
                    },
                    headers={"Content-Range": f"bytes */{total}"},
                )
            partial = blob[start : end + 1]
            headers["Content-Range"] = f"bytes {start}-{end}/{total}"
            return Response(
                content=partial,
                status_code=206,
                media_type="application/octet-stream",
                headers=headers,
            )

        return Response(
            content=blob,
            media_type="application/octet-stream",
            headers=headers,
        )

    @app.post("/delta")
    async def post_delta(
        request: Request,
        worker_id: int,
        round: int,
        val_loss: float | None = None,
        power_watts: float | None = None,
        tokens_per_sec: float | None = None,
        tokens_per_step: int | None = None,
    ):
        body = await request.body()
        sig = request.headers.get("x-delta-signature")
        ack = state.submit_delta(
            worker_id,
            round,
            body,
            val_loss=val_loss,
            power_watts=power_watts,
            tokens_per_sec=tokens_per_sec,
            tokens_per_step=tokens_per_step,
            signature_b64=sig,
        )
        return JSONResponse(ack.model_dump())

    return app
