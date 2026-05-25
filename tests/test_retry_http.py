"""Tests for the worker's retry/backoff HTTP helper.

Real coord links drop bodies mid-stream when pulling the ~200 MB /state blob;
prior to this helper any such blip crashed the worker. These tests cover the
retry surface without needing a real network.
"""
from __future__ import annotations

import httpx
import pytest

from dllm.client.worker import retry_http


class _FakeResponse:
    """Stand-in for httpx.Response — only the status_code field is read by retry_http."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


def _flaky(seq):
    """Build a fn() that, on each call, returns the next item — or raises if it's an Exception."""
    seq = list(seq)
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        item = seq[calls["n"] - 1]
        if isinstance(item, BaseException):
            raise item
        return item

    return fn, calls


def test_retry_passes_through_on_first_success() -> None:
    fn, calls = _flaky([_FakeResponse(200)])
    r = retry_http(fn, label="test", base_delay=0.0)
    assert r.status_code == 200
    assert calls["n"] == 1


def test_retry_recovers_after_truncated_body() -> None:
    """The exact failure mode observed against dllm.planetbass.de: nginx/podman
    closes the stream after ~97 MB of 201 MB. One retry should succeed."""
    err = httpx.RemoteProtocolError(
        "peer closed connection without sending complete message body"
    )
    fn, calls = _flaky([err, err, _FakeResponse(200)])
    r = retry_http(fn, label="GET /state", base_delay=0.0, max_attempts=4)
    assert r.status_code == 200
    assert calls["n"] == 3


def test_retry_recovers_from_5xx() -> None:
    fn, calls = _flaky([_FakeResponse(502), _FakeResponse(503), _FakeResponse(200)])
    r = retry_http(fn, label="GET /state", base_delay=0.0, max_attempts=4)
    assert r.status_code == 200
    assert calls["n"] == 3


def test_retry_passes_through_4xx_without_retrying() -> None:
    """404 means the coord deregistered us — retrying won't help; the caller
    needs to see the 404 to decide whether to re-register or stop."""
    fn, calls = _flaky([_FakeResponse(404), _FakeResponse(200)])
    r = retry_http(fn, label="POST /delta", base_delay=0.0, max_attempts=4)
    assert r.status_code == 404
    assert calls["n"] == 1


def test_retry_gives_up_and_raises_after_max_attempts() -> None:
    err = httpx.ConnectError("connection refused")
    fn, calls = _flaky([err] * 5)
    with pytest.raises(httpx.ConnectError):
        retry_http(fn, label="POST /delta", base_delay=0.0, max_attempts=3)
    assert calls["n"] == 3


def test_retry_does_not_swallow_unexpected_exceptions() -> None:
    fn, _ = _flaky([ValueError("boom")])
    with pytest.raises(ValueError):
        retry_http(fn, label="GET /state", base_delay=0.0)
