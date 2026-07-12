"""Tests for the FotMob HTTP client (gzip + buildId discovery)."""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any, cast

import httpx

from twelveyards.fotmob import client as client_module
from twelveyards.fotmob.client import FotMobClient


class _StubResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        body: bytes = b"",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.content = body
        self.headers = httpx.Headers(headers or {})

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            msg = f"HTTP {self.status_code}"
            raise httpx.HTTPStatusError(
                msg,
                request=httpx.Request("GET", "x"),
                response=cast("httpx.Response", self),
            )


def _stub_httpx_client(
    monkeypatch, response: _StubResponse, captures: list[str] | None = None,
) -> None:
    class _StubHTTP:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def __enter__(self) -> _StubHTTP:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def close(self) -> None:
            pass

        def get(self, url: str, headers: dict[str, str] | None = None) -> _StubResponse:
            if captures is not None:
                captures.append(url)
            return response

    monkeypatch.setattr(client_module.httpx, "Client", _StubHTTP)


def test_discover_build_id_caches_per_client(tmp_path: Path, monkeypatch) -> None:
    discover_calls: list[str] = []

    def fake_discover(c: FotMobClient) -> str:
        discover_calls.append("discover")
        return "fake-build-id"

    monkeypatch.setattr(client_module, "_discover_build_id", fake_discover)
    _stub_httpx_client(
        monkeypatch,
        _StubResponse(status_code=200, body=b"{}", headers={"etag": "abc"}),
    )
    c = FotMobClient()
    c.get("matches/anything/abc123")
    c.get("matches/other/def456")
    assert discover_calls == ["discover"]


def test_distinct_clients_discover_independently(tmp_path: Path, monkeypatch) -> None:
    state = {"count": 0}

    def fake_discover(c: FotMobClient) -> str:
        state["count"] += 1
        return f"build-{state['count']}"

    monkeypatch.setattr(client_module, "_discover_build_id", fake_discover)
    _stub_httpx_client(
        monkeypatch,
        _StubResponse(status_code=200, body=b"{}", headers={"etag": "abc"}),
    )
    c1 = FotMobClient()
    c1.get("matches/x/y")
    assert state["count"] == 1
    c2 = FotMobClient()
    c2.get("matches/x/y")
    assert state["count"] == 2


def test_get_decompresses_gzipped_response_body(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(client_module, "_discover_build_id", lambda c: "fake-build-id")
    payload = b'{"decompressed": true}'
    gz = gzip.compress(payload)
    _stub_httpx_client(
        monkeypatch,
        _StubResponse(
            status_code=200,
            body=gz,
            headers={"content-encoding": "gzip", "etag": "abc"},
        ),
    )
    client = FotMobClient()
    result = client.get("matches/x/y")
    assert result == {"decompressed": True}


def test_get_handles_uncompressed_body_with_gzip_header(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(client_module, "_discover_build_id", lambda c: "fake-build-id")
    payload = b'{"uncompressed": true}'
    _stub_httpx_client(
        monkeypatch,
        _StubResponse(
            status_code=200,
            body=payload,
            headers={"content-encoding": "gzip", "etag": "abc"},
        ),
    )
    client = FotMobClient()
    result = client.get("matches/x/y")
    assert result == {"uncompressed": True}


# ---------------------------------------------------------------------------
# shared httpx.Client tests (PRD-v5)
# ---------------------------------------------------------------------------


def test_shared_http_client_reused_across_calls(tmp_path: Path, monkeypatch) -> None:
    """Two `client.get` calls use the same httpx.Client instance."""
    monkeypatch.setattr(client_module, "_discover_build_id", lambda c: "fake-build-id")
    instance_count = [0]

    class _StubHTTP:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            instance_count[0] += 1

        def __enter__(self) -> _StubHTTP:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def close(self) -> None:
            pass

        def get(self, url: str, headers: dict[str, str] | None = None) -> _StubResponse:
            return _StubResponse(status_code=200, body=b"{}", headers={"etag": "abc"})

    monkeypatch.setattr(client_module.httpx, "Client", _StubHTTP)
    c = FotMobClient()
    c.get("matches/x/y")
    c.get("matches/a/b")
    assert instance_count[0] == 1


def test_close_releases_http_client(tmp_path: Path) -> None:
    c = FotMobClient()
    _http = c.ensure_http()
    assert c._http is not None
    c.close()
    assert c._http is None
