"""Tests for the FotMob HTTP client."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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

    def json(self) -> Any:
        return json.loads(self.content)


def _stub_httpx_client(
    monkeypatch,
    response: _StubResponse,
    captures: list[str] | None = None,
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


# ---------------------------------------------------------------------------
# build-id discovery
# ---------------------------------------------------------------------------


def test_discover_build_id_caches_per_client(tmp_path: Path, monkeypatch) -> None:
    discover_calls: list[str] = []

    def fake_discover(self: FotMobClient) -> str:
        discover_calls.append("discover")
        return "fake-build-id"

    monkeypatch.setattr(FotMobClient, "_discover_build_id", fake_discover)
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

    def fake_discover(self: FotMobClient) -> str:
        state["count"] += 1
        return f"build-{state['count']}"

    monkeypatch.setattr(FotMobClient, "_discover_build_id", fake_discover)
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


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def test_get_returns_parsed_json(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(FotMobClient, "_discover_build_id", lambda self: "fake-build-id")
    _stub_httpx_client(
        monkeypatch,
        _StubResponse(
            status_code=200,
            body=b'{"key": "value"}',
            headers={"etag": "abc"},
        ),
    )
    client = FotMobClient()
    result = client.get("matches/x/y")
    assert result == {"key": "value"}


# ---------------------------------------------------------------------------
# shared httpx.Client
# ---------------------------------------------------------------------------


def test_shared_http_client_reused_across_calls(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(FotMobClient, "_discover_build_id", lambda self: "fake-build-id")
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


