"""Tests for the FotMob HTTP client."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Self

import httpx

from twelveyards.fotmob import client as client_module
from twelveyards.fotmob.client import FotMobClient

if TYPE_CHECKING:
    from pathlib import Path


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

    def json(self) -> Any:  # noqa: ANN401
        return json.loads(self.content)


def _stub_httpx_client(
    monkeypatch: Any,  # noqa: ANN401
    response: _StubResponse,
    captures: list[str] | None = None,
) -> None:
    class _StubHTTP:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:  # noqa: ANN401
            pass

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def close(self) -> None:
            pass

        def get(self, url: str, headers: dict[str, str] | None = None) -> _StubResponse:  # noqa: ARG002
            if captures is not None:
                captures.append(url)
            return response

    monkeypatch.setattr(client_module.httpx, "Client", _StubHTTP)


# ---------------------------------------------------------------------------
# build-id discovery
# ---------------------------------------------------------------------------


def test_discover_build_id_caches_per_client(  # noqa: D103
    tmp_path: Path, monkeypatch: Any,  # noqa: ARG001, ANN401
) -> None:
    discover_calls: list[str] = []

    def fake_discover(self: FotMobClient) -> str:  # noqa: ARG001
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


def test_distinct_clients_discover_independently(  # noqa: D103
    tmp_path: Path, monkeypatch: Any,  # noqa: ARG001, ANN401
) -> None:
    state: dict[str, int] = {"count": 0}

    def fake_discover(self: FotMobClient) -> str:  # noqa: ARG001
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
    assert state["count"] == 2  # noqa: PLR2004


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def test_get_returns_parsed_json(  # noqa: D103
    tmp_path: Path, monkeypatch: Any,  # noqa: ARG001, ANN401
) -> None:
    monkeypatch.setattr(
        FotMobClient, "_discover_build_id", lambda _self: "fake-build-id",
    )
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


def test_shared_http_client_reused_across_calls(  # noqa: D103
    tmp_path: Path, monkeypatch: Any,  # noqa: ARG001, ANN401
) -> None:
    monkeypatch.setattr(
        FotMobClient, "_discover_build_id", lambda _self: "fake-build-id",
    )
    instance_count: list[int] = [0]

    class _StubHTTP:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:  # noqa: ANN401
            instance_count[0] += 1

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def close(self) -> None:
            pass

        def get(
            self, url: str, headers: dict[str, str] | None = None,  # noqa: ARG002
        ) -> _StubResponse:
            return _StubResponse(status_code=200, body=b"{}", headers={"etag": "abc"})

    monkeypatch.setattr(client_module.httpx, "Client", _StubHTTP)
    c = FotMobClient()
    c.get("matches/x/y")
    c.get("matches/a/b")
    assert instance_count[0] == 1


