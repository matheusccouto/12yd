"""Tests for the FotMob HTTP client (cache + ETag + gzip)."""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from penalty_pred import client as client_module
from penalty_pred.client import FotMobClient


@pytest.fixture(autouse=True)
def _reset_build_id_cache():
    """The BuildId is process-global; reset between tests so each test starts cold."""
    client_module.reset_build_id_cache()
    yield
    client_module.reset_build_id_cache()


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
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=httpx.Request("GET", "x"),
                response=cast("httpx.Response", self),
            )


def _stub_httpx_client(
    monkeypatch, response: _StubResponse, captures: list[str] | None = None
) -> None:
    """Patch httpx.Client to return a fixed response and skip the network."""

    class _StubHTTP:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def __enter__(self) -> _StubHTTP:
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def get(self, url: str, headers: dict[str, str] | None = None) -> _StubResponse:
            if captures is not None:
                captures.append(url)
            return response

    monkeypatch.setattr(client_module.httpx, "Client", _StubHTTP)


def test_discover_build_id_caches_in_process(tmp_path: Path, monkeypatch) -> None:
    """The BuildId is discovered once per process and reused on subsequent calls."""
    discover_calls: list[str] = []

    def fake_discover(c: FotMobClient) -> str:
        discover_calls.append("discover")
        return "fake-build-id"

    monkeypatch.setattr(client_module, "_discover_build_id", fake_discover)
    _stub_httpx_client(
        monkeypatch,
        _StubResponse(status_code=200, body=b"{}", headers={"etag": "abc"}),
    )
    c = FotMobClient(cache_dir=tmp_path)
    c.get("matches/anything/abc123")
    c.get("matches/other/def456")
    assert discover_calls == ["discover"]


def test_reset_build_id_cache_forces_rediscovery(tmp_path: Path, monkeypatch) -> None:
    state = {"count": 0}

    def fake_discover(c: FotMobClient) -> str:
        state["count"] += 1
        return f"build-{state['count']}"

    monkeypatch.setattr(client_module, "_discover_build_id", fake_discover)
    _stub_httpx_client(
        monkeypatch,
        _StubResponse(status_code=200, body=b"{}", headers={"etag": "abc"}),
    )
    c1 = FotMobClient(cache_dir=tmp_path)
    c1.get("matches/x/y")
    assert state["count"] == 1
    client_module.reset_build_id_cache()
    c2 = FotMobClient(cache_dir=tmp_path)
    c2.get("matches/x/y")
    assert state["count"] == 2


def test_gzipped_disk_cache_is_deterministic(tmp_path: Path) -> None:
    """Writing the same payload twice produces the same on-disk size."""
    payload = b'{"hello": "world"}'
    gz_path = tmp_path / "x.json.gz"
    gz_path.write_bytes(gzip.compress(payload))
    first_size = gz_path.stat().st_size
    gz_path.write_bytes(gzip.compress(payload))
    assert gz_path.stat().st_size == first_size
    # Roundtrip is lossless.
    assert gzip.decompress(gz_path.read_bytes()) == payload


def test_304_response_serves_from_disk_cache(tmp_path: Path, monkeypatch) -> None:
    """A 304 returns the cached body without refetching."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    # Pin the build id by using a fake discover.
    monkeypatch.setattr(client_module, "_discover_build_id", lambda c: "fake-build-id")
    # Pre-seed a cache file so the 304 handler has something to load.
    body = b'{"x": 1}'
    url = "https://www.fotmob.com/_next/data/fake-build-id/matches/x/y.json"
    cache_key = client_module._cache_key(url)
    (cache_dir / cache_key).write_bytes(gzip.compress(body))
    (cache_dir / cache_key.with_suffix(".etag")).write_text('"abc"')

    _stub_httpx_client(
        monkeypatch,
        _StubResponse(status_code=304, body=b""),
    )
    client = FotMobClient(cache_dir=cache_dir)
    result = client.get("matches/x/y")
    assert result == {"x": 1}


def test_cache_key_is_filesystem_safe() -> None:
    key = client_module._cache_key("https://x.y/z?a=1&b=2/")
    assert "/" not in key.name
    assert "?" not in key.name
    assert key.name.endswith(".json.gz")


def test_decompress_handles_uncompressed_body() -> None:
    """A response with content-encoding=gzip header but uncompressed body
    (some FotMob edge cases) is not double-decompressed."""
    payload = b'{"ok": true}'
    resp = _StubResponse(body=payload, headers={"content-encoding": "gzip"})
    assert client_module._decompress(cast("httpx.Response", resp)) == payload


def test_decompress_handles_gzipped_body() -> None:
    payload = b'{"ok": true}'
    gz = gzip.compress(payload)
    resp = _StubResponse(body=gz, headers={"content-encoding": "gzip"})
    assert client_module._decompress(cast("httpx.Response", resp)) == payload
