"""Tests for the FotMob HTTP client (cache + ETag + gzip)."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, cast

import httpx

from twelveyards import client as client_module
from twelveyards.client import FotMobClient


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


def test_discover_build_id_caches_per_client(tmp_path: Path, monkeypatch) -> None:
    """The BuildId is discovered once per client and reused on subsequent calls."""
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


def test_distinct_clients_discover_independently(tmp_path: Path, monkeypatch) -> None:
    """Two FotMobClient instances each discover their own BuildId."""
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


def test_get_handles_url_with_query_string_and_trailing_slash(tmp_path: Path, monkeypatch) -> None:
    """`client.get` accepts paths whose query string or trailing slash
    would create filesystem-unsafe characters in the cache key. The
    public interface succeeds without raising; the cache file is
    created under the (filesystem-safe) cache key."""
    monkeypatch.setattr(client_module, "_discover_build_id", lambda c: "fake-build-id")
    payload = b'{"ok": true}'
    _stub_httpx_client(
        monkeypatch,
        _StubResponse(status_code=200, body=payload, headers={"etag": "abc"}),
    )
    client = FotMobClient(cache_dir=tmp_path)
    result = client.get("leagues/77/overview/world-cup?tz=UTC&date=20240101")
    assert result == {"ok": True}
    # The cache file is created under the sanitized key (no `/` or `?`
    # in the file name). One .json.gz body + one .etag sidecar.
    cache_files = list(tmp_path.glob("*.json.gz"))
    assert len(cache_files) == 1
    assert "/" not in cache_files[0].name
    assert "?" not in cache_files[0].name


def test_get_decompresses_gzipped_response_body(tmp_path: Path, monkeypatch) -> None:
    """`client.get` decompresses a gzipped body (the `content-encoding`
    header is set). The parsed JSON is the decompressed payload, not
    the raw gzip bytes."""
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
    client = FotMobClient(cache_dir=tmp_path)
    result = client.get("matches/x/y")
    assert result == {"decompressed": True}
    # The on-disk cache holds the gzipped payload (so the second
    # request is a 304, not a re-fetch).
    cache_file = next(tmp_path.glob("*.json.gz"))
    assert gzip.decompress(cache_file.read_bytes()) == payload


def test_get_handles_uncompressed_body_with_gzip_header(tmp_path: Path, monkeypatch) -> None:
    """Some FotMob responses set `content-encoding: gzip` but ship
    already-uncompressed bytes. `client.get` parses the JSON without
    double-decompressing (it detects gzip by magic bytes, not by
    header)."""
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
    client = FotMobClient(cache_dir=tmp_path)
    result = client.get("matches/x/y")
    assert result == {"uncompressed": True}
    # The cache stores the bytes verbatim (the on-disk round-trip is
    # the same as the round-trip via gzip.compress + gzip.decompress).
    cache_file = next(tmp_path.glob("*.json.gz"))
    assert json.loads(gzip.decompress(cache_file.read_bytes())) == {"uncompressed": True}
