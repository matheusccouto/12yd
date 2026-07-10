"""Tests for the FotMob HTTP client (cache + ETag + gzip + buildId-strip)."""

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
    c = FotMobClient(cache_dir=tmp_path)
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
    c1 = FotMobClient(cache_dir=tmp_path)
    c1.get("matches/x/y")
    assert state["count"] == 1
    c2 = FotMobClient(cache_dir=tmp_path)
    c2.get("matches/x/y")
    assert state["count"] == 2


def test_gzipped_disk_cache_is_deterministic(tmp_path: Path) -> None:
    payload = b'{"hello": "world"}'
    gz_path = tmp_path / "x.json.gz"
    gz_path.write_bytes(gzip.compress(payload))
    first_size = gz_path.stat().st_size
    gz_path.write_bytes(gzip.compress(payload))
    assert gz_path.stat().st_size == first_size
    assert gzip.decompress(gz_path.read_bytes()) == payload


def test_304_response_serves_from_disk_cache(tmp_path: Path, monkeypatch) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(client_module, "_discover_build_id", lambda c: "fake-build-id")
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
    monkeypatch.setattr(client_module, "_discover_build_id", lambda c: "fake-build-id")
    payload = b'{"ok": true}'
    _stub_httpx_client(
        monkeypatch,
        _StubResponse(status_code=200, body=payload, headers={"etag": "abc"}),
    )
    client = FotMobClient(cache_dir=tmp_path)
    result = client.get("leagues/77/overview/world-cup?tz=UTC&date=20240101")
    assert result == {"ok": True}
    cache_files = list(tmp_path.glob("*.json.gz"))
    assert len(cache_files) == 1
    assert "/" not in cache_files[0].name
    assert "?" not in cache_files[0].name


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
    client = FotMobClient(cache_dir=tmp_path)
    result = client.get("matches/x/y")
    assert result == {"decompressed": True}
    cache_file = next(tmp_path.glob("*.json.gz"))
    assert gzip.decompress(cache_file.read_bytes()) == payload


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
    client = FotMobClient(cache_dir=tmp_path)
    result = client.get("matches/x/y")
    assert result == {"uncompressed": True}
    cache_file = next(tmp_path.glob("*.json.gz"))
    assert json.loads(gzip.decompress(cache_file.read_bytes())) == {"uncompressed": True}


# ---------------------------------------------------------------------------
# buildId-strip tests (PRD-v5 linchpin fix)
# ---------------------------------------------------------------------------


def test_cache_key_strips_build_id() -> None:
    url_a = "https://www.fotmob.com/_next/data/abc123/matches/x/y.json"
    url_b = "https://www.fotmob.com/_next/data/def456/matches/x/y.json"
    assert client_module._cache_key(url_a) == client_module._cache_key(url_b)


def test_cache_key_strips_build_id_for_league_paths() -> None:
    url_a = "https://www.fotmob.com/_next/data/build-20250101/leagues/77/overview/world-cup.json"
    url_b = "https://www.fotmob.com/_next/data/build-20250709/leagues/77/overview/world-cup.json"
    assert client_module._cache_key(url_a) == client_module._cache_key(url_b)


def test_cache_key_strips_build_id_for_player_paths() -> None:
    url_a = "https://www.fotmob.com/_next/data/X1Y2Z3/players/12345.json"
    url_b = "https://www.fotmob.com/_next/data/A9B8C7/players/12345.json"
    assert client_module._cache_key(url_a) == client_module._cache_key(url_b)


def test_cache_key_different_resources_produce_different_keys() -> None:
    url_a = "https://www.fotmob.com/_next/data/abc/matches/1.json"
    url_b = "https://www.fotmob.com/_next/data/abc/matches/2.json"
    assert client_module._cache_key(url_a) != client_module._cache_key(url_b)


def test_cache_key_non_next_data_urls_unchanged() -> None:
    url = "https://www.fotmob.com/players/12345"
    assert client_module._cache_key(url) == client_module._cache_key(url)


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
    c = FotMobClient(cache_dir=tmp_path)
    c.get("matches/x/y")
    c.get("matches/a/b")
    assert instance_count[0] == 1


def test_close_releases_http_client(tmp_path: Path) -> None:
    c = FotMobClient(cache_dir=tmp_path)
    http = c.ensure_http()
    assert c._http is not None
    c.close()
    assert c._http is None
