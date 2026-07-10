"""HTTP client for FotMob with gzip, ETag revalidation, and persistent disk cache.

PRD-v5: BuildId-stripped cache keys so cached responses survive FotMob
deployment rotations. A shared httpx.Client (lazy-init, per-instance) reuses
one connection pool across all calls instead of creating a new TLS session
per request. Persistent disk cache bypasses the 1h CloudFront TTL
(docs/fotmob.md).
"""

from __future__ import annotations

import gzip
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import httpx

from .config import HTTP_TIMEOUT_SECONDS, USER_AGENT

if TYPE_CHECKING:
    from collections.abc import Mapping

_MAX_CACHE_KEY_LEN: int = 200
_HTTP_NOT_MODIFIED: int = 304
_GZIP_MAGIC_MIN_LEN: int = 2
_GZIP_MAGIC_BYTE_0: int = 0x1F
_GZIP_MAGIC_BYTE_1: int = 0x8B


class FotMobClientLike(Protocol):
    """Structural type for anything that looks like a FotMob HTTP client."""

    def get(self, path: str, params: Mapping[str, str] | None = None) -> Any:  # noqa: ANN401
        """Fetch a FotMob API path and return the parsed JSON body."""
        ...


@dataclass
class FotMobClient:
    """FotMob HTTP client with gzip, ETag revalidation, and disk cache."""

    cache_dir: Path
    timeout: float = HTTP_TIMEOUT_SECONDS
    build_id: str | None = field(default=None, init=False)
    _http: httpx.Client | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Create the cache directory if it does not exist."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def ensure_http(self) -> httpx.Client:
        """Lazy-init and return a shared httpx.Client."""
        if self._http is None:
            self._http = httpx.Client(timeout=self.timeout, follow_redirects=True)
        return self._http

    def get(self, path: str, params: Mapping[str, str] | None = None) -> Any:  # noqa: ANN401
        """Fetch a FotMob API path, build the URL, and return parsed JSON."""
        if self.build_id is None:
            self.build_id = _discover_build_id(self)
        url = f"https://www.fotmob.com/_next/data/{self.build_id}/{path}.json"
        if params:
            url = f"{url}?{'&'.join(f'{k}={v}' for k, v in params.items())}"
        return _cached_get(self, url)

    def close(self) -> None:
        """Close the underlying httpx.Client if it was created."""
        if self._http is not None:
            self._http.close()
            self._http = None


def _discover_build_id(client: FotMobClient) -> str:
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"}
    http = client.ensure_http()
    response = http.get("https://www.fotmob.com/", headers=headers)
    response.raise_for_status()
    match = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.+?)</script>', response.text, re.DOTALL)
    if match is None:
        msg = "Could not find __NEXT_DATA__ script tag on FotMob homepage"
        raise RuntimeError(msg)
    return str(json.loads(match.group(1))["buildId"])


def _cache_key(url: str) -> Path:
    """Deterministic cache file path for a URL, with buildId stripped.

    FotMob's /_next/data/<buildId>/ segment rotates every few hours.
    Stripping the buildId means the same logical resource has the same
    cache key across deployments, so ETag revalidation survives rotations.
    """
    stripped = re.sub(r"/_next/data/[^/]+/", "/_next/data/_/", url)
    sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", stripped)
    if len(sanitized) > _MAX_CACHE_KEY_LEN:
        sanitized = sanitized[:_MAX_CACHE_KEY_LEN]
    return Path(sanitized + ".json.gz")


def _cached_get(client: FotMobClient, url: str) -> Any:  # noqa: ANN401
    cache_file = client.cache_dir / _cache_key(url)
    etag_file = cache_file.with_suffix(".etag")
    headers: dict[str, str] = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"}
    if etag_file.exists():
        headers["If-None-Match"] = etag_file.read_text(encoding="utf-8").strip()

    http = client.ensure_http()
    response = http.get(url, headers=headers)
    if response.status_code == _HTTP_NOT_MODIFIED:
        return _load_cached(cache_file)

    response.raise_for_status()
    body = _decompress(response)
    cache_file.write_bytes(gzip.compress(body))
    if "etag" in response.headers:
        etag_file.write_text(response.headers["etag"], encoding="utf-8")
    return json.loads(body)


def _decompress(response: httpx.Response) -> bytes:
    raw = bytes(response.content)
    if (
        len(raw) >= _GZIP_MAGIC_MIN_LEN
        and raw[0] == _GZIP_MAGIC_BYTE_0
        and raw[1] == _GZIP_MAGIC_BYTE_1
    ):
        return gzip.decompress(raw)
    return raw


def _load_cached(cache_file: Path) -> Any:  # noqa: ANN401
    return json.loads(gzip.decompress(cache_file.read_bytes()))
