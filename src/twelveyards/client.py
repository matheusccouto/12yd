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
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import httpx

from .config import HTTP_TIMEOUT_SECONDS, USER_AGENT


class FotMobClientLike(Protocol):
    def get(self, path: str, params: Mapping[str, str] | None = None) -> Any: ...


@dataclass
class FotMobClient:
    cache_dir: Path
    timeout: float = HTTP_TIMEOUT_SECONDS
    build_id: str | None = field(default=None, init=False)
    _http: httpx.Client | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_http(self) -> httpx.Client:
        if self._http is None:
            self._http = httpx.Client(timeout=self.timeout, follow_redirects=True)
        return self._http

    def get(self, path: str, params: Mapping[str, str] | None = None) -> Any:
        if self.build_id is None:
            self.build_id = _discover_build_id(self)
        url = f"https://www.fotmob.com/_next/data/{self.build_id}/{path}.json"
        if params:
            url = f"{url}?{'&'.join(f'{k}={v}' for k, v in params.items())}"
        return _cached_get(self, url)

    def close(self) -> None:
        if self._http is not None:
            self._http.close()
            self._http = None


def _discover_build_id(client: FotMobClient) -> str:
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"}
    http = client._ensure_http()
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
    if len(sanitized) > 200:
        sanitized = sanitized[:200]
    return Path(sanitized + ".json.gz")


def _cached_get(client: FotMobClient, url: str) -> Any:
    cache_file = client.cache_dir / _cache_key(url)
    etag_file = cache_file.with_suffix(".etag")
    headers: dict[str, str] = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"}
    if etag_file.exists():
        headers["If-None-Match"] = etag_file.read_text(encoding="utf-8").strip()

    http = client._ensure_http()
    response = http.get(url, headers=headers)
    if response.status_code == 304:
        return _load_cached(cache_file)

    response.raise_for_status()
    body = _decompress(response)
    cache_file.write_bytes(gzip.compress(body))
    if "etag" in response.headers:
        etag_file.write_text(response.headers["etag"], encoding="utf-8")
    return json.loads(body)


def _decompress(response: httpx.Response) -> bytes:
    raw = bytes(response.content)
    if len(raw) >= 2 and raw[0] == 0x1F and raw[1] == 0x8B:
        return gzip.decompress(raw)
    return raw


def _load_cached(cache_file: Path) -> Any:
    return json.loads(gzip.decompress(cache_file.read_bytes()))
