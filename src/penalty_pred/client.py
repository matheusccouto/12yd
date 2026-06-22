"""HTTP client for FotMob with gzip, ETag revalidation, and persistent disk cache.

PRD: BuildId discovery is one-time per process; cached for the lifetime of the run.
ETag revalidation means 304 responses return zero bytes on cache hit.
Persistent disk cache bypasses the 1h CloudFront TTL (docs/fotmob.md).
"""

from __future__ import annotations

import gzip
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from .config import HTTP_TIMEOUT_SECONDS, USER_AGENT

# Process-level BuildId cache (one discovery per Python process).
_build_id_cache: str | None = None


@dataclass
class FotMobClient:
    cache_dir: Path
    timeout: float = HTTP_TIMEOUT_SECONDS
    # Discovered BuildId (process-level). Populated lazily on first use.
    build_id: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, path: str, params: Mapping[str, str] | None = None) -> Any:
        """GET a `__next/data` path under the cached BuildId.

        `path` is the post-buildId segment, e.g. `matches/argentina-vs-france/1hox8a`.
        Returns the parsed JSON body. 304 responses are served from disk cache.
        """
        self._ensure_build_id()
        url = f"https://www.fotmob.com/_next/data/{self.build_id}/{path}.json"
        if params:
            url = f"{url}?{'&'.join(f'{k}={v}' for k, v in params.items())}"
        return _cached_get(self, url)

    def _ensure_build_id(self) -> None:
        global _build_id_cache
        if _build_id_cache is None:
            _build_id_cache = _discover_build_id(self)
        self.build_id = _build_id_cache


def _discover_build_id(client: FotMobClient) -> str:
    """Fetch https://www.fotmob.com/ once and extract buildId from __NEXT_DATA__."""
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"}
    with httpx.Client(timeout=client.timeout, follow_redirects=True) as http:
        response = http.get("https://www.fotmob.com/", headers=headers)
    response.raise_for_status()
    match = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.+?)</script>', response.text, re.DOTALL)
    if match is None:
        msg = "Could not find __NEXT_DATA__ script tag on FotMob homepage"
        raise RuntimeError(msg)
    return str(json.loads(match.group(1))["buildId"])


def _cache_key(url: str) -> Path:
    """Deterministic cache file path for a URL (filesystem-safe)."""
    sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", url)
    if len(sanitized) > 200:
        sanitized = sanitized[:200]
    return Path(sanitized + ".json.gz")


def _cached_get(client: FotMobClient, url: str) -> Any:
    """HTTP GET with ETag revalidation and gzipped disk cache."""
    cache_file = client.cache_dir / _cache_key(url)
    etag_file = cache_file.with_suffix(".etag")
    headers: dict[str, str] = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"}
    if etag_file.exists():
        headers["If-None-Match"] = etag_file.read_text(encoding="utf-8").strip()

    # Decode the body ourselves: httpx auto-decompresses by default, but the
    # `content-encoding` header is sometimes present on already-uncompressed
    # bodies, so we need to detect gzip by magic bytes to avoid double-decompress.
    with httpx.Client(timeout=client.timeout, follow_redirects=True) as http:
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
    # Sniff gzip magic bytes: 0x1f 0x8b.
    if len(raw) >= 2 and raw[0] == 0x1F and raw[1] == 0x8B:
        return gzip.decompress(raw)
    return raw


def _load_cached(cache_file: Path) -> Any:
    return json.loads(gzip.decompress(cache_file.read_bytes()))


def reset_build_id_cache() -> None:
    """Clear the process-level BuildId cache. Tests use this to force re-discovery."""
    global _build_id_cache
    _build_id_cache = None
