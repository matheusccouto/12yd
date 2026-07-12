"""HTTP client for FotMob with gzip decompression."""

from __future__ import annotations

import gzip
import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import httpx

from twelveyards.config import HTTP_TIMEOUT_SECONDS, USER_AGENT

if TYPE_CHECKING:
    from collections.abc import Mapping

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
    """FotMob HTTP client with gzip decompression."""

    timeout: float = HTTP_TIMEOUT_SECONDS
    build_id: str | None = field(default=None, init=False)
    _http: httpx.Client | None = field(default=None, init=False, repr=False)

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
        http = self.ensure_http()
        response = http.get(url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"})
        response.raise_for_status()
        body = _decompress(response)
        return json.loads(body)

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


def _decompress(response: httpx.Response) -> bytes:
    raw = bytes(response.content)
    if (
        len(raw) >= _GZIP_MAGIC_MIN_LEN
        and raw[0] == _GZIP_MAGIC_BYTE_0
        and raw[1] == _GZIP_MAGIC_BYTE_1
    ):
        return gzip.decompress(raw)
    return raw
