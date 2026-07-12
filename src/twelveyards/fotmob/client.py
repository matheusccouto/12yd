"""HTTP client for FotMob."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from collections.abc import Mapping

USER_AGENT: str = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
HTTP_TIMEOUT_SECONDS: float = 15.0


class FotMobClient:
    """FotMob HTTP client."""

    def __init__(self, timeout: float = HTTP_TIMEOUT_SECONDS) -> None:
        """Create a FotMob HTTP client with the given timeout."""
        self._build_id: str | None = None
        self._http = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            event_hooks={"response": [lambda r: r.raise_for_status()]},
        )

    @property
    def build_id(self) -> str:
        """Lazily discover and return the current FotMob Next.js build ID."""
        if self._build_id is None:
            self._build_id = self._discover_build_id()
        return self._build_id

    def _discover_build_id(self) -> str:
        response = self._http.get("https://www.fotmob.com/", headers={"User-Agent": USER_AGENT})
        match = re.search(
            pattern=r'<script id="__NEXT_DATA__"[^>]*>(.+?)</script>',
            string=response.text,
            flags=re.DOTALL,
        )
        if match is None:
            msg = "Could not find __NEXT_DATA__ script tag on FotMob homepage"
            raise RuntimeError(msg)
        return str(json.loads(match.group(1))["buildId"])

    def get(self, path: str, params: Mapping[str, str] | None = None) -> Any:  # noqa: ANN401
        """Fetch a FotMob API path, build the URL, and return parsed JSON."""
        url = f"https://www.fotmob.com/_next/data/{self.build_id}/{path}.json"
        if params:
            url = f"{url}?{'&'.join(f'{k}={v}' for k, v in params.items())}"
        response = self._http.get(url, headers={"User-Agent": USER_AGENT})
        return response.json()
