"""HTTP client for FotMob Next.js API."""

from __future__ import annotations

import json
import re
from typing import Any

import httpx

from .leagues import LEAGUES
from .models import League, LeagueDetails, Match, MatchRef, Season

USER_AGENT: str = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
HTTP_TIMEOUT_SECONDS: float = 15.0


def _extract_season_year(season: str) -> str:
    """Extract the season query parameter identifier from a season name string."""
    parts = season.strip().split()
    if not parts:
        msg = f"Empty season string: {season!r}"
        raise ValueError(msg)
    return parts[0]


class FotMob:
    """FotMob Next.js API client."""

    def __init__(self, timeout: float = HTTP_TIMEOUT_SECONDS) -> None:
        """Create a FotMob client with connection pool.

        Includes automatic raise-on-status hook.
        """
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
        response = self._http.get(
            "https://www.fotmob.com/",
            headers={"User-Agent": USER_AGENT},
        )
        match = re.search(
            pattern=r'<script id="__NEXT_DATA__"[^>]*>(.+?)</script>',
            string=response.text,
            flags=re.DOTALL,
        )
        if match is None:
            msg = "Could not find __NEXT_DATA__ script tag on FotMob homepage"
            raise RuntimeError(msg)
        return str(json.loads(match.group(1))["buildId"])

    def _get(self, path: str, **params: Any) -> Any:
        """Fetch a Next.js JSON data route and return raw parsed JSON."""
        url = f"https://www.fotmob.com/_next/data/{self.build_id}/{path}.json"
        response = self._http.get(
            url, params=params, headers={"User-Agent": USER_AGENT},
        )
        return response.json()

    def get(self, path: str, params: dict[str, str] | None = None) -> Any:
        """Fetch a Next.js JSON data route and return raw parsed JSON."""
        return self._get(path, **(params or {}))



    def get_leagues(self) -> list[League]:
        """Return the list of targeted international leagues."""
        return list(LEAGUES)

    def get_league(self, league_id: int) -> LeagueDetails:
        """Get details for a given league."""
        data = self._get(f"leagues/{league_id}")
        details_data = data.get("pageProps", {}).get("details", {})
        return LeagueDetails.model_validate(details_data)

    def get_league_seasons(self, league_id: int) -> list[Season]:
        """Get the available seasons for a given league."""
        data = self._get(f"leagues/{league_id}")
        seasons_data = data.get("pageProps", {}).get("seasons", [])
        return [Season.model_validate(s) for s in seasons_data]

    def get_league_matches(self, league_id: int, season: str) -> list[MatchRef]:
        """Get all match references for a given league and season."""
        year = _extract_season_year(season)
        league_details = self.get_league(league_id)
        slug = league_details.seopath
        data = self._get(f"leagues/{league_id}/overview/{slug}", season=year)

        page_props = data.get("pageProps", {})
        fixtures = page_props.get("fixtures", {})
        matches_data = fixtures.get("allMatches")
        if matches_data is None:
            overview = page_props.get("overview", {})
            matches_data = overview.get("matches", {}).get("allMatches", [])

        return [MatchRef.model_validate(m) for m in matches_data]

    def get_match(self, match_id: str) -> Match:
        """Get full match details including shotmap and lineups by match ID."""
        data = self._get(f"match/{match_id}")
        return Match.model_validate(data)



