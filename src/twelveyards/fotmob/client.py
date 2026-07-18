"""HTTP client for FotMob Next.js API."""

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from functools import cache
from typing import Any

import httpx
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from .models import (
    League,
    Match,
    Penalty,
    Period,
    Player,
    Position,
    Round,
    Score,
    Shot,
    Status,
    Team,
)

USER_AGENT: str = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
HTTP_TIMEOUT_SECONDS: float = 5.0
FLOOR_DATETIME = datetime(2026, 1, 1, tzinfo=UTC)
CEIL_DATETIME = datetime.now(tz=UTC)


logger = logging.getLogger(__name__)


class NoShotsDataError(Exception):
    """Shots data field is missing or empty."""


class FotMob:
    """FotMob Next.js API client."""

    def __init__(self, timeout: float = HTTP_TIMEOUT_SECONDS) -> None:
        """Create a FotMob client with connection pool."""
        self._build_id: str | None = None
        self._http = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            event_hooks={"response": [lambda r: r.raise_for_status()]},
        )

        def is_transient_error(exception: Exception) -> bool:
            if isinstance(exception, httpx.HTTPStatusError):
                return exception.response.status_code in (429, 500, 502, 503, 504)
            return isinstance(exception, httpx.RequestError)

        # Wrap the HTTP client's send method with a retry strategy for transient errors
        self._http.send = retry(
            stop=stop_after_attempt(4),
            wait=wait_exponential_jitter(initial=0.5, max=5.0, jitter=0.1),
            retry=retry_if_exception(is_transient_error),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )(self._http.send)

    @property
    def build_id(self) -> str:
        """Lazily discover and return the current FotMob Next.js build ID."""
        if self._build_id is None:
            self._build_id = self._get_build_id()
        return self._build_id

    def _get_build_id(self) -> str:
        logger.info("Discovering build ID")
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

    def get(self, path: str, params: dict[str, str] | None = None) -> Any:
        """Fetch a Next.js JSON data route and return raw parsed JSON."""
        return self._http.get(
            f"https://www.fotmob.com/_next/data/{self.build_id}/{path}.json",
            params=params,
            headers={"User-Agent": USER_AGENT},
        ).json()

    @cache  # noqa: B019
    def get_league(self, league_id: int) -> League:
        """Get details for a given league."""
        logger.info("Scraping league %s", league_id)
        data = self.get(f"leagues/{league_id}")["pageProps"]
        return League(
            id=int(data["details"]["id"]),
            name=data["details"]["name"],
            seasons=data["allAvailableSeasons"],
            country=data["details"].get("country"),
            gender=data["details"].get("gender"),
        )

    @cache  # noqa: B019
    def get_leagues(
        self,
        max_workers: int = 1,
        limit: int | None = None,
    ) -> list[League]:
        """Return the list of all leagues."""
        logger.info("Scraping all leagues")
        data = self.get("")["pageProps"]
        mapping = data["fallback"]["/api/translationmapping?locale=en"]
        all_ids = list(mapping["TournamentTemplates"] | mapping["TournamentPrefixes"])
        if limit is not None:
            all_ids = all_ids[:limit]

        def get_league_safely(league_id: int) -> League | None:
            try:
                return self.get_league(league_id)
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    return None
                raise  # Re-raise other errors

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return [
                result
                for result in executor.map(get_league_safely, all_ids)
                if result is not None
            ]

    @cache  # noqa: B019
    def get_match(self, match_id: int) -> Match:
        """Get all match for a given league and season."""
        logger.info("Scraping match %s", match_id)
        data = self.get(f"match/{match_id}")["pageProps"]
        if not data["content"]["shotmap"]["shots"]:
            raise NoShotsDataError
        return Match(
            id=int(data["general"]["matchId"]),
            league_id=int(data["general"]["parentLeagueId"] or 0),
            home_team=Team(
                id=int(data["general"]["homeTeam"]["id"]),
                name=data["general"]["homeTeam"]["name"],
            ),
            away_team=Team(
                id=int(data["general"]["awayTeam"]["id"]),
                name=data["general"]["awayTeam"]["name"],
            ),
            round=Round(
                match=data["general"].get("matchRound"),
                league=data["general"].get("leagueRoundName"),
            ),
            start_at=data["general"]["matchTimeUTCDate"],
            status=Status(
                started=data["header"]["status"]["started"],
                finished=data["header"]["status"]["finished"],
                cancelled=data["header"]["status"].get("cancelled", False),
                awarded=data["header"]["status"].get("awarded", False),
                period=Period(
                    slug=data["header"]["status"]["reason"]["longKey"],
                    name=data["header"]["status"]["reason"]["long"],
                ),
            ),
            score=Score(
                label=data["header"]["status"]["scoreStr"],
            ),
            penalties=[
                Penalty(
                    id=x["id"],
                    player_id=x["playerId"],
                    team_id=x["teamId"],
                    period=x["period"],
                    shot=Shot(
                        x=x["onGoalShot"]["x"],
                        y=x["onGoalShot"]["y"],
                        zoom=x["onGoalShot"]["zoomRatio"],
                    ),
                    outcome=x["eventType"],
                )
                for x in data["content"]["shotmap"]["shots"]
                if x["situation"] == "Penalty"
            ],
            players=[
                Player(
                    id=x["id"],
                    name=x["name"],
                    age=x.get("age"),
                    position=Position(
                        id=x.get("usualPlayingPositionId"),
                    ),
                    market_value=x.get("marketValue"),
                )
                for x in (
                    data["content"]["lineup"]["homeTeam"]["starters"]
                    + data["content"]["lineup"]["homeTeam"]["subs"]
                    + data["content"]["lineup"]["awayTeam"]["starters"]
                    + data["content"]["lineup"]["awayTeam"]["subs"]
                )
            ],
        )

    @cache  # noqa: B019
    def get_matches(
        self,
        league_id: int,
        season: str,
        max_workers: int = 1,
        limit: int | None = None,
    ) -> list[Match]:
        """Get all match for a given league and season."""
        logger.info("Scraping matches for league %s season %s", league_id, season)
        data = self.get(f"leagues/{league_id}", params={"season": season})["pageProps"]
        all_ids = [
            int(match["id"])
            for match in data["fixtures"]["allMatches"]
            if match["status"]["started"]
            and match["status"]["finished"]
            and FLOOR_DATETIME
            < datetime.fromisoformat(match["status"]["utcTime"])
            < CEIL_DATETIME
        ]
        if limit is not None:
            all_ids = all_ids[:limit]

        def get_match_safely(match_id: int) -> Match | None:
            try:
                return self.get_match(match_id)
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:  # noqa: PLR2004
                    return None
                raise
            except NoShotsDataError:
                return None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return [
                result
                for result in executor.map(get_match_safely, all_ids)
                if result is not None
            ]
