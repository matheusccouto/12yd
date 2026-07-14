"""FotMob league IDs, slugs, and kinds targeted by the shootout ingest."""

from __future__ import annotations

from .models import League

LEAGUES: tuple[League, ...] = (
    League.model_validate(
        {
            "id": 77,
            "slug": "world-cup",
            "name": "World Cup",
            "kind": "international",
        },
    ),
    League.model_validate(
        {
            "id": 50,
            "slug": "euro",
            "name": "Euro",
            "kind": "international",
        },
    ),
    League.model_validate(
        {
            "id": 44,
            "slug": "copa-america",
            "name": "Copa América",
            "kind": "international",
        },
    ),
    League.model_validate(
        {
            "id": 298,
            "slug": "concacaf-gold-cup",
            "name": "CONCACAF Gold Cup",
            "kind": "international",
        },
    ),
    League.model_validate(
        {
            "id": 290,
            "slug": "afc-asian-cup",
            "name": "AFC Asian Cup",
            "kind": "international",
        },
    ),
    League.model_validate(
        {
            "id": 289,
            "slug": "africa-cup-of-nations",
            "name": "Africa Cup of Nations",
            "kind": "international",
        },
    ),
)

LEAGUE_BY_ID: dict[int, League] = {
    league.league_id: league for league in LEAGUES
}

WC_2026_LEAGUE: League = LEAGUE_BY_ID[77]
WC_2026_SEASON: int = 2026
