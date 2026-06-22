"""The 6 in-scope tournaments (FotMob leagueId + SEO slug pairs).

PRD: The leagueId is the FotMob integer ID, e.g. World Cup = 77, Euro = 50,
Copa América = 44, Gold Cup = 298, Asian Cup = 290, AFCON = 289.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class League:
    league_id: int
    slug: str
    name: str


LEAGUES: tuple[League, ...] = (
    League(77, "world-cup", "World Cup"),
    League(50, "euro", "Euro"),
    League(44, "copa-america", "Copa América"),
    League(298, "concacaf-gold-cup", "CONCACAF Gold Cup"),
    League(290, "afc-asian-cup", "AFC Asian Cup"),
    League(289, "africa-cup-of-nations", "Africa Cup of Nations"),
)

LEAGUE_BY_ID: dict[int, League] = {league.league_id: league for league in LEAGUES}
