"""FotMob league IDs, slugs, and kinds the scraper targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

LeagueKind = Literal["international", "club", "domestic_only"]


@dataclass(frozen=True)
class League:
    """A FotMob league identifier: league_id, SEO slug, display name, and kind."""

    league_id: int
    slug: str
    name: str
    kind: LeagueKind = "domestic_only"


# The 6 in-scope international tournaments for shootout scraping (PRD).
LEAGUES: tuple[League, ...] = (
    League(77, "world-cup", "World Cup", kind="international"),
    League(50, "euro", "Euro", kind="international"),
    League(44, "copa-america", "Copa América", kind="international"),
    League(298, "concacaf-gold-cup", "CONCACAF Gold Cup", kind="international"),
    League(290, "afc-asian-cup", "AFC Asian Cup", kind="international"),
    League(289, "africa-cup-of-nations", "Africa Cup of Nations", kind="international"),
)

# The 7 in-scope club tournaments for shootout scraping (Phase 3, Issue #51).
# The schema is identical to the international shootout scraper (the same
# `pageProps.content.shotmap.shots` payload), so the existing
# `extract_shootout_kicks` runs unchanged. The training rows gain a
# `tournament_kind` attribute (per the ADR §"Schema — what changes") so
# the LOTO CV can group or slice by kind.
CLUB_LEAGUES: tuple[League, ...] = (
    League(41, "copa-libertadores", "Copa Libertadores", kind="club"),
    League(42, "champions-league", "Champions League", kind="club"),
    League(138, "copa-del-rey", "Copa del Rey", kind="club"),
    League(133, "coupe-de-france", "Coupe de France", kind="club"),
    League(132, "fa-cup", "FA Cup", kind="club"),
    League(125, "dfb-pokal", "DFB-Pokal", kind="club"),
    League(137, "coppa-italia", "Coppa Italia", kind="club"),
)

# Leagues the player-history fetcher needs slugs for. This is a
# representative subset of the major domestic leagues and the
# continental cups — not a comprehensive list. Any player whose
# career only touches leagues not in this table will still produce
# rows for the leagues we DO have (the orchestrator skips lookups
# with an unknown leagueId).
EXTENDED_LEAGUES: tuple[League, ...] = (
    # Domestic
    League(87, "la-liga", "LaLiga", kind="domestic_only"),
    League(53, "ligue-1", "Ligue 1", kind="domestic_only"),
    League(47, "premier-league", "Premier League", kind="domestic_only"),
    League(54, "bundesliga", "Bundesliga", kind="domestic_only"),
    League(55, "serie-a", "Serie A", kind="domestic_only"),
    League(130, "mls", "MLS", kind="domestic_only"),
    # Continental cups (player history only — the shootout scraper
    # uses `CLUB_LEAGUES` for Champions League + Copa Libertadores).
    League(73, "europa-league", "Europa League", kind="domestic_only"),
    # International friendlies + qualifiers
    League(114, "friendlies", "Friendlies", kind="domestic_only"),
    League(76, "uefa-euro-qualification", "UEFA Euro Qualifiers", kind="domestic_only"),
    League(
        84, "fifa-world-cup-qualifiers-conmebol", "CONMEBOL WC Qualifiers", kind="domestic_only",
    ),
    # Club continental (Americas)
    League(292, "concacaf-champions-cup", "CONCACAF Champions Cup", kind="domestic_only"),
    League(195, "leagues-cup", "Leagues Cup", kind="domestic_only"),
)

# All leagues the player-history fetcher can look up, including the 6
# international and 7 club tournaments. The shootout scraper filters
# to `kind in {"international", "club"}` (i.e. the union of `LEAGUES`
# and `CLUB_LEAGUES`).
LEAGUE_BY_ID: dict[int, League] = {
    league.league_id: league for league in (*LEAGUES, *CLUB_LEAGUES, *EXTENDED_LEAGUES)
}

# Convenience sets used by the shootout scraper and the tournament
# scope filters. The shootout scraper's filter is
# `kind in {"international", "club"}`; these sets make the intent
# explicit at the call site and the test surface.
INTERNATIONAL_LEAGUE_IDS: frozenset[int] = frozenset(league.league_id for league in LEAGUES)
CLUB_LEAGUE_IDS: frozenset[int] = frozenset(league.league_id for league in CLUB_LEAGUES)

WC_2026_LEAGUE: League = LEAGUE_BY_ID[77]
WC_2026_SEASON: int = 2026
