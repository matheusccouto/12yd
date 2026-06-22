"""The 6 in-scope tournaments (FotMob leagueId + SEO slug pairs).

PRD: The leagueId is the FotMob integer ID, e.g. World Cup = 77, Euro = 50,
Copa América = 44, Gold Cup = 298, Asian Cup = 290, AFCON = 289.

The player-history fetcher also needs slugs for leagues a player may have
played in *outside* the in-scope tournaments — e.g. LaLiga, Ligue 1,
Champions League, MLS, Copa del Rey. We keep a separate `EXTENDED_LEAGUES`
table for those, merged into `LEAGUE_BY_ID` for slug lookups. The shootout
scraper still uses the in-scope `LEAGUES` list for its `LEAGUE_SEASONS_PREDICT_WINDOW`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class League:
    league_id: int
    slug: str
    name: str


# The 6 in-scope tournaments for shootout scraping (PRD).
LEAGUES: tuple[League, ...] = (
    League(77, "world-cup", "World Cup"),
    League(50, "euro", "Euro"),
    League(44, "copa-america", "Copa América"),
    League(298, "concacaf-gold-cup", "CONCACAF Gold Cup"),
    League(290, "afc-asian-cup", "AFC Asian Cup"),
    League(289, "africa-cup-of-nations", "Africa Cup of Nations"),
)

# Leagues the player-history fetcher needs slugs for. This is a
# representative subset of the major domestic leagues and the
# continental cups — not a comprehensive list. Any player whose
# career only touches leagues not in this table will still produce
# rows for the leagues we DO have (the orchestrator skips lookups
# with an unknown leagueId).
EXTENDED_LEAGUES: tuple[League, ...] = (
    # Domestic
    League(87, "la-liga", "LaLiga"),
    League(53, "ligue-1", "Ligue 1"),
    League(47, "premier-league", "Premier League"),
    League(54, "bundesliga", "Bundesliga"),
    League(55, "serie-a", "Serie A"),
    League(130, "mls", "MLS"),
    # Continental cups
    League(42, "champions-league", "Champions League"),
    League(73, "europa-league", "Europa League"),
    League(41, "copa-libertadores", "Copa Libertadores"),
    # Domestic cups
    League(138, "copa-del-rey", "Copa del Rey"),
    League(133, "coupe-de-france", "Coupe de France"),
    League(132, "fa-cup", "FA Cup"),
    League(125, "dfb-pokal", "DFB-Pokal"),
    League(137, "coppa-italia", "Coppa Italia"),
    # International friendlies + qualifiers
    League(114, "friendlies", "Friendlies"),
    League(76, "uefa-euro-qualification", "UEFA Euro Qualifiers"),
    League(84, "fifa-world-cup-qualifiers-conmebol", "CONMEBOL WC Qualifiers"),
    # Club continental (Americas)
    League(292, "concacaf-champions-cup", "CONCACAF Champions Cup"),
    League(195, "leagues-cup", "Leagues Cup"),
)

# All leagues the player-history fetcher can look up, including the 6
# in-scope tournaments. The shootout scraper only iterates `LEAGUES`.
LEAGUE_BY_ID: dict[int, League] = {
    league.league_id: league for league in (*LEAGUES, *EXTENDED_LEAGUES)
}
