"""
Tests for the 6 in-scope international + 7 in-scope club tournaments
constant table.

Phase 3 (Issue #51): the scope extended from 6 international tournaments
to 13 (6 international + 7 club). The `League` dataclass gained a
`kind` field that discriminates the three disjoint tuples (`LEAGUES`,
`CLUB_LEAGUES`, `EXTENDED_LEAGUES`). The shootout scraper filters
to `kind in {"international", "club"}`; the player-history fetcher
uses the full `LEAGUE_BY_ID` for slug lookups.
"""

from __future__ import annotations

from twelveyards.fotmob.leagues import (
    CLUB_LEAGUE_IDS,
    CLUB_LEAGUES,
    EXTENDED_LEAGUES,
    INTERNATIONAL_LEAGUE_IDS,
    LEAGUE_BY_ID,
    LEAGUES,
    League,
)


def test_leagues_count() -> None:
    """Phase 3 (Issue #51): the 6 in-scope international tournaments."""
    assert len(LEAGUES) == 6


def test_leagues_have_required_fields() -> None:
    for league in LEAGUES:
        assert isinstance(league, League)
        assert league.league_id > 0
        assert league.slug
        assert league.name
        assert league.kind == "international", (
            f"international league {league.league_id} has kind={league.kind!r}"
        )


def test_leagues_are_unique() -> None:
    ids = [league.league_id for league in LEAGUES]
    slugs = [league.slug for league in LEAGUES]
    assert len(set(ids)) == len(ids)
    assert len(set(slugs)) == len(slugs)


def test_league_ids_match_prd() -> None:
    """
    PRD: World Cup = 77, Euro = 50, Copa América = 44,
    Gold Cup = 298, Asian Cup = 290, AFCON = 289.
    """
    assert LEAGUE_BY_ID[77].slug == "world-cup"
    assert LEAGUE_BY_ID[50].slug == "euro"
    assert LEAGUE_BY_ID[44].slug == "copa-america"
    assert LEAGUE_BY_ID[298].slug == "concacaf-gold-cup"
    assert LEAGUE_BY_ID[290].slug == "afc-asian-cup"
    assert LEAGUE_BY_ID[289].slug == "africa-cup-of-nations"


# --- Phase 3 club leagues (Issue #51) --------------------------------------


def test_club_leagues_count() -> None:
    """Phase 3 (Issue #51): the 7 in-scope club tournaments."""
    assert len(CLUB_LEAGUES) == 7


def test_club_leagues_have_required_fields() -> None:
    for league in CLUB_LEAGUES:
        assert isinstance(league, League)
        assert league.league_id > 0
        assert league.slug
        assert league.name
        assert league.kind == "club", (
            f"club league {league.league_id} has kind={league.kind!r}, expected 'club'"
        )


def test_club_leagues_are_unique() -> None:
    ids = [league.league_id for league in CLUB_LEAGUES]
    slugs = [league.slug for league in CLUB_LEAGUES]
    assert len(set(ids)) == len(ids)
    assert len(set(slugs)) == len(slugs)


def test_club_league_ids_match_prd() -> None:
    """
    Phase 3 (Issue #51) + ADR
    `docs/adr/0004-phase-3-data-source.md`: Copa Libertadores = 41,
    Champions League = 42, FA Cup = 132, Coupe de France = 133,
    DFB-Pokal = 125, Coppa Italia = 137, Copa del Rey = 138.
    """
    assert LEAGUE_BY_ID[41].slug == "copa-libertadores"
    assert LEAGUE_BY_ID[42].slug == "champions-league"
    assert LEAGUE_BY_ID[132].slug == "fa-cup"
    assert LEAGUE_BY_ID[133].slug == "coupe-de-france"
    assert LEAGUE_BY_ID[125].slug == "dfb-pokal"
    assert LEAGUE_BY_ID[137].slug == "coppa-italia"
    assert LEAGUE_BY_ID[138].slug == "copa-del-rey"


def test_club_league_ids_constant() -> None:
    """`CLUB_LEAGUE_IDS` is the 7-id set of in-scope club leagues."""
    assert frozenset({41, 42, 125, 132, 133, 137, 138}) == CLUB_LEAGUE_IDS


def test_international_league_ids_constant() -> None:
    """
    `INTERNATIONAL_LEAGUE_IDS` is the 6-id set of in-scope
    international leagues (the existing `LEAGUES` IDs).
    """
    assert frozenset({league.league_id for league in LEAGUES}) == INTERNATIONAL_LEAGUE_IDS


# --- `kind` field on every League (Issue #51) -----------------------------


def test_every_league_has_a_kind() -> None:
    """
    The `kind` field on `League` is one of `"international"`,
    `"club"`, or `"domestic_only"`. A future kind (e.g. `"youth"`)
    needs a deliberate update here so the discrimination is
    explicit at the type level.
    """
    for league in LEAGUE_BY_ID.values():
        assert league.kind in ("international", "club", "domestic_only"), (
            f"league {league.league_id} ({league.name}) has unknown kind={league.kind!r}"
        )


def test_every_league_kind_is_a_literal_value() -> None:
    """
    Type-system-level check: the `kind` field is a `LeagueKind`
    literal. A drift (e.g. someone sets `kind="International"` with
    a capital I) is caught here at runtime.
    """
    valid_kinds = ("international", "club", "domestic_only")
    for league in LEAGUE_BY_ID.values():
        assert league.kind in valid_kinds, (
            f"league {league.league_id} has non-literal kind={league.kind!r}"
        )


def test_extended_leagues_are_domestic_only() -> None:
    """
    The 12 `EXTENDED_LEAGUES` (LaLiga, Ligue 1, Premier League,
    etc.) are all `kind="domestic_only"`. A regression that marks
    one of them as `international` or `club` is caught here — that
    would put the league in scope and break the test
    `test_scope_excludes_domestic_only_extended_leagues` in
    `test_tournaments.py`.
    """
    assert len(EXTENDED_LEAGUES) == 12
    for league in EXTENDED_LEAGUES:
        assert league.kind == "domestic_only", (
            f"extended league {league.league_id} ({league.name}) has "
            f"kind={league.kind!r}, expected 'domestic_only'"
        )


def test_in_scope_and_extended_leagues_are_disjoint() -> None:
    """
    The in-scope leagues (`LEAGUES` + `CLUB_LEAGUES`) are
    disjoint from `EXTENDED_LEAGUES`. The 7 club leagues that
    used to live in `EXTENDED_LEAGUES` (Copa Libertadores,
    Champions League, FA Cup, Coupe de France, DFB-Pokal, Coppa
    Italia, Copa del Rey) have moved to `CLUB_LEAGUES`.
    """
    in_scope_ids = {league.league_id for league in LEAGUES} | {
        league.league_id for league in CLUB_LEAGUES
    }
    extended_ids = {league.league_id for league in EXTENDED_LEAGUES}
    assert in_scope_ids.isdisjoint(extended_ids)
