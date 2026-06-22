"""Tests for the 6 in-scope tournaments constant table."""

from __future__ import annotations

from penalty_pred.leagues import LEAGUE_BY_ID, LEAGUES, League


def test_leagues_count() -> None:
    """PRD: 6 in-scope tournaments."""
    assert len(LEAGUES) == 6


def test_leagues_have_required_fields() -> None:
    for league in LEAGUES:
        assert isinstance(league, League)
        assert league.league_id > 0
        assert league.slug
        assert league.name


def test_leagues_are_unique() -> None:
    ids = [league.league_id for league in LEAGUES]
    slugs = [league.slug for league in LEAGUES]
    assert len(set(ids)) == len(ids)
    assert len(set(slugs)) == len(slugs)


def test_league_ids_match_prd() -> None:
    """PRD: World Cup = 77, Euro = 50, Copa América = 44,
    Gold Cup = 298, Asian Cup = 290, AFCON = 289."""
    assert LEAGUE_BY_ID[77].slug == "world-cup"
    assert LEAGUE_BY_ID[50].slug == "euro"
    assert LEAGUE_BY_ID[44].slug == "copa-america"
    assert LEAGUE_BY_ID[298].slug == "concacaf-gold-cup"
    assert LEAGUE_BY_ID[290].slug == "afc-asian-cup"
    assert LEAGUE_BY_ID[289].slug == "africa-cup-of-nations"
