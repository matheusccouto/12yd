"""Tests for the FotMob client."""

from twelveyards.fotmob.client import FotMob


def test_get_league() -> None:
    """Test getting a league by ID."""
    client = FotMob()
    league = client.get_league(77)
    assert league.id == 77
    assert league.name == "World Cup"


def test_get_match() -> None:
    """Test getting a match by ID."""
    client = FotMob()
    match = client.get_match(3370572)
    assert match.id == 3370572
    assert match.league_id == 77
    assert match.home_team.name == "Argentina"
    assert match.away_team.name == "France"
    assert match.score.label == "3 - 3"
