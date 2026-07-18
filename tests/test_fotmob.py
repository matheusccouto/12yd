"""Tests for the FotMob API client and data models."""

from twelveyards.fotmob.client import FotMob
from twelveyards.fotmob.models import League, Match, Score

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def test_score_parses_label() -> None:
    s = Score(label=" 3 - 5 ")
    assert s.home == 3
    assert s.away == 5


def test_score_handles_none_label() -> None:
    s = Score(label=None)
    assert s.home is None
    assert s.away is None


def test_score_handles_malformed_label() -> None:
    s = Score(label="N/A")
    assert s.home is None
    assert s.away is None


# ---------------------------------------------------------------------------
# Live API (integration)
# ---------------------------------------------------------------------------


def test_discover_build_id() -> None:
    client = FotMob()
    assert len(client.build_id) > 0


def test_get_league() -> None:
    client = FotMob()
    league = client.get_league(77)
    assert isinstance(league, League)
    assert league.id == 77
    assert league.name == "World Cup"


def test_get_match() -> None:
    client = FotMob()
    match = client.get_match(3370572)
    assert isinstance(match, Match)
    assert match.id == 3370572
    assert match.league_id == 77
    assert match.home_team.name == "Argentina"
    assert match.away_team.name == "France"
    assert match.score.label == "3 - 3"
