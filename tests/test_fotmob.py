"""Tests for the FotMob API client, data models, and targeted leagues list."""

from __future__ import annotations

from datetime import UTC, datetime

from twelveyards.fotmob.client import FotMob
from twelveyards.fotmob.leagues import (
    LEAGUE_BY_ID,
    LEAGUES,
)
from twelveyards.fotmob.models import (
    League,
    LeagueDetails,
    Match,
    Match,
    Season,
    get_player_position_from_lineup,
)

# ---------------------------------------------------------------------------
# Leagues Catalog Tests
# ---------------------------------------------------------------------------


def test_leagues_count() -> None:
    """The 6 in-scope international tournaments."""
    assert len(LEAGUES) == 6


def test_leagues_have_required_fields() -> None:
    for league in LEAGUES:
        assert isinstance(league, League)
        assert league.league_id > 0
        assert league.slug
        assert league.name
        assert league.kind == "international"


def test_leagues_are_unique() -> None:
    ids = [league.league_id for league in LEAGUES]
    slugs = [league.slug for league in LEAGUES]
    assert len(set(ids)) == len(ids)
    assert len(set(slugs)) == len(slugs)


def test_league_ids_match_prd() -> None:
    assert LEAGUE_BY_ID[77].slug == "world-cup"
    assert LEAGUE_BY_ID[50].slug == "euro"
    assert LEAGUE_BY_ID[44].slug == "copa-america"
    assert LEAGUE_BY_ID[298].slug == "concacaf-gold-cup"
    assert LEAGUE_BY_ID[290].slug == "afc-asian-cup"
    assert LEAGUE_BY_ID[289].slug == "africa-cup-of-nations"


# ---------------------------------------------------------------------------
# Data Models Parsing Tests
# ---------------------------------------------------------------------------


def test_get_player_position_from_lineup() -> None:
    # Test usualPlayingPositionId
    assert get_player_position_from_lineup({"usualPlayingPositionId": 0}) == "Keeper"
    assert get_player_position_from_lineup({"usualPlayingPositionId": 1}) == "Defender"
    assert get_player_position_from_lineup({"usualPlayingPositionId": 2}) == "Midfielder"
    assert get_player_position_from_lineup({"usualPlayingPositionId": 3}) == "Forward"

    # Test positionId fallback
    assert get_player_position_from_lineup({"positionId": 11}) == "Keeper"
    assert get_player_position_from_lineup({"positionId": 34}) == "Defender"
    assert get_player_position_from_lineup({"positionId": 75}) == "Midfielder"
    assert get_player_position_from_lineup({"positionId": 105}) == "Forward"

    # Missing
    assert get_player_position_from_lineup({}) == ""


def test_league_model() -> None:
    l1 = League.model_validate({"id": 77, "slug": "world-cup", "name": "World Cup"})
    assert l1.league_id == 77
    assert l1.slug == "world-cup"
    assert l1.kind == "domestic_only"


def test_season_model() -> None:
    s = Season.model_validate({"seasonName": "2022 Qatar"})
    assert s.season_name == "2022 Qatar"


def test_match_ref_model() -> None:
    fixture = {
        "pageUrl": "/matches/argentina-vs-france/1hox8a#3370572",
        "roundName": "Final",
        "home": {"id": "6706", "name": "Argentina"},
        "away": {"id": "9825", "name": "France"},
        "status": {
            "utcTime": "2022-12-18T15:00:00Z",
            "scoreStr": "3 - 3",
            "reason": {"shortKey": "penalties_short"},
        },
    }
    ref = Match.model_validate(fixture)
    assert ref.match_id == "3370572"
    assert ref.home_id == 6706
    assert ref.home_name == "Argentina"
    assert ref.away_id == 9825
    assert ref.away_name == "France"
    assert ref.round_name == "Final"
    assert ref.match_date == datetime(2022, 12, 18, 15, 0, tzinfo=UTC)
    assert ref.score_str == "3 - 3"
    assert ref.is_shootout is True


def test_match_model_parsing() -> None:
    payload = {
        "pageProps": {
            "general": {
                "matchId": 3370572,
                "leagueId": 77,
                "matchTimeUTC": "2022-12-18T15:00:00.000Z",
                "homeTeam": {"id": 6706},
                "awayTeam": {"id": 6723},
            },
            "content": {
                "shotmap": {
                    "shots": [
                        {
                            "playerId": 701154,
                            "teamId": 6723,
                            "situation": "Penalty",
                            "period": "PenaltyShootout",
                            "onGoalShot": {"x": 0.15, "y": 0.25},
                            "eventType": "Goal",
                            "shotType": "RightFoot",
                        },
                    ],
                },
                "lineup": {
                    "homeTeam": {
                        "starters": [
                            {"id": 268375, "name": "E. Martinez", "positionId": 11},
                        ],
                        "subs": [],
                    },
                    "awayTeam": {
                        "starters": [
                            {"id": 701154, "name": "K. Mbappe", "usualPlayingPositionId": 3},
                        ],
                        "subs": [],
                    },
                },
            },
        },
    }
    match = Match.model_validate(payload)
    assert match.match_id == "3370572"
    assert match.league_id == 77
    assert match.home_team_id == 6706
    assert match.away_team_id == 6723
    assert len(match.shotmap) == 1
    assert match.shotmap[0].player_id == 701154
    assert match.shotmap[0].team_id == 6723
    assert match.shotmap[0].outcome == "Goal"
    assert match.shotmap[0].x == 0.15
    assert match.shotmap[0].y == 0.25
    assert match.player_positions == {268375: "Keeper", 701154: "Forward"}
    assert match.player_teams == {268375: 6706, 701154: 6723}


# ---------------------------------------------------------------------------
# Real API Client Live Integration Tests (No Mocks)
# ---------------------------------------------------------------------------


def test_discover_build_id() -> None:
    client = FotMob()
    assert len(client.build_id) > 0


def test_get_league() -> None:
    client = FotMob()
    details = client.get_league_details(77)
    assert isinstance(details, LeagueDetails)
    assert details.id == 77
    assert details.name == "World Cup"
    assert details.seopath == "world-cup"


def test_get_league_seasons() -> None:
    client = FotMob()
    seasons = client.get_league_seasons(77)
    assert len(seasons) > 0
    assert all(isinstance(s, Season) for s in seasons)
    assert any(s.season_name == "2022 Qatar" for s in seasons)


def test_get_league_matches() -> None:
    client = FotMob()
    matches = client.get_league_matches(77, "2022 Qatar")
    assert len(matches) > 0
    assert isinstance(matches[0], Match)
    assert any(m.match_id == "3370572" for m in matches)


def test_get_match() -> None:
    client = FotMob()
    match = client.get_match("3370572")
    assert isinstance(match, Match)
    assert match.match_id == "3370572"
    assert match.home_team_id == 6706
    assert match.away_team_id == 6723

