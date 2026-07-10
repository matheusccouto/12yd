"""Tests for the v5 dashboard library logic.

v5 changes:
- predictions_for_match takes home_team_id, away_team_id (drops MatchContext)
- drops load_upcoming_knockouts, is_placeholder_team
- KickerPrediction has v5 fields: short_name, photo_url, total_penalties
"""

from __future__ import annotations

import pytest

from twelveyards.dashboard import (
    KickerPrediction,
    distinct_teams,
    most_likely_side,
    opposite_side,
    predictions_for_match,
    recommended_dive,
)
from twelveyards.predict import PredictionRow


def _pred(
    *,
    player_id: int = 1,
    player_name: str = "A",
    team_id: int = 100,
    team_name: str = "Team A",
    p_L: float = 0.5,
    p_C: float = 0.2,
    p_R: float = 0.3,
    total_penalties: int = 0,
    short_name: str = "A",
) -> PredictionRow:
    return PredictionRow(
        player_id=player_id,
        player_name=player_name,
        short_name=short_name,
        team_id=team_id,
        team_name=team_name,
        country_code="",
        kicking_foot="right",
        photo_url=f"https://images.fotmob.com/image_resources/playerimages/{player_id}.png",
        p_L=p_L,
        p_C=p_C,
        p_R=p_R,
        total_penalties=total_penalties,
    )


# ---------------------------------------------------------------------------
# predictions_for_match
# ---------------------------------------------------------------------------


def test_predictions_for_match_splits_home_away() -> None:
    home_id = 100
    away_id = 200
    predictions = [
        _pred(player_id=1, player_name="H1", team_id=home_id),
        _pred(player_id=2, player_name="H2", team_id=home_id),
        _pred(player_id=3, player_name="A1", team_id=away_id),
        _pred(player_id=4, player_name="X", team_id=999),
    ]
    home, away = predictions_for_match(predictions, home_id, away_id)
    assert [k.player_id for k in home] == [1, 2]
    assert [k.player_id for k in away] == [3]


def test_predictions_for_match_sets_recommended_dive() -> None:
    predictions = [_pred(player_id=1, team_id=100, p_L=0.1, p_C=0.6, p_R=0.3)]
    home, _away = predictions_for_match(predictions, 100, 200)
    assert home[0].recommended_dive == "L"


def test_predictions_for_match_sorts_by_total_penalties_desc() -> None:
    predictions = [
        _pred(player_id=1, player_name="Zara", team_id=100, total_penalties=2),
        _pred(player_id=2, player_name="Aaron", team_id=100, total_penalties=8),
        _pred(player_id=3, player_name="Mike", team_id=100, total_penalties=5),
    ]
    home, _away = predictions_for_match(predictions, 100, 200)
    assert [k.player_name for k in home] == ["Aaron", "Mike", "Zara"]


def test_predictions_for_match_name_tiebreaker() -> None:
    predictions = [
        _pred(player_id=1, player_name="Zara", team_id=100, total_penalties=3),
        _pred(player_id=2, player_name="Aaron", team_id=100, total_penalties=3),
    ]
    home, _away = predictions_for_match(predictions, 100, 200)
    assert [k.player_name for k in home] == ["Aaron", "Zara"]


def test_predictions_for_match_falls_back_to_name_sort_with_zero_penalties() -> None:
    predictions = [
        _pred(player_id=1, player_name="Zara", team_id=100),
        _pred(player_id=2, player_name="Aaron", team_id=100),
        _pred(player_id=3, player_name="Mike", team_id=100),
    ]
    home, _away = predictions_for_match(predictions, 100, 200)
    assert [k.player_name for k in home] == ["Aaron", "Mike", "Zara"]


def test_predictions_for_match_returns_empty_when_no_matching_teams() -> None:
    predictions = [_pred(player_id=1, team_id=999)]
    home, away = predictions_for_match(predictions, 100, 200)
    assert home == []
    assert away == []


def test_predictions_for_match_empty_predictions() -> None:
    home, away = predictions_for_match([], 100, 200)
    assert home == []
    assert away == []


def test_predictions_for_match_kicker_prediction_has_v5_fields() -> None:
    predictions = [
        _pred(
            player_id=42,
            player_name="Lionel Messi",
            short_name="Messi",
            team_id=100,
            total_penalties=10,
        ),
    ]
    home, _away = predictions_for_match(predictions, 100, 200)
    k = home[0]
    assert k.player_id == 42
    assert k.player_name == "Lionel Messi"
    assert k.short_name == "Messi"
    assert k.total_penalties == 10
    assert "images.fotmob.com" in k.photo_url
    assert k.recommended_dive in ("L", "C", "R")


# ---------------------------------------------------------------------------
# KickerPrediction
# ---------------------------------------------------------------------------


def test_kicker_prediction_construction() -> None:
    k = KickerPrediction(
        player_id=1,
        player_name="A",
        short_name="A",
        team_id=100,
        team_name="T",
        kicking_foot="right",
        photo_url="http://x.com/1.png",
        total_penalties=5,
        p_L=0.4,
        p_C=0.3,
        p_R=0.3,
        recommended_dive="C",
    )
    assert k.recommended_dive == "C"
    assert k.total_penalties == 5


# ---------------------------------------------------------------------------
# recommended_dive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("p_L", "p_C", "p_R", "expected"),
    [
        (0.1, 0.5, 0.4, "L"),
        (0.4, 0.1, 0.5, "C"),
        (0.3, 0.6, 0.1, "R"),
        (0.33, 0.33, 0.34, "L"),
        (0.34, 0.33, 0.33, "C"),
        (0.33, 0.34, 0.33, "L"),
        (0.33, 0.33, 0.33, "L"),
        (1.0, 0.0, 0.0, "C"),
        (0.0, 1.0, 0.0, "L"),
        (0.0, 0.0, 1.0, "L"),
    ],
)
def test_recommended_dive(p_L: float, p_C: float, p_R: float, expected: str) -> None:
    assert recommended_dive(p_L, p_C, p_R) == expected


# ---------------------------------------------------------------------------
# opposite_side
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("side", "expected"),
    [("L", "R"), ("R", "L"), ("C", "C")],
)
def test_opposite_side(side: str, expected: str) -> None:
    assert opposite_side(side) == expected


def test_opposite_side_unknown_passthrough() -> None:
    assert opposite_side("?") == "?"


# ---------------------------------------------------------------------------
# most_likely_side
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("p_L", "p_C", "p_R", "expected"),
    [
        (0.55, 0.20, 0.25, "L"),
        (0.30, 0.25, 0.45, "R"),
        (0.20, 0.60, 0.20, "C"),
        (0.33, 0.33, 0.34, "R"),
        (0.34, 0.33, 0.33, "L"),
        (0.33, 0.34, 0.33, "C"),
        (0.33, 0.33, 0.33, "L"),
    ],
)
def test_most_likely_side(p_L: float, p_C: float, p_R: float, expected: str) -> None:
    assert most_likely_side(p_L, p_C, p_R) == expected


# ---------------------------------------------------------------------------
# distinct_teams
# ---------------------------------------------------------------------------


def test_distinct_teams_empty() -> None:
    assert distinct_teams([]) == []


def test_distinct_teams_deduplicates() -> None:
    predictions = [
        _pred(player_id=1, team_id=100, team_name="Argentina"),
        _pred(player_id=2, team_id=100, team_name="Argentina"),
        _pred(player_id=3, team_id=200, team_name="Brazil"),
    ]
    teams = distinct_teams(predictions)
    assert teams == [(100, "Argentina"), (200, "Brazil")]


def test_distinct_teams_sorted_by_name() -> None:
    predictions = [
        _pred(player_id=1, team_id=100, team_name="Canada"),
        _pred(player_id=2, team_id=200, team_name="Brazil"),
        _pred(player_id=3, team_id=300, team_name="Argentina"),
    ]
    teams = distinct_teams(predictions)
    assert [t[1] for t in teams] == ["Argentina", "Brazil", "Canada"]


def test_distinct_teams_single_team() -> None:
    predictions = [_pred(player_id=1, team_id=100, team_name="France")]
    assert distinct_teams(predictions) == [(100, "France")]
