"""Centralised test data factories for the twelveyards test suite."""

from __future__ import annotations

from twelveyards.artifacts import (
    PlayerMetadata,
    PlayerPenalty,
    PredictionRow,
    RosterPlayer,
)

# ---------------------------------------------------------------------------
# PlayerPenalty (one row of player_history.jsonl)
# ---------------------------------------------------------------------------


def make_history_row(  # noqa: PLR0913
    match_id: int = 1,
    match_date: str = "2024-06-01T00:00:00+00:00",
    *,
    side: str = "L",
    shot_type: str = "RightFoot",
    kicker_id: int = 1,
    league_id: int = 77,
    league_name: str = "World Cup",
    team_id: int = 100,
    is_home: bool = True,
    x: float = 0.5,
    is_on_target: bool = True,
    outcome: str = "Goal",
) -> PlayerPenalty:
    return PlayerPenalty(
        kicker_id=kicker_id,
        match_id=match_id,
        match_date=match_date,
        league_id=league_id,
        league_name=league_name,
        team_id=team_id,
        is_home=is_home,
        x=x,
        side=side,
        is_on_target=is_on_target,
        outcome=outcome,
        shot_type=shot_type,
    )


# ---------------------------------------------------------------------------
# PlayerMetadata
# ---------------------------------------------------------------------------


def make_metadata(
    player_id: int = 1,
    *,
    player_name: str = "Alpha",
    position_key: str = "striker",
    birth_date: str = "1995-01-01",
    preferred_foot: str = "",
) -> PlayerMetadata:
    return PlayerMetadata(
        player_id=player_id,
        player_name=player_name,
        position_key=position_key,
        birth_date=birth_date,
        preferred_foot=preferred_foot,
    )


# ---------------------------------------------------------------------------
# RosterPlayer
# ---------------------------------------------------------------------------


def make_roster_player(
    player_id: int = 1,
    *,
    player_name: str = "Alpha",
    team_id: int = 100,
    team_name: str = "Argentina",
    country_code: str = "ARG",
) -> RosterPlayer:
    return RosterPlayer(
        player_id=player_id,
        player_name=player_name,
        team_id=team_id,
        team_name=team_name,
        country_code=country_code,
    )


# ---------------------------------------------------------------------------
# PredictionRow (one row of predictions.jsonl) — for test helpers
# ---------------------------------------------------------------------------


def make_prediction_row(  # noqa: PLR0913
    player_id: int = 1,
    *,
    player_name: str = "Alpha",
    short_name: str = "Alpha",
    team_id: int = 100,
    team_name: str = "Argentina",
    country_code: str = "ARG",
    kicking_foot: str = "right",
    photo_url: str = "https://images.fotmob.com/image_resources/playerimages/1.png",
    p_l: float = 0.5,
    p_c: float = 0.25,
    p_r: float = 0.25,
    total_penalties: int = 5,
) -> PredictionRow:
    return PredictionRow(
        player_id=player_id,
        player_name=player_name,
        short_name=short_name,
        team_id=team_id,
        team_name=team_name,
        country_code=country_code,
        kicking_foot=kicking_foot,
        photo_url=photo_url,
        p_L=p_l,
        p_C=p_c,
        p_R=p_r,
        total_penalties=total_penalties,
    )
