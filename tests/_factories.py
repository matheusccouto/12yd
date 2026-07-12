"""Centralised test data factories for the v5 twelveyards test suite.

v5 drops TrainingRow, ShootoutKick, and the old modules (model.py,
evaluate.py, shootouts.py, rsssf.py, validate.py). Surviving builders:

- make_history_row — a PlayerPenalty (one row of player_history.jsonl)
- make_metadata — a PlayerMetadata (per-player data for features)
- make_roster_player — a RosterPlayer (one WC squad entry)

Schema constants are derived from dataclasses.fields() so adding a field
is a one-line change here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields as _fields

from twelveyards.player_history import PlayerMetadata, PlayerPenalty
from twelveyards.predict import PredictionRow
from twelveyards.rosters import RosterPlayer


class FakeFotMobClient:
    """Stub FotMobClient that satisfies the FotMobClientLike protocol."""

    def get(self, path: str, params: Mapping[str, str] | None = None) -> object:
        raise NotImplementedError("tests must monkeypatch the inner fetcher")

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

PLAYER_PENALTY_FIELDS: frozenset[str] = frozenset(f.name for f in _fields(PlayerPenalty))
PLAYER_METADATA_FIELDS: frozenset[str] = frozenset(f.name for f in _fields(PlayerMetadata))
ROSTER_PLAYER_FIELDS: frozenset[str] = frozenset(f.name for f in _fields(RosterPlayer))
PREDICTION_ROW_FIELDS: frozenset[str] = frozenset(f.name for f in _fields(PredictionRow))


# ---------------------------------------------------------------------------
# PlayerPenalty (one row of player_history.jsonl)
# ---------------------------------------------------------------------------


def make_history_row(
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


def make_prediction_row(
    player_id: int = 1,
    *,
    player_name: str = "Alpha",
    short_name: str = "Alpha",
    team_id: int = 100,
    team_name: str = "Argentina",
    country_code: str = "ARG",
    kicking_foot: str = "right",
    photo_url: str = "https://images.fotmob.com/image_resources/playerimages/1.png",
    p_L: float = 0.5,
    p_C: float = 0.25,
    p_R: float = 0.25,
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
        p_L=p_L,
        p_C=p_C,
        p_R=p_R,
        total_penalties=total_penalties,
    )
