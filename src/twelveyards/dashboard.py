"""Streamlit dashboard logic.

PRD-v5: Two independent team dropdowns, no live FotMob. Reads
predictions.jsonl from the working tree. Drops load_upcoming_knockouts,
MatchContext, is_placeholder_team, _parse_kickoff_utc.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .predict import PredictionRow


@dataclass(frozen=True)
class KickerPrediction:
    player_id: int
    player_name: str
    short_name: str
    team_id: int
    team_name: str
    kicking_foot: str
    photo_url: str
    total_penalties: int
    p_L: float
    p_C: float
    p_R: float
    recommended_dive: str


def predictions_for_match(
    predictions: Iterable[PredictionRow],
    home_team_id: int,
    away_team_id: int,
) -> tuple[list[KickerPrediction], list[KickerPrediction]]:
    home_rows: list[KickerPrediction] = []
    away_rows: list[KickerPrediction] = []
    for r in predictions:
        pred = KickerPrediction(
            player_id=r.player_id,
            player_name=r.player_name,
            short_name=r.short_name,
            team_id=r.team_id,
            team_name=r.team_name,
            kicking_foot=r.kicking_foot,
            photo_url=r.photo_url,
            total_penalties=r.total_penalties,
            p_L=r.p_L,
            p_C=r.p_C,
            p_R=r.p_R,
            recommended_dive=recommended_dive(r.p_L, r.p_C, r.p_R),
        )
        if r.team_id == home_team_id:
            home_rows.append(pred)
        elif r.team_id == away_team_id:
            away_rows.append(pred)
    home_rows.sort(key=lambda k: (-k.total_penalties, k.player_name))
    away_rows.sort(key=lambda k: (-k.total_penalties, k.player_name))
    return home_rows, away_rows


def recommended_dive(p_L: float, p_C: float, p_R: float) -> str:
    minimum = min(p_L, p_C, p_R)
    for side, value in (("L", p_L), ("C", p_C), ("R", p_R)):
        if value == minimum:
            return side
    return "L"


def opposite_side(side: str) -> str:
    if side == "L":
        return "R"
    if side == "R":
        return "L"
    if side == "C":
        return "C"
    return side


def most_likely_side(p_L: float, p_C: float, p_R: float) -> str:
    maximum = max(p_L, p_C, p_R)
    for side, value in (("L", p_L), ("C", p_C), ("R", p_R)):
        if value == maximum:
            return side
    return "L"


def distinct_teams(predictions: Iterable[PredictionRow]) -> list[tuple[int, str]]:
    seen: set[int] = set()
    teams: list[tuple[int, str]] = []
    for r in predictions:
        if r.team_id not in seen:
            seen.add(r.team_id)
            teams.append((r.team_id, r.team_name))
    teams.sort(key=lambda t: t[1])
    return teams


__all__ = [
    "KickerPrediction",
    "distinct_teams",
    "most_likely_side",
    "opposite_side",
    "predictions_for_match",
    "recommended_dive",
]
