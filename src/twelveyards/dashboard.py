"""Streamlit dashboard logic.

PRD-v5: Two independent team dropdowns, no live FotMob. Reads
predictions.jsonl from the working tree. Drops load_upcoming_knockouts,
MatchContext, is_placeholder_team, _parse_kickoff_utc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .predict import PredictionRow


@dataclass(frozen=True)
class KickerPrediction:
    """One kicker's predicted side distribution, prepared for card rendering."""

    player_id: int
    player_name: str
    short_name: str
    team_id: int
    team_name: str
    kicking_foot: str
    photo_url: str
    total_penalties: int
    p_L: float  # noqa: N815
    p_C: float  # noqa: N815
    p_R: float  # noqa: N815
    recommended_dive: str


def predictions_for_match(
    predictions: Iterable[PredictionRow],
    home_team_id: int,
    away_team_id: int,
) -> tuple[list[KickerPrediction], list[KickerPrediction]]:
    """Filter predictions into (home, away) KickerPrediction lists."""
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


def recommended_dive(p_l: float, p_c: float, p_r: float) -> str:
    """Return the side the kicker is least likely to aim for (best dive choice)."""
    minimum = min(p_l, p_c, p_r)
    for side, value in (("L", p_l), ("C", p_c), ("R", p_r)):
        if value == minimum:
            return side
    return "L"


def opposite_side(side: str) -> str:
    """Return the opposite goal side (L↔R, C↔C)."""
    if side == "L":
        return "R"
    if side == "R":
        return "L"
    if side == "C":
        return "C"
    return side


def most_likely_side(p_l: float, p_c: float, p_r: float) -> str:
    """Return the side the kicker is most likely to aim for."""
    maximum = max(p_l, p_c, p_r)
    for side, value in (("L", p_l), ("C", p_c), ("R", p_r)):
        if value == maximum:
            return side
    return "L"


def distinct_teams(predictions: Iterable[PredictionRow]) -> list[tuple[int, str]]:
    """Return sorted list of distinct (team_id, team_name) pairs from predictions."""
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
