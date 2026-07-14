"""Pydantic data models for FotMob API responses."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import AliasChoices, BaseModel, Field, model_validator

SHOTMAP_EVENT_TYPE_TO_OUTCOME: dict[str, str] = {
    "Goal": "Goal",
    "AttemptSaved": "Saved",
    "Miss": "Missed",
    "Post": "Missed",
}


def parse_match_date(value: Any) -> datetime:
    """Parse FotMob date strings into UTC timezone-aware datetime."""
    from email.utils import parsedate_to_datetime  # noqa: PLC0415
    if not value:
        return datetime.now(UTC)
    text = str(value)
    try:
        return parsedate_to_datetime(text).astimezone(UTC)
    except (TypeError, ValueError):
        pass
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).astimezone(UTC)
    except ValueError as err:
        msg = f"Could not parse match date: {text!r}"
        raise ValueError(msg) from err


def get_player_position_from_lineup(  # noqa: C901, PLR0911
    player: dict[str, Any],
) -> str:
    """Derive position key from lineup player dict.

    Uses usualPlayingPositionId or positionId.
    """
    usual_id = player.get("usualPlayingPositionId")
    if usual_id is not None:
        try:
            usual_id_int = int(usual_id)
            if usual_id_int == 0:
                return "Keeper"
            if usual_id_int == 1:
                return "Defender"
            if usual_id_int == 2:
                return "Midfielder"
            if usual_id_int == 3:
                return "Forward"
        except (ValueError, TypeError):
            pass

    pos_id = player.get("positionId")
    if pos_id is not None:
        try:
            pos_id_int = int(pos_id)
            if pos_id_int == 11:
                return "Keeper"
            if 30 <= pos_id_int < 50:
                return "Defender"
            if 70 <= pos_id_int < 90:
                return "Midfielder"
            if 100 <= pos_id_int < 110:
                return "Forward"
        except (ValueError, TypeError):
            pass

    return ""


class League(BaseModel):
    """Pydantic model representing a league catalog item."""

    model_config = {"extra": "ignore", "populate_by_name": True}

    league_id: int = Field(validation_alias=AliasChoices("league_id", "id"))
    slug: str
    name: str
    kind: str = "domestic_only"


class LeagueDetails(BaseModel):
    """Pydantic model representing the detailed overview of a league."""

    model_config = {"extra": "ignore"}

    id: int
    name: str
    latest_season: str = Field(alias="latestSeason")
    seopath: str


class Season(BaseModel):
    """Pydantic model representing a season metadata block."""

    model_config = {"extra": "ignore", "populate_by_name": True}

    season_name: str = Field(alias="seasonName")


class MatchRef(BaseModel):
    """Pydantic model representing a match reference from a season fixture list."""

    model_config = {"extra": "ignore", "populate_by_name": True}

    match_id: str
    home_id: int
    home_name: str
    away_id: int
    away_name: str
    round_name: str
    match_date: datetime
    score_str: str
    is_shootout: bool

    @model_validator(mode="before")
    @classmethod
    def parse_fixture(cls, data: Any) -> Any:
        """Parse raw fixture data from FotMob before validation."""
        if not isinstance(data, dict):
            return data
        if "match_id" in data:
            return data

        match_id = data.get("id")
        if match_id is not None:
            match_id = str(match_id)
        else:
            page_url = data.get("pageUrl", "")
            if page_url:
                anchor_idx = page_url.find("#")
                if anchor_idx != -1:
                    match_id = page_url[anchor_idx + 1 :]

        home = data.get("home", {})
        away = data.get("away", {})
        status = data.get("status", {})
        reason = status.get("reason", {})
        is_shootout = reason.get("shortKey") == "penalties_short"

        round_val = data.get("roundName") or data.get("round") or ""
        round_name = str(round_val)

        utc_time = status.get("utcTime")

        return {
            "match_id": match_id,
            "home_id": int(home.get("id") or 0),
            "home_name": str(home.get("name") or ""),
            "away_id": int(away.get("id") or 0),
            "away_name": str(away.get("name") or ""),
            "round_name": round_name,
            "match_date": parse_match_date(utc_time),
            "score_str": str(status.get("scoreStr") or ""),
            "is_shootout": is_shootout,
        }


class Shot(BaseModel):
    """Pydantic model representing a single penalty kick shot in the match shotmap."""

    model_config = {"extra": "ignore", "populate_by_name": True}

    player_id: int = Field(alias="playerId")
    team_id: int = Field(alias="teamId")
    situation: str
    period: str
    x: float
    y: float
    outcome: str
    shot_type: str = Field(alias="shotType")

    @model_validator(mode="before")
    @classmethod
    def parse_shot(cls, data: Any) -> Any:
        """Parse raw shotmap data from FotMob before validation."""
        if not isinstance(data, dict):
            return data
        event_type = data.get("eventType")
        outcome = SHOTMAP_EVENT_TYPE_TO_OUTCOME.get(
            str(event_type),
            str(event_type),
        )
        on_goal = data.get("onGoalShot", {})
        return {
            "player_id": int(data.get("playerId") or 0),
            "team_id": int(data.get("teamId") or 0),
            "situation": data.get("situation"),
            "period": data.get("period"),
            "x": float(on_goal.get("x") or 0.0),
            "y": float(on_goal.get("y") or 0.0),
            "outcome": outcome,
            "shot_type": data.get("shotType"),
        }


class Match(BaseModel):
    """Pydantic model representing full match details.

    Includes the shotmap and lineup.
    """

    model_config = {"extra": "ignore", "populate_by_name": True}

    match_id: str
    league_id: int
    match_date: datetime
    home_team_id: int
    away_team_id: int
    shotmap: list[Shot]
    player_positions: dict[int, str]
    player_teams: dict[int, int]

    @model_validator(mode="before")
    @classmethod
    def parse_match(cls, data: Any) -> Any:
        """Parse raw match details data from FotMob before validation."""
        if not isinstance(data, dict):
            return data
        if "match_id" in data:
            return data

        page_props = data.get("pageProps", {})
        general = page_props.get("general", {})
        content = page_props.get("content", {})

        match_id = str(general.get("matchId") or "")
        league_id = int(general.get("leagueId") or 0)

        match_date_raw = general.get("matchTimeUTC")

        home_team = general.get("homeTeam", {})
        away_team = general.get("awayTeam", {})
        home_team_id = int(home_team.get("id") or 0)
        away_team_id = int(away_team.get("id") or 0)

        shots_data = content.get("shotmap", {}).get("shots", [])

        lineup = content.get("lineup", {})
        players_list = [
            (int(p["id"]), team_id, get_player_position_from_lineup(p))
            for team_key, team_id in (
                ("homeTeam", home_team_id),
                ("awayTeam", away_team_id),
            )
            for p in lineup.get(team_key, {}).get("starters", [])
            + lineup.get(team_key, {}).get("subs", [])
            if p.get("id")
        ]
        player_positions = {p_id: pos for p_id, _, pos in players_list}
        player_teams = {p_id: t_id for p_id, t_id, _ in players_list}

        return {
            "match_id": match_id,
            "league_id": league_id,
            "match_date": parse_match_date(match_date_raw),
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "shotmap": shots_data,
            "player_positions": player_positions,
            "player_teams": player_teams,
        }


class PenaltyKick(BaseModel):
    """Pydantic model representing the persisted row of a shootout penalty kick."""

    model_config = {"extra": "ignore"}

    match_id: str
    league_id: int
    season: str
    match_date: datetime
    player_id: int
    team_id: int
    is_home: bool
    x: float
    y: float
    outcome: str
    shot_type: str
    player_position: str
