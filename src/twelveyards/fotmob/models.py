"""Pydantic data models for FotMob API responses."""

from pydantic import AwareDatetime, BaseModel, computed_field


class League(BaseModel):
    """League."""

    id: int
    name: str
    seasons: list[str]
    country: str | None = None
    gender: str | None = None


class Team(BaseModel):
    """Team."""

    id: int
    name: str


class Round(BaseModel):
    """Round."""

    match: str
    league: str


class Score(BaseModel):
    """Score."""

    label: str | None = None

    @computed_field
    @property
    def home(self) -> int | None:
        """Dynamically extracts the home score from the label."""
        if not self.label or "-" not in self.label:
            return None
        return int(self.label.split("-")[0].strip())

    @computed_field
    @property
    def away(self) -> int | None:
        """Dynamically extracts the away score from the label."""
        if not self.label or "-" not in self.label:
            return None
        return int(self.label.split("-")[-1].strip())


class Period(BaseModel):
    """Period."""

    slug: str
    name: str


class Status(BaseModel):
    """Status."""

    period: Period
    started: bool
    finished: bool
    cancelled: bool
    awarded: bool


class Position(BaseModel):
    """Position."""

    id: int | None


class Player(BaseModel):
    """Player."""

    id: int
    name: str
    position: Position
    age: int | None
    market_value: float | None


class Shot(BaseModel):
    """Shot."""

    x: float
    y: float
    zoom: float


class Penalty(BaseModel):
    """Penalty kick."""

    id: int
    player_id: int
    team_id: int
    period: str
    shot: Shot
    outcome: str


class Match(BaseModel):
    """Match."""

    id: int
    league_id: int
    home_team: Team
    away_team: Team
    round: Round
    start_at: AwareDatetime
    status: Status
    score: Score
    penalties: list[Penalty]
    players: list[Player]
