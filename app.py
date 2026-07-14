"""12yd — Penalty Shootout Side Prediction. Streamlit dashboard entry point."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import pandas as pd
import streamlit as st

from twelveyards.artifacts import Artifacts

if TYPE_CHECKING:
    from collections.abc import Iterable

    from twelveyards.artifacts import PredictionRow

_BadgeColor = Literal[
    "red", "orange", "yellow", "blue", "green", "violet", "gray", "grey", "primary",
]

_TEAM_COLORS: dict[str, _BadgeColor] = {
    "Argentina": "blue",
    "Brazil": "yellow",
    "England": "gray",
    "France": "blue",
    "Germany": "gray",
    "Italy": "blue",
    "Netherlands": "orange",
    "Portugal": "red",
    "Spain": "red",
    "Uruguay": "blue",
    "Croatia": "red",
    "Belgium": "red",
    "Mexico": "green",
    "Japan": "red",
    "Senegal": "green",
    "Morocco": "red",
    "United States": "red",
    "USA": "red",
    "Canada": "red",
    "Korea Republic": "red",
    "South Korea": "red",
    "Norway": "red",
    "Poland": "gray",
    "Saudi Arabia": "green",
    "Serbia": "red",
    "Switzerland": "red",
    "Türkiye": "red",
    "Turkey": "red",
    "Australia": "yellow",
}
_NEUTRAL_COLOR: _BadgeColor = "gray"


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


def load_predictions() -> list[PredictionRow]:
    """Load predictions.jsonl from the working tree and return deserialized rows."""
    art = Artifacts()
    path = art.predictions
    if not path.exists():
        return []
    return art.read_predictions()


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
    """Return the side the kicker is least likely to aim for."""
    minimum = min(p_l, p_c, p_r)
    for side, value in (("L", p_l), ("C", p_c), ("R", p_r)):
        if value == minimum:
            return side
    return "L"


def distinct_teams(predictions: Iterable[PredictionRow]) -> list[tuple[int, str]]:
    """Return sorted distinct (team_id, team_name) pairs from predictions."""
    seen: set[int] = set()
    teams: list[tuple[int, str]] = []
    for r in predictions:
        if r.team_id not in seen:
            seen.add(r.team_id)
            teams.append((r.team_id, r.team_name))
    teams.sort(key=lambda t: t[1])
    return teams


def badge_color(team_name: str) -> _BadgeColor:
    """Return the color associated with a team for badge rendering."""
    return _TEAM_COLORS.get(team_name, _NEUTRAL_COLOR)


def foot_label(kicking_foot: str) -> str:
    """Return a short label for the foot pill."""
    foot = kicking_foot.strip().lower()
    if foot == "right":
        return "R"
    if foot == "left":
        return "L"
    if foot == "both":
        return "L/R"
    return ""


def foot_color(kicking_foot: str) -> _BadgeColor:
    """Return the theme color associated with a kicking foot."""
    foot = kicking_foot.strip().lower()
    if foot == "right":
        return "blue"
    if foot == "left":
        return "orange"
    if foot == "both":
        return "green"
    return "gray"


@st.cache_data(show_spinner="Loading predictions…")
def _cached_load_predictions() -> list[PredictionRow]:
    """Cache and load predictions using library logic."""
    return load_predictions()


@st.cache_data
def _cached_distinct_teams() -> list[tuple[int, str]]:
    """Cache and extract distinct (team_id, team_name) pairs."""
    return distinct_teams(_cached_load_predictions())


def render_card(kicker: KickerPrediction) -> None:
    """Render one kicker's predictions as a Streamlit card with a bar chart."""
    with st.container(border=True):
        label = foot_label(kicker.kicking_foot)
        if label:
            st.markdown(
                f"**{kicker.player_name}** "
                f":{foot_color(kicker.kicking_foot)}[{label}] "
                f"· {kicker.total_penalties} pen",
            )
        else:
            st.markdown(f"**{kicker.player_name}** · {kicker.total_penalties} pen")
        df = pd.DataFrame(
            {"probability": [kicker.p_L * 100, kicker.p_C * 100, kicker.p_R * 100]},
            index=pd.Index(["Left", "Center", "Right"]),
        )
        st.bar_chart(df, sort=False, height=180)


def render_team_block(
    kickers: list[KickerPrediction],
    *,
    heading: str,
) -> None:
    """Render all kicker cards for one team with a heading badge."""
    st.badge(heading.upper(), color=badge_color(heading))
    for k in kickers:
        render_card(k)


def main() -> None:
    """Entry point: configure the page, render sidebar dropdowns, and display cards."""
    st.set_page_config(
        page_title="12yd — Penalty Shootout Prediction",
        page_icon="⚽",
        layout="wide",
    )
    st.sidebar.title("12yd")
    st.sidebar.caption("Penalty shootout side prediction.")

    teams = _cached_distinct_teams()
    if not teams:
        st.sidebar.warning("No predictions found. Run the pipeline first.")
        return

    team_options = {name: tid for tid, name in teams}
    team_names = [name for _, name in teams]

    team_a_name = st.sidebar.selectbox("Team A", options=team_names)
    team_b_name = st.sidebar.selectbox("Team B", options=team_names)

    team_a_id = team_options[team_a_name]
    team_b_id = team_options[team_b_name]

    predictions = _cached_load_predictions()
    home_kickers, away_kickers = predictions_for_match(
        predictions, team_a_id, team_b_id,
    )

    if not home_kickers and not away_kickers:
        st.warning("No roster players found for the selected teams.")
        return

    left, right = st.columns(2)
    with left:
        render_team_block(home_kickers, heading=team_a_name)
    with right:
        render_team_block(away_kickers, heading=team_b_name)


if __name__ == "__main__":
    main()
