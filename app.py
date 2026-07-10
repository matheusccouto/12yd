"""Streamlit dashboard entry point — v5 two-team dropdowns.

PRD-v5: Two independent team dropdowns, no live FotMob fixture fetch.
Reads predictions.jsonl and player_history.jsonl from the working tree.
Cards rendered with st.container(border=True) + st.bar_chart.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
import streamlit as st

from twelveyards.artifacts import Artifacts
from twelveyards.dashboard import (
    KickerPrediction,
    distinct_teams,
    predictions_for_match,
)

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


@st.cache_data(show_spinner="Loading predictions…")
def load_predictions() -> list:
    art = Artifacts()
    path = art.predictions
    if not path.exists():
        return []
    return art.read_predictions()


@st.cache_data
def get_distinct_teams() -> list[tuple[int, str]]:
    predictions = load_predictions()
    return distinct_teams(predictions)


@st.cache_data(show_spinner="Loading player history…")
def load_player_history() -> dict[int, list]:
    art = Artifacts()
    path = art.player_history
    if not path.exists():
        return {}
    from twelveyards.player_history import PlayerPenalty

    rows = art.read_player_history()
    grouped: dict[int, list[PlayerPenalty]] = {}
    for row in rows:
        grouped.setdefault(row.kicker_id, []).append(row)
    return grouped


def _badge_color(team_name: str) -> _BadgeColor:
    return _TEAM_COLORS.get(team_name, _NEUTRAL_COLOR)


def render_card(kicker: KickerPrediction) -> None:
    with st.container(border=True):
        st.markdown(f"**{kicker.player_name}** · {kicker.total_penalties} pen")
        df = pd.DataFrame(
            {"probability": [kicker.p_L * 100, kicker.p_C * 100, kicker.p_R * 100]},
            index=["Left", "Center", "Right"],
        )
        st.bar_chart(df, sort=False, height=180)


def render_team_block(
    kickers: list[KickerPrediction],
    *,
    heading: str,
) -> None:
    color = _badge_color(heading)
    st.badge(heading.upper(), color=color)
    for k in kickers:
        render_card(k)


def main() -> None:
    st.set_page_config(
        page_title="12yd — Penalty Shootout Prediction",
        page_icon="⚽",
        layout="wide",
    )
    st.sidebar.title("12yd")
    st.sidebar.caption("Penalty shootout side prediction.")

    teams = get_distinct_teams()
    if not teams:
        st.sidebar.warning("No predictions found. Run the pipeline first.")
        return

    team_options = {name: tid for tid, name in teams}
    team_names = [name for _, name in teams]

    team_a_name = st.sidebar.selectbox("Team A", options=team_names)
    team_b_name = st.sidebar.selectbox("Team B", options=team_names)

    team_a_id = team_options[team_a_name]
    team_b_id = team_options[team_b_name]

    predictions = load_predictions()
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
