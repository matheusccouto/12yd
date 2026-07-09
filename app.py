"""Streamlit dashboard entry point — v8 bar chart cards (Issue #48).

A single-page Streamlit app on Streamlit Cloud that surfaces live
shootout predictions. At load time, the app fetches the WC 2026
fixture list from FotMob, filters to upcoming matches with both
teams decided (any round — R32, R16, QF, SF, F), and lets the user
pick a match from a sidebar selectbox. For the selected match, the
app loads `predictions.jsonl` and `player_history.jsonl` from
Hugging Face, joins the two to compute the per-kicker career penalty
count, filters to the match's two teams, and renders a **card per
kicker**: name + career penalty count + a bar chart showing L/C/R
probabilities on a fixed 0–100 scale.

The v8 design uses only native Streamlit elements. Each card is a
bordered container with:
- Line 1: Player name + penalty count (inline)
- Line 2: A `st.bar_chart` with Left/Center/Right bars, y-axis fixed 0–100.

The data + match-filter logic lives in `penalty_pred.dashboard` —
this file is a thin Streamlit layer over the library, so the same
code can be unit-tested (the library) and exercised end-to-end (the
app).

Deployment: Streamlit Cloud via the GitHub repo. The default entry
point at the repo root is what Streamlit Cloud's "deploy from repo"
wizard expects.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

from penalty_pred.artifacts import Artifacts
from penalty_pred.client import FotMobClient
from penalty_pred.dashboard import (
    KickerPrediction,
    MatchContext,
    load_upcoming_knockouts,
    predictions_for_match,
)
from penalty_pred.player_history import PlayerPenalty

_BadgeColor = Literal[
    "red", "orange", "yellow", "blue", "green", "violet", "gray", "grey", "primary"
]

HF_REPO_ID: str = "couto/12yd"

# Team-color palette for badges. The map covers the 32 WC 2026
# finalists; a neutral gray falls through for any team not in the
# map (e.g. a friendly-roster player whose team is not a finalist).
_TEAM_COLORS: dict[str, _BadgeColor] = {
    "Argentina": "blue",
    "Australia": "yellow",
    "Brazil": "yellow",
    "Canada": "red",
    "Croatia": "red",
    "England": "gray",
    "France": "blue",
    "Germany": "gray",
    "Italy": "blue",
    "Japan": "red",
    "Korea Republic": "red",
    "Mexico": "green",
    "Morocco": "red",
    "Netherlands": "orange",
    "Norway": "red",
    "Poland": "gray",
    "Portugal": "red",
    "Saudi Arabia": "green",
    "Senegal": "green",
    "Serbia": "red",
    "South Korea": "red",
    "Spain": "red",
    "Switzerland": "red",
    "Türkiye": "red",
    "Turkey": "red",
    "United States": "red",
    "USA": "red",
    "Uruguay": "blue",
}
_NEUTRAL_COLOR: _BadgeColor = "gray"


# ---------------------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Loading predictions from Hugging Face…")
def load_predictions_from_hf() -> list:
    """Download `data/predictions.jsonl` from HF and parse it.

    v3 (Issue #36): the dashboard reads the artifact directly instead
    of re-scoring on the fly. The model is round-agnostic, so the
    per-kicker probabilities on disk are the same for every match.
    `st.cache_data` keeps the parsed list in memory for the lifetime
    of the Streamlit process.
    """
    path = hf_hub_download(HF_REPO_ID, "data/predictions.jsonl")
    return Artifacts().read_predictions(path=Path(path))


@st.cache_data(show_spinner="Loading player history from Hugging Face…")
def load_player_history_from_hf() -> dict[int, list]:
    """Download `data/player_history.jsonl` from HF and group by kicker.

    The v6 card layout needs the per-kicker career penalty count to
    render inline on each card. The history is grouped here so
    `predictions_for_match` can read it in O(1) per kicker without
    re-parsing the JSONL inside the per-match view.
    """
    path = hf_hub_download(HF_REPO_ID, "data/player_history.jsonl")
    rows: list[PlayerPenalty] = Artifacts().read_player_history(path=Path(path))
    grouped: dict[int, list[PlayerPenalty]] = {}
    for row in rows:
        grouped.setdefault(row.kicker_id, []).append(row)
    return grouped


@st.cache_data
def predictions_artifact_mtime() -> datetime:
    """The artifact's last-updated timestamp (mtime of the downloaded file).

    Surfaced as a `st.caption` at the bottom of the page so the viewer
    knows how fresh the prediction is. `st.cache_data` keeps the
    timestamp for the lifetime of the Streamlit process — the file's
    mtime is stable across re-runs.
    """
    path = hf_hub_download(HF_REPO_ID, "data/predictions.jsonl")
    return datetime.fromtimestamp(Path(path).stat().st_mtime, tz=UTC)


@st.cache_resource
def build_fotmob_client() -> FotMobClient:
    """Build a `FotMobClient` for the live fixture fetch.

    The client lives in the Streamlit container's ephemeral disk by
    default — the dashboard is a read-only consumer of FotMob, so the
    ETag/gzip cache is built up on first load and re-used for the
    lifetime of the process.
    """
    return Artifacts().fotmob_client()


# ---------------------------------------------------------------------------
# Sidebar: minimal — title + match selector
# ---------------------------------------------------------------------------


def render_sidebar() -> None:
    """Render the title and the match selector in the sidebar.

    v6 (Issue #48): the sidebar is minimal. No model block, no data
    block, no explainer paragraph, no legend. The title + a single
    `st.selectbox` is the entire surface.
    """
    st.sidebar.title("12yd")
    st.sidebar.caption("Penalty shootout side prediction.")


def render_match_selector() -> MatchContext | None:
    """Render the match selectbox in the sidebar; return the selected match.

    The fixture list is fetched once per page load (the
    `@st.cache_data` on `load_upcoming_knockouts` would cache across
    page reruns but not across days — a stale fixture list would be a
    bug here, so we always re-fetch).
    """
    matches = load_upcoming_knockouts(build_fotmob_client())
    if not matches:
        st.sidebar.info("No upcoming matches with both teams decided.")
        return None
    labels = [_match_label(m) for m in matches]
    index = st.sidebar.selectbox(
        "Match",
        options=range(len(matches)),
        format_func=lambda i: labels[i],
        index=0,
    )
    return matches[index]


# ---------------------------------------------------------------------------
# Card rendering
# ---------------------------------------------------------------------------


def _kicker_badge_color(team_name: str) -> _BadgeColor:
    """The team-color for the badge.

    Falls back to a neutral gray for any team not in the palette
    (e.g. a friendly-roster player whose team is not a WC finalist).
    """
    return _TEAM_COLORS.get(team_name, _NEUTRAL_COLOR)


def render_card(kicker: KickerPrediction) -> None:
    """Render one kicker card: name + penalties inline, bar chart.

    The card is a `st.container(border=True)` with:
    - Line 1: Player name + penalty count (inline)
    - Line 2: A `st.bar_chart` with Left/Center/Right bars.
      `sort=False` keeps the data order (Left, Center, Right).
    """
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
    """Render one team's column of cards with a heading.

    The team block is a single column with a heading (the team name
    as a badge with the team color) and a stack of `render_card` calls.
    Kickers with `total_penalties == 0` still get a card — the v6
    design renders every kicker, with the three near-white cards
    being the honest "no history" signal.
    """
    color = _kicker_badge_color(heading)
    st.badge(heading.upper(), color=color)
    for k in kickers:
        render_card(k)


# ---------------------------------------------------------------------------
# Header + main
# ---------------------------------------------------------------------------


def _match_label(m: MatchContext) -> str:
    """A one-line selectbox label: 'Sat 04 Jul · 1/8 · Canada vs Mexico'."""
    kickoff_local = m.kickoff_utc.astimezone().strftime("%a %d %b %H:%M")
    return f"{kickoff_local} · {_round_label(m.round)} · {m.home_team_name} vs {m.away_team_name}"


def _round_label(round_code: str) -> str:
    """A human-readable round label for the FotMob codes."""
    return {
        "1/16": "Round of 32",
        "1/8": "Round of 16",
        "1/4": "Quarter-finals",
        "1/2": "Semi-finals",
        "final": "Final",
    }.get(round_code, round_code)


def render_match_header(context: MatchContext) -> None:
    """The header line above the cards: 'Brazil vs France · Quarter-final · Sat 11 Jul 21:00'."""
    kickoff_local = context.kickoff_utc.astimezone().strftime("%a %d %b %H:%M")
    st.markdown(
        f"## {context.home_team_name} vs {context.away_team_name} "
        f"· {_round_label(context.round)} · {kickoff_local}"
    )


def main() -> None:
    """The Streamlit page entry point."""
    st.set_page_config(
        page_title="12yd — Penalty Shootout Prediction",
        page_icon="⚽",
        layout="wide",
    )
    render_sidebar()
    match = render_match_selector()
    if match is not None:
        render_match_header(match)
        predictions = load_predictions_from_hf()
        history = load_player_history_from_hf()
        kickers = predictions_for_match(predictions, match, player_history=history)
        if not kickers:
            st.warning(
                "No roster players found for either team in this match. "
                "The WC 2026 squad list may not yet have one of the teams."
            )
        else:
            home_kickers = [k for k in kickers if k.team_id == match.home_team_id]
            away_kickers = [k for k in kickers if k.team_id == match.away_team_id]
            left, right = st.columns(2)
            with left:
                render_team_block(home_kickers, heading=match.home_team_name)
            with right:
                render_team_block(away_kickers, heading=match.away_team_name)

    st.caption(
        f"Last updated: {predictions_artifact_mtime().strftime('%Y-%m-%d %H:%M UTC')} · "
        f"Data: `{HF_REPO_ID}/data/predictions.jsonl`"
    )


if __name__ == "__main__":
    main()
