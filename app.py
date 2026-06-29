"""Streamlit dashboard entry point.

A single-page app on Streamlit Cloud that surfaces live shootout
predictions. At load time, the app fetches the WC 2026 fixture list
from FotMob, filters to upcoming matches with both teams decided
(any round — R32, R16, QF, SF, F), and lets the user pick a match
from a selectbox. For the selected match, the app loads
`predictions.jsonl` from Hugging Face, filters to the match's two
teams, and shows a per-kicker table: name, team, kicking foot, P(L),
P(C), P(R), and the recommended dive (`argmin`).

The data + match-filter logic lives in `penalty_pred.dashboard` —
this file is a thin Streamlit layer over the library, so the same
code can be unit-tested (the library) and exercised end-to-end (the
app). v3 (Issue #36) collapsed the per-match re-score path: the
model is round-agnostic and `predictions.jsonl` is the source of
truth for every match.

Deployment: Streamlit Cloud via the GitHub repo. The default entry
point at the repo root is what Streamlit Cloud's "deploy from repo"
wizard expects.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

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

HF_REPO_ID: str = "couto/12yd"


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
# Sidebar: data summary
# ---------------------------------------------------------------------------


def render_sidebar() -> None:
    """Render the data summary in the sidebar."""
    st.sidebar.header("12yd — penalty shootout side prediction")
    st.sidebar.markdown(
        "Predicts P(L) / P(C) / P(R) for each kicker, so the goalkeeper "
        "can pick the lowest-probability side to dive."
    )
    predictions = load_predictions_from_hf()
    st.sidebar.markdown(f"**Predictions:** {len(predictions)} WC 2026 players")
    st.sidebar.caption(f"Loaded from `{HF_REPO_ID}/data/predictions.jsonl`")


# ---------------------------------------------------------------------------
# Match selector
# ---------------------------------------------------------------------------


def render_match_selector() -> MatchContext | None:
    """Render the upcoming-matches selectbox; return the selected match.

    The fixture list is fetched once per page load (the
    `@st.cache_data` on `load_upcoming_knockouts` would cache across
    page reruns but not across days — a stale fixture list would be a
    bug here, so we always re-fetch).
    """
    st.subheader("Upcoming matches")
    matches = load_upcoming_knockouts(build_fotmob_client())
    if not matches:
        st.info("No upcoming matches with both teams decided.")
        return None
    labels = [_match_label(m) for m in matches]
    index = st.selectbox(
        "Pick a match",
        options=range(len(matches)),
        format_func=lambda i: labels[i],
        index=0,
    )
    selected = matches[index]
    st.caption(f"Match round: **{_round_label(selected.round)}**")
    return selected


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


# ---------------------------------------------------------------------------
# Per-kicker predictions table
# ---------------------------------------------------------------------------


def render_predictions_table(context: MatchContext) -> None:
    """Filter predictions.jsonl to the match's teams and render the per-kicker table."""
    predictions = load_predictions_from_hf()
    kickers = predictions_for_match(predictions, context)
    if not kickers:
        st.warning(
            "No roster players found for either team in this match. "
            "The WC 2026 squad list may not yet have one of the teams."
        )
        return
    df = _kickers_to_dataframe(kickers)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "P(L)": st.column_config.ProgressColumn(
                "P(L)", min_value=0.0, max_value=1.0, format="%.3f"
            ),
            "P(C)": st.column_config.ProgressColumn(
                "P(C)", min_value=0.0, max_value=1.0, format="%.3f"
            ),
            "P(R)": st.column_config.ProgressColumn(
                "P(R)", min_value=0.0, max_value=1.0, format="%.3f"
            ),
        },
    )
    st.caption(
        "The **Recommended Dive** is `argmin` over P(L), P(C), P(R) — the side "
        "the model says the kicker is least likely to use."
    )


def _kickers_to_dataframe(kickers: list[KickerPrediction]) -> pd.DataFrame:
    """Package the per-kicker predictions into the Streamlit table shape."""
    return pd.DataFrame(
        {
            "Kicker": [k.player_name for k in kickers],
            "Team": [k.team_name for k in kickers],
            "Foot": [k.kicking_foot for k in kickers],
            "P(L)": [k.p_L for k in kickers],
            "P(C)": [k.p_C for k in kickers],
            "P(R)": [k.p_R for k in kickers],
            "Recommended Dive": [k.recommended_dive for k in kickers],
        }
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """The Streamlit page entry point."""
    st.set_page_config(
        page_title="12yd — Penalty Shootout Prediction",
        page_icon="⚽",
        layout="wide",
    )
    render_sidebar()
    st.title("Live penalty shootout predictions")
    st.markdown(
        "Pick an upcoming WC 2026 knockout match. For each kicker on each "
        "team, the table shows the predicted P(L), P(C), P(R) and the "
        "recommended dive (`argmin`)."
    )

    match = render_match_selector()
    if match is not None:
        render_predictions_table(match)

    st.caption(
        f"Last loaded: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')} · "
        f"Data: `{HF_REPO_ID}/data/predictions.jsonl`"
    )


if __name__ == "__main__":
    main()
