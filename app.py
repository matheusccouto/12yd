"""Streamlit dashboard entry point.

A single-page app on Streamlit Cloud that surfaces live shootout
predictions. At load time, the app fetches the WC 2026 fixture list
from FotMob, filters to upcoming knockout matches (R16, QF, SF, F)
with both teams decided, and lets the user pick a match from a
selectbox. For the selected match, the app loads the frozen LightGBM
from Hugging Face, re-scores each likely kicker with the match's
actual round, and shows a per-kicker table: name, team, kicking foot,
P(L), P(C), P(R), and the recommended dive (`argmin`).

The data + re-score logic lives in `penalty_pred.dashboard` — this
file is a thin Streamlit layer over the library, so the same code
can be unit-tested (the library) and exercised end-to-end (the app).

Deployment: Streamlit Cloud via the GitHub repo. The default entry
point at the repo root is what Streamlit Cloud's "deploy from repo"
wizard expects.
"""

from __future__ import annotations

import pickle
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
    predict_match,
)
from penalty_pred.features import fetcher_from_client
from penalty_pred.rosters import RosterPlayer

HF_REPO_ID: str = "couto/12yd"


# ---------------------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading model from Hugging Face…")
def load_model() -> dict:
    """Download `model/lightgbm.pkl` from HF and unpickle it.

    `st.cache_resource` keeps the unpickled dict in process memory for
    the lifetime of the Streamlit process — a re-load would be a
    multi-MB download + pickle parse for no reason.
    """
    path = hf_hub_download(HF_REPO_ID, "model/lightgbm.pkl")
    with Path(path).open("rb") as f:
        return pickle.load(f)


@st.cache_resource
def build_fotmob_client() -> FotMobClient:
    """Build a `FotMobClient` for the live fixture fetch.

    The client lives in the Streamlit container's ephemeral disk by
    default — the dashboard is a read-only consumer of FotMob, so the
    ETag/gzip cache is built up on first load and re-used for the
    lifetime of the process.
    """
    return Artifacts().fotmob_client()


@st.cache_data(show_spinner="Loading roster from Hugging Face…")
def load_roster_from_hf() -> list[RosterPlayer]:
    """Download `data/wc2026_roster.jsonl` from HF and parse it."""
    path = hf_hub_download(HF_REPO_ID, "data/wc2026_roster.jsonl")
    return Artifacts().read_roster(path=Path(path))


@st.cache_data(show_spinner="Loading player history from Hugging Face…")
def load_history_from_hf() -> dict[int, list]:
    """Download `data/player_history.jsonl` from HF and parse it.

    Returns a dict keyed by `kicker_id`; each value is the kicker's
    unsorted list of `PlayerPenalty` rows (the dashboard's re-score
    re-filters per row by `target_date`).
    """
    from penalty_pred.features import load_player_history

    path = hf_hub_download(HF_REPO_ID, "data/player_history.jsonl")
    return load_player_history(Path(path))


# ---------------------------------------------------------------------------
# Sidebar: model + data summary
# ---------------------------------------------------------------------------


def render_sidebar() -> None:
    """Render the model + data summary in the sidebar."""
    st.sidebar.header("12yd — penalty shootout side prediction")
    st.sidebar.markdown(
        "Predicts P(L) / P(C) / P(R) for each kicker, so the goalkeeper "
        "can pick the lowest-probability side to dive."
    )
    artifact = load_model()
    st.sidebar.markdown(
        f"**Model:** {artifact.get('model_kind', 'lightgbm').title()} "
        f"({len(artifact.get('feature_columns', []))} features)"
    )
    st.sidebar.caption(f"Loaded from `{HF_REPO_ID}/model/`")

    roster = load_roster_from_hf()
    history = load_history_from_hf()
    st.sidebar.markdown(f"**Roster:** {len(roster)} WC 2026 players")
    st.sidebar.markdown(
        f"**History:** {sum(len(v) for v in history.values())} penalty rows "
        f"across {len(history)} kickers"
    )


# ---------------------------------------------------------------------------
# Match selector
# ---------------------------------------------------------------------------


def render_match_selector() -> MatchContext | None:
    """Render the upcoming-knockouts selectbox; return the selected match.

    The fixture list is fetched once per page load (the
    `@st.cache_data` on `load_upcoming_knockouts` would cache across
    page reruns but not across days — a stale fixture list would be a
    bug here, so we always re-fetch).
    """
    st.subheader("Upcoming knockout matches")
    matches = load_upcoming_knockouts(build_fotmob_client())
    if not matches:
        st.info(
            "No upcoming knockout matches with both teams decided. "
            "The dashboard will populate as group-stage results decide the knockout slots."
        )
        return None
    labels = [_match_label(m) for m in matches]
    index = st.selectbox(
        "Pick a match",
        options=range(len(matches)),
        format_func=lambda i: labels[i],
        index=0,
    )
    selected = matches[index]
    st.caption(f"Re-scoring with the match's actual round: **{_round_label(selected.round)}**")
    return selected


def _match_label(m: MatchContext) -> str:
    """A one-line selectbox label: 'Sat 04 Jul · 1/8 · Canada vs Mexico'."""
    kickoff_local = m.kickoff_utc.astimezone().strftime("%a %d %b %H:%M")
    return f"{kickoff_local} · {_round_label(m.round)} · {m.home_team_name} vs {m.away_team_name}"


def _round_label(round_code: str) -> str:
    """A human-readable round label for the FotMob codes."""
    return {
        "1/8": "Round of 16",
        "1/4": "Quarter-finals",
        "1/2": "Semi-finals",
        "final": "Final",
    }.get(round_code, round_code)


# ---------------------------------------------------------------------------
# Per-kicker predictions table
# ---------------------------------------------------------------------------


def render_predictions_table(context: MatchContext) -> None:
    """Re-score the match and render the per-kicker table."""
    artifact = load_model()
    model = artifact["model"]
    roster = load_roster_from_hf()
    history = load_history_from_hf()
    client = build_fotmob_client()
    metadata_fetcher = fetcher_from_client(client)

    kickers = predict_match(
        roster=roster,
        player_history=history,
        metadata_fetcher=metadata_fetcher,
        model=model,
        context=context,
    )
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
        "Sorted by total career penalties (most-experienced kicker first). "
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
            "Penalties": [k.total_penalties for k in kickers],
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
        f"Model: `{HF_REPO_ID}/model/lightgbm.pkl`"
    )


if __name__ == "__main__":
    main()
