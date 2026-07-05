"""Streamlit dashboard entry point — v4 card layout (Issue #48).

A single-page Streamlit app on Streamlit Cloud that surfaces live
shootout predictions. At load time, the app fetches the WC 2026
fixture list from FotMob, filters to upcoming matches with both
teams decided (any round — R32, R16, QF, SF, F), and lets the user
pick a match from a sidebar selectbox. For the selected match, the
app loads `predictions.jsonl` and `player_history.jsonl` from
Hugging Face, joins the two to compute the per-kicker career penalty
count, filters to the match's two teams, and renders a **card per
kicker**: photo placeholder + name + career penalty count + foot
pill + a Plotly goal drawing (3 coloured segments with a star on the
most-likely side) + a one-line prediction row.

The v4 design (locked in `docs/prototype-card-layout.png`) is:

- Both teams' cards on the same page, sorted by `total_penalties`
  descending (name as tiebreaker). A goalkeeper can scan the squad
  in one screen.
- Each card has a team-color left stripe (yellow for Brazil, blue
  for France, etc.) for at-a-glance team identification.
- The Kicker-PoV frame is explicit end to end: the per-kicker
  probabilities, the "WILL AIM" headline, and the "GK dive ↔ X"
  hint are all in the Kicker-PoV (per `CONTEXT.md`). The dashboard
  reads consistently kicker-PoV; the L/R letters in the dive hint
  are the Kicker's mirror (the Kicker's L is the Goalkeeper's R).
- Kickers with no penalty history in the 5-year lookback window
  render with three near-equal light cells (honest about the
  absence of signal) and a `0 career penalties` caption.

The data + match-filter logic lives in `penalty_pred.dashboard` —
this file is a thin Streamlit layer over the library, so the same
code can be unit-tested (the library) and exercised end-to-end (the
app). The Plotly goal-drawing helper is built inline in this module
so the figure is testable on its structure (3 shapes, 4 annotations,
the most-likely border, the star) without launching Streamlit.

Deployment: Streamlit Cloud via the GitHub repo. The default entry
point at the repo root is what Streamlit Cloud's "deploy from repo"
wizard expects.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import plotly.graph_objects as go
import streamlit as st
from huggingface_hub import hf_hub_download

from penalty_pred.artifacts import Artifacts
from penalty_pred.client import FotMobClient
from penalty_pred.dashboard import (
    KickerPrediction,
    MatchContext,
    load_upcoming_knockouts,
    most_likely_side,
    opposite_side,
    predictions_for_match,
)
from penalty_pred.player_history import PlayerPenalty

_BadgeColor = Literal[
    "red", "orange", "yellow", "blue", "green", "violet", "gray", "grey", "primary"
]

HF_REPO_ID: str = "couto/12yd"

# Team-color palette for the v4 card layout (Issue #48). The left
# stripe on each card and the avatar's circle background are filled
# with the team's primary color. The map covers the 32 WC 2026
# finalists; a neutral gray falls through for any team not in the
# map (e.g. a friendly-roster player whose team is not a finalist).
_TEAM_COLORS: dict[str, str] = {
    "Argentina": "#75AADB",
    "Australia": "#FFCD00",
    "Brazil": "#FFCD00",
    "Canada": "#FF0000",
    "Croatia": "#FF0000",
    "England": "#FFFFFF",
    "France": "#002654",
    "Germany": "#FFFFFF",
    "Italy": "#0066B3",
    "Japan": "#BC002D",
    "Korea Republic": "#C60C30",
    "Mexico": "#006847",
    "Morocco": "#C1272D",
    "Netherlands": "#FF6F00",
    "Norway": "#EF2B2D",
    "Poland": "#FFFFFF",
    "Portugal": "#FF0000",
    "Saudi Arabia": "#006C35",
    "Senegal": "#00853F",
    "Serbia": "#C7363D",
    "South Korea": "#C60C30",
    "Spain": "#AA151B",
    "Switzerland": "#FF0000",
    "Türkiye": "#E30A17",
    "Turkey": "#E30A17",
    "United States": "#B22234",
    "USA": "#B22234",
    "Uruguay": "#0038A8",
}
_NEUTRAL_COLOR: str = "#9AA0A6"  # Streamlit's neutral gray for unknown teams

# The colormap for the goal drawing: light → deep blue. The base is a
# near-white blue (low probability) and the dark end is a deep navy
# (high probability). Three colours is the minimum the figure needs
# to look like a heatmap; interpolation gives a smooth gradient.
_GOAL_BASE: tuple[int, int, int] = (220, 235, 250)  # rgb
_GOAL_DARK: tuple[int, int, int] = (20, 60, 140)  # rgb
_GOAL_ACCENT: str = "#0A2540"  # thick border + annotation text
_GOAL_NEUTRAL_BORDER: str = "#7A8DA3"  # thin border on non-most-likely sides


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

    The v4 card layout needs the per-kicker career penalty count to
    render "N career penalties" on each card and to sort the squad by
    experience. The history is grouped here so `predictions_for_match`
    can read it in O(1) per kicker without re-parsing the JSONL
    inside the per-match view.
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

    v4 (Issue #48): the sidebar is minimal. No model block, no data
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
# Goal drawing figure (inline Plotly helper, testable in isolation)
# ---------------------------------------------------------------------------


def goal_drawing_figure(p_L: float, p_C: float, p_R: float) -> go.Figure:
    """Build the Plotly goal-drawing figure: 3 coloured segments + a star.

    The v4 card layout's visual goal drawing. Three equal-width
    rectangles (L, C, R) coloured by the kicker-PoV probability on a
    light-to-deep blue colormap. The most-likely side gets a thick
    accent border and a `★` annotation. Each segment is annotated
    with `L · 55%`-style text so the values are readable on the card.

    The figure is built inline (per the v4 PRD) so the dashboard seam
    stays at one level. The function is testable on its structure:
    3 shapes, 4 annotations (one per side + the star), the
    most-likely side's `line.color` is the accent and the others are
    the neutral border.

    Kickers with no penalty history render with three near-equal
    light cells — the colormap is `p = 0.33` for each side when the
    sum is uniform, so the visual is honest about the absence of
    signal without a special flag.
    """
    sides = (("L", p_L, 0, 1), ("C", p_C, 1, 2), ("R", p_R, 2, 3))
    most = most_likely_side(p_L, p_C, p_R)

    fig = go.Figure()
    for side, p, x0, x1 in sides:
        t = min(max(float(p), 0.0), 1.0)
        r = int(_GOAL_BASE[0] + (_GOAL_DARK[0] - _GOAL_BASE[0]) * t)
        g = int(_GOAL_BASE[1] + (_GOAL_DARK[1] - _GOAL_BASE[1]) * t)
        b = int(_GOAL_BASE[2] + (_GOAL_DARK[2] - _GOAL_BASE[2]) * t)
        fill = f"rgb({r}, {g}, {b})"
        is_most = side == most
        line_color = _GOAL_ACCENT if is_most else _GOAL_NEUTRAL_BORDER
        line_width = 4 if is_most else 1
        fig.add_shape(
            type="rect",
            x0=x0,
            x1=x1,
            y0=0,
            y1=1,
            fillcolor=fill,
            line=dict(color=line_color, width=line_width),
        )
        # Per-side text annotation: "L · 55%"
        fig.add_annotation(
            x=(x0 + x1) / 2,
            y=0.5,
            text=f"<b>{side}</b> · {p * 100:.0f}%",
            showarrow=False,
            font=dict(size=14, color=_GOAL_ACCENT),
        )
    # The star — the 4th annotation.
    if most:
        x0_star, x1_star = {"L": (0, 1), "C": (1, 2), "R": (2, 3)}[most]
        fig.add_annotation(
            x=(x0_star + x1_star) / 2,
            y=1.10,
            text="★",
            showarrow=False,
            font=dict(size=22, color=_GOAL_ACCENT),
        )
    fig.update_xaxes(range=[0, 3], visible=False)
    fig.update_yaxes(range=[0, 1.2], visible=False)
    fig.update_layout(
        margin=dict(l=0, r=0, t=20, b=0),
        height=80,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Card rendering
# ---------------------------------------------------------------------------


def _kicker_color(team_name: str) -> str:
    """The team-color for the card's left stripe and the avatar's circle.

    Falls back to a neutral gray for any team not in the palette
    (e.g. a friendly-roster player whose team is not a WC finalist).
    """
    return _TEAM_COLORS.get(team_name, _NEUTRAL_COLOR)


def _initials(player_name: str) -> str:
    """Two-letter initials for the avatar. Names with one word get the
    first two letters; names with multiple words get the first letter of
    the first and last words (skipping diacritics where possible).
    """
    # Normalize: strip diacritics, collapse whitespace.
    normalized = " ".join(player_name.split())
    if not normalized:
        return "??"
    parts = normalized.split(" ")
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()


def _avatar_svg(initials: str, color: str, size: int = 64) -> str:
    """A circular SVG with the initials centred. Rendered via `st.image`.

    Streamlit's `st.image` accepts an SVG string and renders it
    natively, so the avatar needs no HTML or JavaScript. The text
    is centred with `text-anchor="middle"` and the y-coordinate is
    tuned for visual balance (the visual centre of a glyph is
    roughly 35% from the top, not 50%).
    """
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}">'
        f'<circle cx="{size / 2}" cy="{size / 2}" r="{size / 2}" fill="{color}"/>'
        f'<text x="{size / 2}" y="{size * 0.62}" font-size="{size * 0.4}" '
        f'font-family="sans-serif" font-weight="700" text-anchor="middle" '
        f'fill="#0A2540">{initials}</text>'
        f"</svg>"
    )


def _foot_badge_color(kicking_foot: str) -> _BadgeColor:
    """Map the declared foot to a Streamlit `st.badge` color.

    The screenshot uses pink for left, blue for right, and yellow for
    both. Streamlit's `st.badge` accepts only the named palette
    (`red`, `blue`, `yellow`, `green`, `orange`, `violet`, `gray`,
    `primary`); "red" is the closest to the pink in the screenshot.
    """
    if kicking_foot == "left":
        return "red"
    if kicking_foot == "right":
        return "blue"
    if kicking_foot == "both":
        return "yellow"
    return "gray"


def _foot_label(kicking_foot: str) -> str:
    """The all-caps label for the foot pill."""
    if kicking_foot == "left":
        return "LEFT FOOT"
    if kicking_foot == "right":
        return "RIGHT FOOT"
    if kicking_foot == "both":
        return "BOTH FEET"
    return "UNKNOWN"


def _prediction_row(kicker: KickerPrediction) -> None:
    """The one-line 'Kicker will aim: X [%]  ·  GK dive ↔ Y' row.

    Both labels are in the Kicker-PoV frame (per `CONTEXT.md`). The
    WILL AIM side is `argmax` of the three probabilities (the Kicker's
    most-likely aim); the GK dive side is the Kicker's mirror of the
    WILL AIM side (L ↔ R, C ↔ C). For a Kicker whose distribution is
    single-sided, the dive hint matches `argmin` (`recommended_dive`)
    by construction; for a flatter distribution they may differ, and
    the headline `argmax` is the more honest display of where the
    Kicker is most likely to aim.
    """
    will_aim = most_likely_side(kicker.p_L, kicker.p_C, kicker.p_R)
    dive = opposite_side(will_aim)
    will_pct = max(kicker.p_L, kicker.p_C, kicker.p_R) * 100
    st.markdown(f"**WILL AIM {will_aim}** {will_pct:.0f}% &nbsp;·&nbsp; GK dive ↔ **{dive}**")


def render_card(kicker: KickerPrediction) -> None:
    """Render one kicker card: photo, name, goal drawing, prediction row.

    The card is a `st.container(border=True)` with a `st.columns` layout:
    the meta block (avatar + name + career penalty count + foot pill)
    on the left, the goal drawing + prediction row on the right. The
    team-color stripe is a thin SVG rectangle in a leftmost column.

    The card uses only native Streamlit elements — no
    `st.markdown(unsafe_allow_html=True)`, no custom JS, no
    third-party card components. The avatar is rendered via
    `st.image` with an inline SVG string; the foot is a `st.badge`;
    the goal is a `st.plotly_chart`.
    """
    team_color = _kicker_color(kicker.team_name)
    initials = _initials(kicker.player_name)
    with st.container(border=True):
        stripe, meta, goal = st.columns([0.04, 0.42, 0.54])
        with stripe:
            # The team-color stripe: a 4-px wide SVG bar at full
            # container height. `st.image` with an SVG keeps the
            # rendering native (no `unsafe_allow_html`).
            stripe_svg = (
                '<svg xmlns="http://www.w3.org/2000/svg" width="4" height="100">'
                f'<rect width="4" height="100" fill="{team_color}"/>'
                "</svg>"
            )
            st.image(stripe_svg, width=4)
        with meta:
            st.image(_avatar_svg(initials, team_color, size=64), width=64)
            st.markdown(f"**{kicker.player_name}**")
            st.caption(f"{kicker.total_penalties} career penalties")
            st.badge(
                _foot_label(kicker.kicking_foot),
                color=_foot_badge_color(kicker.kicking_foot),
            )
        with goal:
            st.plotly_chart(
                goal_drawing_figure(kicker.p_L, kicker.p_C, kicker.p_R),
                width="stretch",
                config={"displayModeBar": False},
                key=f"goal-{kicker.player_id}",
            )
            _prediction_row(kicker)


def render_team_block(
    kickers: list[KickerPrediction],
    *,
    heading: str,
) -> None:
    """Render one team's column of cards with a heading.

    The team block is a single column with a heading (the team name
    with a small colored dot prefix, matching the team stripe) and a
    stack of `render_card` calls. Kickers with `total_penalties == 0`
    still get a card — the v4 design renders every kicker, with the
    three near-equal light cells being the honest "no history" signal.

    The team-color dot is a small SVG circle rendered via `st.image`
    (no `unsafe_allow_html` — the card layout uses only native
    Streamlit elements).
    """
    color = _kicker_color(heading)
    dot_svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14">'
        f'<circle cx="7" cy="7" r="6" fill="{color}"/>'
        "</svg>"
    )
    head_cols = st.columns([0.06, 0.94])
    with head_cols[0]:
        st.image(dot_svg, width=14)
    with head_cols[1]:
        st.markdown(f"**{heading.upper()}**")
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
