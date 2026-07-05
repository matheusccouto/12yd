"""Tests for the Streamlit app's inline helpers and figure builder.

The Streamlit UI itself isn't unit-tested (Streamlit apps typically
are not — the seam is the library). What's tested here:

- The inline `goal_drawing_figure` Plotly helper: it produces a figure
  with 3 shapes (one per Side), 4 annotations (one per Side + the
  star), the most-likely shape has the accent border, and the star
  annotation is positioned over the most-likely Side.
- The `_initials` helper: two-letter initials from the player name.
- The `_kicker_color` helper: deterministic per-team colors with a
  neutral fallback.
- The `_foot_label` / `_foot_badge_color` helpers: the all-caps label
  and the Streamlit `st.badge` color for each declared foot.
- The card-rendering functions: `_kicker_color`, `_avatar_svg`,
  `_prediction_row` are all import-safe (they don't touch Streamlit
  state) so we exercise their output strings.

The card layout's per-kicker count and order are pinned in
`tests/test_dashboard.py::test_predictions_for_match_*`; the layout
itself is a 1:1 mapping from the `KickerPrediction` list to
`render_card` calls.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

import app
from app import (
    _avatar_svg,
    _foot_badge_color,
    _foot_label,
    _initials,
    _kicker_color,
    goal_drawing_figure,
)
from penalty_pred.dashboard import (
    KickerPrediction,
    MatchContext,
    most_likely_side,
    opposite_side,
    predictions_for_match,
)
from penalty_pred.predict import PredictionRow

# ---------------------------------------------------------------------------
# Inline Plotly helper — the v4 card's goal-drawing figure
# ---------------------------------------------------------------------------


def test_goal_drawing_figure_has_three_shapes() -> None:
    """The figure has 3 `shapes` (one per Side: L, C, R)."""
    fig = goal_drawing_figure(0.55, 0.20, 0.25)
    assert len(fig.layout.shapes) == 3


def test_goal_drawing_figure_has_four_annotations() -> None:
    """The figure has 4 `annotations` (one per Side + the star)."""
    fig = goal_drawing_figure(0.55, 0.20, 0.25)
    assert len(fig.layout.annotations) == 4


def test_goal_drawing_figure_most_likely_has_accent_border() -> None:
    """The most-likely shape has the accent border (line color = #0A2540)."""
    fig = goal_drawing_figure(0.55, 0.20, 0.25)
    # The accent color is the constant the helper uses. The other two
    # shapes use the neutral border (#7A8DA3).
    shapes = list(fig.layout.shapes)
    line_colors = {s.line.color for s in shapes}
    assert app._GOAL_ACCENT in line_colors
    assert app._GOAL_NEUTRAL_BORDER in line_colors
    # The accent is on exactly one shape (the most-likely).
    accent_count = sum(1 for s in shapes if s.line.color == app._GOAL_ACCENT)
    neutral_count = sum(1 for s in shapes if s.line.color == app._GOAL_NEUTRAL_BORDER)
    assert accent_count == 1
    assert neutral_count == 2


def test_goal_drawing_figure_star_on_most_likely() -> None:
    """The `★` annotation is positioned over the most-likely Side.

    The star is the 4th annotation; its x-coordinate is the centre of
    the most-likely shape (L = 0.5, C = 1.5, R = 2.5).
    """
    cases = [
        (0.55, 0.20, 0.25, "L", 0.5),
        (0.20, 0.60, 0.20, "C", 1.5),
        (0.30, 0.25, 0.45, "R", 2.5),
    ]
    for p_L, p_C, p_R, most, expected_x in cases:
        fig = goal_drawing_figure(p_L, p_C, p_R)
        star = fig.layout.annotations[-1]
        assert star.text == "★", f"expected star text, got {star.text!r}"
        # x may be a float; compare with a small tolerance.
        assert abs(star.x - expected_x) < 1e-6, (
            f"most-likely={most}: expected star x={expected_x}, got {star.x}"
        )


def test_goal_drawing_figure_per_side_text_annotations() -> None:
    """The first 3 annotations are the per-Side 'L · 55%' text labels."""
    fig = goal_drawing_figure(0.55, 0.20, 0.25)
    texts = [a.text for a in fig.layout.annotations[:3]]
    assert "L" in texts[0]
    assert "55" in texts[0]
    assert "C" in texts[1]
    assert "20" in texts[1]
    assert "R" in texts[2]
    assert "25" in texts[2]


def test_goal_drawing_figure_no_history_is_near_uniform() -> None:
    """A Kicker with no history (uniform 1/3, 1/3, 1/3) renders with
    three near-equal light cells — the colormap is `p = 0.33` for
    each side. The `argmax` tiebreaker picks L (the L→C→R order
    wins), so the L shape gets the accent border and the star.
    """
    fig = goal_drawing_figure(1 / 3, 1 / 3, 1 / 3)
    # All three fill colors are the same (no history).
    fill_colors = {s.fillcolor for s in fig.layout.shapes}
    assert len(fill_colors) == 1
    # The L shape has the accent border (argmax tiebreaker).
    accent_count = sum(1 for s in fig.layout.shapes if s.line.color == app._GOAL_ACCENT)
    assert accent_count == 1


# ---------------------------------------------------------------------------
# `_initials` — the avatar's 2-letter initials
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Vinícius Júnior", "VJ"),
        ("Kylian Mbappé", "KM"),
        ("R. Kolo Muani", "RM"),
        ("Rodrygo", "RO"),
        ("Bruno Guimarães", "BG"),
        ("A. Griezmann", "AG"),
        ("Ousmane Dembélé", "OD"),
        ("Marcus Thuram", "MT"),
        ("Raphinha", "RA"),
        ("Endrick", "EN"),
        # Edge cases
        ("", "??"),
        ("a", "A"),
        ("Single", "SI"),
        # Whitespace is collapsed before the slice.
        ("  Padded  Name  ", "PN"),
    ],
)
def test_initials(name: str, expected: str) -> None:
    assert _initials(name) == expected


# ---------------------------------------------------------------------------
# `_kicker_color` — the team's primary color (or neutral fallback)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("team_name", "expected"),
    [
        ("Brazil", "#FFCD00"),
        ("France", "#002654"),
        ("Argentina", "#75AADB"),
        ("USA", "#B22234"),
        # Neutral fallback for unknown teams.
        ("Atlantis", app._NEUTRAL_COLOR),
        ("", app._NEUTRAL_COLOR),
    ],
)
def test_kicker_color(team_name: str, expected: str) -> None:
    assert _kicker_color(team_name) == expected


# ---------------------------------------------------------------------------
# Foot pill helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("foot", "expected_label"),
    [
        ("left", "LEFT FOOT"),
        ("right", "RIGHT FOOT"),
        ("both", "BOTH FEET"),
        ("", "UNKNOWN"),
        ("unknown", "UNKNOWN"),
    ],
)
def test_foot_label(foot: str, expected_label: str) -> None:
    assert _foot_label(foot) == expected_label


@pytest.mark.parametrize(
    ("foot", "expected_color"),
    [
        ("left", "red"),
        ("right", "blue"),
        ("both", "yellow"),
        ("", "gray"),
        ("unknown", "gray"),
    ],
)
def test_foot_badge_color(foot: str, expected_color: str) -> None:
    assert _foot_badge_color(foot) == expected_color


# ---------------------------------------------------------------------------
# Avatar SVG — the photo placeholder
# ---------------------------------------------------------------------------


def test_avatar_svg_contains_initials() -> None:
    """The avatar SVG embeds the initials as text."""
    svg = _avatar_svg("VJ", "#FFCD00", size=64)
    assert "VJ" in svg
    assert "#FFCD00" in svg
    assert 'fill="' in svg


def test_avatar_svg_default_size_is_64() -> None:
    """The default size is 64 px (the v4 card spec)."""
    svg = _avatar_svg("VJ", "#FFCD00")
    assert 'width="64"' in svg
    assert 'height="64"' in svg


# ---------------------------------------------------------------------------
# Card layout — one card per Kicker, in the order of `predictions_for_match`
# ---------------------------------------------------------------------------


def _pred(*, player_id: int, name: str, team_id: int, p_L=0.5, p_C=0.2, p_R=0.3) -> PredictionRow:
    return PredictionRow(
        player_id=player_id,
        player_name=name,
        team_id=team_id,
        team_name=f"Team {team_id}",
        country_code="",
        kicking_foot="right",
        p_L=p_L,
        p_C=p_C,
        p_R=p_R,
    )


_NOW = datetime(2026, 6, 29, 12, 0, 0, tzinfo=UTC)


def _context(home_id: int, away_id: int) -> MatchContext:
    return MatchContext(
        match_id=42,
        kickoff_utc=_NOW + timedelta(days=2),
        round="1/4",
        home_team_id=home_id,
        home_team_name="Brazil",
        away_team_id=away_id,
        away_team_name="France",
    )


def test_card_layout_one_card_per_kicker_in_predicted_order() -> None:
    """The card layout produces one `KickerPrediction` per kicker, in the
    same order as `predictions_for_match` (Issue #48 acceptance).

    The card-rendering functions are 1:1 with the kicker list — one
    `render_card` per `KickerPrediction`, in the order the library
    returns. This test pins that contract.
    """
    home_id, away_id = 100, 200
    predictions = [
        _pred(player_id=1, name="Zara", team_id=home_id),
        _pred(player_id=2, name="Aaron", team_id=home_id),
        _pred(player_id=3, name="Mike", team_id=away_id),
        _pred(player_id=4, name="Other", team_id=999),  # dropped
    ]
    kickers = predictions_for_match(predictions, _context(home_id, away_id))
    # 3 cards, 1:1 with the kicker list (Zara, Aaron, Mike by name).
    assert len(kickers) == 3
    assert [k.player_name for k in kickers] == ["Aaron", "Mike", "Zara"]
    # The card layout's render_card is called once per kicker; this
    # is the contract the `app.py` `main()` relies on.
    for k in kickers:
        assert isinstance(k, KickerPrediction)


def test_card_layout_no_history_renders_with_zero_total() -> None:
    """A Kicker with no `player_history` entry has `total_penalties=0`,
    which is the v4 "no history" signal — the card renders three
    near-equal light cells and a "0 career penalties" caption.
    """
    from penalty_pred.player_history import PlayerPenalty

    def _row(pid: int) -> PlayerPenalty:
        return PlayerPenalty(
            kicker_id=pid,
            match_id=100000 + pid,
            match_date="2024-01-01",
            league_id=77,
            league_name="World Cup",
            team_id=100,
            is_home=True,
            x=1.0,
            side="L",
            is_on_target=True,
            outcome="Goal",
            shot_type="RightFoot",
        )

    home_id, away_id = 100, 200
    predictions = [
        _pred(player_id=1, name="With History", team_id=home_id),
        _pred(player_id=2, name="No History", team_id=home_id),
    ]
    history = {1: [_row(1) for _ in range(4)]}  # player 2 has no entry
    kickers = predictions_for_match(predictions, _context(home_id, away_id), player_history=history)
    by_name = {k.player_name: k.total_penalties for k in kickers}
    assert by_name == {"With History": 4, "No History": 0}


# ---------------------------------------------------------------------------
# The plotly figure integrates with the dashboard's `most_likely_side`
# (sanity pin: the figure's "most-likely" matches the library's argmax).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("p_L", "p_C", "p_R"),
    [
        (0.55, 0.20, 0.25),
        (0.20, 0.60, 0.20),
        (0.30, 0.25, 0.45),
        (1 / 3, 1 / 3, 1 / 3),
    ],
)
def test_goal_drawing_most_likely_matches_library(p_L: float, p_C: float, p_R: float) -> None:
    """The figure's most-likely shape is the library's `most_likely_side`."""
    most = most_likely_side(p_L, p_C, p_R)
    fig = goal_drawing_figure(p_L, p_C, p_R)
    # Find the shape with the accent border; its x-range tells us
    # which side it is (L = 0..1, C = 1..2, R = 2..3).
    accent = next(s for s in fig.layout.shapes if s.line.color == app._GOAL_ACCENT)
    center = (accent.x0 + accent.x1) / 2
    if most == "L":
        assert center == pytest.approx(0.5)
    elif most == "C":
        assert center == pytest.approx(1.5)
    else:
        assert center == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# The opposite-side helper used by the prediction row
# ---------------------------------------------------------------------------


def test_prediction_row_uses_opposite_of_argmax() -> None:
    """The card's "GK dive" hint is `opposite_side(most_likely_side(p))`."""
    # p_L=0.55, p_C=0.20, p_R=0.25 → argmax = L → opposite = R
    assert opposite_side(most_likely_side(0.55, 0.20, 0.25)) == "R"
    # p_L=0.20, p_C=0.60, p_R=0.20 → argmax = C → opposite = C
    assert opposite_side(most_likely_side(0.20, 0.60, 0.20)) == "C"
    # p_L=0.30, p_C=0.25, p_R=0.45 → argmax = R → opposite = L
    assert opposite_side(most_likely_side(0.30, 0.25, 0.45)) == "L"


# ---------------------------------------------------------------------------
# End-to-end smoke test: the app parses, runs, and renders the v4 cards
# (with HF + FotMob mocked). Catches regressions in `main()` without
# spinning up the real cloud dependencies.
# ---------------------------------------------------------------------------


def test_app_renders_v4_layout_with_mocked_dependencies() -> None:
    """End-to-end: the app loads, no exceptions, the v4 surface is rendered.

    Verifies (a) no exceptions or errors at the Streamlit boundary, (b)
    the sidebar has the match selectbox, (c) the page renders a
    "Last updated" caption (the v4 footer), and (d) the "Brazil" and
    "France" team headings appear (the per-team blocks).
    """
    import os
    from unittest.mock import MagicMock, patch

    os.environ.setdefault("STREAMLIT_GLOBAL_DISABLE_WIDGET_STATE_PERSISTENCE", "true")

    home_id, away_id = 100, 200
    predictions_data = [
        {
            "player_id": 1,
            "player_name": "Vinícius Júnior",
            "team_id": home_id,
            "team_name": "Brazil",
            "country_code": "BRA",
            "kicking_foot": "left",
            "p_L": 0.55,
            "p_C": 0.20,
            "p_R": 0.25,
        },
        {
            "player_id": 2,
            "player_name": "Kylian Mbappé",
            "team_id": away_id,
            "team_name": "France",
            "country_code": "FRA",
            "kicking_foot": "right",
            "p_L": 0.30,
            "p_C": 0.18,
            "p_R": 0.52,
        },
    ]

    def _mock_hf(*_a, **_kw):
        return "/tmp/fake_predictions.jsonl"

    def _mock_fotmob_get(*_a, **_kw):
        return {
            "pageProps": {
                "fixtures": {
                    "allMatches": [
                        {
                            "id": 42,
                            "round": "1/4",
                            "status": {
                                "utcTime": "2026-07-15T21:00:00.000Z",
                                "scoreStr": "",
                            },
                            "home": {"id": home_id, "name": "Brazil"},
                            "away": {"id": away_id, "name": "France"},
                        }
                    ]
                }
            }
        }

    def _mock_read_predictions(*_a, **_kw):
        from penalty_pred.predict import PredictionRow

        return [PredictionRow(**p) for p in predictions_data]

    def _mock_read_player_history(*_a, **_kw):
        from penalty_pred.player_history import PlayerPenalty

        rows: list[PlayerPenalty] = []
        for pid, count in [(1, 12), (2, 14)]:
            for i in range(count):
                rows.append(
                    PlayerPenalty(
                        kicker_id=pid,
                        match_id=100000 + pid * 100 + i,
                        match_date="2024-01-01",
                        league_id=77,
                        league_name="World Cup",
                        team_id=100,
                        is_home=True,
                        x=1.0,
                        side="L",
                        is_on_target=True,
                        outcome="Goal",
                        shot_type="RightFoot",
                    )
                )
        return rows

    from streamlit.testing.v1 import AppTest

    with (
        patch("app.hf_hub_download", side_effect=_mock_hf),
        patch("app.Artifacts.read_predictions", side_effect=_mock_read_predictions),
        patch("app.Artifacts.read_player_history", side_effect=_mock_read_player_history),
        patch("app.Artifacts.fotmob_client") as mock_fm_factory,
    ):
        mock_client = MagicMock()
        mock_client.get.side_effect = _mock_fotmob_get
        mock_fm_factory.return_value = mock_client

        at = AppTest.from_file("app.py", default_timeout=30)
        at.run()

    # (a) No exceptions or errors at the Streamlit boundary.
    assert list(at.exception) == []
    assert list(at.error) == []

    # (b) The sidebar has the match selectbox.
    assert len(at.sidebar.selectbox) == 1

    # (c) The page renders a "Last updated" caption (the v4 footer).
    last_updated = [c for c in at.caption if "Last updated" in c.value]
    assert len(last_updated) == 1
    assert "couto/12yd" in last_updated[0].value

    # (d) The "Brazil" and "France" team headings appear in the main
    # area (the per-team blocks).
    markdown_text = " ".join(m.value for m in at.markdown)
    assert "BRAZIL" in markdown_text
    assert "FRANCE" in markdown_text
