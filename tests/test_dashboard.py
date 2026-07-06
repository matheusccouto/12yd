"""Tests for the dashboard's library logic.

The dashboard is a thin Streamlit app over four library functions
(plus `is_placeholder_team`). The tests pin the library behaviour so
the Streamlit UI doesn't need to be unit-tested. The end-to-end check
(Streamlit Cloud deployment) is a manual checklist per the PRD.

v3 (Issue #36) collapsed the per-match re-score path: the dashboard
now reads `predictions.jsonl` directly. The `predict_match` /
`predict_roster_with_context` tests were removed; the per-match
view is `predictions_for_match(predictions, context)`.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from penalty_pred.dashboard import (
    MatchContext,
    is_placeholder_team,
    load_upcoming_knockouts,
    most_likely_side,
    opposite_side,
    predictions_for_match,
    recommended_dive,
)
from penalty_pred.predict import PredictionRow

# ---------------------------------------------------------------------------
# Fixtures: minimal FotMob fixture payloads + a fake FotMobClient.
# ---------------------------------------------------------------------------


class FakeFotMobClient:
    """A minimal `FotMobClientLike` substitute that returns a canned payload.

    The dashboard's `load_upcoming_knockouts` calls `client.get(path, params)`,
    so the fake just needs to support that one method. The canned
    payload is the `pageProps.fixtures.allMatches` shape.
    """

    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.payload = payload
        self.calls: list[tuple[str, Mapping[str, str] | None]] = []

    def get(self, path: str, params: Mapping[str, str] | None = None) -> Any:
        self.calls.append((path, params))
        return self.payload


def _fixture(
    *,
    match_id: int,
    round_name: str,
    utc_time: str,
    home_id: int,
    home_name: str,
    away_id: int,
    away_name: str,
) -> dict[str, Any]:
    """Build one FotMob fixture entry (the shape inside `allMatches`)."""
    return {
        "id": match_id,
        "pageUrl": f"/matches/foo/bar#{match_id}",
        "round": round_name,
        "status": {"utcTime": utc_time, "scoreStr": ""},
        "home": {"id": home_id, "name": home_name},
        "away": {"id": away_id, "name": away_name},
    }


def _payload(fixtures: list[dict[str, Any]]) -> dict[str, Any]:
    """Wrap a fixture list in the `pageProps.fixtures` shape."""
    return {"pageProps": {"fixtures": {"allMatches": fixtures}}}


# ---------------------------------------------------------------------------
# `is_placeholder_team`
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "team_id", "expected"),
    [
        ("Canada", 5810, False),
        ("Mexico", 6710, False),
        ("Argentina", 1983, False),
        ("", 0, True),
        ("", 5810, True),  # missing name wins over id
        ("Winner EF 1", 1440908, True),
        ("Winner QF 1", 2036, True),
        ("Loser SF 1", 1981, True),
        ("Loser SF 2", 1982, True),
        ("Netherlands/Morocco", 2055187, True),  # group winner TBD
        ("Argentina/Cape Verde", 2056838, True),
        ("Winner", 1440908, True),  # bare "Winner" without space also placeholder
    ],
)
def test_is_placeholder_team(name: str, team_id: int, expected: bool) -> None:
    assert is_placeholder_team(name, team_id) is expected


# ---------------------------------------------------------------------------
# `load_upcoming_knockouts` — the three filters + sort + URL contract
# ---------------------------------------------------------------------------


# An arbitrary "now" pinned for the test cases below. The fixtures
# are chosen so the upcoming/past split lands on either side of this
# instant.
_NOW = datetime(2026, 6, 29, 12, 0, 0, tzinfo=UTC)
_NOW_ISO = _NOW.isoformat()


def _kickoff(days_from_now: int) -> str:
    """A kickoff time `days_from_now` after the pinned `_NOW`."""
    return (_NOW + timedelta(days=days_from_now)).isoformat()


@pytest.mark.parametrize(
    "round_name",
    ["1/16", "1/8", "1/4", "1/2", "final"],
)
def test_load_upcoming_knockouts_accepts_every_round(round_name: str) -> None:
    """Any round passes the filter when both teams are real and the kickoff is upcoming.

    The 48-team WC adds Round of 32 (FotMob code `"1/16"`); the same
    code must work whether the tournament is in R32, R16, QF, SF, or
    F. The round is display-only.
    """
    fixtures = [
        _fixture(
            match_id=42,
            round_name=round_name,
            utc_time=_kickoff(5),
            home_id=5810,
            home_name="Brazil",
            away_id=6710,
            away_name="Japan",
        ),
    ]
    out = load_upcoming_knockouts(FakeFotMobClient(_payload(fixtures)), now=_NOW)
    assert [m.match_id for m in out] == [42]
    assert out[0].round == round_name


def test_load_upcoming_knockouts_returns_only_knockout_rounds() -> None:
    """The round is no longer a filter; this test exercises the placeholder check
    for fixtures whose teams are group-stage placeholders (`A/B`) or
    undecided-knockout placeholders (`Winner N` / `Loser N`).

    Historical behaviour — kept as a regression pin for the case where
    the fixture list mixes group-stage and knockout rounds. The current
    filter relies on the placeholder check to drop group-stage matches
    (their opponents are joined by `/`, e.g. "Netherlands/Morocco"),
    so this test only exercises the placeholder drop, not a round
    allowlist.
    """
    fixtures = [
        # group stage — opponent joined by `/`, dropped by placeholder check
        _fixture(
            match_id=1,
            round_name="1",
            utc_time=_kickoff(5),
            home_id=10,
            home_name="Netherlands/Morocco",
            away_id=20,
            away_name="Croatia",
        ),
        # bronze — both teams are "Loser SF N" placeholders, dropped
        _fixture(
            match_id=3,
            round_name="bronze",
            utc_time=_kickoff(20),
            home_id=12,
            home_name="Loser SF 1",
            away_id=13,
            away_name="Loser SF 2",
        ),
        # knockout — should be kept
        _fixture(
            match_id=4,
            round_name="1/8",
            utc_time=_kickoff(7),
            home_id=30,
            home_name="Canada",
            away_id=40,
            away_name="Mexico",
        ),
    ]
    out = load_upcoming_knockouts(FakeFotMobClient(_payload(fixtures)), now=_NOW)
    assert [m.match_id for m in out] == [4]


def test_load_upcoming_knockouts_drops_past_matches() -> None:
    """A match with kickoff <= now is dropped (the user can't pick a past match)."""
    fixtures = [
        _fixture(
            match_id=10,
            round_name="1/8",
            utc_time=_kickoff(-1),  # yesterday — past
            home_id=30,
            home_name="Canada",
            away_id=40,
            away_name="Mexico",
        ),
        _fixture(
            match_id=11,
            round_name="1/8",
            utc_time=_kickoff(0),  # exactly now — past (strict inequality)
            home_id=30,
            home_name="Canada",
            away_id=40,
            away_name="Mexico",
        ),
        _fixture(
            match_id=12,
            round_name="1/8",
            utc_time=_kickoff(1),  # tomorrow — upcoming
            home_id=30,
            home_name="Canada",
            away_id=40,
            away_name="Mexico",
        ),
    ]
    out = load_upcoming_knockouts(FakeFotMobClient(_payload(fixtures)), now=_NOW)
    assert [m.match_id for m in out] == [12]


@pytest.mark.parametrize(
    ("round_name", "home_name", "home_id", "away_name", "away_id"),
    [
        # "Winner N" — undecided knockout slot naming the prior round
        ("1/4", "Winner EF 1", 1440908, "France", 9826),
        ("1/2", "Winner QF 1", 2036, "Brazil", 5810),
        ("final", "Winner SF 1", 1981, "Winner SF 2", 1982),
        # "Loser N" — undecided knockout slot (3rd-place play-off etc.)
        ("1/2", "Loser QF 1", 2001, "Loser QF 2", 2002),
        ("1/4", "Loser R16 1", 2001, "Loser R16 2", 2002),
        # "A/B" — group stage opponent whose winner will fill the slot
        ("1/8", "Netherlands/Morocco", 2055187, "France", 9826),
        ("1/16", "Argentina/Cape Verde", 2056838, "France", 9826),
        # empty name with a real id — name wins over id
        ("final", "", 5810, "France", 9826),
        # empty name with id=0 — placeholder
        ("1/4", "", 0, "France", 9826),
        # R32 round (1/16) is the most recent addition; it must be filtered
        # by the placeholder check too, not the round
        ("1/16", "Winner AB 1", 2055187, "France", 9826),
    ],
)
def test_load_upcoming_knockouts_drops_placeholder_teams(
    round_name: str,
    home_name: str,
    home_id: int,
    away_name: str,
    away_id: int,
) -> None:
    """A match with a placeholder team on either side is dropped, regardless of round.

    Covers the four placeholder shapes FotMob emits: `Winner N` /
    `Loser N` (undecided knockout slot), `A/B` (group stage opponent
    TBD), and empty name (with or without a non-zero id). The round
    is not consulted; only the placeholder check matters.
    """
    fixtures = [
        _fixture(
            match_id=20,
            round_name=round_name,
            utc_time=_kickoff(10),
            home_id=home_id,
            home_name=home_name,
            away_id=away_id,
            away_name=away_name,
        ),
        # Real teams on both sides — should be kept as a sanity check.
        _fixture(
            match_id=22,
            round_name="1/8",
            utc_time=_kickoff(5),
            home_id=5810,
            home_name="Canada",
            away_id=6710,
            away_name="Mexico",
        ),
    ]
    out = load_upcoming_knockouts(FakeFotMobClient(_payload(fixtures)), now=_NOW)
    assert [m.match_id for m in out] == [22]


def test_load_upcoming_knockouts_drops_placeholders() -> None:
    """A match with a placeholder team (Winner X, Loser X, or "A/B" group TBD) is dropped.

    Kept as a non-parametrized regression pin so the test name remains
    self-documenting in pytest output; the parametrized version above
    covers the full placeholder × round cross product.
    """
    fixtures = [
        _fixture(
            match_id=20,
            round_name="1/4",
            utc_time=_kickoff(10),
            home_id=1440908,
            home_name="Winner EF 1",
            away_id=1440909,
            away_name="Winner EF 2",
        ),
        _fixture(
            match_id=21,
            round_name="1/8",
            utc_time=_kickoff(5),
            home_id=2055187,
            home_name="Netherlands/Morocco",
            away_id=2056841,
            away_name="France",
        ),
        # Real teams on both sides — should be kept.
        _fixture(
            match_id=22,
            round_name="1/8",
            utc_time=_kickoff(5),
            home_id=5810,
            home_name="Canada",
            away_id=6710,
            away_name="Mexico",
        ),
    ]
    out = load_upcoming_knockouts(FakeFotMobClient(_payload(fixtures)), now=_NOW)
    assert [m.match_id for m in out] == [22]


def test_load_upcoming_knockouts_sorts_by_kickoff() -> None:
    """The returned list is sorted by `kickoff_utc` ascending (nearest first)."""
    fixtures = [
        _fixture(
            match_id=30,
            round_name="1/8",
            utc_time=_kickoff(10),
            home_id=30,
            home_name="Canada",
            away_id=40,
            away_name="Mexico",
        ),
        _fixture(
            match_id=31,
            round_name="1/8",
            utc_time=_kickoff(2),
            home_id=30,
            home_name="Canada",
            away_id=40,
            away_name="Mexico",
        ),
        _fixture(
            match_id=32,
            round_name="1/4",
            utc_time=_kickoff(5),
            home_id=30,
            home_name="Canada",
            away_id=40,
            away_name="Mexico",
        ),
    ]
    out = load_upcoming_knockouts(FakeFotMobClient(_payload(fixtures)), now=_NOW)
    assert [m.match_id for m in out] == [31, 32, 30]


def test_load_upcoming_knockouts_calls_fotmob_with_known_args() -> None:
    """The FotMob URL is `leagues/77/overview/world-cup?season=2026`."""
    client = FakeFotMobClient(_payload([]))
    load_upcoming_knockouts(client, now=_NOW)
    assert client.calls == [("leagues/77/overview/world-cup", {"season": "2026"})]


def test_load_upcoming_knockouts_drops_unparseable_kickoffs() -> None:
    """A fixture with a missing/malformed `utcTime` is dropped (not raised)."""
    fixtures = [
        _fixture(
            match_id=40,
            round_name="1/8",
            utc_time="not-a-date",
            home_id=30,
            home_name="Canada",
            away_id=40,
            away_name="Mexico",
        ),
        _fixture(
            match_id=41,
            round_name="1/8",
            utc_time=_kickoff(3),
            home_id=30,
            home_name="Canada",
            away_id=40,
            away_name="Mexico",
        ),
    ]
    out = load_upcoming_knockouts(FakeFotMobClient(_payload(fixtures)), now=_NOW)
    assert [m.match_id for m in out] == [41]


# ---------------------------------------------------------------------------
# `recommended_dive`
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("p_L", "p_C", "p_R", "expected"),
    [
        (0.1, 0.5, 0.4, "L"),
        (0.4, 0.1, 0.5, "C"),
        (0.3, 0.6, 0.1, "R"),
        # L→C→R tiebreaker: when two or more sides tie for the minimum,
        # the earliest in (L, C, R) order wins.
        (0.33, 0.33, 0.34, "L"),  # L and C tied; L wins
        (0.34, 0.33, 0.33, "C"),  # C and R tied; C wins
        (0.33, 0.34, 0.33, "L"),  # L and R tied; L wins
        (0.33, 0.33, 0.33, "L"),  # all three tied; L wins
        # zero / one extremes still work
        (1.0, 0.0, 0.0, "C"),  # C and R tied at 0; L→C→R → C
        (0.0, 1.0, 0.0, "L"),
        (0.0, 0.0, 1.0, "L"),  # L and C tied at 0; L→C→R → L
    ],
)
def test_recommended_dive_argmin(p_L: float, p_C: float, p_R: float, expected: str) -> None:
    assert recommended_dive(p_L, p_C, p_R) == expected


def test_recommended_dive_docstring_pins_kicker_pov() -> None:
    """The docstring makes the Kicker-PoV frame explicit (Issue #47 / v4).

    Cheap pin so a future maintainer doesn't quietly re-frame the
    L/C/R labels to the Goalkeeper's PoV (the v3 dashboard's
    "Recommended Dive" column invited exactly that re-reading).
    """
    assert recommended_dive.__doc__ is not None
    assert "Kicker-PoV" in recommended_dive.__doc__


# ---------------------------------------------------------------------------
# v4 (Issue #48): `opposite_side` and `most_likely_side` — the card's
# "GK dive ↔ X" hint and the "Kicker will aim: X [%]" headline.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("side", "expected"),
    [("L", "R"), ("R", "L"), ("C", "C")],
)
def test_opposite_side(side: str, expected: str) -> None:
    """`opposite_side` is the Kicker's mirror: L↔R, C↔C."""
    assert opposite_side(side) == expected


def test_opposite_side_unknown_passthrough() -> None:
    """An unknown side passes through (the renderer treats the output as a label)."""
    assert opposite_side("?") == "?"


@pytest.mark.parametrize(
    ("p_L", "p_C", "p_R", "expected"),
    [
        (0.55, 0.20, 0.25, "L"),
        (0.30, 0.25, 0.45, "R"),
        (0.20, 0.60, 0.20, "C"),
        # L→C→R tiebreaker mirrors `recommended_dive`'s order.
        (0.33, 0.33, 0.34, "R"),  # R is the unique max
        (0.34, 0.33, 0.33, "L"),
        (0.33, 0.34, 0.33, "C"),
        (0.33, 0.33, 0.33, "L"),  # all tied → L wins
    ],
)
def test_most_likely_side_argmax(p_L: float, p_C: float, p_R: float, expected: str) -> None:
    """`most_likely_side` is the argmax; ties break L→C→R (deterministic)."""
    assert most_likely_side(p_L, p_C, p_R) == expected


def test_most_likely_side_docstring_pins_kicker_pov() -> None:
    """`most_likely_side`'s docstring stays in the Kicker's frame."""
    assert most_likely_side.__doc__ is not None
    assert "Kicker" in most_likely_side.__doc__


# ---------------------------------------------------------------------------
# `predictions_for_match` — per-match view over the round-agnostic
# `predictions.jsonl`
# ---------------------------------------------------------------------------


def _pred(
    *, player_id: int, name: str, team_id: int, p_L: float = 0.5, p_C: float = 0.2, p_R: float = 0.3
) -> PredictionRow:
    """Build a `PredictionRow` for the dashboard's per-match view tests."""
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


def test_predictions_for_match_filters_to_match_teams() -> None:
    """Only predictions on the home or away team are kept; other teams are dropped."""
    home_id = 100
    away_id = 200
    other_id = 999
    predictions = [
        _pred(player_id=1, name="Home Striker", team_id=home_id),
        _pred(player_id=2, name="Away Striker", team_id=away_id),
        _pred(player_id=3, name="Other Striker", team_id=other_id),  # dropped
    ]
    context = MatchContext(
        match_id=42,
        kickoff_utc=_NOW + timedelta(days=2),
        round="1/4",
        home_team_id=home_id,
        home_team_name="Home FC",
        away_team_id=away_id,
        away_team_name="Away FC",
    )
    out = predictions_for_match(predictions, context)
    assert {k.player_id for k in out} == {1, 2}


def test_predictions_for_match_sets_recommended_dive() -> None:
    """`recommended_dive` is the argmin over the prediction's probabilities."""
    predictions = [_pred(player_id=1, name="K", team_id=100, p_L=0.1, p_C=0.6, p_R=0.3)]
    context = MatchContext(
        match_id=42,
        kickoff_utc=_NOW + timedelta(days=2),
        round="final",
        home_team_id=100,
        home_team_name="H",
        away_team_id=200,
        away_team_name="A",
    )
    [kicker] = predictions_for_match(predictions, context)
    assert kicker.recommended_dive == "L"
    assert kicker.p_L == pytest.approx(0.1)
    assert kicker.p_C == pytest.approx(0.6)
    assert kicker.p_R == pytest.approx(0.3)


def test_predictions_for_match_sorts_by_total_penalties_desc() -> None:
    """v4 (Issue #48): cards are sorted by `total_penalties` desc, name asc as tiebreaker.

    The sort reads `total_penalties` from the per-kicker
    `player_history` length. Most-experienced kickers float to the
    top of the page; a name tiebreaker keeps the order stable.
    """
    from penalty_pred.player_history import PlayerPenalty

    def _row(pid: int, d: str) -> PlayerPenalty:
        return PlayerPenalty(
            kicker_id=pid,
            match_id=100000 + pid,
            match_date=d,
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

    predictions = [
        _pred(player_id=1, name="Zara", team_id=100),
        _pred(player_id=2, name="Aaron", team_id=100),
        _pred(player_id=3, name="Mike", team_id=100),
    ]
    context = MatchContext(
        match_id=42,
        kickoff_utc=_NOW + timedelta(days=2),
        round="1/4",
        home_team_id=100,
        home_team_name="H",
        away_team_id=200,
        away_team_name="A",
    )
    history = {
        1: [_row(1, "2024-01-01") for _ in range(2)],  # Zara: 2 penalties
        2: [_row(2, "2024-01-01") for _ in range(8)],  # Aaron: 8 penalties (top)
        3: [_row(3, "2024-01-01") for _ in range(5)],  # Mike: 5 penalties
    }
    out = predictions_for_match(predictions, context, player_history=history)
    assert [k.player_name for k in out] == ["Aaron", "Mike", "Zara"]
    assert [k.total_penalties for k in out] == [8, 5, 2]


def test_predictions_for_match_name_tiebreaker_when_penalties_equal() -> None:
    """When two kickers have the same `total_penalties`, name (ascending) is the tiebreaker."""
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

    predictions = [
        _pred(player_id=1, name="Zara", team_id=100),
        _pred(player_id=2, name="Aaron", team_id=100),
    ]
    context = MatchContext(
        match_id=42,
        kickoff_utc=_NOW + timedelta(days=2),
        round="1/4",
        home_team_id=100,
        home_team_name="H",
        away_team_id=200,
        away_team_name="A",
    )
    history = {
        1: [_row(1) for _ in range(3)],
        2: [_row(2) for _ in range(3)],
    }
    out = predictions_for_match(predictions, context, player_history=history)
    assert [k.player_name for k in out] == ["Aaron", "Zara"]


def test_predictions_for_match_falls_back_to_name_sort_without_history() -> None:
    """Without `player_history`, `total_penalties` is 0 for every kicker
    and the sort degrades to name-ascending (the v3 behaviour, kept
    for backward compatibility with callers that don't load history).
    """
    predictions = [
        _pred(player_id=1, name="Zara", team_id=100),
        _pred(player_id=2, name="Aaron", team_id=100),
        _pred(player_id=3, name="Mike", team_id=100),
    ]
    context = MatchContext(
        match_id=42,
        kickoff_utc=_NOW + timedelta(days=2),
        round="1/4",
        home_team_id=100,
        home_team_name="H",
        away_team_id=200,
        away_team_name="A",
    )
    out = predictions_for_match(predictions, context)  # no player_history
    assert [k.player_name for k in out] == ["Aaron", "Mike", "Zara"]
    assert all(k.total_penalties == 0 for k in out)


def test_predictions_for_match_no_history_key_means_zero() -> None:
    """A kicker with no entry in `player_history` has `total_penalties=0`
    (the v4 "no history" signal — the card renders three near-equal
    light cells).
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

    predictions = [
        _pred(player_id=1, name="With History", team_id=100),
        _pred(player_id=2, name="No History", team_id=100),
    ]
    context = MatchContext(
        match_id=42,
        kickoff_utc=_NOW + timedelta(days=2),
        round="1/4",
        home_team_id=100,
        home_team_name="H",
        away_team_id=200,
        away_team_name="A",
    )
    history = {1: [_row(1) for _ in range(4)]}  # player 2 has no entry
    out = predictions_for_match(predictions, context, player_history=history)
    by_id = {k.player_id: k.total_penalties for k in out}
    assert by_id == {1: 4, 2: 0}


def test_predictions_for_match_drops_zero_team_id() -> None:
    """A prediction with `team_id=0` (defensive — shouldn't appear in
    `predictions.jsonl` but the live roster may emit a 0 placeholder)
    is dropped (it's neither home nor away)."""
    home_id = 100
    away_id = 200
    predictions = [
        _pred(player_id=1, name="Home", team_id=home_id),
        _pred(player_id=2, name="Zero", team_id=0),
    ]
    context = MatchContext(
        match_id=42,
        kickoff_utc=_NOW + timedelta(days=2),
        round="1/4",
        home_team_id=home_id,
        home_team_name="H",
        away_team_id=away_id,
        away_team_name="A",
    )
    out = predictions_for_match(predictions, context)
    assert [k.player_id for k in out] == [1]


# ---------------------------------------------------------------------------
# Test doubles — keep the dashboard tests independent of the model + the
# FotMob client.
# ---------------------------------------------------------------------------


# v3 (Issue #36): the previous `_ConstantModel` test double is gone
# (the re-score path is gone; `predictions_for_match` doesn't take
# a model). The `_NoFetcher` double is also no longer needed (the
# per-match view doesn't need a metadata fetcher).
