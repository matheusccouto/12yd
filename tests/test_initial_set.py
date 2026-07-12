"""Tests for the Initial Set assembly."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tests._factories import make_roster_player
from twelveyards.artifacts import Artifacts
from twelveyards.scraper.initial_set import (
    InitialSetFetchResult,
    InitialSetKicker,
    MissingKicker,
    iter_initial_set_kickers,
)

if TYPE_CHECKING:
    from pathlib import Path

_roster = make_roster_player


# ---------------------------------------------------------------------------
# iter_initial_set_kickers
# ---------------------------------------------------------------------------


def test_iter_initial_set_kickers_empty() -> None:  # noqa: D103
    assert list(iter_initial_set_kickers([])) == []


def test_iter_initial_set_kickers_yields_one_per_roster_player() -> None:  # noqa: D103
    roster = [
        _roster(player_id=1, player_name="Alpha", team_id=100, team_name="Argentina"),
        _roster(player_id=2, player_name="Bravo", team_id=200, team_name="Brazil"),
    ]
    kickers = list(iter_initial_set_kickers(roster))
    assert len(kickers) == 2  # noqa: PLR2004
    assert kickers[0].player_id == 1
    assert kickers[0].team_name == "Argentina"
    assert kickers[1].player_id == 2  # noqa: PLR2004
    assert kickers[1].team_name == "Brazil"


def test_iter_initial_set_kickers_preserves_all_fields() -> None:  # noqa: D103
    roster = [
        _roster(
            player_id=42, player_name="Neymar", team_id=5810, team_name="Brazil",
        ),
    ]
    [k] = list(iter_initial_set_kickers(roster))
    assert k.player_id == 42  # noqa: PLR2004
    assert k.player_name == "Neymar"
    assert k.team_id == 5810  # noqa: PLR2004
    assert k.team_name == "Brazil"


def test_iter_initial_set_kickers_returns_initial_set_kicker_type() -> None:  # noqa: D103
    roster = [_roster(player_id=1)]
    [k] = list(iter_initial_set_kickers(roster))
    assert isinstance(k, InitialSetKicker)


# ---------------------------------------------------------------------------
# InitialSetKicker
# ---------------------------------------------------------------------------


def test_initial_set_kicker_fields() -> None:  # noqa: D103
    k = InitialSetKicker(
        player_id=100, player_name="Alpha", team_id=1, team_name="Argentina",
    )
    assert k.player_id == 100  # noqa: PLR2004
    assert k.player_name == "Alpha"
    assert k.team_id == 1
    assert k.team_name == "Argentina"


# ---------------------------------------------------------------------------
# MissingKicker
# ---------------------------------------------------------------------------


def test_missing_kicker_fields() -> None:  # noqa: D103
    m = MissingKicker(
        player_id=1, player_name="No History", team_id=100, team_name="Argentina",
    )
    assert m.player_id == 1
    assert m.player_name == "No History"
    assert m.team_id == 100  # noqa: PLR2004
    assert m.team_name == "Argentina"


# ---------------------------------------------------------------------------
# InitialSetFetchResult
# ---------------------------------------------------------------------------


def test_initial_set_fetch_result_defaults() -> None:  # noqa: D103
    kicker = InitialSetKicker(player_id=1, player_name="A", team_id=100, team_name="T")
    r = InitialSetFetchResult(kicker=kicker, rows=[])
    assert r.kicker == kicker
    assert r.rows == []
    assert r.error is None


def test_initial_set_fetch_result_with_error() -> None:  # noqa: D103
    kicker = InitialSetKicker(player_id=1, player_name="A", team_id=100, team_name="T")
    r = InitialSetFetchResult(kicker=kicker, rows=[], error="simulated error")
    assert r.error == "simulated error"


# ---------------------------------------------------------------------------
# JSONL roundtrip
# ---------------------------------------------------------------------------


def test_missing_kicker_jsonl_roundtrip(tmp_path: Path) -> None:  # noqa: D103
    rows = [
        MissingKicker(
            player_id=42, player_name="No History", team_id=1, team_name="Argentina",
        ),
        MissingKicker(
            player_id=43, player_name="Also Empty", team_id=2, team_name="Brazil",
        ),
    ]
    out = tmp_path / "missing.jsonl"
    art = Artifacts(root=tmp_path)
    n = art.write_missing_history(rows, path=out)
    assert n == 2  # noqa: PLR2004
    back = art.read_missing_history(path=out)
    assert back == rows
