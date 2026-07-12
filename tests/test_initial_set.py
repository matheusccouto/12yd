"""Tests for the v5 Initial Set assembly.

v5 changes:
- iter_initial_set_kickers takes only roster (drops shootout_kicks)
- No Training Initial Set — only Prediction Initial Set from the roster
- fetch_all_initial_set_penalty_history uses fetch_player_penalty_history
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests._factories import FakeFotMobClient, make_roster_player
from twelveyards.artifacts import Artifacts
from twelveyards.initial_set import (
    InitialSetFetchResult,
    InitialSetKicker,
    MissingKicker,
    fetch_all_initial_set_penalty_history,
    iter_initial_set_kickers,
)

_roster = make_roster_player


# ---------------------------------------------------------------------------
# iter_initial_set_kickers
# ---------------------------------------------------------------------------


def test_iter_initial_set_kickers_empty() -> None:
    assert list(iter_initial_set_kickers([])) == []


def test_iter_initial_set_kickers_yields_one_per_roster_player() -> None:
    roster = [
        _roster(player_id=1, player_name="Alpha", team_id=100, team_name="Argentina"),
        _roster(player_id=2, player_name="Bravo", team_id=200, team_name="Brazil"),
    ]
    kickers = list(iter_initial_set_kickers(roster))
    assert len(kickers) == 2
    assert kickers[0].player_id == 1
    assert kickers[0].team_name == "Argentina"
    assert kickers[1].player_id == 2
    assert kickers[1].team_name == "Brazil"


def test_iter_initial_set_kickers_preserves_all_fields() -> None:
    roster = [_roster(player_id=42, player_name="Neymar", team_id=5810, team_name="Brazil")]
    [k] = list(iter_initial_set_kickers(roster))
    assert k.player_id == 42
    assert k.player_name == "Neymar"
    assert k.team_id == 5810
    assert k.team_name == "Brazil"


def test_iter_initial_set_kickers_returns_initial_set_kicker_type() -> None:
    roster = [_roster(player_id=1)]
    [k] = list(iter_initial_set_kickers(roster))
    assert isinstance(k, InitialSetKicker)


# ---------------------------------------------------------------------------
# InitialSetKicker
# ---------------------------------------------------------------------------


def test_initial_set_kicker_fields() -> None:
    k = InitialSetKicker(player_id=100, player_name="Alpha", team_id=1, team_name="Argentina")
    assert k.player_id == 100
    assert k.player_name == "Alpha"
    assert k.team_id == 1
    assert k.team_name == "Argentina"


# ---------------------------------------------------------------------------
# MissingKicker
# ---------------------------------------------------------------------------


def test_missing_kicker_fields() -> None:
    m = MissingKicker(player_id=1, player_name="No History", team_id=100, team_name="Argentina")
    assert m.player_id == 1
    assert m.player_name == "No History"
    assert m.team_id == 100
    assert m.team_name == "Argentina"


# ---------------------------------------------------------------------------
# InitialSetFetchResult
# ---------------------------------------------------------------------------


def test_initial_set_fetch_result_defaults() -> None:
    kicker = InitialSetKicker(player_id=1, player_name="A", team_id=100, team_name="T")
    r = InitialSetFetchResult(kicker=kicker, rows=[])
    assert r.kicker == kicker
    assert r.rows == []
    assert r.error is None


def test_initial_set_fetch_result_with_error() -> None:
    kicker = InitialSetKicker(player_id=1, player_name="A", team_id=100, team_name="T")
    r = InitialSetFetchResult(kicker=kicker, rows=[], error="simulated error")
    assert r.error == "simulated error"


# ---------------------------------------------------------------------------
# fetch_all_initial_set_penalty_history (stubbed client)
# ---------------------------------------------------------------------------


def test_fetch_all_initial_set_empty() -> None:
    results = list(fetch_all_initial_set_penalty_history(FakeFotMobClient(), []))
    assert results == []


def test_fetch_all_initial_set_yields_one_per_kicker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from twelveyards import player_history as player_history_module

    def fake_fetch(
        client, player_id, player_slug="", target_date=None, lookback_years=5, history_floor=None,
    ):
        return []

    monkeypatch.setattr(player_history_module, "fetch_player_penalty_history", fake_fetch)

    kickers = [
        InitialSetKicker(player_id=1, player_name="A", team_id=100, team_name="T"),
        InitialSetKicker(player_id=2, player_name="B", team_id=200, team_name="U"),
    ]
    results = list(fetch_all_initial_set_penalty_history(FakeFotMobClient(), kickers))
    assert len(results) == 2
    assert results[0].kicker.player_id == 1
    assert results[1].kicker.player_id == 2
    assert results[0].error is None
    assert results[1].error is None


def test_fetch_all_initial_set_captures_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from twelveyards import player_history as player_history_module

    call_count = 0

    def fake_fetch(
        client, player_id, player_slug="", target_date=None, lookback_years=5, history_floor=None,
    ):
        nonlocal call_count
        call_count += 1
        if player_id == 1:
            raise RuntimeError("simulated fetch failure")
        return []

    monkeypatch.setattr(player_history_module, "fetch_player_penalty_history", fake_fetch)

    kickers = [
        InitialSetKicker(player_id=1, player_name="Bad", team_id=100, team_name="T"),
        InitialSetKicker(player_id=2, player_name="Good", team_id=200, team_name="U"),
    ]
    results = list(fetch_all_initial_set_penalty_history(FakeFotMobClient(), kickers))
    assert len(results) == 2
    assert results[0].error is not None
    assert "simulated fetch failure" in results[0].error
    assert results[0].rows == []
    assert results[1].error is None
    assert results[1].rows == []
    assert call_count == 2


# ---------------------------------------------------------------------------
# JSONL roundtrip
# ---------------------------------------------------------------------------


def test_missing_kicker_jsonl_roundtrip(tmp_path: Path) -> None:
    rows = [
        MissingKicker(player_id=42, player_name="No History", team_id=1, team_name="Argentina"),
        MissingKicker(player_id=43, player_name="Also Empty", team_id=2, team_name="Brazil"),
    ]
    out = tmp_path / "missing.jsonl"
    art = Artifacts(root=tmp_path)
    n = art.write_missing_history(rows, path=out)
    assert n == 2
    back = art.read_missing_history(path=out)
    assert back == rows
