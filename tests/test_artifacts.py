"""Tests for the v5 Artifacts on-disk layout adapter.

v5 changes:
- Artifacts root is "data" (was "output")
- Surviving artifacts: player_history, missing_history, roster, predictions
- Dropped: shootout_kicks, training_table, model pickles, metrics, cv, diagnostics,
  discrepancies, tournament_success_rate, lightgbm_model, baseline_model
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests._factories import (
    make_history_row,
    make_prediction_row,
    make_roster_player,
)
from twelveyards.artifacts import Artifacts
from twelveyards.fotmob.client import FotMobClient

# ---------------------------------------------------------------------------
# Path accessors
# ---------------------------------------------------------------------------


def test_default_paths_point_at_canonical_filenames() -> None:
    art = Artifacts()
    assert art.root == Path("data")
    assert art.player_history == Path("data/player_history.jsonl")
    assert art.missing_history == Path("data/missing_history.jsonl")
    assert art.roster == Path("data/wc2026_roster.jsonl")
    assert art.predictions == Path("data/predictions.jsonl")


def test_custom_root_redirects_every_artifact() -> None:
    art = Artifacts(root=Path("/tmp/foo"))
    assert art.player_history == Path("/tmp/foo/player_history.jsonl")
    assert art.missing_history == Path("/tmp/foo/missing_history.jsonl")
    assert art.roster == Path("/tmp/foo/wc2026_roster.jsonl")
    assert art.predictions == Path("/tmp/foo/predictions.jsonl")


# ---------------------------------------------------------------------------
# fotmob_client factory
# ---------------------------------------------------------------------------


def test_fotmob_client_factory() -> None:
    art = Artifacts()
    client = art.fotmob_client()
    assert isinstance(client, FotMobClient)


# ---------------------------------------------------------------------------
# v5 has no old artifacts
# ---------------------------------------------------------------------------


def test_v5_has_no_shootout_kicks() -> None:
    art = Artifacts()
    assert not hasattr(art, "shootout_kicks")


def test_v5_has_no_training_table() -> None:
    art = Artifacts()
    assert not hasattr(art, "training_table")


def test_v5_has_no_lightgbm_model() -> None:
    art = Artifacts()
    assert not hasattr(art, "lightgbm_model")


def test_v5_has_no_baseline_model() -> None:
    art = Artifacts()
    assert not hasattr(art, "baseline_model")


def test_v5_has_no_metrics() -> None:
    art = Artifacts()
    assert not hasattr(art, "metrics")


# ---------------------------------------------------------------------------
# JSONL round-trips
# ---------------------------------------------------------------------------


def test_player_history_round_trip(tmp_path: Path) -> None:
    art = Artifacts(root=tmp_path)
    rows = [make_history_row(match_id=99, match_date="2022-01-01T00:00:00+00:00")]
    n = art.write_player_history(rows, path=art.player_history)
    assert n == 1
    assert art.read_player_history() == rows


def test_missing_history_round_trip(tmp_path: Path) -> None:
    from twelveyards.scraper.initial_set import MissingKicker

    art = Artifacts(root=tmp_path)
    rows = [MissingKicker(player_id=1, player_name="No History", team_id=100, team_name="T")]
    n = art.write_missing_history(rows, path=art.missing_history)
    assert n == 1
    assert art.read_missing_history() == rows


def test_roster_round_trip(tmp_path: Path) -> None:
    art = Artifacts(root=tmp_path)
    rows = [make_roster_player()]
    n = art.write_roster(rows, path=art.roster)
    assert n == 1
    assert art.read_roster() == rows


def test_predictions_round_trip(tmp_path: Path) -> None:
    art = Artifacts(root=tmp_path)
    rows = [make_prediction_row()]
    n = art.write_predictions(rows, path=art.predictions)
    assert n == 1
    assert art.read_predictions() == rows


def test_predictions_round_trip_with_v5_fields(tmp_path: Path) -> None:
    art = Artifacts(root=tmp_path)
    rows = [
        make_prediction_row(
            player_id=42,
            player_name="Lionel Messi",
            short_name="Messi",
            photo_url="https://images.fotmob.com/image_resources/playerimages/42.png",
            total_penalties=12,
        ),
    ]
    n = art.write_predictions(rows, path=art.predictions)
    assert n == 1
    back = art.read_predictions()
    assert back[0].short_name == "Messi"
    assert back[0].total_penalties == 12
    assert "fotmob" in back[0].photo_url


# ---------------------------------------------------------------------------
# Read raises for missing JSONL
# ---------------------------------------------------------------------------


def test_read_missing_jsonl_raises(tmp_path: Path) -> None:
    art = Artifacts(root=tmp_path)
    with pytest.raises(FileNotFoundError):
        art.read_player_history()
    with pytest.raises(FileNotFoundError):
        art.read_roster()
    with pytest.raises(FileNotFoundError):
        art.read_predictions()
    with pytest.raises(FileNotFoundError):
        art.read_missing_history()


def test_write_creates_parent_directories(tmp_path: Path) -> None:
    art = Artifacts(root=tmp_path / "deep" / "nested")
    art.write_player_history([make_history_row()])
    assert art.player_history.exists()


# ---------------------------------------------------------------------------
# serialize_row
# ---------------------------------------------------------------------------


def test_serialize_row_matches_write_shape(tmp_path: Path) -> None:
    import json

    art = Artifacts(root=tmp_path)
    row = make_roster_player()
    text = art.serialize_row(row)
    payload = json.loads(text)
    assert payload["player_id"] == row.player_id
    assert payload["player_name"] == row.player_name


# ---------------------------------------------------------------------------
# Explicit path override on read/write
# ---------------------------------------------------------------------------


def test_explicit_path_round_trip(tmp_path: Path) -> None:
    art = Artifacts(root=tmp_path)
    custom = tmp_path / "custom.jsonl"
    rows = [make_roster_player(player_id=99)]
    n = art.write_roster(rows, path=custom)
    assert n == 1
    assert art.read_roster(path=custom) == rows
