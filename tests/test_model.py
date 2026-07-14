"""Tests for the model feature matrix builders and TabPFN prediction pipeline."""

from __future__ import annotations

import json
from datetime import date
from typing import TYPE_CHECKING

import numpy as np

from tests._factories import make_history_row, make_metadata, make_roster_player
from twelveyards.model import (
    CATEGORICAL_INDICES,
    CLASSES,
    PRIOR_PROB,
    build_prediction_matrix,
    build_training_matrix,
    compute_features,
    load_player_history,
    predict_and_write,
    side_distribution,
)

if TYPE_CHECKING:
    from pathlib import Path


_penalty = make_history_row
_metadata = make_metadata
_roster = make_roster_player

TARGET_DATE = date(2025, 7, 1)


# ---------------------------------------------------------------------------
# side_distribution
# ---------------------------------------------------------------------------


def test_side_distribution_empty_returns_prior() -> None:
    assert side_distribution([], 5) == PRIOR_PROB
    assert side_distribution([], 20) == PRIOR_PROB


def test_side_distribution_single_side() -> None:
    assert side_distribution(["L"], 5) == (1.0, 0.0, 0.0)
    assert side_distribution(["C"], 10) == (0.0, 1.0, 0.0)
    assert side_distribution(["R"], 20) == (0.0, 0.0, 1.0)


def test_side_distribution_takes_last_n() -> None:
    p_l, p_c, p_r = side_distribution(["L"] * 10 + ["R"] * 10, 5)
    assert (p_l, p_c, p_r) == (0.0, 0.0, 1.0)


def test_side_distribution_horizons_nest() -> None:
    sides = ["L"] * 10 + ["C"] * 5 + ["R"] * 5
    p5 = side_distribution(sides, 5)
    p10 = side_distribution(sides, 10)
    p20 = side_distribution(sides, 20)
    assert p5 == (0.0, 0.0, 1.0)
    assert p10 == (0.0, 0.5, 0.5)
    assert p20 == (0.5, 0.25, 0.25)


def test_side_distribution_history_shorter_than_horizon() -> None:
    p_l, p_c, p_r = side_distribution(["L", "C"], 20)
    assert p_l == 0.5
    assert p_c == 0.5
    assert p_r == 0.0


# ---------------------------------------------------------------------------
# compute_features
# ---------------------------------------------------------------------------


def test_compute_features_no_history_uses_prior() -> None:
    feats = compute_features(
        [], _metadata(player_id=1, preferred_foot="right"), TARGET_DATE,
    )
    assert feats["p_L"] == 1.0 / 3.0
    assert feats["p_C"] == 1.0 / 3.0
    assert feats["p_R"] == 1.0 / 3.0
    assert feats["last_side"] == ""
    assert feats["preferred_foot"] == "right"
    assert feats["career_penalty_count"] == 0
    assert feats["position"] == "striker"


def test_compute_features_no_metadata() -> None:
    feats = compute_features([], None, TARGET_DATE)
    assert feats["preferred_foot"] == ""
    assert feats["position"] == ""


def test_compute_features_with_history() -> None:
    history = [
        _penalty(1, "2024-01-01T00:00:00+00:00", side="L"),
        _penalty(2, "2024-02-01T00:00:00+00:00", side="L"),
        _penalty(3, "2024-03-01T00:00:00+00:00", side="R"),
        _penalty(4, "2024-04-01T00:00:00+00:00", side="L"),
        _penalty(5, "2024-05-01T00:00:00+00:00", side="R"),
    ]
    feats = compute_features(
        history,
        _metadata(player_id=1, preferred_foot="left", position_key="midfielder"),
        TARGET_DATE,
    )
    assert (feats["p_L"], feats["p_C"], feats["p_R"]) == (0.6, 0.0, 0.4)
    assert feats["last_side"] == "R"
    assert feats["preferred_foot"] == "left"
    assert feats["career_penalty_count"] == 5
    assert feats["position"] == "midfielder"


def test_compute_features_excludes_kicks_on_or_after_target_date() -> None:
    history = [
        _penalty(1, "2024-01-01T00:00:00+00:00", side="L"),
        _penalty(2, "2025-07-01T00:00:00+00:00", side="R"),
    ]
    feats = compute_features(history, None, TARGET_DATE)
    assert feats["career_penalty_count"] == 1
    assert feats["last_side"] == "L"


def test_compute_features_filters_outside_lookback() -> None:
    history = [
        _penalty(1, "2018-01-01T00:00:00+00:00", side="L"),
        _penalty(2, "2024-01-01T00:00:00+00:00", side="R"),
    ]
    feats = compute_features(history, None, date(2025, 1, 1), lookback_years=5)
    assert feats["career_penalty_count"] == 1
    assert feats["last_side"] == "R"


def test_compute_features_returns_seven_keys() -> None:
    feats = compute_features([], _metadata(player_id=1), TARGET_DATE)
    assert set(feats.keys()) == {
        "p_L", "p_C", "p_R", "last_side",
        "preferred_foot", "career_penalty_count", "position",
    }


# ---------------------------------------------------------------------------
# CLASSES and CATEGORICAL_INDICES
# ---------------------------------------------------------------------------


def test_classes() -> None:
    assert CLASSES == ("L", "C", "R")


def test_categorical_indices() -> None:
    assert CATEGORICAL_INDICES == [4, 5, 6]


# ---------------------------------------------------------------------------
# build_training_matrix
# ---------------------------------------------------------------------------


def test_build_training_matrix_empty() -> None:
    x, y = build_training_matrix({}, {})
    assert x.shape == (0, 7)
    assert y.shape == (0,)


def test_build_training_matrix_shapes() -> None:
    p1_kicks = [
        _penalty(1, "2024-01-15T00:00:00+00:00", side="L"),
        _penalty(2, "2024-06-15T00:00:00+00:00", side="R"),
    ]
    x, y = build_training_matrix(
        {1: p1_kicks},
        {1: _metadata(player_id=1, preferred_foot="right", position_key="striker")},
    )
    assert x.shape == (2, 7)
    assert y.shape == (2,)
    assert list(y) == [CLASSES.index("L"), CLASSES.index("R")]


def test_build_training_matrix_no_metadata_for_player() -> None:
    p1_kicks = [_penalty(1, "2024-01-15T00:00:00+00:00", side="L")]
    x, y = build_training_matrix({1: p1_kicks}, {})
    assert x.shape == (1, 7)
    assert y.shape == (1,)


def test_build_training_matrix_respects_train_floor() -> None:
    kicks_before = [_penalty(1, "2019-01-01T00:00:00+00:00", side="L")]
    kicks_after = [
        _penalty(2, "2024-01-15T00:00:00+00:00", side="L"),
        _penalty(3, "2024-06-15T00:00:00+00:00", side="R"),
    ]
    x, _ = build_training_matrix(
        {1: kicks_before + kicks_after},
        {1: _metadata(player_id=1)},
    )
    assert x.shape == (2, 7)


# ---------------------------------------------------------------------------
# build_prediction_matrix
# ---------------------------------------------------------------------------


def test_build_prediction_matrix_empty() -> None:
    x = build_prediction_matrix([], {}, {})
    assert x.shape == (0, 7)


def test_build_prediction_matrix_shapes() -> None:
    x = build_prediction_matrix(
        [1, 2],
        {1: [_penalty(1, "2024-01-01T00:00:00+00:00", side="L")]},
        {1: _metadata(player_id=1, preferred_foot="right")},
        target_date=TARGET_DATE,
    )
    assert x.shape == (2, 7)


def test_build_prediction_matrix_no_history_player_gets_prior() -> None:
    x = build_prediction_matrix(
        [1],
        {},
        {1: _metadata(player_id=1)},
        target_date=TARGET_DATE,
    )
    assert x.shape == (1, 7)
    np.testing.assert_allclose(x[0, 0:3], [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    assert x[0, 3] == 0.0


# ---------------------------------------------------------------------------
# predict_and_write (Live TabPFN Cloud API - No Mocks)
# ---------------------------------------------------------------------------


def test_predict_and_write_writes_jsonl(tmp_path: Path) -> None:
    roster = [_roster(player_id=1, player_name="Alpha")]
    output_path = tmp_path / "preds.jsonl"
    rows = predict_and_write(
        roster,
        {
            1: [
                _penalty(1, "2024-01-01T00:00:00+00:00", side="L"),
                _penalty(2, "2024-02-01T00:00:00+00:00", side="R"),
            ],
            2: [
                _penalty(3, "2024-01-01T00:00:00+00:00", side="C"),
            ],
        },
        {
            1: _metadata(player_id=1, preferred_foot="right", position_key="striker"),
            2: _metadata(player_id=2, preferred_foot="left", position_key="goalkeeper"),
        },
        output_path,
        target_date=date(2026, 7, 1),
    )
    assert len(rows) == 1
    assert rows[0].player_id == 1
    assert rows[0].short_name == "Alpha"
    assert rows[0].total_penalties == 2
    assert output_path.exists()
    with output_path.open() as f:
        data = json.loads(f.readline())
    assert data["player_id"] == 1
    assert data["short_name"] == "Alpha"


def test_predict_and_write_multiple_players(tmp_path: Path) -> None:
    roster = [
        _roster(player_id=1, player_name="Alpha", team_id=100),
        _roster(player_id=2, player_name="Bravo", team_id=200),
    ]
    history = {
        1: [
            _penalty(1, "2024-01-01T00:00:00+00:00", side="L"),
            _penalty(2, "2024-02-01T00:00:00+00:00", side="R"),
        ],
        2: [
            _penalty(3, "2024-01-01T00:00:00+00:00", side="C"),
        ],
    }
    metadata = {
        1: _metadata(player_id=1, preferred_foot="right", position_key="striker"),
        2: _metadata(player_id=2, preferred_foot="left", position_key="goalkeeper"),
    }
    output_path = tmp_path / "preds.jsonl"
    rows = predict_and_write(
        roster, history, metadata, output_path, target_date=date(2026, 7, 1),
    )
    assert len(rows) == 2
    assert rows[0].team_id == 100
    assert rows[1].team_id == 200


def test_predict_and_write_empty_roster(tmp_path: Path) -> None:
    output_path = tmp_path / "preds.jsonl"
    rows = predict_and_write([], {}, {}, output_path, target_date=date(2026, 7, 1))
    assert rows == []
    assert output_path.read_text() == ""


# ---------------------------------------------------------------------------
# load_player_history
# ---------------------------------------------------------------------------


def test_load_player_history_groups_by_kicker(tmp_path: Path) -> None:
    path = tmp_path / "h.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for kicker_id, match_id in ((1, 100), (1, 101), (2, 200)):
            row = {
                "kicker_id": kicker_id,
                "match_id": match_id,
                "match_date": "2022-01-01T00:00:00+00:00",
                "league_id": 77,
                "league_name": "WC",
                "team_id": 1,
                "is_home": True,
                "x": 0.5,
                "side": "L",
                "is_on_target": True,
                "outcome": "Goal",
                "shot_type": "RightFoot",
            }
            f.write(json.dumps(row) + "\n")
    history = load_player_history(path)
    assert set(history.keys()) == {1, 2}
    assert len(history[1]) == 2
    assert len(history[2]) == 1
