"""Tests for the v5 feature builder.

v5: 7 player-only, match-agnostic features:
- A1: (p_L, p_C, p_R) — side distribution over the 5-year rolling window
- A2: last_side — side of the most recent kick in the window
- A3: preferred_foot — declared foot from FotMob metadata
- A4: career_penalty_count — count of kicks in the window
- C1: position — from FotMob metadata

Categorical column indices are [4, 5, 6] (last_side, preferred_foot, position).
"""

from __future__ import annotations

from datetime import date

import numpy as np

from tests._factories import make_history_row, make_metadata, make_roster_player
from twelveyards.model.features import (
    CATEGORICAL_INDICES,
    CLASSES,
    PRIOR_PROB,
    build_prediction_matrix,
    build_training_matrix,
    compute_features,
    side_distribution,
)

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
    p_L, p_C, p_R = side_distribution(["L"] * 10 + ["R"] * 10, 5)
    assert (p_L, p_C, p_R) == (0.0, 0.0, 1.0)


def test_side_distribution_horizons_nest() -> None:
    sides = ["L"] * 10 + ["C"] * 5 + ["R"] * 5
    p5 = side_distribution(sides, 5)
    p10 = side_distribution(sides, 10)
    p20 = side_distribution(sides, 20)
    assert p5 == (0.0, 0.0, 1.0)
    assert p10 == (0.0, 0.5, 0.5)
    assert p20 == (0.5, 0.25, 0.25)


def test_side_distribution_history_shorter_than_horizon() -> None:
    p_L, p_C, p_R = side_distribution(["L", "C"], 20)
    assert p_L == 0.5
    assert p_C == 0.5
    assert p_R == 0.0


# ---------------------------------------------------------------------------
# compute_features
# ---------------------------------------------------------------------------


def test_compute_features_no_history_uses_prior() -> None:
    feats = compute_features([], _metadata(player_id=1, preferred_foot="right"), TARGET_DATE)
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


def test_compute_features_position_from_metadata() -> None:
    feats = compute_features([], _metadata(player_id=1, position_key="centreback"), TARGET_DATE)
    assert feats["position"] == "centreback"


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
    X, y = build_training_matrix({}, {})
    assert X.shape == (0, 7)
    assert y.shape == (0,)


def test_build_training_matrix_shapes() -> None:
    p1_kicks = [
        _penalty(1, "2024-01-15T00:00:00+00:00", side="L"),
        _penalty(2, "2024-06-15T00:00:00+00:00", side="R"),
    ]
    X, y = build_training_matrix(
        {1: p1_kicks},
        {1: _metadata(player_id=1, preferred_foot="right", position_key="striker")},
    )
    assert X.shape == (2, 7)
    assert y.shape == (2,)
    assert list(y) == [CLASSES.index("L"), CLASSES.index("R")]


def test_build_training_matrix_no_metadata_for_player() -> None:
    p1_kicks = [_penalty(1, "2024-01-15T00:00:00+00:00", side="L")]
    X, y = build_training_matrix({1: p1_kicks}, {})
    assert X.shape == (1, 7)
    assert y.shape == (1,)


def test_build_training_matrix_respects_train_floor() -> None:
    kicks_before = [_penalty(1, "2019-01-01T00:00:00+00:00", side="L")]
    kicks_after = [
        _penalty(2, "2024-01-15T00:00:00+00:00", side="L"),
        _penalty(3, "2024-06-15T00:00:00+00:00", side="R"),
    ]
    X, _ = build_training_matrix(
        {1: kicks_before + kicks_after},
        {1: _metadata(player_id=1)},
    )
    assert X.shape == (2, 7)


def test_build_training_matrix_multiple_players() -> None:
    history = {
        1: [_penalty(1, "2024-01-15T00:00:00+00:00", kicker_id=1, side="L")],
        2: [_penalty(2, "2024-03-15T00:00:00+00:00", kicker_id=2, side="R")],
    }
    metadata = {
        1: _metadata(player_id=1),
        2: _metadata(player_id=2, preferred_foot="left"),
    }
    X, y = build_training_matrix(history, metadata)
    assert X.shape == (2, 7)
    assert y.shape == (2,)


# ---------------------------------------------------------------------------
# build_prediction_matrix
# ---------------------------------------------------------------------------


def test_build_prediction_matrix_empty() -> None:
    X = build_prediction_matrix([], {}, {})
    assert X.shape == (0, 7)


def test_build_prediction_matrix_shapes() -> None:
    X = build_prediction_matrix(
        [1, 2],
        {1: [_penalty(1, "2024-01-01T00:00:00+00:00", side="L")]},
        {1: _metadata(player_id=1, preferred_foot="right")},
        target_date=TARGET_DATE,
    )
    assert X.shape == (2, 7)


def test_build_prediction_matrix_default_target_date() -> None:
    X = build_prediction_matrix([1], {}, {})
    assert X.shape == (1, 7)


def test_build_prediction_matrix_no_history_player_gets_prior() -> None:
    X = build_prediction_matrix(
        [1],
        {},
        {1: _metadata(player_id=1)},
        target_date=TARGET_DATE,
    )
    assert X.shape == (1, 7)
    np.testing.assert_allclose(X[0, 0:3], [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    assert X[0, 3] == 0.0


def test_build_prediction_matrix_float_dtype() -> None:
    X = build_prediction_matrix([1], {}, {}, target_date=TARGET_DATE)
    assert X.dtype == np.float64


def test_build_prediction_matrix_encodes_categoricals() -> None:
    X = build_prediction_matrix(
        [1],
        {},
        {1: _metadata(player_id=1, preferred_foot="left", position_key="striker")},
        target_date=TARGET_DATE,
    )
    assert X[0, 4] == 0.0
    assert X[0, 5] >= 0
    assert X[0, 6] >= 0


def test_build_prediction_matrix_historical_data_shapes_features() -> None:
    history = [_penalty(i + 1, f"2024-{i+1:02d}-01T00:00:00+00:00", side="L") for i in range(5)]
    X = build_prediction_matrix(
        [1],
        {1: history},
        {1: _metadata(player_id=1, preferred_foot="right", position_key="striker")},
        target_date=TARGET_DATE,
    )
    assert X.shape == (1, 7)
    np.testing.assert_allclose(X[0, 0:3], [1.0, 0.0, 0.0])
    assert X[0, 3] == 5.0
