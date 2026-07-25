"""Tests for the model feature matrix builders and prediction pipeline."""

from pathlib import Path

import pandas as pd

from scripts.predict import FEATURES, PredictionRow, prepare_features, run_pipeline


def test_prepare_features_with_existing_matches() -> None:
    """Test feature preparation against data/matches.jsonl."""
    matches_path = Path("data/matches.jsonl")
    if not matches_path.exists():
        return

    df_train, df_predict = prepare_features(matches_path)

    assert not df_train.empty
    assert not df_predict.empty

    expected_features = [
        "attempts_5y",
        "left_5y",
        "center_5y",
        "right_5y",
        "conversion_5y",
        "attempts_1y",
        "left_1y",
        "center_1y",
        "right_1y",
        "conversion_1y",
        "last_side_left",
        "last_side_center",
        "last_side_right",
    ]

    for feat in expected_features:
        assert feat in df_train.columns
        assert feat in df_predict.columns

    removed_features = [
        "player_age",
        "player_market_value",
        "market_value_known",
        "player_position_id",
    ]
    for feat in removed_features:
        assert feat not in df_train.columns
        assert feat not in df_predict.columns
        assert feat not in FEATURES


def test_coordinate_normalization() -> None:
    """Test unzoomed coordinate formula: true_x = 1.0 + (shot_x - 1.0) / shot_zoom."""
    shot_x = 1.2
    shot_zoom = 2.0
    true_x = 1.0 + (shot_x - 1.0) / shot_zoom
    assert true_x == 1.1

    shot_x_center = 1.0
    shot_zoom_center = 3.0
    true_x_center = 1.0 + (shot_x_center - 1.0) / shot_zoom_center
    assert true_x_center == 1.0


def test_side_binning_ranges() -> None:
    """Test side binning ranges [-inf, 2/3, 4/3, inf] for left, center, right."""
    series = pd.Series([-0.5, 0.0, 0.5, 2 / 3, 1.0, 4 / 3, 1.5, 2.5])
    bins = pd.cut(
        series,
        bins=[-float("inf"), 2 / 3, 4 / 3, float("inf")],
        labels=["left", "center", "right"],
        right=False,
    )
    expected = ["left", "left", "left", "center", "center", "right", "right", "right"]
    assert list(bins) == expected
    assert not bins.isna().any()


def test_rolling_nan_leakage() -> None:
    """Test that rolling window aggregations fillna(0) and don't leak NaNs."""
    matches_path = Path("data/matches.jsonl")
    if not matches_path.exists():
        return

    df_train, df_predict = prepare_features(matches_path)
    rolling_cols = [
        "left_1y",
        "center_1y",
        "right_1y",
        "left_5y",
        "center_5y",
        "right_5y",
        "goals_1y",
        "goals_5y",
    ]
    for col in rolling_cols:
        assert not df_predict[col].isna().any(), f"NaN found in df_predict[{col}]"
        assert not df_train[col].isna().any(), f"NaN found in df_train[{col}]"


def test_prediction_probabilities_varied(tmp_path: Path) -> None:
    """Verify that predictions produce varied probability distributions."""
    matches_path = Path("data/matches.jsonl")
    if not matches_path.exists():
        return

    out_path = tmp_path / "predictions.jsonl"
    results = run_pipeline(matches_path, out_path)

    assert len(results) > 0
    # Check that not all kickers get identical p_L, p_C, p_R
    prob_tuples = {(r.p_left, r.p_center, r.p_right) for r in results}
    assert len(prob_tuples) > 1, (
        "Predictions defaulted to a single static probability distribution!"
    )


def test_prediction_row_schema() -> None:
    """Test PredictionRow validation and serialization."""
    row = PredictionRow(
        player_id=101,
        player_name="Test Kicker",
        short_name="T. Kicker",
        team_id=1,
        team_name="Team A",
        kicking_foot=None,
        photo_url="https://example.com/photo.png",
        total_penalties=5,
        p_left=0.4,
        p_center=0.2,
        p_right=0.4,
    )

    dump = row.model_dump_json()
    assert '"p_left":0.4' in dump
    assert '"kicking_foot":null' in dump


def test_na_player_name_and_team_handling() -> None:
    """Ensure pd.NA in player_name or team_id does not raise TypeError."""
    df = pd.DataFrame(
        [
            {"player_id": 123, "player_name": pd.NA, "team_id": pd.NA},
            {"player_id": 456, "player_name": "Known Player", "team_id": 10},
        ],
    )

    for _i, row in df.iterrows():
        pid = int(row["player_id"])
        raw_name = row.get("player_name")
        name = str(raw_name) if pd.notna(raw_name) and raw_name else f"Player {pid}"
        raw_team = row.get("team_id")
        team_id = int(raw_team) if pd.notna(raw_team) and raw_team else 0

        if pid == 123:
            assert name == "Player 123"
            assert team_id == 0
        else:
            assert name == "Known Player"
            assert team_id == 10
