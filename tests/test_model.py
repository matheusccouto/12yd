"""Tests for the model feature matrix builders and prediction pipeline."""

from pathlib import Path

from scripts.predict import PredictionRow, prepare_features


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
        "player_age",
        "player_market_value",
        "market_value_known",
        "player_position_id",
    ]

    for feat in expected_features:
        assert feat in df_train.columns
        assert feat in df_predict.columns


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
        p_L=0.4,
        p_C=0.2,
        p_R=0.4,
    )

    dump = row.model_dump_json(by_alias=True)
    assert '"p_L":0.4' in dump
    assert '"kicking_foot":null' in dump
