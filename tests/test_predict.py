"""Tests for the v5 predict slice.

v5 changes:
- TabPFN classifier instead of LightGBM
- PredictionRow now has photo_url, short_name, total_penalties
- predict_and_write is the main orchestrator (replaces predict_roster)
- _derive_short_name extracts the last name
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from tests._factories import (
    PREDICTION_ROW_FIELDS,
    make_history_row,
    make_metadata,
    make_roster_player,
)
from twelveyards.artifacts import Artifacts
from twelveyards.predict import (
    PredictionRow,
    _derive_short_name,
    load_player_history,
    predict_and_write,
)

_roster = make_roster_player
_penalty = make_history_row
_metadata = make_metadata

TARGET_DATE = date(2026, 7, 1)


# ---------------------------------------------------------------------------
# _derive_short_name
# ---------------------------------------------------------------------------


def test_derive_short_name_single_word() -> None:
    assert _derive_short_name("Messi") == "Messi"


def test_derive_short_name_last_name() -> None:
    assert _derive_short_name("Lionel Messi") == "Messi"


def test_derive_short_name_three_parts() -> None:
    assert _derive_short_name("Kylian Mbappé Lottin") == "Lottin"


def test_derive_short_name_whitespace_handling() -> None:
    assert _derive_short_name("  Neymar  ") == "Neymar"


def test_derive_short_name_empty_raises() -> None:
    with pytest.raises(IndexError):
        _derive_short_name("")


# ---------------------------------------------------------------------------
# PredictionRow
# ---------------------------------------------------------------------------


def test_prediction_row_fields() -> None:
    assert "player_id" in PREDICTION_ROW_FIELDS
    assert "player_name" in PREDICTION_ROW_FIELDS
    assert "short_name" in PREDICTION_ROW_FIELDS
    assert "team_id" in PREDICTION_ROW_FIELDS
    assert "team_name" in PREDICTION_ROW_FIELDS
    assert "country_code" in PREDICTION_ROW_FIELDS
    assert "kicking_foot" in PREDICTION_ROW_FIELDS
    assert "photo_url" in PREDICTION_ROW_FIELDS
    assert "p_L" in PREDICTION_ROW_FIELDS
    assert "p_C" in PREDICTION_ROW_FIELDS
    assert "p_R" in PREDICTION_ROW_FIELDS
    assert "total_penalties" in PREDICTION_ROW_FIELDS


def test_prediction_row_has_v5_fields() -> None:
    row = PredictionRow(
        player_id=1,
        player_name="Lionel Messi",
        short_name="Messi",
        team_id=6706,
        team_name="Argentina",
        country_code="ARG",
        kicking_foot="left",
        photo_url="https://images.fotmob.com/image_resources/playerimages/1.png",
        p_L=0.4,
        p_C=0.3,
        p_R=0.3,
        total_penalties=12,
    )
    assert row.player_id == 1
    assert row.short_name == "Messi"
    assert row.photo_url == "https://images.fotmob.com/image_resources/playerimages/1.png"
    assert row.total_penalties == 12
    assert row.kicking_foot == "left"
    assert row.p_L + row.p_C + row.p_R == pytest.approx(1.0)


def test_prediction_row_total_penalties_defaults_to_zero() -> None:
    row = PredictionRow(
        player_id=1,
        player_name="X",
        short_name="X",
        team_id=1,
        team_name="A",
        country_code="",
        kicking_foot="",
        photo_url="",
        p_L=0.33,
        p_C=0.34,
        p_R=0.33,
    )
    assert row.total_penalties == 0


def test_prediction_row_is_frozen() -> None:
    row = PredictionRow(
        player_id=1,
        player_name="X",
        short_name="X",
        team_id=1,
        team_name="A",
        country_code="",
        kicking_foot="",
        photo_url="",
        p_L=0.33,
        p_C=0.34,
        p_R=0.33,
    )
    with pytest.raises(Exception):
        row.p_L = 0.9  # type: ignore[misc]


# ---------------------------------------------------------------------------
# predict_and_write (stubbed TabPFN via monkeypatch)
# ---------------------------------------------------------------------------


def test_predict_and_write_writes_jsonl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from twelveyards import predict as predict_module

    class _FakeModel:
        def fit(self, X, y):
            pass

        def predict_proba(self, X):
            import numpy as np
            return np.full((len(X), 3), 1.0 / 3.0)

    monkeypatch.setattr(predict_module, "tabpfn_init", lambda: None)
    monkeypatch.setattr(predict_module, "TabPFN", lambda categorical_features_indices=None: _FakeModel())

    roster = [_roster(player_id=1, player_name="Alpha")]
    output_path = tmp_path / "preds.jsonl"
    rows = predict_and_write(
        roster,
        {1: [_penalty(1, "2024-01-01T00:00:00+00:00", side="L")]},
        {1: _metadata(player_id=1, preferred_foot="right")},
        output_path,
        target_date=TARGET_DATE,
    )
    assert len(rows) == 1
    assert rows[0].player_id == 1
    assert rows[0].short_name == "Alpha"
    assert rows[0].total_penalties == 1
    assert rows[0].photo_url == "https://images.fotmob.com/image_resources/playerimages/1.png"
    assert output_path.exists()
    with output_path.open() as f:
        data = json.loads(f.readline())
    assert data["player_id"] == 1
    assert data["short_name"] == "Alpha"
    assert "photo_url" in data


def test_predict_and_write_multiple_players(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from twelveyards import predict as predict_module

    call_count = 0

    class _FakeModel:
        def fit(self, X, y):
            nonlocal call_count
            call_count += 1

        def predict_proba(self, X):
            import numpy as np
            return np.full((len(X), 3), 1.0 / 3.0)

    monkeypatch.setattr(predict_module, "tabpfn_init", lambda: None)
    monkeypatch.setattr(predict_module, "TabPFN", lambda categorical_features_indices=None: _FakeModel())

    roster = [
        _roster(player_id=1, player_name="Alpha", team_id=100),
        _roster(player_id=2, player_name="Bravo", team_id=200),
    ]
    history = {
        1: [_penalty(1, "2024-01-01T00:00:00+00:00", side="L")],
        2: [_penalty(2, "2024-01-01T00:00:00+00:00", side="R")],
    }
    metadata = {
        1: _metadata(player_id=1),
        2: _metadata(player_id=2),
    }
    output_path = tmp_path / "preds.jsonl"
    rows = predict_and_write(roster, history, metadata, output_path, target_date=TARGET_DATE)

    assert len(rows) == 2
    assert rows[0].team_id == 100
    assert rows[1].team_id == 200
    assert call_count == 1


def test_predict_and_write_uses_metadata_foot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from twelveyards import predict as predict_module

    class _FakeModel:
        def fit(self, X, y):
            pass

        def predict_proba(self, X):
            import numpy as np
            return np.full((len(X), 3), 1.0 / 3.0)

    monkeypatch.setattr(predict_module, "tabpfn_init", lambda: None)
    monkeypatch.setattr(predict_module, "TabPFN", lambda categorical_features_indices=None: _FakeModel())

    roster = [_roster(player_id=1)]
    output_path = tmp_path / "preds.jsonl"
    rows = predict_and_write(
        roster,
        {},
        {1: _metadata(player_id=1, preferred_foot="both")},
        output_path,
        target_date=TARGET_DATE,
    )
    assert rows[0].kicking_foot == "both"


def test_predict_and_write_no_metadata_foot_defaults_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from twelveyards import predict as predict_module

    class _FakeModel:
        def fit(self, X, y):
            pass

        def predict_proba(self, X):
            import numpy as np
            return np.full((len(X), 3), 1.0 / 3.0)

    monkeypatch.setattr(predict_module, "tabpfn_init", lambda: None)
    monkeypatch.setattr(predict_module, "TabPFN", lambda categorical_features_indices=None: _FakeModel())

    roster = [_roster(player_id=1)]
    output_path = tmp_path / "preds.jsonl"
    rows = predict_and_write(roster, {}, {}, output_path, target_date=TARGET_DATE)
    assert rows[0].kicking_foot == ""


def test_predict_and_write_empty_roster(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from twelveyards import predict as predict_module

    class _FakeModel:
        def fit(self, X, y):
            pass

        def predict_proba(self, X):
            import numpy as np
            return np.full((len(X), 3), 1.0 / 3.0)

    monkeypatch.setattr(predict_module, "tabpfn_init", lambda: None)
    monkeypatch.setattr(predict_module, "TabPFN", lambda categorical_features_indices=None: _FakeModel())

    output_path = tmp_path / "preds.jsonl"
    rows = predict_and_write([], {}, {}, output_path, target_date=TARGET_DATE)
    assert rows == []
    assert output_path.read_text() == ""


def test_predict_and_write_probabilities_sum_to_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    from twelveyards import predict as predict_module

    class _FakeModel:
        def fit(self, X, y):
            pass

        def predict_proba(self, X):
            return np.array([[0.5, 0.3, 0.2]])

    monkeypatch.setattr(predict_module, "tabpfn_init", lambda: None)
    monkeypatch.setattr(predict_module, "TabPFN", lambda categorical_features_indices=None: _FakeModel())

    roster = [_roster(player_id=1)]
    output_path = tmp_path / "preds.jsonl"
    rows = predict_and_write(roster, {}, {}, output_path, target_date=TARGET_DATE)
    assert len(rows) == 1
    total = rows[0].p_L + rows[0].p_C + rows[0].p_R
    assert total == pytest.approx(1.0)


def test_predict_and_write_country_code_passthrough(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from twelveyards import predict as predict_module

    class _FakeModel:
        def fit(self, X, y):
            pass

        def predict_proba(self, X):
            import numpy as np
            return np.full((len(X), 3), 1.0 / 3.0)

    monkeypatch.setattr(predict_module, "tabpfn_init", lambda: None)
    monkeypatch.setattr(predict_module, "TabPFN", lambda categorical_features_indices=None: _FakeModel())

    roster = [_roster(player_id=1, country_code="FRA")]
    output_path = tmp_path / "preds.jsonl"
    rows = predict_and_write(roster, {}, {}, output_path, target_date=TARGET_DATE)
    assert rows[0].country_code == "FRA"


# ---------------------------------------------------------------------------
# JSONL roundtrip with Artifacts
# ---------------------------------------------------------------------------


def test_predictions_jsonl_roundtrip(tmp_path: Path) -> None:
    from tests._factories import make_prediction_row

    art = Artifacts(root=tmp_path)
    rows = [
        make_prediction_row(player_id=1, player_name="Alpha", short_name="Alpha"),
        make_prediction_row(player_id=2, player_name="Bravo Beta", short_name="Beta"),
    ]
    n = art.write_predictions(rows, path=art.predictions)
    assert n == 2
    back = art.read_predictions()
    assert back == rows
    assert back[0].short_name == "Alpha"
    assert back[1].short_name == "Beta"


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
