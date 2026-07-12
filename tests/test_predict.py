"""Tests for the predict pipeline."""

from __future__ import annotations

import json
from datetime import date
from typing import TYPE_CHECKING

import pytest

from tests._factories import make_history_row, make_metadata, make_roster_player
from twelveyards.artifacts import Artifacts
from twelveyards.model.predict import load_player_history, predict_and_write

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

_roster = make_roster_player
_penalty = make_history_row
_metadata = make_metadata

TARGET_DATE = date(2026, 7, 1)


# ---------------------------------------------------------------------------
# predict_and_write (stubbed TabPFN via monkeypatch)
# ---------------------------------------------------------------------------


class _FakeModel:
    def fit(self, x: np.ndarray, y_val: np.ndarray) -> None:
        pass

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        import numpy as np  # noqa: PLC0415
        return np.full((len(x), 3), 1.0 / 3.0)


def _stub_tabpfn(monkeypatch: pytest.MonkeyPatch) -> None:
    from twelveyards.model import predict as predict_module  # noqa: PLC0415
    monkeypatch.setattr(predict_module, "_init_tabpfn", lambda: None)
    monkeypatch.setattr(predict_module, "_TabPFN", lambda **_: _FakeModel())


def test_predict_and_write_writes_jsonl(  # noqa: D103
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_tabpfn(monkeypatch)
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


def test_predict_and_write_multiple_players(  # noqa: D103
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_tabpfn(monkeypatch)
    roster = [
        _roster(player_id=1, player_name="Alpha", team_id=100),
        _roster(player_id=2, player_name="Bravo", team_id=200),
    ]
    history = {
        1: [_penalty(1, "2024-01-01T00:00:00+00:00", side="L")],
        2: [_penalty(2, "2024-01-01T00:00:00+00:00", side="R")],
    }
    metadata = {1: _metadata(player_id=1), 2: _metadata(player_id=2)}
    output_path = tmp_path / "preds.jsonl"
    rows = predict_and_write(
        roster, history, metadata, output_path, target_date=TARGET_DATE,
    )
    assert len(rows) == 2  # noqa: PLR2004
    assert rows[0].team_id == 100  # noqa: PLR2004
    assert rows[1].team_id == 200  # noqa: PLR2004


def test_predict_and_write_uses_metadata_foot(  # noqa: D103
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_tabpfn(monkeypatch)
    roster = [_roster(player_id=1)]
    output_path = tmp_path / "preds.jsonl"
    rows = predict_and_write(
        roster, {}, {1: _metadata(player_id=1, preferred_foot="both")},
        output_path, target_date=TARGET_DATE,
    )
    assert rows[0].kicking_foot == "both"


def test_predict_and_write_no_metadata_foot_defaults_empty(  # noqa: D103
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_tabpfn(monkeypatch)
    roster = [_roster(player_id=1)]
    output_path = tmp_path / "preds.jsonl"
    rows = predict_and_write(roster, {}, {}, output_path, target_date=TARGET_DATE)
    assert rows[0].kicking_foot == ""


def test_predict_and_write_empty_roster(  # noqa: D103
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_tabpfn(monkeypatch)
    output_path = tmp_path / "preds.jsonl"
    rows = predict_and_write([], {}, {}, output_path, target_date=TARGET_DATE)
    assert rows == []
    assert output_path.read_text() == ""


def test_predict_and_write_probabilities_sum_to_one(  # noqa: D103
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np  # noqa: PLC0415

    from twelveyards.model import predict as predict_module  # noqa: PLC0415

    class _FakeModelCustom:
        def fit(self, x: np.ndarray, y_val: np.ndarray) -> None:
            pass
        def predict_proba(self, _x: np.ndarray) -> np.ndarray:
            return np.array([[0.5, 0.3, 0.2]])

    monkeypatch.setattr(predict_module, "_init_tabpfn", lambda: None)
    monkeypatch.setattr(predict_module, "_TabPFN", lambda **_: _FakeModelCustom())

    roster = [_roster(player_id=1)]
    output_path = tmp_path / "preds.jsonl"
    rows = predict_and_write(roster, {}, {}, output_path, target_date=TARGET_DATE)
    assert len(rows) == 1
    total = rows[0].p_L + rows[0].p_C + rows[0].p_R
    assert total == pytest.approx(1.0)


def test_predict_and_write_country_code_passthrough(  # noqa: D103
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_tabpfn(monkeypatch)
    roster = [_roster(player_id=1, country_code="FRA")]
    output_path = tmp_path / "preds.jsonl"
    rows = predict_and_write(roster, {}, {}, output_path, target_date=TARGET_DATE)
    assert rows[0].country_code == "FRA"


# ---------------------------------------------------------------------------
# JSONL roundtrip with Artifacts
# ---------------------------------------------------------------------------


def test_predictions_jsonl_roundtrip(tmp_path: Path) -> None:  # noqa: D103
    from tests._factories import make_prediction_row  # noqa: PLC0415

    art = Artifacts(root=tmp_path)
    rows = [
        make_prediction_row(player_id=1, player_name="Alpha", short_name="Alpha"),
        make_prediction_row(player_id=2, player_name="Bravo Beta", short_name="Beta"),
    ]
    n = art.write_predictions(rows, path=art.predictions)
    assert n == 2  # noqa: PLR2004
    back = art.read_predictions()
    assert back == rows
    assert back[0].short_name == "Alpha"
    assert back[1].short_name == "Beta"


# ---------------------------------------------------------------------------
# load_player_history
# ---------------------------------------------------------------------------


def test_load_player_history_groups_by_kicker(tmp_path: Path) -> None:  # noqa: D103
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
    assert len(history[1]) == 2  # noqa: PLR2004
    assert len(history[2]) == 1
