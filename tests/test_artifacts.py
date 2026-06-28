"""Tests for the `Artifacts` on-disk layout adapter (Issue #29).

The adapter owns the on-disk layout — paths, JSONL shape, model pickle,
metrics JSON. These tests pin the contract: the path accessors point at
the canonical filenames, the read/write methods round-trip, NaN ages
are emitted as `null` (not `NaN`), the metrics report survives a
`to_dict` → JSON → `from_dict` round-trip, and the `fotmob_client`
factory returns a `FotMobClient` rooted at the configured cache_dir.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import pytest

from penalty_pred.artifacts import Artifacts
from penalty_pred.evaluate import BaselineMetrics, MetricsReport
from penalty_pred.features import TrainingRow
from penalty_pred.initial_set import MissingKicker
from penalty_pred.player_history import PlayerPenalty
from penalty_pred.predict import PredictionRow
from penalty_pred.rosters import RosterPlayer
from penalty_pred.shootouts import ShootoutKick


# A module-level dataclass (not nested in a test) so pickle can
# serialise it — local classes are not picklable.
@dataclass
class _StubPickleModel:
    weights: list[float]


# ---------------------------------------------------------------------------
# Path accessors
# ---------------------------------------------------------------------------


def test_default_paths_point_at_canonical_filenames() -> None:
    """The default instance's path accessors point at the v1
    canonical on-disk filenames (the script surface is preserved)."""
    art = Artifacts()
    assert art.root == Path("output")
    assert art.cache_dir == Path("data/fotmob_cache")
    assert art.shootout_kicks == Path("output/shootout_kicks.jsonl")
    assert art.player_history == Path("output/player_history.jsonl")
    assert art.missing_history == Path("output/missing_history.jsonl")
    assert art.roster == Path("output/wc2026_roster.jsonl")
    assert art.training_table == Path("output/training_table.jsonl")
    assert art.predictions == Path("output/predictions.jsonl")
    assert art.lightgbm_model == Path("output/lightgbm.pkl")
    assert art.baseline_model == Path("output/baseline.pkl")
    assert art.metrics == Path("output/metrics.json")
    assert art.discrepancies == Path("output/discrepancies.json")


def test_custom_root_redirects_every_artifact() -> None:
    """A non-default `root` redirects every path accessor (the
    scripts can be re-parameterised via `--root` without touching
    individual artifact paths)."""
    art = Artifacts(root=Path("/tmp/foo"))
    assert art.shootout_kicks == Path("/tmp/foo/shootout_kicks.jsonl")
    assert art.predictions == Path("/tmp/foo/predictions.jsonl")
    assert art.metrics == Path("/tmp/foo/metrics.json")
    # The cache_dir is independent of `root`.
    assert art.cache_dir == Path("data/fotmob_cache")


def test_fotmob_client_factory_uses_cache_dir() -> None:
    """`fotmob_client()` returns a `FotMobClient` whose `cache_dir`
    matches the adapter's `cache_dir`."""
    from penalty_pred.client import FotMobClient

    art = Artifacts(cache_dir=Path("/tmp/foo_cache"))
    client = art.fotmob_client()
    assert isinstance(client, FotMobClient)
    assert client.cache_dir == Path("/tmp/foo_cache")


# ---------------------------------------------------------------------------
# JSONL round-trips (one per dataclass)
# ---------------------------------------------------------------------------


def _shootout_kick(match_id: int = 1, kick_number: int = 1) -> ShootoutKick:
    return ShootoutKick(
        match_id=match_id,
        match_date="2022-12-18T15:00:00+00:00",
        tournament_id=77,
        tournament_name="World Cup",
        round="Final",
        kick_number=kick_number,
        kicker_id=42,
        kicker_name="Stub",
        team_id=100,
        is_home=True,
        x=0.5,
        side="L",
        is_on_target=True,
        outcome="Goal",
        pen_score_before=[0, 0],
        pen_score_after=[1, 0],
        match_score_home=3,
        match_score_away=3,
    )


def test_shootout_kicks_round_trip(tmp_path: Path) -> None:
    art = Artifacts(root=tmp_path)
    kicks = [_shootout_kick(1), _shootout_kick(1, 2)]
    n = art.write_shootout_kicks(kicks, path=art.shootout_kicks)
    assert n == 2
    assert art.read_shootout_kicks() == kicks


def test_player_history_round_trip(tmp_path: Path) -> None:
    art = Artifacts(root=tmp_path)
    rows = [
        PlayerPenalty(
            kicker_id=42,
            match_id=99,
            match_date="2022-01-01T00:00:00+00:00",
            league_id=77,
            league_name="World Cup",
            team_id=100,
            is_home=True,
            x=0.5,
            side="L",
            is_on_target=True,
            outcome="Goal",
            shot_type="RightFoot",
        )
    ]
    n = art.write_player_history(rows, path=art.player_history)
    assert n == 1
    assert art.read_player_history() == rows


def test_missing_history_round_trip(tmp_path: Path) -> None:
    art = Artifacts(root=tmp_path)
    rows = [
        MissingKicker(
            player_id=1,
            player_name="No History",
            team_id=1,
            team_name="Argentina",
        )
    ]
    n = art.write_missing_history(rows, path=art.missing_history)
    assert n == 1
    assert art.read_missing_history() == rows


def test_roster_round_trip(tmp_path: Path) -> None:
    art = Artifacts(root=tmp_path)
    rows = [
        RosterPlayer(
            player_id=1,
            player_name="Alpha",
            team_id=100,
            team_name="Argentina",
            country_code="ARG",
        )
    ]
    n = art.write_roster(rows, path=art.roster)
    assert n == 1
    assert art.read_roster() == rows


def test_predictions_round_trip(tmp_path: Path) -> None:
    art = Artifacts(root=tmp_path)
    rows = [
        PredictionRow(
            player_id=1,
            player_name="Alpha",
            team_id=100,
            team_name="Argentina",
            country_code="ARG",
            kicking_foot="RightFoot",
            p_L=0.5,
            p_C=0.25,
            p_R=0.25,
        )
    ]
    n = art.write_predictions(rows, path=art.predictions)
    assert n == 1
    assert art.read_predictions() == rows


def test_training_table_round_trip_with_nan_age(tmp_path: Path) -> None:
    """`write_training_table` emits NaN ages as JSON `null` (strict
    JSON, not `NaN`); `read_training_table` recovers them as `None`."""
    art = Artifacts(root=tmp_path)
    rows = [
        TrainingRow(
            match_id=1,
            kick_number=1,
            kicker_id=1,
            kicker_name="X",
            match_date="2022-12-18T15:00:00+00:00",
            tournament_id=77,
            tournament_name="World Cup",
            round="Final",
            team_id=1,
            is_home=True,
            label="L",
            is_on_target=True,
            p_L_5=1.0,
            p_C_5=0.0,
            p_R_5=0.0,
            p_L_10=1.0,
            p_C_10=0.0,
            p_R_10=0.0,
            p_L_20=1.0,
            p_C_20=0.0,
            p_R_20=0.0,
            last_side="R",
            kicking_foot="RightFoot",
            career_penalty_count=5,
            b1_kick_number=1,
            pen_score_home=0,
            pen_score_away=0,
            is_decisive=False,
            b3_round="Final",
            position="striker",
            age=math.nan,
        )
    ]
    n = art.write_training_table(rows, path=art.training_table)
    assert n == 1
    raw = art.training_table.read_text(encoding="utf-8")
    assert '"age": null' in raw
    assert "NaN" not in raw
    back = art.read_training_table()
    assert back[0].age is None  # NaN → null → None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _make_report() -> MetricsReport:
    return MetricsReport(
        model=BaselineMetrics(
            name="model",
            log_loss=1.1,
            accuracy=0.5,
            save_rate=0.46,
            n_kicks=28,
        ),
        random_baseline=BaselineMetrics(
            name="random",
            log_loss=math.log(3),
            accuracy=1.0 / 3.0,
            save_rate=0.55,
            n_kicks=28,
        ),
        kicker_most_frequent_baseline=BaselineMetrics(
            name="last_side",
            log_loss=None,
            accuracy=None,
            save_rate=0.40,
            n_kicks=28,
        ),
        actual_keeper_baseline=BaselineMetrics(
            name="actual_keeper",
            log_loss=None,
            accuracy=None,
            save_rate=None,
            n_kicks=28,
        ),
        n_train=151,
        n_holdout=28,
        holdout_cutoff_date="2026-01-01",
        baseline=BaselineMetrics(
            name="baseline",
            log_loss=1.05,
            accuracy=0.5,
            save_rate=0.43,
            n_kicks=28,
        ),
        extras={
            "model_kind": "lightgbm",
            "classes": ["L", "C", "R"],
            "feature_columns": ["p_L_5", "b3_round"],
        },
    )


def test_metrics_round_trip_preserves_extras(tmp_path: Path) -> None:
    """`write_metrics` then `read_metrics` returns a `MetricsReport`
    with the same fields, including the optional `baseline` and the
    `extras` dict (the model_kind/classes/feature_columns metadata
    that the model layer stashes in `extras`)."""
    art = Artifacts(root=tmp_path)
    report = _make_report()
    art.write_metrics(report, path=art.metrics)
    back = art.read_metrics()
    assert back.n_train == 151
    assert back.n_holdout == 28
    assert back.holdout_cutoff_date == "2026-01-01"
    assert back.model.save_rate == 0.46
    assert back.random_baseline.save_rate == 0.55
    assert back.actual_keeper_baseline.save_rate is None
    assert back.baseline is not None
    assert back.baseline.save_rate == 0.43
    assert back.extras["model_kind"] == "lightgbm"
    assert back.extras["feature_columns"] == ["p_L_5", "b3_round"]


def test_metrics_round_trip_without_optional_baseline(tmp_path: Path) -> None:
    """A report without the optional `baseline` section round-trips
    with `baseline=None` (not a KeyError)."""
    art = Artifacts(root=tmp_path)
    report = _make_report()
    object.__setattr__(report, "baseline", None)  # type: ignore[attr-defined]
    art.write_metrics(report, path=art.metrics)
    back = art.read_metrics()
    assert back.baseline is None


# ---------------------------------------------------------------------------
# Model artifact
# ---------------------------------------------------------------------------


def test_model_round_trip_records_feature_columns(tmp_path: Path) -> None:
    """`write_model` pickles `{model, feature_columns, model_kind, params}`
    and `read_model` returns the same dict. The `feature_columns` order
    is preserved (the predict path relies on it for column alignment)."""
    import pickle

    art = Artifacts(root=tmp_path)
    model = _StubPickleModel(weights=[1.0, 2.0, 3.0])
    art.write_model(
        model,
        feature_columns=["p_L_5", "p_C_5", "p_R_5"],
        model_kind="stub",
        params={"C": 1.0},
        path=art.baseline_model,
    )
    raw = pickle.loads(art.baseline_model.read_bytes())
    assert raw["model"] == model
    assert raw["feature_columns"] == ["p_L_5", "p_C_5", "p_R_5"]
    assert raw["model_kind"] == "stub"
    assert raw["params"] == {"C": 1.0}

    back = art.read_model(path=art.baseline_model)
    assert back["model"] == model
    assert back["feature_columns"] == ["p_L_5", "p_C_5", "p_R_5"]


# ---------------------------------------------------------------------------
# Streaming serialise
# ---------------------------------------------------------------------------


def test_serialize_row_matches_write_shape(tmp_path: Path) -> None:
    """`serialize_row` returns a JSON string compatible with the
    per-row shape of `write_*` so a streaming writer can mix
    per-row `serialize_row` calls with the same file format."""
    art = Artifacts(root=tmp_path)
    kick = _shootout_kick()
    # Write one row via the streaming helper…
    out = tmp_path / "kicks.jsonl"
    out.write_text(art.serialize_row(kick) + "\n", encoding="utf-8")
    # …and read it back via the normal read path.
    assert art.read_shootout_kicks(path=out) == [kick]


def test_serialize_row_emits_nan_as_null() -> None:
    """`serialize_row(nan_to_null=True)` emits NaN ages as JSON `null`
    (strict JSON), matching `write_training_table`'s per-row shape."""
    art = Artifacts()
    row = TrainingRow(
        match_id=1,
        kick_number=1,
        kicker_id=1,
        kicker_name="X",
        match_date="2022-12-18T15:00:00+00:00",
        tournament_id=77,
        tournament_name="World Cup",
        round="Final",
        team_id=1,
        is_home=True,
        label="L",
        is_on_target=True,
        p_L_5=1.0,
        p_C_5=0.0,
        p_R_5=0.0,
        p_L_10=1.0,
        p_C_10=0.0,
        p_R_10=0.0,
        p_L_20=1.0,
        p_C_20=0.0,
        p_R_20=0.0,
        last_side="R",
        kicking_foot="RightFoot",
        career_penalty_count=5,
        b1_kick_number=1,
        pen_score_home=0,
        pen_score_away=0,
        is_decisive=False,
        b3_round="Final",
        position="striker",
        age=math.nan,
    )
    text = art.serialize_row(row, nan_to_null=True)
    payload = json.loads(text)
    assert payload["age"] is None
    # The bare `serialize_row(row)` (without nan_to_null) is the
    # non-strict path used for the rest of the dataclasses.
    text2 = art.serialize_row(row)
    payload2 = json.loads(text2)
    assert math.isnan(payload2["age"])


# ---------------------------------------------------------------------------
# Read raises FileNotFoundError for missing JSONL
# ---------------------------------------------------------------------------


def test_read_missing_jsonl_raises(tmp_path: Path) -> None:
    """A missing JSONL raises `FileNotFoundError` on read; the slice
    scripts gate on `path.exists()` first. Pinning this behaviour
    here so a future change to "return [] on missing" is a deliberate
    decision, not a side effect."""
    art = Artifacts(root=tmp_path)
    with pytest.raises(FileNotFoundError):
        art.read_shootout_kicks()
    with pytest.raises(FileNotFoundError):
        art.read_player_history()
    with pytest.raises(FileNotFoundError):
        art.read_roster()
    with pytest.raises(FileNotFoundError):
        art.read_training_table()
    with pytest.raises(FileNotFoundError):
        art.read_predictions()
    with pytest.raises(FileNotFoundError):
        art.read_missing_history()


def test_write_creates_parent_directories(tmp_path: Path) -> None:
    """`write_*` calls `mkdir -p` on the parent (the slice scripts
    can write the artifact without first creating `output/`)."""
    art = Artifacts(root=tmp_path / "deep" / "nested")
    art.write_shootout_kicks([_shootout_kick()])
    assert art.shootout_kicks.exists()
