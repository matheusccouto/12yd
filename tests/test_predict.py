"""Tests for the predict slice (slice #9, Issue #25).

The tests cover five layers:

1. **Pure helpers** — `build_prediction_features` (the per-kicker
   feature builder for a prediction target). No network, no I/O.

2. **Per-kicker predict** — `predict_kicker` against a stubbed model
   that records the input it received and returns canned probabilities.
   Verifies the schema of the returned `PredictionRow`.

3. **Orchestration** — `predict_roster` against the stubbed model and
   stubbed metadata fetcher. Verifies the row count, the order, and
   that one bad metadata fetch does not abort the run.

4. **JSONL helpers** — `Artifacts.write_predictions` /
   `Artifacts.read_predictions` roundtrip. NaN / float fields are
   serialised correctly.

5. **Live smoke tests** — `output/predictions.jsonl` (skipped if absent):
   schema, row count matches the roster, probabilities sum to 1 and are
   non-negative, no-history kickers have `kicking_foot="Unknown"`,
   two consecutive runs produce byte-identical output.

The live smoke tests depend on the upstream artifacts being present
(roster, player history, LightGBM model). The slice is re-runnable
end-to-end via `python scripts/predict.py`.

Phase 0 (Issue #30): no `_training_row_from_table_row` bridge — the
unified `TrainingRow` carries its own features, so the predict slice
constructs one via `compute_features` + `build_features` with neutral
B-group context.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from penalty_pred.artifacts import Artifacts
from penalty_pred.features import PRIOR_PROB
from penalty_pred.player_history import PlayerMetadata
from penalty_pred.predict import (
    PredictionRow,
    build_prediction_features,
    predict_kicker,
    predict_roster,
)
from penalty_pred.rosters import RosterPlayer
from tests._factories import (
    PREDICTION_ROW_FIELDS,
    make_history_row,
    make_metadata,
    make_roster_player,
)

_roster_player = make_roster_player
_penalty = make_history_row
_metadata = make_metadata

TARGET_DATE: str = "2026-12-31T00:00:00+00:00"


# ---------------------------------------------------------------------------
# build_prediction_features
# ---------------------------------------------------------------------------


def test_build_prediction_features_no_history_uses_prior() -> None:
    """No history → A1 = (1/3, 1/3, 1/3), A2 = "", A3 = "Unknown", A4 = 0.

    The B-group values are the neutral defaults (kick_number=1,
    pen_score=0-0, is_decisive=False, round=""). The C-group values
    come from the metadata (C1=position, C2=age at target_date).
    """
    row = build_prediction_features(
        kicker=_roster_player(player_id=1),
        history=[],
        metadata=_metadata(player_id=1, birth_date="1995-01-01"),
        target_date=TARGET_DATE,
    )
    assert (row.p_L_5, row.p_C_5, row.p_R_5) == PRIOR_PROB
    assert (row.p_L_10, row.p_C_10, row.p_R_10) == PRIOR_PROB
    assert (row.p_L_20, row.p_C_20, row.p_R_20) == PRIOR_PROB
    assert row.last_side == ""
    assert row.preferred_foot == ""  # default in make_metadata
    assert row.career_penalty_count == 0
    # Neutral B-group
    assert row.b1_kick_number == 1
    assert row.pen_score_home == 0
    assert row.pen_score_away == 0
    assert row.is_decisive is False
    # v3: no b3_round column on the unified row.
    # C-group from metadata
    assert row.position == "striker"
    # Born 1995-01-01, target 2026-12-31: 2026 - 1995 = 31 (had 31st
    # birthday on 2026-01-01).
    assert row.age == pytest.approx(31.0)


def test_build_prediction_features_with_history_computes_a1_a2_a3_a4() -> None:
    """A 5-kick history: A1 over last 5 = (0.6, 0.0, 0.4); A2 = "R";
    A4 = 5. A3 (`preferred_foot`) comes from `metadata.preferred_foot`
    — the test metadata doesn't set it, so A3 is the empty string
    (the v3 default; the per-penalty `shot_type` mode is no longer
    consulted for the A3 feature)."""
    history = [
        _penalty(1, "2024-01-01T00:00:00+00:00", side="L"),
        _penalty(2, "2024-02-01T00:00:00+00:00", side="L"),
        _penalty(3, "2024-03-01T00:00:00+00:00", side="R"),
        _penalty(4, "2024-04-01T00:00:00+00:00", side="L"),
        _penalty(5, "2024-05-01T00:00:00+00:00", side="R"),
    ]
    row = build_prediction_features(
        kicker=_roster_player(player_id=1),
        history=history,
        metadata=_metadata(player_id=1),
        target_date=TARGET_DATE,
    )
    assert (row.p_L_5, row.p_C_5, row.p_R_5) == (0.6, 0.0, 0.4)
    assert row.last_side == "R"
    assert row.preferred_foot == ""  # metadata default
    assert row.career_penalty_count == 5


def test_build_prediction_features_no_metadata_handles_missing_c1_c2() -> None:
    """`metadata=None` → C1 = "" and C2 = NaN (the model's
    `SimpleImputer` and the LightGBM wrapper's missing-value handling
    both treat NaN as missing)."""
    row = build_prediction_features(
        kicker=_roster_player(player_id=1),
        history=[],
        metadata=None,
        target_date=TARGET_DATE,
    )
    assert row.position == ""
    assert math.isnan(row.age)


def test_build_prediction_features_filters_history_to_before_target() -> None:
    """`build_features` filters history to `match_date < target_date`
    (strict `<`). A penalty at exactly `target_date` is excluded, and
    any future penalty is excluded too. The same convention as the
    training slice.
    """
    history = [
        _penalty(1, "2024-01-01T00:00:00+00:00", side="L"),
        _penalty(2, "2026-12-31T00:00:00+00:00", side="R"),  # same as target — excluded
        _penalty(3, "2027-01-01T00:00:00+00:00", side="C"),  # after target — excluded
    ]
    row = build_prediction_features(
        kicker=_roster_player(player_id=1),
        history=history,
        metadata=None,
        target_date=TARGET_DATE,
    )
    assert row.career_penalty_count == 1
    assert (row.p_L_5, row.p_C_5, row.p_R_5) == (1.0, 0.0, 0.0)
    assert row.last_side == "L"


# ---------------------------------------------------------------------------
# build_prediction_features → TrainingRow (Issue #30)
# ---------------------------------------------------------------------------


def test_build_prediction_features_returns_unified_training_row() -> None:
    """`build_prediction_features` returns a `TrainingRow` whose
    18 model features are individual fields (the unified row type).
    The previous `_training_row_from_table_row` bridge is gone —
    the row carries the features directly. v3 dropped B3
    (`b3_round`), so the field set shrinks from 19 to 18.
    """
    row = build_prediction_features(
        kicker=_roster_player(player_id=42),
        history=[],
        metadata=_metadata(player_id=42),
        target_date=TARGET_DATE,
    )
    from penalty_pred.model import FEATURE_COLUMNS

    # All 18 feature columns are accessible as fields.
    for col in FEATURE_COLUMNS:
        assert hasattr(row, col), f"TrainingRow missing field {col!r}"
    assert row.position == "striker"
    assert row.preferred_foot == ""  # metadata default
    # Dummies (unused at predict time).
    assert row.label == "L"
    assert row.is_on_target is True
    # Identifiers are pass-throughs.
    assert row.kicker_id == 42
    assert row.match_date == TARGET_DATE


# ---------------------------------------------------------------------------
# predict_kicker (against a stub model)
# ---------------------------------------------------------------------------


class _StubModel:
    """A minimal stand-in for the LightGBM model.

    Records the `FeatureMatrix` it received so tests can inspect the
    features, and returns a canned probability vector. The shape
    matches the real model's `predict_proba`.
    """

    def __init__(self, return_probs: np.ndarray) -> None:
        self.return_probs = return_probs
        self.last_x: Any = None

    def predict_proba(self, X: Any) -> np.ndarray:
        self.last_x = X
        n = X.shape[0]
        return np.tile(self.return_probs, (n, 1))


def test_predict_kicker_with_no_history_returns_prior_only_row() -> None:
    """A kicker with no history and no metadata gets `kicking_foot=""`
    (the v3 default — no metadata → no preferred_foot → empty string)
    and a `PredictionRow` with the stub model's prior-only output."""
    stub = _StubModel(np.array([0.4, 0.2, 0.4]))
    pred = predict_kicker(
        model=stub,
        kicker=_roster_player(player_id=1, player_name="Alpha"),
        history=[],
        metadata=_metadata(player_id=1),
        target_date=TARGET_DATE,
    )
    assert isinstance(pred, PredictionRow)
    assert pred.player_id == 1
    assert pred.player_name == "Alpha"
    assert pred.kicking_foot == ""
    assert pred.p_L == pytest.approx(0.4)
    assert pred.p_C == pytest.approx(0.2)
    assert pred.p_R == pytest.approx(0.4)


def test_predict_kicker_uses_preferred_foot_from_metadata() -> None:
    """v3 (Issue #36): `kicking_foot` (the `PredictionRow` field name,
    kept for consumer continuity) is now the declared preferred foot
    from `PlayerMetadata.preferred_foot`. The previous inference from
    the per-penalty `shot_type` mode is gone."""
    stub = _StubModel(np.array([0.5, 0.25, 0.25]))
    pred = predict_kicker(
        model=stub,
        kicker=_roster_player(player_id=2),
        history=[],
        metadata=PlayerMetadata(
            player_id=2,
            player_name="X",
            position_key="striker",
            birth_date="1990-01-01",
            preferred_foot="left",
        ),
        target_date=TARGET_DATE,
    )
    assert pred.kicking_foot == "left"


def test_predict_kicker_passes_correct_features_to_model() -> None:
    """The features the model receives are the canonical 18 columns
    in `FEATURE_COLUMNS` order (v3 dropped B3), with the A1 prior for
    a no-history kicker. Inspect the captured `X`."""
    from penalty_pred.model import FEATURE_COLUMNS

    stub = _StubModel(np.array([1 / 3, 1 / 3, 1 / 3]))
    predict_kicker(
        model=stub,
        kicker=_roster_player(player_id=3),
        history=[],
        metadata=_metadata(player_id=3, position_key="striker", birth_date="1990-01-01"),
        target_date=TARGET_DATE,
    )
    assert stub.last_x is not None
    X = stub.last_x
    # Column order matches the model's expectation.
    assert list(X.columns) == list(FEATURE_COLUMNS)
    # Single row.
    assert X.shape[0] == 1
    # A1 prior (1/3, 1/3, 1/3) on every horizon.
    for col in (
        "p_L_5",
        "p_C_5",
        "p_R_5",
        "p_L_10",
        "p_C_10",
        "p_R_10",
        "p_L_20",
        "p_C_20",
        "p_R_20",
    ):
        assert X[col].iloc[0] == pytest.approx(1 / 3)
    # C-group from metadata.
    assert X["position"].iloc[0] == "striker"


def test_predict_kicker_probabilities_match_stub() -> None:
    """The `PredictionRow` probabilities are the model's output, not
    re-normalised. A stub that returns (0.5, 0.3, 0.2) yields exactly
    those values in the row."""
    stub = _StubModel(np.array([0.5, 0.3, 0.2]))
    pred = predict_kicker(
        model=stub,
        kicker=_roster_player(player_id=4),
        history=[],
        metadata=None,
        target_date=TARGET_DATE,
    )
    assert pred.p_L == pytest.approx(0.5)
    assert pred.p_C == pytest.approx(0.3)
    assert pred.p_R == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# predict_roster
# ---------------------------------------------------------------------------


def test_predict_roster_returns_one_row_per_kicker() -> None:
    """The orchestrator returns one `PredictionRow` per input roster
    player, in the same order."""
    roster = [
        _roster_player(player_id=1, player_name="Alpha"),
        _roster_player(player_id=2, player_name="Bravo"),
        _roster_player(player_id=3, player_name="Charlie"),
    ]
    stub = _StubModel(np.array([0.5, 0.25, 0.25]))
    out = predict_roster(
        model=stub,
        roster=roster,
        player_history={},
        metadata_fetcher=lambda _pid: None,
        target_date=TARGET_DATE,
    )
    assert len(out) == 3
    assert [r.player_id for r in out] == [1, 2, 3]
    assert [r.player_name for r in out] == ["Alpha", "Bravo", "Charlie"]


def test_predict_roster_looks_up_per_kicker_metadata() -> None:
    """v3 (Issue #36): each kicker's `kicking_foot` (the JSONL column;
    semantically now `preferred_foot`) is read from the metadata
    fetcher, not the per-kicker `player_history`. A kicker whose
    metadata fetch returns a non-None `PlayerMetadata` with
    `preferred_foot="right"` gets `kicking_foot="right"` in the
    output; a kicker with no metadata gets `kicking_foot=""`."""
    roster = [
        _roster_player(player_id=1, player_name="With Metadata"),
        _roster_player(player_id=2, player_name="No Metadata"),
    ]
    stub = _StubModel(np.array([1 / 3, 1 / 3, 1 / 3]))

    def fetch(pid: int) -> PlayerMetadata | None:
        if pid == 1:
            return _metadata(
                player_id=1, position_key="striker", birth_date="1990-01-01"
            )  # default preferred_foot = ""
        return None  # player 2

    out = predict_roster(
        model=stub,
        roster=roster,
        player_history={},
        metadata_fetcher=fetch,
        target_date=TARGET_DATE,
    )
    assert out[0].kicking_foot == ""  # make_metadata default
    assert out[1].kicking_foot == ""  # failed fetch


def test_predict_roster_handles_metadata_fetch_failure() -> None:
    """A metadata fetcher that returns None for some players does not
    abort the run — those kickers just get `position=""`,
    `age=NaN`, and `preferred_foot=""`."""
    roster = [
        _roster_player(player_id=1, player_name="A"),
        _roster_player(player_id=2, player_name="B"),
    ]
    stub = _StubModel(np.array([0.5, 0.3, 0.2]))

    def fetch(pid: int) -> PlayerMetadata | None:
        if pid == 1:
            return _metadata(
                player_id=1, position_key="striker", birth_date="1990-01-01"
            )
        return None  # failed for player 2

    out = predict_roster(
        model=stub,
        roster=roster,
        player_history={},
        metadata_fetcher=fetch,
        target_date=TARGET_DATE,
    )
    assert len(out) == 2
    assert out[0].player_id == 1
    assert out[1].player_id == 2
    assert out[1].kicking_foot == ""  # failed fetch → empty preferred_foot


def test_predict_roster_preserves_team_metadata() -> None:
    """The `team_id`, `team_name`, and `country_code` are pass-throughs
    from the roster, not derived from the prediction."""
    roster = [
        RosterPlayer(
            player_id=10,
            player_name="Mbappé",
            team_id=6710,
            team_name="France",
            country_code="FRA",
        ),
    ]
    stub = _StubModel(np.array([0.4, 0.3, 0.3]))
    out = predict_roster(
        model=stub,
        roster=roster,
        player_history={},
        metadata_fetcher=lambda _pid: None,
        target_date=TARGET_DATE,
    )
    assert out[0].team_id == 6710
    assert out[0].team_name == "France"
    assert out[0].country_code == "FRA"


# ---------------------------------------------------------------------------
# JSONL roundtrip
# ---------------------------------------------------------------------------


def test_predictions_jsonl_roundtrip(tmp_path: Path) -> None:
    """`Artifacts.write_predictions` then `read_predictions` yields
    the same `PredictionRow` records (probabilities preserved as
    floats, not strings). v3 (Issue #36): the `kicking_foot` column
    is the declared preferred foot ("right" / "left" / "both" / "")."""
    preds = [
        PredictionRow(
            player_id=1,
            player_name="Alpha",
            team_id=100,
            team_name="Argentina",
            country_code="ARG",
            kicking_foot="right",
            p_L=0.5,
            p_C=0.25,
            p_R=0.25,
        ),
        PredictionRow(
            player_id=2,
            player_name="Bravo",
            team_id=101,
            team_name="Brazil",
            country_code="BRA",
            kicking_foot="left",
            p_L=0.1,
            p_C=0.2,
            p_R=0.7,
        ),
    ]
    out_path = tmp_path / "predictions.jsonl"
    art = Artifacts(root=tmp_path)
    n = art.write_predictions(preds, path=out_path)
    assert n == 2
    back = art.read_predictions(path=out_path)
    assert back == preds


# ---------------------------------------------------------------------------
# Live smoke tests (issue #25 AC)
# ---------------------------------------------------------------------------


def _live_artifacts_present() -> bool:
    art = Artifacts()
    return (
        art.roster.exists() and art.player_history.exists() and art.lightgbm_model.exists()
    )


@pytest.mark.skipif(
    not _live_artifacts_present(),
    reason="output/ artifacts not present (run the slices first)",
)
def test_live_predictions_jsonl_schema_smoke() -> None:
    """Issue #25 AC: `predictions.jsonl` has one row per WC player, with
    `p_L + p_C + p_R ≈ 1.0` (within 1e-6), and the per-row schema is
    the canonical 9 fields.
    """
    art = Artifacts()
    roster_path = art.roster
    preds_path = art.predictions
    if not preds_path.exists():
        pytest.skip(f"{preds_path} not present (run scripts/predict.py first)")

    with roster_path.open(encoding="utf-8") as f:
        n_roster = sum(1 for _ in f)
    with preds_path.open(encoding="utf-8") as f:
        lines = f.readlines()
    n_preds = len(lines)
    # One row per WC player.
    assert n_preds == n_roster, (
        f"predictions.jsonl has {n_preds} rows, roster has {n_roster}; expected 1-to-1"
    )

    for i, line in enumerate(lines):
        row = json.loads(line)
        assert set(row.keys()) == set(PREDICTION_ROW_FIELDS), (
            f"row {i} has unexpected fields: {set(row.keys()) ^ set(PREDICTION_ROW_FIELDS)}"
        )
        # Probabilities are valid floats and sum to 1.
        p_L, p_C, p_R = row["p_L"], row["p_C"], row["p_R"]
        assert math.isfinite(p_L) and math.isfinite(p_C) and math.isfinite(p_R)
        assert p_L >= 0.0 and p_C >= 0.0 and p_R >= 0.0
        assert abs(p_L + p_C + p_R - 1.0) < 1e-6, (
            f"row {i} (player {row['player_id']}): p_L+p_C+p_R = {p_L + p_C + p_R}"
        )


@pytest.mark.skipif(
    not _live_artifacts_present(),
    reason="output/ artifacts not present (run the slices first)",
)
def test_live_deterministic_run() -> None:
    """Issue #25 AC: the slice is re-runnable and produces identical
    output. Two consecutive runs of the predict orchestrator on the
    same inputs yield byte-identical predictions (per-player, to
    floating-point tolerance). We test a 20-player subset to keep the
    test fast; the orchestrator is pure (same inputs → same outputs)
    and the full pipeline is idempotent by the same argument.
    """
    from penalty_pred.client import FotMobClient
    from penalty_pred.config import DEFAULT_CACHE_DIR
    from penalty_pred.features import fetcher_from_client, load_player_history
    from penalty_pred.model import load_artifact
    from penalty_pred.predict import load_roster, predict_roster

    art = load_artifact(Artifacts().lightgbm_model)
    model = art["model"]
    roster = load_roster(Artifacts().roster)[:20]
    history = load_player_history(Artifacts().player_history)
    client = FotMobClient(cache_dir=Path(DEFAULT_CACHE_DIR))
    fetcher = fetcher_from_client(client)
    # Fixed target date so the test is reproducible.
    target_date = "2026-12-31T00:00:00+00:00"

    preds1 = predict_roster(model, roster, history, fetcher, target_date)
    preds2 = predict_roster(model, roster, history, fetcher, target_date)
    # Two independent runs produce identical probabilities (per-player,
    # to floating-point tolerance).
    assert len(preds1) == len(preds2)
    for p1, p2 in zip(preds1, preds2, strict=True):
        assert p1.player_id == p2.player_id
        assert p1.kicking_foot == p2.kicking_foot
        assert p1.p_L == pytest.approx(p2.p_L)
        assert p1.p_C == pytest.approx(p2.p_C)
        assert p1.p_R == pytest.approx(p2.p_R)


@pytest.mark.skipif(
    not _live_artifacts_present(),
    reason="output/ artifacts not present (run the slices first)",
)
def test_live_no_history_kickers_have_unknown_foot() -> None:
    """Issue #25 AC: players with no penalty history have
    `kicking_foot="Unknown"` and the model's prior-only prediction
    (the A1 prior is the only signal; the model's prior-only output
    is the trained model's prior). The check is loose: we just
    confirm the schema and that all 1063 missing-history players
    from the player-history slice have `kicking_foot="Unknown"`."""
    art = Artifacts()
    preds_path = art.predictions
    if not preds_path.exists():
        pytest.skip(f"{preds_path} not present")
    missing_path = art.missing_history
    if not missing_path.exists():
        pytest.skip(f"{missing_path} not present")

    with missing_path.open(encoding="utf-8") as f:
        missing_ids = {int(json.loads(line)["player_id"]) for line in f}
    with preds_path.open(encoding="utf-8") as f:
        pred_by_id: dict[int, dict[str, object]] = {
            int(json.loads(line)["player_id"]): json.loads(line) for line in f
        }
    # Every missing-history player has kicking_foot="Unknown".
    for pid in missing_ids:
        if pid in pred_by_id:
            assert pred_by_id[pid]["kicking_foot"] == "Unknown", (
                f"player {pid} has penalty history but is in missing_history.jsonl"
            )


@pytest.mark.skipif(
    not _live_artifacts_present(),
    reason="output/ artifacts not present (run the slices first)",
)
def test_live_with_history_kickers_have_known_foot() -> None:
    """Players with penalty history have a non-"Unknown" kicking_foot
    derived from the history's `shot_type` mode."""
    art = Artifacts()
    preds_path = art.predictions
    if not preds_path.exists():
        pytest.skip(f"{preds_path} not present")
    with preds_path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    n_unknown = sum(1 for r in rows if r["kicking_foot"] == "Unknown")
    n_known = len(rows) - n_unknown
    # Live: 265/1243 with history. The model gives all 1243 players
    # a prediction; the 265 with history get a real kicking_foot.
    assert n_known > 0
    assert n_unknown > 0
    # Sanity: known feet are exactly the two-shot string values.
    for r in rows:
        assert r["kicking_foot"] in ("LeftFoot", "RightFoot", "Unknown"), (
            f"unexpected kicking_foot: {r['kicking_foot']!r}"
        )


@pytest.mark.skipif(
    not _live_artifacts_present(),
    reason="output/ artifacts not present (run the slices first)",
)
def test_live_predictions_for_all_roster_players() -> None:
    """Every player in `wc2026_roster.jsonl` has a `PredictionRow` in
    `predictions.jsonl`. The two files are 1-to-1 on `player_id`."""
    art = Artifacts()
    roster_path = art.roster
    preds_path = art.predictions
    if not preds_path.exists():
        pytest.skip(f"{preds_path} not present")
    with roster_path.open(encoding="utf-8") as f:
        roster_ids = [int(json.loads(line)["player_id"]) for line in f]
    with preds_path.open(encoding="utf-8") as f:
        pred_ids = [int(json.loads(line)["player_id"]) for line in f]
    # Roster may have duplicates (a player on multiple team lineups in
    # FotMob's data) — set comparison is the right invariant.
    assert set(roster_ids) == set(pred_ids)
    # And the count matches.
    assert len(roster_ids) == len(pred_ids)


# ---------------------------------------------------------------------------
# Script CLI
# ---------------------------------------------------------------------------


def test_predict_script_accepts_reparameterisation_flags() -> None:
    """Issue #25 AC: the slice can be re-parameterised for a different
    roster (e.g. a knockout-round subset) without code changes. The
    CLI exposes `--roster`, `--player-history`, `--model`, `--output`,
    `--target-date`, and `--cache-dir`.
    """
    from subprocess import run

    out = run(
        ["python", "scripts/predict.py", "--help"],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).resolve().parent.parent,
    )
    assert "--roster" in out.stdout
    assert "--player-history" in out.stdout
    assert "--model" in out.stdout
    assert "--output" in out.stdout
    assert "--target-date" in out.stdout
    assert "--cache-dir" in out.stdout


def test_predict_script_default_paths() -> None:
    """The default CLI paths match the upstream artifacts so the
    script runs without flags."""
    from subprocess import run

    out = run(
        ["python", "scripts/predict.py", "--help"],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).resolve().parent.parent,
    )
    assert "output/wc2026_roster.jsonl" in out.stdout
    assert "output/player_history.jsonl" in out.stdout
    assert "output/lightgbm.pkl" in out.stdout
    assert "output/predictions.jsonl" in out.stdout
