"""Tests for the model module (slice #7, Issue #23).

Four layers:

1. **Pure helpers** — `TrainingRow` (re-exported from `features`),
   `build_feature_matrix`, `temporal_split`, `load_training_table`,
   `is_numeric_nan`. No network, no I/O.

2. **Pipeline builder / fit / predict** — `make_logistic_regression`,
   `fit_logistic_regression`. The pipeline is fitted on a constructed
   `FeatureMatrix` and verified to (a) sum to 1, (b) be non-negative,
   (c) be deterministic across runs (same seed). Predict is via
   `model.predict_proba(matrix.X)` directly (no shim, per the
   `PredictProba` Protocol).

3. **Artifact I/O** — `save_artifact` / `load_artifact` roundtrip;
   the `feature_columns` are recorded in the artifact.

4. **Live smoke test** — `output/baseline.pkl` and `metrics.json`
   (skipped if absent): the artifact has the expected shape; the
   metrics report contains all four sections; the model beats random
   on log loss.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from penalty_pred.artifacts import Artifacts
from penalty_pred.model import (
    CATEGORICAL_FEATURES,
    CLASSES,
    FEATURE_COLUMNS,
    HOLDOUT_CUTOFF_DATE,
    LOGREG_DEFAULTS,
    NUMERIC_FEATURES,
    RANDOM_SEED,
    TrainingRow,
    build_feature_matrix,
    fit_logistic_regression,
    is_numeric_nan,
    is_on_target_by_key,
    load_artifact,
    load_training_table,
    make_logistic_regression,
    save_artifact,
    temporal_split,
)

# ---------------------------------------------------------------------------
# Helpers (test-local)
# ---------------------------------------------------------------------------


def _make_row(
    label: str = "L",
    *,
    match_id: int = 1,
    kick_number: int = 1,
    kicker_id: int = 1,
    date: str = "2024-06-01T00:00:00+00:00",
    kicking_foot: str = "RightFoot",
    pos: str = "striker",
    rnd: str = "1/8",
    age: float | None = 25.0,
    last_side: str = "L",
    career: int = 5,
    score: tuple[int, int] = (0, 0),
) -> TrainingRow:
    """Build a complete `TrainingRow` for tests (the unified row type)."""
    return TrainingRow(
        match_id=match_id,
        kick_number=kick_number,
        kicker_id=kicker_id,
        kicker_name="Stub",
        match_date=date,
        tournament_id=77,
        tournament_name="World Cup",
        round=rnd,
        team_id=1,
        is_home=True,
        label=label,
        is_on_target=True,
        # A1 — uniform-ish; tests verify the matrix shape, not the math.
        p_L_5=1.0 if label == "L" else 0.0,
        p_C_5=1.0 if label == "C" else 0.0,
        p_R_5=1.0 if label == "R" else 0.0,
        p_L_10=1.0 if label == "L" else 0.0,
        p_C_10=1.0 if label == "C" else 0.0,
        p_R_10=1.0 if label == "R" else 0.0,
        p_L_20=1.0 if label == "L" else 0.0,
        p_C_20=1.0 if label == "C" else 0.0,
        p_R_20=1.0 if label == "R" else 0.0,
        # A2
        last_side=last_side,
        # A3
        kicking_foot=kicking_foot,
        # A4
        career_penalty_count=career,
        # B1
        b1_kick_number=kick_number,
        # B2
        pen_score_home=score[0],
        pen_score_away=score[1],
        is_decisive=False,
        # B3
        b3_round=rnd,
        # C1
        position=pos,
        # C2
        age=age if age is not None else float("nan"),
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_feature_columns_matches_published_schema() -> None:
    """The canonical feature column list is the 16 PRD features."""
    assert len(FEATURE_COLUMNS) == 19
    assert set(FEATURE_COLUMNS) == set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES)
    # 15 numeric + 4 categorical = 19.
    assert len(NUMERIC_FEATURES) == 15
    assert len(CATEGORICAL_FEATURES) == 4


def test_classes_order_is_lcr() -> None:
    assert CLASSES == ("L", "C", "R")


def test_holdout_cutoff_is_2026() -> None:
    assert HOLDOUT_CUTOFF_DATE == "2026-01-01"


# ---------------------------------------------------------------------------
# is_numeric_nan
# ---------------------------------------------------------------------------


def test_is_numeric_nan_none() -> None:
    assert is_numeric_nan(None) is True


def test_is_numeric_nan_nan() -> None:
    assert is_numeric_nan(float("nan")) is True


def test_is_numeric_nan_number() -> None:
    assert is_numeric_nan(0.0) is False
    assert is_numeric_nan(25) is False


def test_is_numeric_nan_string() -> None:
    # Strings are not "numeric NaN"; they're categorical values.
    assert is_numeric_nan("") is False
    assert is_numeric_nan("L") is False


# ---------------------------------------------------------------------------
# TrainingRow
# ---------------------------------------------------------------------------


def test_training_row_label_index_lcr() -> None:
    row = _make_row(label="L")
    assert row.label_index == 0
    row = _make_row(label="C")
    assert row.label_index == 1
    row = _make_row(label="R")
    assert row.label_index == 2


def test_training_row_is_frozen() -> None:
    """TrainingRow is a frozen dataclass; every field is immutable."""
    row = _make_row()
    with pytest.raises((AttributeError, Exception)):
        row.label = "R"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# build_feature_matrix
# ---------------------------------------------------------------------------


def test_build_feature_matrix_shape_and_dtypes() -> None:
    """The matrix is a DataFrame with the right shape and numeric
    columns coerced to float."""
    rows = [_make_row(label="L"), _make_row(label="R"), _make_row(label="C")]
    matrix = build_feature_matrix(rows)
    assert matrix.X.shape == (3, len(FEATURE_COLUMNS))
    assert list(matrix.X.columns) == list(FEATURE_COLUMNS)
    for col in NUMERIC_FEATURES:
        # Coerced to a numeric dtype (float, int, or bool). The age
        # column with `None` becomes NaN (a float). The other numeric
        # columns are int or bool.
        assert matrix.X[col].dtype.kind in "fiub", (
            f"numeric column {col!r} not coerced to a numeric dtype: {matrix.X[col].dtype}"
        )
    assert matrix.y.tolist() == [0, 2, 1]
    assert matrix.on_target.tolist() == [True, True, True]


def test_build_feature_matrix_none_age_becomes_nan() -> None:
    """A row with `age=None` survives into the DataFrame as NaN so
    the imputer can fill it."""
    row = _make_row(label="L", age=None)
    matrix = build_feature_matrix([row])
    assert np.isnan(matrix.X["age"].iloc[0])


def test_build_feature_matrix_default_columns() -> None:
    """If `feature_columns` is not given, the matrix uses the module
    canonical list."""
    rows = [_make_row()]
    matrix = build_feature_matrix(rows)
    assert matrix.feature_columns == list(FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# temporal_split
# ---------------------------------------------------------------------------


def test_temporal_split_pre_post() -> None:
    """Rows before the cutoff go to train, rows at or after to holdout."""
    pre = _make_row(date="2025-12-31T23:59:00+00:00")
    edge = _make_row(date="2026-01-01T00:00:00+00:00")
    post = _make_row(date="2026-06-15T00:00:00+00:00")
    train, holdout = temporal_split([pre, edge, post])
    assert len(train) == 1
    assert len(holdout) == 2
    assert train[0].match_date == pre.match_date
    assert {r.match_date for r in holdout} == {edge.match_date, post.match_date}


def test_temporal_split_custom_cutoff() -> None:
    pre = _make_row(date="2022-01-01T00:00:00+00:00")
    post = _make_row(date="2022-12-31T00:00:00+00:00")
    train, holdout = temporal_split([pre, post], cutoff_date="2022-06-01")
    assert len(train) == 1
    assert len(holdout) == 1


# ---------------------------------------------------------------------------
# load_training_table
# ---------------------------------------------------------------------------


def test_load_training_table_reads_live(tmp_path: Path) -> None:
    """The loader reads the JSONL and recovers the per-row features
    from the schema, falling back to True for is_on_target when the
    is_on_target_by_key lookup is absent or the row is missing."""
    table_path = tmp_path / "training_table.jsonl"
    row = {
        "match_id": 1,
        "kick_number": 1,
        "kicker_id": 2,
        "kicker_name": "Stub",
        "match_date": "2024-06-01T00:00:00+00:00",
        "tournament_id": 77,
        "tournament_name": "World Cup",
        "round": "1/8",
        "team_id": 1,
        "is_home": True,
        "label": "L",
        "p_L_5": 0.5,
        "p_C_5": 0.3,
        "p_R_5": 0.2,
        "p_L_10": 0.5,
        "p_C_10": 0.3,
        "p_R_10": 0.2,
        "p_L_20": 0.5,
        "p_C_20": 0.3,
        "p_R_20": 0.2,
        "last_side": "L",
        "kicking_foot": "RightFoot",
        "career_penalty_count": 5,
        "b1_kick_number": 1,
        "pen_score_home": 0,
        "pen_score_away": 0,
        "is_decisive": False,
        "b3_round": "1/8",
        "position": "striker",
        "age": 25.0,
    }
    with table_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    rows = load_training_table(table_path)
    assert len(rows) == 1
    r = rows[0]
    assert r.label == "L"
    assert r.p_L_5 == 0.5
    # No is_on_target_by_key supplied → is_on_target defaults True.
    assert r.is_on_target is True


def test_load_training_table_joins_is_on_target(tmp_path: Path) -> None:
    """When `is_on_target_by_key` is passed, the loader joins the
    per-row on-target flag against the supplied lookup. The data
    layer's directory layout is no longer leaked into the model
    layer (the join is the caller's responsibility)."""
    table_path = tmp_path / "training_table.jsonl"
    table_row = {
        "match_id": 1,
        "kick_number": 1,
        "kicker_id": 2,
        "kicker_name": "Stub",
        "match_date": "2024-06-01T00:00:00+00:00",
        "tournament_id": 77,
        "tournament_name": "World Cup",
        "round": "1/8",
        "team_id": 1,
        "is_home": True,
        "label": "L",
        "p_L_5": 0.5,
        "p_C_5": 0.3,
        "p_R_5": 0.2,
        "p_L_10": 0.5,
        "p_C_10": 0.3,
        "p_R_10": 0.2,
        "p_L_20": 0.5,
        "p_C_20": 0.3,
        "p_R_20": 0.2,
        "last_side": "L",
        "kicking_foot": "RightFoot",
        "career_penalty_count": 5,
        "b1_kick_number": 1,
        "pen_score_home": 0,
        "pen_score_away": 0,
        "is_decisive": False,
        "b3_round": "1/8",
        "position": "striker",
        "age": 25.0,
    }
    with table_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(table_row) + "\n")
    rows = load_training_table(
        table_path,
        is_on_target_by_key={(1, 1): False},  # off-target
    )
    assert rows[0].is_on_target is False


def test_is_on_target_by_key_builds_lookup() -> None:
    """`is_on_target_by_key` builds the lookup from any iterable of
    objects with `match_id`, `kick_number`, `is_on_target` (the
    `ShootoutKick` dataclass is the canonical caller)."""
    from penalty_pred.shootouts import ShootoutKick

    kicks = [
        ShootoutKick(
            match_id=1,
            kick_number=1,
            match_date="2024-01-01T00:00:00+00:00",
            tournament_id=77,
            tournament_name="WC",
            round="Final",
            kicker_id=1,
            kicker_name="X",
            team_id=1,
            is_home=True,
            x=0.5,
            side="L",
            is_on_target=True,
            outcome="Goal",
            pen_score_before=[0, 0],
            pen_score_after=[1, 0],
            match_score_home=0,
            match_score_away=0,
        ),
        ShootoutKick(
            match_id=1,
            kick_number=2,
            match_date="2024-01-01T00:00:00+00:00",
            tournament_id=77,
            tournament_name="WC",
            round="Final",
            kicker_id=2,
            kicker_name="Y",
            team_id=2,
            is_home=False,
            x=0.5,
            side="R",
            is_on_target=False,  # off-target
            outcome="Missed",
            pen_score_before=[1, 0],
            pen_score_after=[1, 0],
            match_score_home=0,
            match_score_away=0,
        ),
    ]
    lookup = is_on_target_by_key(kicks)
    assert lookup[(1, 1)] is True
    assert lookup[(1, 2)] is False


# ---------------------------------------------------------------------------
# make_logistic_regression
# ---------------------------------------------------------------------------


def test_make_logistic_regression_default_params() -> None:
    """The default pipeline uses the module's LOGREG_DEFAULTS."""
    pipe = make_logistic_regression()
    clf = pipe.named_steps["clf"]
    assert clf.C == LOGREG_DEFAULTS["C"]
    assert clf.solver == LOGREG_DEFAULTS["solver"]
    assert clf.class_weight == LOGREG_DEFAULTS["class_weight"]


def test_make_logistic_regression_override_params() -> None:
    """Caller-supplied params override the defaults."""
    pipe = make_logistic_regression(params={"C": 0.5})
    clf = pipe.named_steps["clf"]
    assert clf.C == 0.5


# ---------------------------------------------------------------------------
# fit / predict
# ---------------------------------------------------------------------------


def _toy_dataset(n_per_class: int = 30) -> list[TrainingRow]:
    """Build a balanced toy dataset of 90 rows (30 L, 30 C, 30 R)."""
    rows: list[TrainingRow] = []
    for label in ("L", "C", "R"):
        for i in range(n_per_class):
            rows.append(
                _make_row(
                    label=label,
                    kicker_id=hash((label, i)) & 0xFFFF,
                    pos={"L": "striker", "C": "midfielder", "R": "defender"}[label],
                    kicking_foot={"L": "RightFoot", "C": "RightFoot", "R": "LeftFoot"}[label],
                )
            )
    return rows


def test_fit_logistic_regression_predicts_valid_distribution() -> None:
    """The pipeline returns (n, 3) arrays that sum to 1 and are
    non-negative for any input."""
    rows = _toy_dataset()
    matrix = build_feature_matrix(rows)
    pipe = fit_logistic_regression(matrix)
    probs = np.asarray(pipe.predict_proba(matrix.X))
    assert probs.shape == (len(rows), 3)
    assert np.all(probs >= 0)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_fit_logistic_regression_handles_none_age() -> None:
    """A row with `age=None` flows through the imputer without error
    and the predictions remain a valid distribution."""
    rows = _toy_dataset() + [_make_row(label="L", age=None)]
    matrix = build_feature_matrix(rows)
    pipe = fit_logistic_regression(matrix)
    probs = np.asarray(pipe.predict_proba(matrix.X))
    assert np.all(probs >= 0)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_fit_logistic_regression_deterministic() -> None:
    """The same inputs + same seed → same predictions."""
    rows = _toy_dataset()
    matrix = build_feature_matrix(rows)
    pipe1 = fit_logistic_regression(matrix, random_state=RANDOM_SEED)
    pipe2 = fit_logistic_regression(matrix, random_state=RANDOM_SEED)
    p1 = np.asarray(pipe1.predict_proba(matrix.X))
    p2 = np.asarray(pipe2.predict_proba(matrix.X))
    assert np.allclose(p1, p2)


def test_predict_proba_columns_in_class_order() -> None:
    """The column order is `CLASSES` (L, C, R). Verify by training on
    a labelled dataset and checking that the predicted class matches
    the side of the feature signal."""
    rows = _toy_dataset()
    matrix = build_feature_matrix(rows)
    pipe = fit_logistic_regression(matrix)
    probs = np.asarray(pipe.predict_proba(matrix.X))
    # The L class's row should have the highest P(L) in column 0;
    # we don't require it to be the argmax (overlap with C is
    # possible given small data), but the L class's P(L) should be
    # larger than the C class's P(L) on average.
    p_L_by_class = np.array([probs[i * 30 : (i + 1) * 30, 0].mean() for i in range(3)])
    assert p_L_by_class[0] > p_L_by_class[2], (
        f"P(L) on L rows ({p_L_by_class[0]}) should exceed P(L) on R rows "
        f"({p_L_by_class[2]}). Per-class means: {p_L_by_class}"
    )


# ---------------------------------------------------------------------------
# save_artifact / load_artifact
# ---------------------------------------------------------------------------


def test_save_and_load_artifact_roundtrip(tmp_path: Path) -> None:
    """The artifact roundtrips and exposes the expected dict keys."""
    rows = _toy_dataset()
    matrix = build_feature_matrix(rows)
    pipe = fit_logistic_regression(matrix)
    out = tmp_path / "model.pkl"
    save_artifact(out, pipe, FEATURE_COLUMNS, "baseline", LOGREG_DEFAULTS)
    art = load_artifact(out)
    assert set(art.keys()) == {"model", "feature_columns", "model_kind", "params"}
    assert art["model_kind"] == "baseline"
    assert art["feature_columns"] == list(FEATURE_COLUMNS)
    assert art["params"]["C"] == LOGREG_DEFAULTS["C"]


def test_save_artifact_is_picklable_independently(tmp_path: Path) -> None:
    """The saved file is a plain pickle — readable by a third-party
    tool with no knowledge of the model module."""
    out = tmp_path / "model.pkl"
    rows = _toy_dataset()
    matrix = build_feature_matrix(rows)
    pipe = fit_logistic_regression(matrix)
    save_artifact(out, pipe, FEATURE_COLUMNS, "baseline")
    with out.open("rb") as f:
        raw = pickle.load(f)
    assert raw["model_kind"] == "baseline"


# ---------------------------------------------------------------------------
# build_feature_matrix with optional labels (Issue #30)
# ---------------------------------------------------------------------------


def test_build_feature_matrix_no_labels() -> None:
    """The predict path uses `build_feature_matrix` with no `y` /
    `on_target` (Issue #30: the old `rows_to_predict_matrix` is gone;
    the same builder takes optional labels)."""
    rows = [_make_row(label="L"), _make_row(label="R")]
    matrix = build_feature_matrix(rows)
    assert matrix.X.shape == (2, len(FEATURE_COLUMNS))
    # No labels supplied — the builder defaults to the rows' labels
    # (so training and predict share the same call site, just with
    # different label-providing paths).
    assert matrix.y.tolist() == [0, 2]
    assert matrix.on_target.tolist() == [True, True]
    assert matrix.feature_columns == list(FEATURE_COLUMNS)


def test_build_feature_matrix_with_explicit_y() -> None:
    """`build_feature_matrix` accepts an explicit `y` and `on_target`
    so callers (e.g. a held-out eval) can override the rows' labels."""
    rows = [_make_row(label="L"), _make_row(label="R")]
    explicit_y = np.array([1, 1], dtype=np.int64)  # both labelled C
    explicit_on_target = np.array([False, True], dtype=bool)
    matrix = build_feature_matrix(
        rows, y=explicit_y, on_target=explicit_on_target
    )
    assert matrix.y.tolist() == [1, 1]
    assert matrix.on_target.tolist() == [False, True]


# ---------------------------------------------------------------------------
# Live smoke test (issue #23 AC: model beats random on log loss)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (
        Artifacts().training_table.exists()
        and Artifacts().baseline_model.exists()
        and Artifacts().metrics.exists()
    ),
    reason="output/ artifacts not present (run the slice first)",
)
def test_baseline_metrics_beat_random_on_log_loss() -> None:
    """Issue #23 AC: the logreg baseline's log loss is below the
    random baseline's on the 2026 holdout.

    On the slice #24 (LightGBM) metrics.json, the `baseline` section
    holds the logreg result (the LightGBM is the new `model`
    section). The test reads the `baseline` section if present
    (slice #24+), otherwise the `model` section (slice #23).
    """
    with Artifacts().metrics.open(encoding="utf-8") as f:
        metrics = json.load(f)
    baseline_section = metrics.get("baseline", metrics["model"])
    baseline_ll = baseline_section["log_loss"]
    random_ll = metrics["random_baseline"]["log_loss"]
    assert baseline_ll < random_ll, (
        f"baseline log loss {baseline_ll} did not beat random {random_ll} on the 2026 holdout"
    )


@pytest.mark.skipif(
    not Artifacts().baseline_model.exists(),
    reason="output/baseline.pkl not present (run the slice first)",
)
def test_baseline_artifact_smoke() -> None:
    """Issue #23 AC: baseline.pkl is a serializable artifact that
    records the feature column order."""
    art = load_artifact(Artifacts().baseline_model)
    assert art["model_kind"] == "baseline"
    assert art["feature_columns"] == list(FEATURE_COLUMNS)
    assert "model" in art
    # Predict on a stub row to confirm the artifact loads cleanly.
    rows = [_make_row(label="L")]
    matrix = build_feature_matrix(rows)
    probs = np.asarray(art["model"].predict_proba(matrix.X))
    assert probs.shape == (1, 3)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert (probs >= 0).all()
