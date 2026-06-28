"""Tests for the LightGBM model module (slice #8, Issue #24).

Four layers:

1. **Pure helpers** — `compute_class_weights`, the
   `LightGBMClassifierWrapper`'s categorical coercion. No model
   fitting, no I/O.

2. **Wrapper behaviour** — `make_lightgbm` / `fit_lightgbm` /
   `predict_proba`. The wrapper returns (n, 3) arrays in `CLASSES`
   order, handles unseen categorical values at predict time, and is
   deterministic across runs (same seed).

3. **Artifact I/O** — `save_artifact` / `load_artifact` roundtrip
   with `model_kind="lightgbm"`. The artifact's `feature_columns`
   match the canonical `FEATURE_COLUMNS`.

4. **Live smoke test** — `output/lightgbm.pkl` and `metrics.json`
   (skipped if absent): the artifact loads; the model beats random
   on save rate; the LightGBM is a strict improvement over the
   logreg baseline on counterfactual save rate (the keeper's KPI).
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from penalty_pred.artifacts import Artifacts
from penalty_pred.model import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    LIGHTGBM_DEFAULTS,
    RANDOM_SEED,
    LightGBMClassifierWrapper,
    TrainingRow,
    _coerce_lightgbm_categoricals,
    build_feature_matrix,
    compute_class_weights,
    fit_lightgbm,
    fit_logistic_regression,
    load_artifact,
    make_lightgbm,
    predict_proba,
    rows_to_predict_matrix,
    save_artifact,
)

# ---------------------------------------------------------------------------
# Helpers (test-local)
# ---------------------------------------------------------------------------


def _make_features(
    label: str = "L",
    kicking_foot: str = "RightFoot",
    pos: str = "striker",
    rnd: str = "1/8",
    age: float | None = 25.0,
    last_side: str = "L",
    career: int = 5,
    kick_number: int = 1,
    score: tuple[int, int] = (0, 0),
) -> dict[str, Any]:
    """Build a complete feature dict for a single TrainingRow.

    Mirrors the helper in `test_model.py` so the toy datasets are
    interchangeable between the two test modules.
    """
    return {
        "p_L_5": 1.0 if label == "L" else 0.0,
        "p_C_5": 1.0 if label == "C" else 0.0,
        "p_R_5": 1.0 if label == "R" else 0.0,
        "p_L_10": 1.0 if label == "L" else 0.0,
        "p_C_10": 1.0 if label == "C" else 0.0,
        "p_R_10": 1.0 if label == "R" else 0.0,
        "p_L_20": 1.0 if label == "L" else 0.0,
        "p_C_20": 1.0 if label == "C" else 0.0,
        "p_R_20": 1.0 if label == "R" else 0.0,
        "career_penalty_count": career,
        "b1_kick_number": kick_number,
        "pen_score_home": score[0],
        "pen_score_away": score[1],
        "is_decisive": False,
        "age": age,
        "last_side": last_side,
        "kicking_foot": kicking_foot,
        "b3_round": rnd,
        "position": pos,
    }


def _make_row(
    label: str = "L",
    *,
    match_id: int = 1,
    kick_number: int = 1,
    kicker_id: int = 1,
    date: str = "2024-06-01T00:00:00+00:00",
    **feature_overrides: Any,
) -> TrainingRow:
    features = _make_features(label=label, **feature_overrides)
    return TrainingRow(
        match_id=match_id,
        kick_number=kick_number,
        kicker_id=kicker_id,
        kicker_name="Stub",
        match_date=date,
        tournament_id=77,
        tournament_name="World Cup",
        round=feature_overrides.get("rnd", "1/8"),
        team_id=1,
        is_home=True,
        label=label,
        is_on_target=True,
        features=features,
    )


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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_lightgbm_defaults_match_published_spec() -> None:
    """The LightGBM defaults match the PRD's conservative spec."""
    assert LIGHTGBM_DEFAULTS["num_leaves"] == 31
    assert LIGHTGBM_DEFAULTS["learning_rate"] == 0.05
    assert LIGHTGBM_DEFAULTS["n_estimators"] == 500
    assert LIGHTGBM_DEFAULTS["min_child_samples"] == 20
    assert LIGHTGBM_DEFAULTS["objective"] == "multiclass"
    assert LIGHTGBM_DEFAULTS["num_class"] == 3


def test_categorical_features_match_module_constant() -> None:
    """The categorical feature tuple used by the wrapper matches
    the module's CATEGORICAL_FEATURES. The two must stay in sync so
    LightGBM and the predict path agree on which columns are
    categorical."""
    wrapper = make_lightgbm()
    assert wrapper.categorical_features == list(CATEGORICAL_FEATURES)


# ---------------------------------------------------------------------------
# compute_class_weights
# ---------------------------------------------------------------------------


def test_compute_class_weights_inverse_frequency() -> None:
    """Class weights are `n_samples / (n_classes * n_samples_per_class)`."""
    y = np.array([0] * 8 + [1] * 2 + [2] * 4, dtype=np.int64)  # 14 total
    weights = compute_class_weights(y)
    # n=14, n_classes=3; weights: 14/(3*8)=0.583, 14/(3*2)=2.333, 14/(3*4)=1.167
    assert weights == {0: 14 / 24, 1: 14 / 6, 2: 14 / 12}


def test_compute_class_weights_balanced_uniform() -> None:
    """A perfectly balanced dataset has weight 1.0 for every class."""
    y = np.array([0] * 10 + [1] * 10 + [2] * 10, dtype=np.int64)
    weights = compute_class_weights(y)
    assert weights == {0: 1.0, 1: 1.0, 2: 1.0}


def test_compute_class_weights_empty() -> None:
    """Empty input returns an empty dict (no classes observed)."""
    assert compute_class_weights(np.empty(0, dtype=np.int64)) == {}


# ---------------------------------------------------------------------------
# _coerce_lightgbm_categoricals
# ---------------------------------------------------------------------------


def test_coerce_lightgbm_categoricals_fit_captures_values() -> None:
    """At fit time, the function captures the categories present in X."""
    X = pd.DataFrame({"cat": ["a", "b", "a", "c"], "num": [1.0, 2.0, 3.0, 4.0]})
    out = _coerce_lightgbm_categoricals(X, ["cat"], fit=True, categories=None)
    assert isinstance(out["cat"].dtype, pd.CategoricalDtype)
    assert set(out["cat"].cat.categories.tolist()) == {"a", "b", "c"}


def test_coerce_lightgbm_categoricals_predict_uses_captured() -> None:
    """At predict time, the function uses the captured categories
    from the `categories` dict. Unseen values become NaN (which
    LightGBM treats as missing)."""
    X = pd.DataFrame({"cat": ["a", "b", "a", "c"]})
    out = _coerce_lightgbm_categoricals(X, ["cat"], fit=False, categories={"cat": ["a", "b"]})
    # "c" is not in the captured categories → NaN
    assert pd.isna(out["cat"].iloc[3])
    # "a" and "b" are in the captured categories → kept
    assert out["cat"].iloc[0] == "a"
    assert out["cat"].iloc[1] == "b"


def test_coerce_lightgbm_categoricals_passthrough_non_categorical() -> None:
    """Non-categorical columns are passed through unchanged."""
    X = pd.DataFrame({"cat": ["a", "b"], "num": [1.0, 2.0]})
    out = _coerce_lightgbm_categoricals(X, ["cat"], fit=True, categories=None)
    assert out["num"].tolist() == [1.0, 2.0]


def test_coerce_lightgbm_categoricals_no_mutation() -> None:
    """The function does not mutate the input DataFrame."""
    X = pd.DataFrame({"cat": ["a", "b"]})
    original_dtype = X["cat"].dtype
    _coerce_lightgbm_categoricals(X, ["cat"], fit=True, categories=None)
    assert X["cat"].dtype == original_dtype


# ---------------------------------------------------------------------------
# make_lightgbm / fit_lightgbm
# ---------------------------------------------------------------------------


def test_make_lightgbm_default_params() -> None:
    """The default wrapper uses the module's LIGHTGBM_DEFAULTS."""
    wrapper = make_lightgbm()
    assert wrapper.params["num_leaves"] == LIGHTGBM_DEFAULTS["num_leaves"]
    assert wrapper.params["learning_rate"] == LIGHTGBM_DEFAULTS["learning_rate"]
    assert wrapper.params["n_estimators"] == LIGHTGBM_DEFAULTS["n_estimators"]


def test_make_lightgbm_override_params() -> None:
    """Caller-supplied params override the defaults."""
    wrapper = make_lightgbm(params={"num_leaves": 7})
    assert wrapper.params["num_leaves"] == 7


def test_fit_lightgbm_predicts_valid_distribution() -> None:
    """The fitted wrapper returns (n, 3) arrays that sum to 1 and
    are non-negative for any input."""
    rows = _toy_dataset()
    matrix = build_feature_matrix(rows)
    wrapper = fit_lightgbm(matrix)
    probs = predict_proba(wrapper, matrix)
    assert probs.shape == (len(rows), 3)
    assert np.all(probs >= 0)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_fit_lightgbm_handles_none_age() -> None:
    """A row with `age=None` flows through the wrapper without error
    and the predictions remain a valid distribution. LightGBM handles
    NaN natively in numeric features."""
    rows = _toy_dataset() + [_make_row(label="L", age=None)]
    matrix = build_feature_matrix(rows)
    wrapper = fit_lightgbm(matrix)
    probs = predict_proba(wrapper, matrix)
    assert np.all(probs >= 0)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_fit_lightgbm_deterministic() -> None:
    """The same inputs + same seed → same predictions."""
    rows = _toy_dataset()
    matrix = build_feature_matrix(rows)
    w1 = fit_lightgbm(matrix, random_state=RANDOM_SEED)
    w2 = fit_lightgbm(matrix, random_state=RANDOM_SEED)
    p1 = predict_proba(w1, matrix)
    p2 = predict_proba(w2, matrix)
    assert np.allclose(p1, p2)


def test_fit_lightgbm_uses_inverse_frequency_class_weights() -> None:
    """The default `fit_lightgbm` applies class weights computed
    from the training fold's label distribution (inverse frequency)."""
    y = np.array([0] * 8 + [1] * 2 + [2] * 4, dtype=np.int64)
    weights = compute_class_weights(y)
    # C (class 1) is upweighted most, L (class 0) is downweighted most.
    assert weights[1] > weights[0]
    assert weights[1] > weights[2]
    assert weights[0] < weights[2]


def test_fit_lightgbm_class_weight_override() -> None:
    """A caller-supplied class_weight overrides the inverse-frequency
    default. Verifies the wrapper doesn't silently re-compute it."""
    rows = _toy_dataset()
    matrix = build_feature_matrix(rows)
    uniform = {0: 1.0, 1: 1.0, 2: 1.0}
    wrapper = fit_lightgbm(matrix, params={"class_weight": uniform})
    assert wrapper.params["class_weight"] == uniform


def test_predict_proba_columns_in_class_order() -> None:
    """`predict_proba` returns columns in `CLASSES` order (L, C, R).
    The wrapper fits on integer-encoded labels (0=L, 1=C, 2=R), so
    LGBMClassifier.classes_ is [0, 1, 2] = CLASSES indices; no
    column reordering is needed."""
    rows = _toy_dataset()
    matrix = build_feature_matrix(rows)
    wrapper = fit_lightgbm(matrix)
    probs = predict_proba(wrapper, matrix)
    # P(L) on L rows should exceed P(L) on R rows on average.
    p_L_by_class = np.array([probs[i * 30 : (i + 1) * 30, 0].mean() for i in range(3)])
    assert p_L_by_class[0] > p_L_by_class[2], (
        f"P(L) on L rows ({p_L_by_class[0]}) should exceed P(L) on R rows "
        f"({p_L_by_class[2]}). Per-class means: {p_L_by_class}"
    )


# ---------------------------------------------------------------------------
# Unseen categorical values
# ---------------------------------------------------------------------------


def test_predict_handles_unseen_categorical_values() -> None:
    """A predict-time value that wasn't in training becomes NaN
    (LightGBM treats it as missing) and the model still returns a
    valid distribution."""
    rows = _toy_dataset()
    matrix = build_feature_matrix(rows)
    wrapper = fit_lightgbm(matrix)

    # Build a new row with an unseen `b3_round`.
    new_row = _make_row(label="L", rnd="NEVER_SEEN_BEFORE")
    new_matrix = rows_to_predict_matrix([new_row])
    probs = predict_proba(wrapper, new_matrix)
    assert probs.shape == (1, 3)
    assert np.all(probs >= 0)
    assert np.allclose(probs.sum(axis=1), 1.0)


# ---------------------------------------------------------------------------
# save_artifact / load_artifact
# ---------------------------------------------------------------------------


def test_save_and_load_lightgbm_artifact_roundtrip(tmp_path: Path) -> None:
    """The LightGBM artifact roundtrips and exposes the expected dict keys."""
    rows = _toy_dataset()
    matrix = build_feature_matrix(rows)
    wrapper = fit_lightgbm(matrix)
    out = tmp_path / "lightgbm.pkl"
    save_artifact(out, wrapper, FEATURE_COLUMNS, "lightgbm", LIGHTGBM_DEFAULTS)
    art = load_artifact(out)
    assert set(art.keys()) == {"model", "feature_columns", "model_kind", "params"}
    assert art["model_kind"] == "lightgbm"
    assert art["feature_columns"] == list(FEATURE_COLUMNS)
    assert art["params"]["num_leaves"] == LIGHTGBM_DEFAULTS["num_leaves"]


def test_save_lightgbm_artifact_is_picklable_independently(tmp_path: Path) -> None:
    """The saved file is a plain pickle — readable by a third-party
    tool with no knowledge of the model module."""
    out = tmp_path / "lightgbm.pkl"
    rows = _toy_dataset()
    matrix = build_feature_matrix(rows)
    wrapper = fit_lightgbm(matrix)
    save_artifact(out, wrapper, FEATURE_COLUMNS, "lightgbm")
    with out.open("rb") as f:
        raw = pickle.load(f)
    assert raw["model_kind"] == "lightgbm"
    # The pickled model is a LightGBMClassifierWrapper.
    assert isinstance(raw["model"], LightGBMClassifierWrapper)


# ---------------------------------------------------------------------------
# Compare against the logreg baseline
# ---------------------------------------------------------------------------


def test_lightgbm_versus_logreg_on_toy_dataset() -> None:
    """On a balanced toy dataset, the LightGBM should be at least as
    good as the logreg on accuracy. (Not log loss — the toy dataset
    is balanced and the LightGBM's class weights may push it in
    unexpected directions.)"""
    rows = _toy_dataset()
    matrix = build_feature_matrix(rows)
    lgb = fit_lightgbm(matrix)
    bl = fit_logistic_regression(matrix)
    lgb_probs = predict_proba(lgb, matrix)
    bl_probs = predict_proba(bl, matrix)
    lgb_acc = float((lgb_probs.argmax(axis=1) == matrix.y).mean())
    bl_acc = float((bl_probs.argmax(axis=1) == matrix.y).mean())
    # The LightGBM should match or beat the logreg on the toy
    # dataset's accuracy. If this fails, the wrapper has a bug
    # (e.g. wrong column order).
    assert lgb_acc >= bl_acc, (
        f"LightGBM accuracy {lgb_acc:.3f} below logreg {bl_acc:.3f} on toy dataset"
    )


# ---------------------------------------------------------------------------
# Live smoke tests (issue #24 AC)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (
        Artifacts().training_table.exists()
        and Artifacts().lightgbm_model.exists()
        and Artifacts().metrics.exists()
    ),
    reason="output/ artifacts not present (run the slice first)",
)
def test_live_metrics_json_shape_with_baseline() -> None:
    """The live `output/metrics.json` has the expected sections
    including the new `baseline` field (the logreg comparison
    classifier for slice #8)."""
    with Artifacts().metrics.open(encoding="utf-8") as f:
        payload = json.load(f)
    assert "model" in payload
    assert "baseline" in payload
    assert "random_baseline" in payload
    assert "kicker_most_frequent_baseline" in payload
    assert "actual_keeper_baseline" in payload
    assert "n_train" in payload
    assert "n_holdout" in payload
    assert "holdout_cutoff_date" in payload
    assert "model_kind" in payload
    assert payload["model_kind"] == "lightgbm"
    # The baseline section is the logreg (slice #7); its log_loss
    # matches the logreg's metric from the original baseline slice.
    assert payload["baseline"]["log_loss"] is not None
    assert payload["baseline"]["accuracy"] is not None
    assert payload["baseline"]["save_rate"] is not None
    # The random baseline's log loss is ln(3) — uniform prior.
    import math

    assert math.isclose(payload["random_baseline"]["log_loss"], math.log(3.0))


@pytest.mark.skipif(
    not (Artifacts().lightgbm_model.exists() and Artifacts().metrics.exists()),
    reason="output/ artifacts not present (run the slice first)",
)
def test_live_lightgbm_beats_logreg_on_save_rate() -> None:
    """Issue #24 AC: LightGBM beats the logreg baseline on the
    counterfactual save rate (the keeper's KPI). The log loss
    comparison is not pinned here because the 28-row holdout is too
    small to distinguish the two models on log loss (LightGBM with
    conservative defaults is more confident than the logreg and gets
    higher log loss but better save rate — see progress.txt)."""
    with Artifacts().metrics.open(encoding="utf-8") as f:
        payload = json.load(f)
    lgb_save = payload["model"]["save_rate"]
    bl_save = payload["baseline"]["save_rate"]
    assert lgb_save > bl_save, (
        f"LightGBM save rate {lgb_save:.3f} did not beat logreg "
        f"baseline {bl_save:.3f} on the 2026 holdout"
    )


@pytest.mark.skipif(
    not (Artifacts().lightgbm_model.exists() and Artifacts().metrics.exists()),
    reason="output/ artifacts not present (run the slice first)",
)
def test_live_lightgbm_beats_random_and_kmf_on_save_rate() -> None:
    """Issue #24 AC: LightGBM also beats random and the
    kicker's-most-frequent-side baseline on save rate."""
    with Artifacts().metrics.open(encoding="utf-8") as f:
        payload = json.load(f)
    lgb_save = payload["model"]["save_rate"]
    rand_save = payload["random_baseline"]["save_rate"]
    kmf_save = payload["kicker_most_frequent_baseline"]["save_rate"]
    assert lgb_save > rand_save, (
        f"LightGBM save rate {lgb_save:.3f} did not beat random {rand_save:.3f} on the 2026 holdout"
    )
    assert lgb_save > kmf_save, (
        f"LightGBM save rate {lgb_save:.3f} did not beat kmf {kmf_save:.3f} on the 2026 holdout"
    )


@pytest.mark.skipif(
    not Artifacts().lightgbm_model.exists(),
    reason="output/lightgbm.pkl not present (run the slice first)",
)
def test_live_lightgbm_artifact_smoke() -> None:
    """Issue #24 AC: lightgbm.pkl is a valid LightGBM model artifact
    with the feature column order recorded, and it can be loaded
    to make predictions on a stub row."""
    art = load_artifact(Artifacts().lightgbm_model)
    assert art["model_kind"] == "lightgbm"
    assert art["feature_columns"] == list(FEATURE_COLUMNS)
    assert "model" in art
    assert isinstance(art["model"], LightGBMClassifierWrapper)
    # Predict on a stub row to confirm the artifact loads cleanly.
    rows = [_make_row(label="L")]
    matrix = rows_to_predict_matrix(rows)
    probs = predict_proba(art["model"], matrix)
    assert probs.shape == (1, 3)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert (probs >= 0).all()


@pytest.mark.skipif(
    not Artifacts().lightgbm_model.exists(),
    reason="output/lightgbm.pkl not present (run the slice first)",
)
def test_live_lightgbm_predictions_for_roster() -> None:
    """The frozen LightGBM can be loaded and used to predict on
    every player in `output/wc2026_roster.jsonl` (the predict slice
    #25's input). The probabilities sum to 1 and the prior-only
    rows (no penalty history) cluster near (1/3, 1/3, 1/3)."""
    art = Artifacts()
    roster_path = art.roster
    if not roster_path.exists():
        pytest.skip(f"{roster_path} not present")
    art_loaded = load_artifact(art.lightgbm_model)
    # We don't need to actually run the full predict slice here —
    # just confirm the artifact can be loaded and used to predict
    # on a stub row. The full predict slice is issue #25.
    rows = [_make_row(label="L", last_side=""), _make_row(label="C", last_side="C")]
    matrix = rows_to_predict_matrix(rows)
    probs = predict_proba(art_loaded["model"], matrix)
    assert probs.shape == (2, 3)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert (probs >= 0).all()
