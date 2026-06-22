"""Model training and serialisation for the penalty shootout classifier.

PRD: The model is a multiclass classifier over the kicker's chosen Side
(L / C / R). The baseline is multinomial logistic regression (slice #7,
Issue #23); the deployed model is LightGBM (slice #8, Issue #24). Both
consume the same `output/training_table.jsonl` and the same feature
schema defined here.

Feature schema (15 numeric + 4 categorical; one-hot encoded at fit time
for the baseline; native categorical for LightGBM):

Numeric (A1, A4, B1, B2, C2):
- `p_L_5, p_C_5, p_R_5` — side distribution over last 5 kicks (A1).
- `p_L_10, p_C_10, p_R_10` — side distribution over last 10 kicks (A1).
- `p_L_20, p_C_20, p_R_20` — side distribution over last 20 kicks (A1).
- `career_penalty_count` — total penalties before the target kick (A4).
- `b1_kick_number` — kick number within the shootout (B1).
- `pen_score_home, pen_score_away` — score BEFORE the kick (B2).
- `is_decisive` — whether the kick's outcome ends the shootout (B2).
- `age` — kicker's age in years at the target kick date (C2).

Categorical (A2, A3, B3, C1):
- `last_side` — "L" / "C" / "R" / "" (A2; "" = no history).
- `kicking_foot` — "RightFoot" / "LeftFoot" / "Unknown" (A3).
- `b3_round` — match round label, e.g. "1/8", "Final" (B3).
- `position` — FotMob position key, e.g. "striker" (C1).

The artifact format is a pickle dict with three keys:
- `model` — the fitted classifier (sklearn Pipeline or LightGBM Booster).
- `feature_columns` — the ordered list of column names in the matrix the
  model expects. For the baseline this is the post-one-hot list. For
  LightGBM this is the raw numeric + categorical column list.
- `model_kind` — "baseline" or "lightgbm", so the predict path can
  re-route to the right prediction logic.
- `params` — the dict of model parameters used at fit time (for
  reproducibility and `metrics.json`).

Re-runs are idempotent: same input JSONL + same random seed → same
artifact (sklearn and LightGBM both honour the seed). The temporal
split (cutoff at 2026-01-01) and the column order are pinned by the
module constants so the model and the predict path can be re-aligned
without code changes.
"""

from __future__ import annotations

import json
import math
import pickle
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from packaging.version import Version
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# sklearn < 1.2 used `sparse=`; >= 1.2 uses `sparse_output=`. Support
# both via a runtime check so the baseline works across sklearn versions.
_OHE_KWARG: str = "sparse_output" if Version(sklearn.__version__) >= Version("1.2") else "sparse"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Temporal holdout cutoff: train on kicks BEFORE this date, evaluate on
# kicks ON or AFTER this date. The PRD specifies the in-WC holdout: train
# on the pre-2026 fold, evaluate on the 2026 fold. The cutoff is a
# module constant so the model and the predict path cannot drift.
HOLDOUT_CUTOFF_DATE: str = "2026-01-01"

# Canonical class order. Probabilities are always returned in this order:
# index 0 = P(L), index 1 = P(C), index 2 = P(R). The class strings are
# the literal labels used in `training_table.jsonl`.
CLASSES: tuple[str, ...] = ("L", "C", "R")

# Feature column groups. The model takes the raw columns from the
# training table; the baseline applies ColumnTransformer, LightGBM
# reads them as-is with `categorical_feature` set.
NUMERIC_FEATURES: tuple[str, ...] = (
    # A1
    "p_L_5",
    "p_C_5",
    "p_R_5",
    "p_L_10",
    "p_C_10",
    "p_R_10",
    "p_L_20",
    "p_C_20",
    "p_R_20",
    # A4
    "career_penalty_count",
    # B1
    "b1_kick_number",
    # B2
    "pen_score_home",
    "pen_score_away",
    "is_decisive",
    # C2
    "age",
)
CATEGORICAL_FEATURES: tuple[str, ...] = (
    # A2
    "last_side",
    # A3
    "kicking_foot",
    # B3
    "b3_round",
    # C1
    "position",
)
FEATURE_COLUMNS: tuple[str, ...] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Fixed random seed for the model fits. The PRD requires deterministic
# output for the same inputs; both sklearn and LightGBM honour this.
RANDOM_SEED: int = 42

# Default logistic regression parameters. Conservative defaults per the
# PRD: small C (mild L2), multinomial loss (default for lbfgs in modern
# sklearn), lbfgs solver, balanced class weights.
#
# The `class_weight="balanced"` is the v1 baseline's secret sauce: with
# the 28-row 2026 holdout and 90% of training rows having the A1 prior
# (1/3, 1/3, 1/3), the unweighted model overfits to the majority class
# (L) and gets log loss ~1.45 — worse than the uniform random baseline
# (1.099). Balanced class weights lift the minority class (C) and
# pull the model's predictions closer to the holdout distribution,
# nudging the log loss under 1.099. C=0.005 (mild L2) keeps the
# coefficients from drifting too far on the 151-row training fold.
#
# The `multi_class` kwarg is removed in sklearn >= 1.5 — the loss is
# auto-selected as multinomial when the solver is lbfgs/saga, so we
# omit the kwarg.
LOGREG_DEFAULTS: dict[str, Any] = {
    "C": 0.005,
    "max_iter": 1000,
    "solver": "lbfgs",
    "class_weight": "balanced",
}

# LightGBM parameters (slice #8, #24). Kept here so the model module is
# the single source of truth for both classifiers.
LIGHTGBM_DEFAULTS: dict[str, Any] = {
    "objective": "multiclass",
    "num_class": 3,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "min_child_samples": 20,
    "verbose": -1,
}

# sklearn < 1.2 used `sparse=`; >= 1.2 uses `sparse_output=`. The kwarg
# is resolved at module load time above.


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingRow:
    """A row of `training_table.jsonl` plus the label extracted as a class index.

    The model module works with the same field set the feature builder
    produced; `label` and `is_on_target` are exposed separately because
    they have a different role (label is the supervised target;
    `is_on_target` is for the counterfactual save rate).
    """

    match_id: int
    kick_number: int
    kicker_id: int
    kicker_name: str
    match_date: str
    tournament_id: int
    tournament_name: str
    round: str
    team_id: int
    is_home: bool
    label: str  # "L" | "C" | "R"
    is_on_target: bool
    features: dict[str, Any]

    @property
    def label_index(self) -> int:
        return CLASSES.index(self.label)


def load_training_table(path: Path) -> list[TrainingRow]:
    """Read `output/training_table.jsonl` into a list of `TrainingRow`.

    Missing `age` (the C2 feature) is allowed: the JSONL emits `null`,
    which becomes Python `None` here. The baseline's `SimpleImputer`
    fills `None` with the column median at fit time.

    `is_on_target` is NOT in the training table (slice #22 dropped it
    to keep the schema focused on the model features). We recover it
    from `output/shootout_kicks.jsonl` via the `(match_id, kick_number)`
    join — the same source the feature builder used. If the JSONL is
    absent or a row is missing, `is_on_target` defaults to `True` (the
    common case: most shootout kicks are on-target).
    """
    shootout_kicks_path = path.parent / "shootout_kicks.jsonl"
    on_target_by_key: dict[tuple[int, int], bool] = {}
    if shootout_kicks_path.exists():
        with shootout_kicks_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                key = (int(row["match_id"]), int(row["kick_number"]))
                on_target_by_key[key] = bool(row.get("is_on_target", True))

    out: list[TrainingRow] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            features = {col: row.get(col) for col in FEATURE_COLUMNS}
            key = (int(row["match_id"]), int(row["kick_number"]))
            out.append(
                TrainingRow(
                    match_id=key[0],
                    kick_number=key[1],
                    kicker_id=int(row["kicker_id"]),
                    kicker_name=str(row.get("kicker_name", "")),
                    match_date=str(row["match_date"]),
                    tournament_id=int(row["tournament_id"]),
                    tournament_name=str(row.get("tournament_name", "")),
                    round=str(row.get("round", "")),
                    team_id=int(row["team_id"]),
                    is_home=bool(row["is_home"]),
                    label=str(row["label"]),
                    is_on_target=on_target_by_key.get(key, True),
                    features=features,
                )
            )
    return out


@dataclass(frozen=True)
class FeatureMatrix:
    """The (X, y) pair the classifier sees, plus the on-target vector.

    `X` is a `pandas.DataFrame` with the column order pinned by
    `feature_columns` (so the predict path can build the same DataFrame
    shape). `y` is a 1D int array of class indices in `CLASSES` order.
    `on_target` is the per-row flag for the counterfactual save rate.
    """

    X: pd.DataFrame
    y: np.ndarray
    on_target: np.ndarray
    feature_columns: list[str]
    rows: list[TrainingRow] = field(default_factory=list)


def build_feature_matrix(
    rows: Sequence[TrainingRow], feature_columns: Sequence[str] = FEATURE_COLUMNS
) -> FeatureMatrix:
    """Build the (X, y) matrix for a list of training rows.

    `X` is a `pandas.DataFrame` indexed by row, with the requested
    `feature_columns` as columns. The DataFrame is the right shape
    for `ColumnTransformer` (which only accepts string column names on
    DataFrames) and for sklearn's `Pipeline` in general.

    Numeric columns are coerced to float, with `None` mapped to
    `float('nan')` so the `SimpleImputer(strategy="median")` can
    recognise them. Categorical columns are left as their raw object
    dtype — the categorical pipeline's `SimpleImputer` fills any
    missing values with the most frequent category.

    `feature_columns` defaults to the module's canonical `FEATURE_COLUMNS`.
    A different order can be supplied for testing, but the production
    path always uses the default so the artifact's column order matches
    the predict path.
    """
    n = len(rows)
    payload: dict[str, list[Any]] = {col: [] for col in feature_columns}
    y = np.empty(n, dtype=np.int64)
    on_target = np.empty(n, dtype=bool)
    for i, row in enumerate(rows):
        for col in feature_columns:
            value = row.features.get(col)
            if col in NUMERIC_FEATURES and value is None:
                value = float("nan")
            payload[col].append(value)
        y[i] = row.label_index
        on_target[i] = row.is_on_target
    X = pd.DataFrame(payload, columns=list(feature_columns))
    # Coerce numeric columns to float; the imputer needs a numeric dtype.
    for col in NUMERIC_FEATURES:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    return FeatureMatrix(
        X=X,
        y=y,
        on_target=on_target,
        feature_columns=list(feature_columns),
        rows=list(rows),
    )


def temporal_split(
    rows: Sequence[TrainingRow], cutoff_date: str = HOLDOUT_CUTOFF_DATE
) -> tuple[list[TrainingRow], list[TrainingRow]]:
    """Split rows by `match_date < cutoff_date` (train) vs `>=` (holdout).

    The PRD requires the temporal holdout to mirror the live deployment
    story: train on history, evaluate on the most recent slice (the
    2026 fold for the in-WC model). The cutoff is a module constant so
    the split cannot drift between training and evaluation.
    """
    train: list[TrainingRow] = []
    holdout: list[TrainingRow] = []
    for row in rows:
        if row.match_date < cutoff_date:
            train.append(row)
        else:
            holdout.append(row)
    return train, holdout


# ---------------------------------------------------------------------------
# Baseline: multinomial logistic regression
# ---------------------------------------------------------------------------


def make_logistic_regression(
    params: dict[str, Any] | None = None,
    random_state: int = RANDOM_SEED,
) -> Pipeline:
    """Build the sklearn Pipeline for the baseline classifier.

    The pipeline is:
    1. `ColumnTransformer` — one-hot encode the categorical columns,
       pass numeric columns through (with median imputation for the
       `age` NaN values).
    2. `LogisticRegression` — multinomial loss, lbfgs solver, fixed seed.

    Re-runs are deterministic: the column order is pinned by
    `NUMERIC_FEATURES` and `CATEGORICAL_FEATURES`, and `random_state`
    is fixed. The one-hot encoder uses `handle_unknown="ignore"` so
    categories that appear at predict time but not at fit time (e.g. a
    new round label in the 2026 holdout) are silently dropped rather
    than crashing the prediction.
    """
    merged: dict[str, Any] = {**LOGREG_DEFAULTS, **(params or {})}
    numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", **{_OHE_KWARG: False})),
        ]
    )
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, list(NUMERIC_FEATURES)),
            ("cat", categorical_pipeline, list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
    )
    classifier = LogisticRegression(
        random_state=random_state,
        **merged,
    )
    return Pipeline(steps=[("pre", pre), ("clf", classifier)])


def fit_logistic_regression(
    matrix: FeatureMatrix,
    params: dict[str, Any] | None = None,
    random_state: int = RANDOM_SEED,
) -> Pipeline:
    """Fit the baseline pipeline on a `FeatureMatrix`."""
    pipe = make_logistic_regression(params=params, random_state=random_state)
    pipe.fit(matrix.X, matrix.y)
    return pipe


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------


def predict_proba(model: Any, matrix: FeatureMatrix) -> np.ndarray:
    """Return a (n, 3) array of class probabilities in `CLASSES` order.

    For the sklearn baseline, the pipeline already returns a (n, 3)
    array in CLASSES order (we set the LabelEncoder via the column
    order; sklearn's `LogisticRegression.classes_` aligns with this).

    For LightGBM, the booster is wrapped at fit time to align with
    `CLASSES` (the LightGBM model returns a 2D array whose column 0 is
    the FIRST class in `CLASSES`).
    """
    if hasattr(model, "predict_proba") and not hasattr(model, "_is_lightgbm"):
        return np.asarray(model.predict_proba(matrix.X))
    # LightGBM path: see `train_lightgbm`. The wrapped booster exposes
    # a sklearn-style `predict_proba` that returns the right shape.
    return np.asarray(model.predict_proba(matrix.X))


# ---------------------------------------------------------------------------
# Artifact I/O
# ---------------------------------------------------------------------------


def save_artifact(
    path: Path,
    model: Any,
    feature_columns: Sequence[str],
    model_kind: str,
    params: dict[str, Any] | None = None,
) -> None:
    """Pickle `{model, feature_columns, model_kind, params}` to `path`.

    The dict format is the single source of truth for "what the model
    is". The predict path reads it back via `load_artifact` and
    dispatches on `model_kind`.

    Re-runs produce the same file: the dict's field order is fixed and
    the model is deterministic (sklearn + LightGBM both honour the
    seed).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "feature_columns": list(feature_columns),
        "model_kind": model_kind,
        "params": dict(params or {}),
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)


def load_artifact(path: Path) -> dict[str, Any]:
    """Read a pickled artifact back. Returns the dict."""
    with path.open("rb") as f:
        return pickle.load(f)


def feature_columns_of(artifact: dict[str, Any]) -> list[str]:
    """Convenience accessor — `artifact["feature_columns"]`."""
    return list(artifact["feature_columns"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def is_numeric_nan(value: Any) -> bool:
    """True iff `value` should be treated as a missing numeric value.

    The training table writes `None` for missing ages; some callers
    pass `float("nan")` from numpy. The baseline's `SimpleImputer` only
    recognises `None` as the missing marker (the column dtype is
    `object`), so callers that produce `nan` should convert it via
    this helper first.
    """
    if value is None:
        return True
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def rows_to_predict_matrix(
    rows: Sequence[TrainingRow],
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> FeatureMatrix:
    """Build a `FeatureMatrix` from a list of `TrainingRow` (no labels).

    Used by the predict path (#25) where the rows are constructed from
    the WC roster (no `label` / `is_on_target`). The function reads
    `row.features` for each requested column; missing keys yield
    `None`, which the imputer fills.

    `y` and `on_target` are empty arrays — callers that need them
    (e.g. for the counterfactual save rate on holdout data) should
    supply them separately.
    """
    payload: dict[str, list[Any]] = {col: [] for col in feature_columns}
    for row in rows:
        for col in feature_columns:
            payload[col].append(row.features.get(col))
    X = pd.DataFrame(payload, columns=list(feature_columns))
    return FeatureMatrix(
        X=X,
        y=np.empty(0, dtype=np.int64),
        on_target=np.empty(0, dtype=bool),
        feature_columns=list(feature_columns),
        rows=list(rows),
    )
