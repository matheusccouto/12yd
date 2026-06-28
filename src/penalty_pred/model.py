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
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMClassifier
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


def is_on_target_by_key(
    shootout_kicks: Iterable[Any],
) -> dict[tuple[int, int], bool]:
    """Build a `(match_id, kick_number) -> is_on_target` lookup from shootout kicks.

    Used by `load_training_table` to recover the per-row on-target flag
    (the training table dropped the column in slice #22 to keep the
    schema focused on the model features). The caller is expected to
    pass the list of `ShootoutKick` (or any object with `match_id`,
    `kick_number`, `is_on_target` attributes / keys).
    """
    out: dict[tuple[int, int], bool] = {}
    for kick in shootout_kicks:
        key = (int(kick.match_id), int(kick.kick_number))
        out[key] = bool(kick.is_on_target)
    return out


def load_training_table(
    path: Path,
    is_on_target_by_key: dict[tuple[int, int], bool] | None = None,
) -> list[TrainingRow]:
    """Read `output/training_table.jsonl` into a list of `TrainingRow`.

    Missing `age` (the C2 feature) is allowed: the JSONL emits `null`,
    which becomes Python `None` here. The baseline's `SimpleImputer`
    fills `None` with the column median at fit time.

    `is_on_target` is NOT in the training table (slice #22 dropped it
    to keep the schema focused on the model features). The caller is
    expected to pass the lookup from `is_on_target_by_key(shootout_kicks)`;
    if the lookup is absent or a row is missing, `is_on_target` defaults
    to `True` (the common case: most shootout kicks are on-target).
    The data layer's directory layout is no longer leaked into the
    model layer — the join is the caller's responsibility, not the
    loader's.
    """
    on_target_by_key = is_on_target_by_key or {}

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
    array in CLASSES order (the column order is pinned by
    `FEATURE_COLUMNS` and `LogisticRegression.classes_` is sorted
    alphabetically — L < R < C, but the coefficients are fit on the
    column order we set, so the output is in CLASSES order by
    construction).

    For LightGBM, the booster is wrapped in
    `LightGBMClassifierWrapper` (see below). The wrapper:
    - Coerces the categorical columns to `pd.Categorical` with the
      categories observed at fit time (so unseen values at predict
      time become NaN, which LightGBM treats as missing).
    - Calls the underlying `LGBMClassifier.predict_proba`, which
      returns columns in the order of the integer-encoded labels
      (0=L, 1=C, 2=R — matching `CLASSES`).
    """
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(matrix.X))
    raise TypeError(f"model of type {type(model).__name__} has no predict_proba")


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


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------


@dataclass
class LightGBMClassifierWrapper:
    """A LightGBM classifier that preserves the project conventions.

    The wrapper:
    1. At fit time, coerces the `categorical_features` columns to
       `pd.Categorical` (with the categories observed in the training
       data) and stores the category list per column.
    2. At predict time, coerces the same columns to `pd.Categorical`
       with the stored categories, so unseen values become `NaN`
       (LightGBM treats them as missing). This is the LightGBM-native
       way to handle "this categorical value wasn't in training".
    3. Forwards `predict_proba` to the underlying `LGBMClassifier`,
       which returns columns in the order of the integer-encoded
       labels (0=L, 1=C, 2=R). The fit method requires `y` to be
       int-encoded in `CLASSES` order.

    The wrapper is pickle-safe: the underlying LGBMClassifier is
    serialised by LightGBM's own Booster pickling, and the
    `_categories` dict is plain Python.

    Parameters
    ----------
    params
        A dict of LGBMClassifier parameters. Defaults to
        `LIGHTGBM_DEFAULTS`. The wrapper does not require `num_class`
        or `objective` to be set — they are added at construction.
    categorical_features
        The list of column names in `X` to treat as categorical. The
        columns must contain string values (or be coercible to
        `pd.Categorical`).
    random_state
        The random seed for the booster.
    """

    params: dict[str, Any]
    categorical_features: list[str]
    random_state: int

    # Populated by `fit`.
    _booster: LGBMClassifier | None = field(default=None, init=False)
    _categories: dict[str, list[Any]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        # Normalise the params: every LGBMClassifier needs `objective`
        # and `num_class` to do multiclass. We keep them in the
        # `LIGHTGBM_DEFAULTS` dict for clarity, but also stamp them on
        # the LGBMClassifier constructor for the case where a caller
        # passes a custom `params` (so we don't accidentally drop them).
        merged: dict[str, Any] = {**LIGHTGBM_DEFAULTS, **self.params}
        merged["random_state"] = self.random_state
        self._resolved_params: dict[str, Any] = merged

    @property
    def classes_(self) -> np.ndarray:
        """The class indices in the order `LGBMClassifier.predict_proba` uses."""
        if self._booster is None:
            return np.array([], dtype=np.int64)
        return np.asarray(self._booster.classes_)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> LightGBMClassifierWrapper:
        """Fit the underlying LGBMClassifier on `X` (DataFrame) and `y` (int array).

        `y` must be int-encoded in `CLASSES` order (0=L, 1=C, 2=R).
        The caller is expected to do the encoding — the wrapper does
        not look at the string labels, only the integer class indices.
        """
        X_fit = _coerce_lightgbm_categoricals(
            X, self.categorical_features, fit=True, categories=None
        )
        # Snapshot the categories for the predict path.
        for col in self.categorical_features:
            if col in X.columns:
                self._categories[col] = sorted(X[col].dropna().astype(str).unique().tolist())
        self._booster = LGBMClassifier(**self._resolved_params)
        self._booster.fit(X_fit, y, categorical_feature=list(self.categorical_features))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return a (n, 3) array of probabilities in `CLASSES` order.

        LightGBM returns columns in the order of the integer-encoded
        labels; we fit on `0, 1, 2 = (L, C, R)`, so the columns are
        already in `CLASSES` order. No reordering is needed.
        """
        if self._booster is None:
            raise RuntimeError("LightGBMClassifierWrapper is not fitted")
        X_pred = _coerce_lightgbm_categoricals(
            X,
            self.categorical_features,
            fit=False,
            categories=self._categories,
        )
        return np.asarray(self._booster.predict_proba(X_pred))


def _coerce_lightgbm_categoricals(
    X: pd.DataFrame,
    categorical_features: Sequence[str],
    *,
    fit: bool,
    categories: dict[str, list[Any]] | None,
) -> pd.DataFrame:
    """Convert the categorical columns to `pd.Categorical` with the right dtype.

    At fit time (`fit=True`), the categories are taken from the input
    `X` (so any string value the booster sees becomes a known
    category). At predict time, the categories are taken from the
    `categories` dict captured at fit time, so unseen values become
    `NaN` (LightGBM treats `NaN` as missing in categorical features).

    The function does not mutate the input `X`; it returns a new
    DataFrame. Non-categorical columns are passed through unchanged.
    """
    X = X.copy()
    for col in categorical_features:
        if col not in X.columns:
            continue
        if fit:
            cats = sorted(X[col].dropna().astype(str).unique().tolist())
        else:
            cats = (categories or {}).get(col, [])
        # `pd.Categorical` with explicit `categories=` treats values
        # outside the category set as NaN. We pre-mask the values so
        # the construction never sees a non-null entry outside the
        # categories — that's the future-pandas contract (panda 4.x
        # will raise on the deprecated path).
        values = X[col].astype(object).where(X[col].notna(), None)
        in_cats = values.isin(cats) | values.isna()
        values = values.where(in_cats, None)
        X[col] = pd.Categorical(values, categories=cats)
    return X


def compute_class_weights(y: np.ndarray) -> dict[int, float]:
    """Compute inverse-frequency class weights for an int-encoded label vector.

    The formula is `n_samples / (n_classes * n_samples_per_class)`,
    which is the "balanced" weight from sklearn. Returns a dict
    mapping class index to weight.

    For our 3-class problem (L=0, C=1, R=2), with the live training
    table's 88 L + 33 C + 58 R rows, the weights are roughly
    {0: 0.68, 1: 1.81, 2: 1.03} — C is upweighted, L is downweighted.
    """
    if y.size == 0:
        return {}
    n_samples = y.size
    classes, counts = np.unique(y, return_counts=True)
    n_classes = len(classes)
    weights: dict[int, float] = {}
    for cls, count in zip(classes, counts, strict=True):
        weights[int(cls)] = n_samples / (n_classes * count)
    return weights


def make_lightgbm(
    params: dict[str, Any] | None = None,
    categorical_features: Sequence[str] = CATEGORICAL_FEATURES,
    random_state: int = RANDOM_SEED,
) -> LightGBMClassifierWrapper:
    """Build an unfitted `LightGBMClassifierWrapper`.

    The default `params` are `LIGHTGBM_DEFAULTS` merged with
    `class_weight=None` (the caller is expected to pass
    `class_weight=<computed weights>` after computing them on the
    training fold's label distribution; this keeps the
    inverse-frequency recipe close to the fit site).
    """
    merged: dict[str, Any] = dict(LIGHTGBM_DEFAULTS)
    if params:
        merged.update(params)
    # class_weight is a LightGBM kwarg (different from sklearn's
    # `class_weight=`). LightGBM accepts a dict of class index -> weight.
    return LightGBMClassifierWrapper(
        params=merged,
        categorical_features=list(categorical_features),
        random_state=random_state,
    )


def fit_lightgbm(
    matrix: FeatureMatrix,
    params: dict[str, Any] | None = None,
    categorical_features: Sequence[str] = CATEGORICAL_FEATURES,
    random_state: int = RANDOM_SEED,
) -> LightGBMClassifierWrapper:
    """Fit a `LightGBMClassifierWrapper` on `matrix`.

    Computes class weights from the training fold's label
    distribution (inverse frequency) and passes them via
    `params["class_weight"]`. The caller can override the weights by
    passing a `params` dict with a `class_weight` key.
    """
    class_weights = compute_class_weights(matrix.y)
    merged: dict[str, Any] = dict(params or {})
    # Only set the class_weight from the data if the caller didn't
    # override it (e.g. for ablation tests that want uniform weights).
    merged.setdefault("class_weight", class_weights)
    model = make_lightgbm(
        params=merged,
        categorical_features=categorical_features,
        random_state=random_state,
    )
    model.fit(matrix.X, matrix.y)
    return model
