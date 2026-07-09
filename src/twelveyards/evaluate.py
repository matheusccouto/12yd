"""Evaluation metrics for the penalty shootout classifier.

PRD: For each historical shootout kick, the keeper's optimal action is
`argmin(p_L, p_C, p_R)` — dive the lowest-probability side, or stay
center if C is the minimum. A "save" is:
- an off-target kick (always a save, the keeper doesn't need to be
  there), or
- an on-target kick where the recommended dive matches the kicker's
  actual side.

The counterfactual save rate is `saves / kicks`. The metric answers:
"if the keeper had dived where the model told them, what fraction of
shootout kicks would the keeper have stopped?"

Three baselines are computed on the same set:
- **Random**: 33.3% (the keeper dives a random side and, on-target
  kicks, matches the kicker's side 1/3 of the time; off-target kicks
  are always a save). The expected rate is `2/3 * 1/3 + 1/3 = 5/9`
  (~55.6%) for the dataset with the actual off-target mix.
- **Kicker-most-frequent**: for each kick, the keeper dives the
  kicker's most frequent historical side (over the lookback window).
  This is the per-kicker "they always go L" strategy.
- **Actual keeper dive**: the keeper's actual dive, when recoverable
  from the data. FotMob's data path does not record the keeper's dive
  direction, so this baseline is reported as `null` for v1; the PRD
  flags it as "when recoverable".

The metrics report is a flat dict that serialises to JSON for
`metrics.json`. The field set is stable across model slices (#23
baseline and #24 LightGBM) so the same downstream tooling can diff
the two reports.

v3 additions:

- **Calibration block** (Issue #43): Brier and ECE for the model, the
  logreg baseline, and the closed-form uniform random baseline. The
  block is `None` only for an empty holdout (where the metrics are
  undefined). `from_dict` accepts pre-#43 metrics without the block
  and sets it to `None`.

- **Cross-validation block** (Issue #45): a leave-one-group-out
  cross-validation report with per-fold metrics and the aggregate
  summary. The `group_by` field is the row attribute used for
  grouping (default `tournament_name`). Folds below `min_fold_size`
  are skipped (recorded in the `skipped` dict). The aggregate
  save rate is the weighted mean across folds, weighted by
  per-fold `n_holdout`. The aggregate SE is the binomial SE on the
  weighted total. `from_dict` accepts pre-#45 metrics without the
  block and sets it to `None`.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .features import CLASSES  # re-exported from .model for backwards-compat
from .model import FeatureMatrix, TrainingRow, build_feature_matrix

# Class indices (mirrors CLASSES in model.py).
L: int = 0
C: int = 1
R: int = 2


# ---------------------------------------------------------------------------
# Counterfactual save rate
# ---------------------------------------------------------------------------


def recommended_dive(probs: np.ndarray) -> np.ndarray:
    """Return the recommended dive per row as an int class index.

    The keeper dives the lowest-probability side. Ties go to the side
    with the lowest index (L < C < R), so the behaviour is
    deterministic across runs.

    Parameters
    ----------
    probs
        A (n, 3) array of probabilities in `CLASSES` order (L, C, R).
    """
    return np.argmin(probs, axis=1)


def counterfactual_save_rate(
    probs: np.ndarray,
    labels: np.ndarray,
    on_target: np.ndarray,
) -> tuple[float, int]:
    """Compute the counterfactual save rate.

    A row is a save if either:
    - the kick is off-target (the keeper's dive is irrelevant), or
    - the kick is on-target AND the recommended dive matches the
      kicker's actual side.

    Parameters
    ----------
    probs
        (n, 3) array of predicted probabilities in `CLASSES` order.
    labels
        (n,) array of int class indices (the kicker's actual side).
    on_target
        (n,) array of bool (whether the kick was on-target).

    Returns
    -------
    (save_rate, n_kicks)
        The save rate as a float in [0, 1] and the number of kicks
        considered (== len(probs)).
    """
    if probs.shape[0] == 0:
        return (0.0, 0)
    dive = recommended_dive(probs)
    matched = dive == labels
    save = (~on_target) | (on_target & matched)
    return (float(save.mean()), int(save.size))


def random_save_rate(labels: np.ndarray, on_target: np.ndarray) -> tuple[float, int]:
    """The expected save rate for a keeper who dives a uniform random side.

    The keeper picks L, C, or R with equal probability. The save
    probability is computed in closed form:
    `P(save) = P(off-target) + P(on-target) * P(dive matches)`
    `P(dive matches) = sum_c P(side=c) * 1/3`
    = `mean(P(L), P(C), P(R))` over the observed labels.

    Returns `(save_rate, n_kicks)`. The expected rate is at most 5/9
    (~55.6%) when every kick is on-target and the labels are
    uniformly distributed.
    """
    n = int(labels.size)
    if n == 0:
        return (0.0, 0)
    class_freq = np.bincount(labels, minlength=3) / n
    p_match = float(class_freq.mean())  # 1/3 weighted by class freq
    p_off = float((~on_target).mean())
    p_on = float(on_target.mean())
    return (p_off + p_on * p_match, n)


def last_side_save_rate(
    rows: Sequence[TrainingRow],
) -> tuple[float | None, int]:
    """The save rate for a keeper who dives the kicker's `last_side` feature.

    For each row, the recommended dive is the A2 `last_side` field
    (the side of the kicker's most recent penalty in the lookback).
    If the kicker has no history, the dive falls back to L (the prior's
    tiebreaker). This matches the model's "use the prior when the data
    is missing" behaviour.

    Rows whose kicker has no historical kicks cannot improve on the
    random baseline in aggregate; the function reports `None` if the
    rate is undefined (zero rows).

    Returns `(save_rate, n_kicks)`. The save rate is the fraction of
    kicks where the keeper's recommended dive (a) matches the actual
    side on-target, or (b) the kick is off-target.
    """
    n = len(rows)
    if n == 0:
        return (None, 0)
    saves = 0
    for row in rows:
        last = row.last_side
        if last == "L":
            dive = L
        elif last == "R":
            dive = R
        elif last == "C":
            dive = C
        else:
            dive = L  # prior fallback: any tie in the prior goes to L
        matched = dive == CLASSES.index(row.label)
        if not row.is_on_target or matched:
            saves += 1
    return (saves / n, n)


def actual_keeper_save_rate(
    rows: Sequence[TrainingRow],
) -> tuple[float | None, int]:
    """The save rate for the actual keeper's dive.

    FotMob's data path does not record the keeper's dive direction for
    shootout kicks (the `penaltyShootoutEvents` block has the kicker,
    the shotmap, and the running score, but not the keeper's lateral
    movement). The baseline is therefore undefined for v1; we report
    `None` and the rate is omitted from the metrics JSON.
    """
    return (None, len(rows))


# ---------------------------------------------------------------------------
# Per-row scoring
# ---------------------------------------------------------------------------


def log_loss(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-15) -> float:
    """Multinomial cross-entropy loss.

    `probs[i, y_i]` is the predicted probability of the true class
    for row i. We clip to `[eps, 1 - eps]` to avoid `log(0) = -inf`
    when the model is over-confident.
    """
    if probs.shape[0] == 0:
        return 0.0
    p_true = probs[np.arange(probs.shape[0]), labels]
    p_true = np.clip(p_true, eps, 1.0)
    return float(-np.mean(np.log(p_true)))


def accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    """Top-1 accuracy: fraction of rows where the argmax matches the label."""
    if probs.shape[0] == 0:
        return 0.0
    return float(np.mean(np.argmax(probs, axis=1) == labels))


def brier_multiclass(probs: np.ndarray, labels: np.ndarray) -> float:
    """Multiclass Brier score: mean squared error against one-hot labels.

    For each row, computes `sum_c (p_c - y_c)^2` where `y` is one-hot
    in `CLASSES` order. The result is the mean over rows. For 3
    classes the range is `[0, 2]`: 0 = perfect calibration, 2 = worst
    possible (the model places zero mass on the true class and unit
    mass on a wrong one). The uniform predictor `(1/3, 1/3, 1/3)`
    scores `2/3` on every row, independent of the label distribution.
    """
    if probs.shape[0] == 0:
        return 0.0
    n = probs.shape[0]
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n), labels] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error.

    Bins rows by their max predicted probability (equal-width bins on
    `[0, 1]`) and computes the weighted mean of
    `|accuracy(bin) - confidence(bin)|` weighted by bin size. ECE
    ranges in `[0, 1]`; 0 means perfectly calibrated. Rows whose max
    probability is exactly 0 fall in the first bin.

    The function ignores empty bins (a bin with zero rows contributes
    nothing, so the metric is well-defined on small holdouts where
    some bins are unreached).
    """
    if probs.shape[0] == 0:
        return 0.0
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    # `np.digitize` returns 1..n_bins+1 for points strictly inside
    # the edges; we want 0..n_bins-1, so subtract 1 and clip.
    bin_indices = np.clip(np.digitize(confidences, bin_edges) - 1, 0, n_bins - 1)
    ece_val = 0.0
    for b in range(n_bins):
        mask = bin_indices == b
        if not mask.any():
            continue
        bin_conf = float(confidences[mask].mean())
        bin_acc = float(accuracies[mask].mean())
        ece_val += (mask.sum() / len(confidences)) * abs(bin_acc - bin_conf)
    return float(ece_val)


# ---------------------------------------------------------------------------
# Metrics report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaselineMetrics:
    """The metrics for a single baseline (model or fixed strategy).

    `save_rate` may be `None` when the baseline is undefined
    (e.g. the actual-keeper baseline, when the data path doesn't
    carry the dive direction). The other fields are always defined.
    """

    name: str
    log_loss: float | None
    accuracy: float | None
    save_rate: float | None
    n_kicks: int


@dataclass(frozen=True)
class CalibrationMetrics:
    """Brier score and expected calibration error for one predictor.

    `brier` is the multiclass Brier score in `[0, 2]` (0 = perfect).
    `ece` is the expected calibration error in `[0, 1]` (0 = perfect),
    computed on `n_bins` equal-width confidence bins. `n_bins` is
    recorded on the dataclass so a future re-evaluation can match the
    binning choice without re-deriving it.
    """

    brier: float
    ece: float
    n_bins: int


@dataclass(frozen=True)
class CalibrationReport:
    """The calibration block of a metrics report (Issue #43).

    `model` is the deployed classifier (the LightGBM in slice #8).
    `baseline` is the optional logreg comparison classifier (None
    when the metrics report has no baseline). `random` is the
    closed-form uniform baseline `(1/3, 1/3, 1/3)`.
    """

    model: CalibrationMetrics
    baseline: CalibrationMetrics | None
    random: CalibrationMetrics


@dataclass(frozen=True)
class MetricsReport:
    """A full evaluation report on one holdout fold.

    `model` is the classifier's metrics (the LightGBM in slice #8;
    the logreg baseline in slice #7). `baseline` is an optional
    comparison classifier — the logreg in slice #8 (so the LightGBM
    is compared apples-to-apples against the previous slice on the
    same holdout fold). The three `*_baseline` fields are the
    fixed-strategy baselines for context. `n_train` and `n_holdout`
    are the row counts in each fold. `holdout_cutoff_date` is the ISO
    8601 cutoff the split used. `calibration` is the Brier / ECE
    block (Issue #43) and is `None` only for an empty holdout
    (where the metrics are undefined). `cv` is the leave-one-group-out
    cross-validation report (Issue #45) and is `None` for any metrics
    report that pre-dates the LOTO CV slice (e.g. a re-run of the
    train script before the CV slice ran).
    """

    model: BaselineMetrics
    random_baseline: BaselineMetrics
    kicker_most_frequent_baseline: BaselineMetrics
    actual_keeper_baseline: BaselineMetrics
    n_train: int
    n_holdout: int
    holdout_cutoff_date: str
    baseline: BaselineMetrics | None = None
    calibration: CalibrationReport | None = None
    cv: CVReport | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dict.

        `None` save rates and log losses are preserved as `null` so the
        report honestly reflects the absence of a metric (rather than
        silently substituting 0.0). The optional `baseline` field
        (the logreg comparison classifier) is serialised when set.
        The optional `calibration` block (Issue #43) is serialised
        when set, with `baseline` nested under it as `None` when the
        metrics report has no baseline classifier. The optional `cv`
        block (Issue #45) is serialised when set, with per-fold
        metrics and the aggregate summary.
        """
        payload: dict[str, Any] = {
            "model": asdict(self.model),
            "random_baseline": asdict(self.random_baseline),
            "kicker_most_frequent_baseline": asdict(self.kicker_most_frequent_baseline),
            "actual_keeper_baseline": asdict(self.actual_keeper_baseline),
            "n_train": self.n_train,
            "n_holdout": self.n_holdout,
            "holdout_cutoff_date": self.holdout_cutoff_date,
        }
        if self.baseline is not None:
            payload["baseline"] = asdict(self.baseline)
        if self.calibration is not None:
            payload["calibration"] = {
                "model": asdict(self.calibration.model),
                "baseline": (
                    asdict(self.calibration.baseline)
                    if self.calibration.baseline is not None
                    else None
                ),
                "random": asdict(self.calibration.random),
            }
        if self.cv is not None:
            payload["cv"] = _cv_to_dict(self.cv)
        payload.update(self.extras)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MetricsReport:
        """Reconstruct a `MetricsReport` from a dict produced by `to_dict`.

        The round-trip preserves every field, including the optional
        `baseline` classifier section, the `calibration` block
        (Issue #43), the `cv` block (Issue #45), and the `extras`
        dict (which carries the model_kind, classes, feature_columns,
        and params). Any keys not in the known schema are kept in
        `extras` so a future report with new metadata is read back
        intact. The `calibration` block is optional: a metrics report
        that pre-dates Issue #43 roundtrips with `calibration=None`.
        The `cv` block is optional: a metrics report that pre-dates
        Issue #45 roundtrips with `cv=None`.
        """
        known = {
            "model",
            "random_baseline",
            "kicker_most_frequent_baseline",
            "actual_keeper_baseline",
            "baseline",
            "calibration",
            "cv",
            "n_train",
            "n_holdout",
            "holdout_cutoff_date",
        }
        extras = {k: v for k, v in payload.items() if k not in known}
        calibration_payload = payload.get("calibration")
        calibration = _calibration_from_dict(calibration_payload) if calibration_payload else None
        cv_payload = payload.get("cv")
        cv = _cv_from_dict(cv_payload) if cv_payload else None
        return cls(
            model=BaselineMetrics(**payload["model"]),
            random_baseline=BaselineMetrics(**payload["random_baseline"]),
            kicker_most_frequent_baseline=BaselineMetrics(
                **payload["kicker_most_frequent_baseline"]
            ),
            actual_keeper_baseline=BaselineMetrics(**payload["actual_keeper_baseline"]),
            baseline=(BaselineMetrics(**payload["baseline"]) if payload.get("baseline") else None),
            calibration=calibration,
            cv=cv,
            n_train=int(payload["n_train"]),
            n_holdout=int(payload["n_holdout"]),
            holdout_cutoff_date=str(payload.get("holdout_cutoff_date", "")),
            extras=extras,
        )


def _calibration_from_dict(payload: dict[str, Any]) -> CalibrationReport:
    """Build a `CalibrationReport` from its JSON-serialised form.

    `baseline` may be `null` in the payload (when the parent
    `MetricsReport` has no baseline classifier); the field is
    `None` on the dataclass in that case.
    """
    model = CalibrationMetrics(**payload["model"])
    baseline_payload = payload.get("baseline")
    baseline = CalibrationMetrics(**baseline_payload) if baseline_payload else None
    random = CalibrationMetrics(**payload["random"])
    return CalibrationReport(model=model, baseline=baseline, random=random)


def _cv_to_dict(report: CVReport) -> dict[str, Any]:
    """Serialise a `CVReport` to a JSON-friendly dict.

    The block has four top-level keys: `folds` (per-fold metrics),
    `aggregate` (the n_holdout-weighted summary), `group_by` (the
    row attribute used for grouping), and `skipped` (groups below
    `min_fold_size` with their row counts).
    """
    return {
        "folds": [asdict(f) for f in report.folds],
        "aggregate": {
            "save_rate": report.aggregate_save_rate,
            "log_loss": report.aggregate_log_loss,
            "accuracy": report.aggregate_accuracy,
            "n_total": report.n_total,
            "se_save_rate": report.se_save_rate,
        },
        "group_by": report.group_by,
        "skipped": dict(report.skipped),
    }


def _cv_from_dict(payload: dict[str, Any]) -> CVReport:
    """Build a `CVReport` from its JSON-serialised form.

    Folds are reconstructed as `CVFold` dataclasses. The aggregate
    is read from the `aggregate` sub-dict. `skipped` defaults to an
    empty dict when absent (older LOTO reports do not have it).
    """
    folds_payload = payload.get("folds", [])
    folds = tuple(CVFold(**f) for f in folds_payload)
    agg = payload.get("aggregate", {})
    return CVReport(
        folds=folds,
        aggregate_save_rate=float(agg.get("save_rate", 0.0)),
        aggregate_log_loss=float(agg.get("log_loss", 0.0)),
        aggregate_accuracy=float(agg.get("accuracy", 0.0)),
        n_total=int(agg.get("n_total", 0)),
        se_save_rate=float(agg.get("se_save_rate", 0.0)),
        group_by=str(payload.get("group_by", "tournament_name")),
        skipped=dict(payload.get("skipped", {})),
    )


def evaluate_predictions(
    probs: np.ndarray,
    holdout_rows: Sequence[TrainingRow],
    baseline_probs: np.ndarray | None = None,
) -> MetricsReport:
    """Compute the full metrics report for a model on a holdout fold.

    The `probs` array is the model's predicted probabilities for the
    `holdout_rows` in `CLASSES` order. The function derives the
    `labels` and `on_target` arrays from the rows.

    If `baseline_probs` is provided, the report includes a `baseline`
    section (the logreg comparison classifier) so the model's metric
    can be diffed apples-to-apples against the previous slice on the
    same holdout fold. The baseline's log loss / accuracy / save
    rate are computed the same way as the model's.

    The `calibration` block (Issue #43) is computed for all three
    probabilistic predictors — the model, the optional baseline, and
    the closed-form uniform random baseline `(1/3, 1/3, 1/3)`. Brier
    and ECE are deterministic on the same `probs` and `labels`; the
    10-bin ECE matches the binning used in `docs/model-review.md`
    (Topic 3).
    """
    n = len(holdout_rows)
    if n == 0:
        empty = BaselineMetrics(
            name="empty", log_loss=None, accuracy=None, save_rate=None, n_kicks=0
        )
        return MetricsReport(
            model=empty,
            random_baseline=empty,
            kicker_most_frequent_baseline=empty,
            actual_keeper_baseline=empty,
            n_train=0,
            n_holdout=0,
            holdout_cutoff_date="",
        )
    labels = np.array([CLASSES.index(r.label) for r in holdout_rows], dtype=np.int64)
    on_target = np.array([r.is_on_target for r in holdout_rows], dtype=bool)
    save_rate, n_kicks = counterfactual_save_rate(probs, labels, on_target)
    model_metrics = BaselineMetrics(
        name="model",
        log_loss=log_loss(probs, labels),
        accuracy=accuracy(probs, labels),
        save_rate=save_rate,
        n_kicks=n_kicks,
    )
    rand_save, rand_n = random_save_rate(labels, on_target)
    random_metrics = BaselineMetrics(
        name="random",
        log_loss=math.log(3),  # uniform over 3 classes: -log(1/3)
        accuracy=1.0 / 3.0,
        save_rate=rand_save,
        n_kicks=rand_n,
    )
    last_save, last_n = last_side_save_rate(holdout_rows)
    last_metrics = BaselineMetrics(
        name="last_side",
        log_loss=None,  # not a probabilistic baseline
        accuracy=None,
        save_rate=last_save,
        n_kicks=last_n,
    )
    ak_save, ak_n = actual_keeper_save_rate(holdout_rows)
    ak_metrics = BaselineMetrics(
        name="actual_keeper",
        log_loss=None,
        accuracy=None,
        save_rate=ak_save,
        n_kicks=ak_n,
    )
    baseline_metrics: BaselineMetrics | None = None
    if baseline_probs is not None:
        b_save, b_n = counterfactual_save_rate(baseline_probs, labels, on_target)
        baseline_metrics = BaselineMetrics(
            name="baseline",
            log_loss=log_loss(baseline_probs, labels),
            accuracy=accuracy(baseline_probs, labels),
            save_rate=b_save,
            n_kicks=b_n,
        )

    # Issue #43: Brier + ECE for all three probabilistic predictors.
    # The closed-form uniform random baseline is constructed inline
    # (one row per holdout kick, all three classes at 1/3) so its
    # calibration is computed on the same `labels` as the others.
    random_probs = np.full((n, 3), 1.0 / 3.0)
    calibration = CalibrationReport(
        model=CalibrationMetrics(
            brier=brier_multiclass(probs, labels),
            ece=ece(probs, labels, n_bins=10),
            n_bins=10,
        ),
        baseline=(
            CalibrationMetrics(
                brier=brier_multiclass(baseline_probs, labels),
                ece=ece(baseline_probs, labels, n_bins=10),
                n_bins=10,
            )
            if baseline_probs is not None
            else None
        ),
        random=CalibrationMetrics(
            brier=brier_multiclass(random_probs, labels),
            ece=ece(random_probs, labels, n_bins=10),
            n_bins=10,
        ),
    )

    return MetricsReport(
        model=model_metrics,
        baseline=baseline_metrics,
        random_baseline=random_metrics,
        kicker_most_frequent_baseline=last_metrics,
        actual_keeper_baseline=ak_metrics,
        n_train=0,  # filled in by the caller
        n_holdout=n,
        holdout_cutoff_date="",  # filled in by the caller
        calibration=calibration,
    )


def write_metrics_json(path: Path, report: MetricsReport) -> None:
    """Write the metrics report to a JSON file at `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Cross-validation (Issue #45)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CVFold:
    """The metrics for one leave-one-group-out fold.

    `name` is the group label (e.g. "World Cup Final Stage" for the
    `tournament_name` group). `n_train` and `n_holdout` are the
    per-fold row counts. `save_rate`, `log_loss`, and `accuracy` are
    the model's metrics on the holdout fold. `random_save_rate` is
    the closed-form uniform random baseline on the same holdout, for
    context.
    """

    name: str
    n_train: int
    n_holdout: int
    save_rate: float
    log_loss: float
    accuracy: float
    random_save_rate: float


@dataclass(frozen=True)
class CVReport:
    """The aggregate LOTO CV report (Issue #45).

    `folds` is the per-fold list, in the order produced by the
    `cross_validate` helper (largest fold first, ties broken by name).
    `aggregate_save_rate`, `aggregate_log_loss`, and
    `aggregate_accuracy` are the n_holdout-weighted means across
    folds. `n_total` is the sum of per-fold `n_holdout` (the aggregate
    holdout size). `se_save_rate` is the binomial standard error
    `sqrt(p * (1 - p) / n_total)` on the aggregate save rate, where
    `p` is `aggregate_save_rate`. `group_by` is the row attribute
    used for grouping (e.g. `"tournament_name"`). `skipped` is a
    `{group_name: n_rows}` map of groups that were skipped because
    they had fewer than `min_fold_size` rows; the report is still
    valid when this is non-empty (an empty group has no signal to
    contribute and no fold to score).
    """

    folds: tuple[CVFold, ...]
    aggregate_save_rate: float
    aggregate_log_loss: float
    aggregate_accuracy: float
    n_total: int
    se_save_rate: float
    group_by: str
    skipped: dict[str, int] = field(default_factory=dict)


def _score_fold(
    probs: np.ndarray, holdout_rows: Sequence[TrainingRow]
) -> tuple[float, float, float, float]:
    """Score one fold: (save_rate, log_loss, accuracy, random_save_rate).

    `probs` is the model's (n, 3) prediction array in `CLASSES` order.
    `holdout_rows` is the list of `TrainingRow` for the fold. The
    random save rate is the closed-form uniform baseline on the same
    labels (the function delegates to `random_save_rate`).
    """
    labels = np.array([CLASSES.index(r.label) for r in holdout_rows], dtype=np.int64)
    on_target = np.array([r.is_on_target for r in holdout_rows], dtype=bool)
    save_rate, _ = counterfactual_save_rate(probs, labels, on_target)
    rand_save, _ = random_save_rate(labels, on_target)
    return (
        float(save_rate),
        float(log_loss(probs, labels)),
        float(accuracy(probs, labels)),
        float(rand_save),
    )


def cross_validate(
    model_factory: Callable[[FeatureMatrix], Any],
    rows: Sequence[TrainingRow],
    group_by: str = "tournament_name",
    min_fold_size: int = 1,
) -> CVReport:
    """Leave-one-group-out cross-validation (Issue #45).

    Groups `rows` by `getattr(row, group_by)`. For each non-empty
    group G, fits a fresh model on the rows NOT in G via
    `model_factory(train_matrix)`, then evaluates on the rows in G.
    Returns a `CVReport` with the per-fold metrics and the aggregate
    summary.

    Parameters
    ----------
    model_factory
        A callable that takes a `FeatureMatrix` (the training fold)
        and returns a fitted model with a `predict_proba(X) -> (n, 3)`
        method. The wrapper does not know about a specific model
        type; `scripts/evaluate_cv.py` passes
        `lambda matrix: fit_lightgbm(matrix)` and
        `lambda matrix: fit_logistic_regression(matrix)` for the
        LightGBM and logreg respectively.
    rows
        The list of `TrainingRow` to CV over. The caller is expected
        to pass the same rows the model was originally trained on
        (i.e. the full training table, not a single train/holdout
        split).
    group_by
        The row attribute to group on. Default `"tournament_name"`
        gives a leave-one-tournament-out CV. Any attribute on
        `TrainingRow` works.
    min_fold_size
        Folds with fewer than this many rows are skipped. The
        skipped groups are recorded in the `CVReport.skipped` dict
        so a future maintainer can see why a fold was dropped (e.g.
        Asian Cup has 0 rows in the current data). Default `1`
        (only skip truly empty folds).

    Notes
    -----
    The aggregate save rate is the `n_holdout`-weighted mean across
    folds. The aggregate SE is the binomial SE on the weighted
    total: `sqrt(p * (1 - p) / n_total)` where `p` is the aggregate
    save rate. This is a standard simplification; the exact LOTO SE
    depends on within-fold correlation and is rarely reported in
    practice.

    The function is deterministic for the same `model_factory` +
    `rows` + `group_by` + `min_fold_size`. The fit step is
    deterministic for both sklearn and LightGBM at fixed random
    seed (`RANDOM_SEED` in `model.py`).
    """
    if min_fold_size < 1:
        raise ValueError(f"min_fold_size must be >= 1, got {min_fold_size}")
    if not rows:
        return CVReport(
            folds=(),
            aggregate_save_rate=0.0,
            aggregate_log_loss=0.0,
            aggregate_accuracy=0.0,
            n_total=0,
            se_save_rate=0.0,
            group_by=group_by,
            skipped={},
        )

    # Group rows by the `group_by` attribute. `defaultdict` keeps the
    # insertion order (Python 3.7+ guarantee) so the fold order is
    # deterministic for the same input.
    groups: dict[str, list[TrainingRow]] = {}
    for row in rows:
        key = str(getattr(row, group_by))
        groups.setdefault(key, []).append(row)

    folds: list[CVFold] = []
    skipped: dict[str, int] = {}
    for name, fold_rows in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        if len(fold_rows) < min_fold_size:
            skipped[name] = len(fold_rows)
            continue
        train_rows = [r for r in rows if getattr(r, group_by) != getattr(fold_rows[0], group_by)]
        train_matrix = build_feature_matrix(train_rows)
        holdout_matrix = build_feature_matrix(fold_rows)
        model = model_factory(train_matrix)
        probs = np.asarray(model.predict_proba(holdout_matrix.X))
        save_rate, ll, acc, rand_save = _score_fold(probs, fold_rows)
        folds.append(
            CVFold(
                name=name,
                n_train=len(train_rows),
                n_holdout=len(fold_rows),
                save_rate=save_rate,
                log_loss=ll,
                accuracy=acc,
                random_save_rate=rand_save,
            )
        )

    if not folds:
        return CVReport(
            folds=(),
            aggregate_save_rate=0.0,
            aggregate_log_loss=0.0,
            aggregate_accuracy=0.0,
            n_total=0,
            se_save_rate=0.0,
            group_by=group_by,
            skipped=skipped,
        )

    n_total = sum(f.n_holdout for f in folds)
    if n_total == 0:
        return CVReport(
            folds=tuple(folds),
            aggregate_save_rate=0.0,
            aggregate_log_loss=0.0,
            aggregate_accuracy=0.0,
            n_total=0,
            se_save_rate=0.0,
            group_by=group_by,
            skipped=skipped,
        )
    agg_save = sum(f.save_rate * f.n_holdout for f in folds) / n_total
    agg_ll = sum(f.log_loss * f.n_holdout for f in folds) / n_total
    agg_acc = sum(f.accuracy * f.n_holdout for f in folds) / n_total
    se = math.sqrt(agg_save * (1.0 - agg_save) / n_total) if 0.0 < agg_save < 1.0 else 0.0
    return CVReport(
        folds=tuple(folds),
        aggregate_save_rate=agg_save,
        aggregate_log_loss=agg_ll,
        aggregate_accuracy=agg_acc,
        n_total=n_total,
        se_save_rate=se,
        group_by=group_by,
        skipped=skipped,
    )
