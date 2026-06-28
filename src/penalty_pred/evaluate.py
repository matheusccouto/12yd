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
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .model import CLASSES, TrainingRow

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
        last = row.features.get("last_side", "")
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
class MetricsReport:
    """A full evaluation report on one holdout fold.

    `model` is the classifier's metrics (the LightGBM in slice #8;
    the logreg baseline in slice #7). `baseline` is an optional
    comparison classifier — the logreg in slice #8 (so the LightGBM
    is compared apples-to-apples against the previous slice on the
    same holdout fold). The three `*_baseline` fields are the
    fixed-strategy baselines for context. `n_train` and `n_holdout`
    are the row counts in each fold. `holdout_cutoff_date` is the ISO
    8601 cutoff the split used.
    """

    model: BaselineMetrics
    random_baseline: BaselineMetrics
    kicker_most_frequent_baseline: BaselineMetrics
    actual_keeper_baseline: BaselineMetrics
    n_train: int
    n_holdout: int
    holdout_cutoff_date: str
    baseline: BaselineMetrics | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dict.

        `None` save rates and log losses are preserved as `null` so the
        report honestly reflects the absence of a metric (rather than
        silently substituting 0.0). The optional `baseline` field
        (the logreg comparison classifier) is serialised when set.
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
        payload.update(self.extras)
        return payload


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
    return MetricsReport(
        model=model_metrics,
        baseline=baseline_metrics,
        random_baseline=random_metrics,
        kicker_most_frequent_baseline=last_metrics,
        actual_keeper_baseline=ak_metrics,
        n_train=0,  # filled in by the caller
        n_holdout=n,
        holdout_cutoff_date="",  # filled in by the caller
    )


def write_metrics_json(path: Path, report: MetricsReport) -> None:
    """Write the metrics report to a JSON file at `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
