"""Tests for the evaluation module (slice #7, Issue #23).

Three layers:

1. **Counterfactual save rate** — `recommended_dive`,
   `counterfactual_save_rate`. Verifies the off-target-always-saves
   rule and the on-target-matches-the-dive rule.

2. **Baselines** — `random_save_rate`, `last_side_save_rate`,
   `actual_keeper_save_rate`. Verifies the closed-form random rate,
   the per-row kicker mode, and the N/A keeper baseline.

3. **Per-row metrics** — `log_loss`, `accuracy`.

4. **Report** — `evaluate_predictions`, `write_metrics_json`. Verifies
   the report shape and the JSON roundtrip.

5. **Live smoke test** — `output/metrics.json` (skipped if absent):
   the model section has a `save_rate` and `log_loss`, the random
   baseline's `log_loss` is `ln(3)`, the keeper baseline's `save_rate`
   is `None`.

6. **Cross-validation** (Issue #45) — `cross_validate`, `CVReport`,
   `CVFold`, plus the `cv` block on `MetricsReport` (roundtrip and
   backward compat).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from twelveyards.artifacts import Artifacts
from twelveyards.evaluate import (
    BaselineMetrics,
    CalibrationMetrics,
    CalibrationReport,
    CVFold,
    CVReport,
    MetricsReport,
    _cv_from_dict,
    _cv_to_dict,
    accuracy,
    actual_keeper_save_rate,
    brier_multiclass,
    counterfactual_save_rate,
    cross_validate,
    ece,
    evaluate_predictions,
    last_side_save_rate,
    log_loss,
    random_save_rate,
    recommended_dive,
    write_metrics_json,
)
from tests._factories import make_training_row

_make_row = make_training_row


# ---------------------------------------------------------------------------
# recommended_dive
# ---------------------------------------------------------------------------


def test_recommended_dive_picks_lowest_probability() -> None:
    """The keeper dives the side with the lowest predicted probability.
    For a row with probs [0.8, 0.1, 0.1], the dive is C (index 1)."""
    probs = np.array([[0.8, 0.1, 0.1]])
    assert recommended_dive(probs).tolist() == [1]


def test_recommended_dive_tie_breaks_to_lower_index() -> None:
    """On ties, the lower index wins (deterministic)."""
    probs = np.array([[0.1, 0.1, 0.1]])
    assert recommended_dive(probs).tolist() == [0]


# ---------------------------------------------------------------------------
# counterfactual_save_rate
# ---------------------------------------------------------------------------


def test_counterfactual_save_rate_off_target_always_saves() -> None:
    """Off-target kicks are always saves, regardless of the dive."""
    probs = np.array([[0.1, 0.1, 0.8]] * 3)  # dive = L (index 0)
    labels = np.array([2, 2, 2], dtype=np.int64)  # all R — dive mismatches
    on_target = np.array([False, False, False])
    sr, n = counterfactual_save_rate(probs, labels, on_target)
    assert sr == 1.0
    assert n == 3


def test_counterfactual_save_rate_on_target_requires_match() -> None:
    """On-target kicks are saves only when the dive matches the
    kicker's actual side."""
    probs = np.array([[0.1, 0.1, 0.8]] * 4)  # dive = L
    labels = np.array([0, 0, 2, 2], dtype=np.int64)
    on_target = np.array([True, True, True, True])
    sr, n = counterfactual_save_rate(probs, labels, on_target)
    # matches: [True, True, False, False] → 2/4
    assert sr == 0.5
    assert n == 4


def test_counterfactual_save_rate_mixed() -> None:
    """A mix of on-target and off-target kicks: the off-target ones
    save regardless; the on-target ones need a match."""
    probs = np.array([[0.1, 0.1, 0.8]] * 3)
    labels = np.array([0, 0, 0], dtype=np.int64)  # all on the highest side
    on_target = np.array([True, False, False])
    sr, _ = counterfactual_save_rate(probs, labels, on_target)
    # dive = L; matches L on row 0 → save; rows 1, 2 off-target → save
    # = 3/3 = 1.0
    assert sr == 1.0


def test_counterfactual_save_rate_empty() -> None:
    """An empty input returns (0.0, 0) without error."""
    sr, n = counterfactual_save_rate(
        np.empty((0, 3)), np.empty(0, dtype=np.int64), np.empty(0, dtype=bool)
    )
    assert sr == 0.0
    assert n == 0


# ---------------------------------------------------------------------------
# random_save_rate
# ---------------------------------------------------------------------------


def test_random_save_rate_uniform_labels_all_on_target() -> None:
    """Uniform labels (one of each class), all on-target: the random
    baseline's save rate is 1/3."""
    labels = np.array([0, 1, 2], dtype=np.int64)
    on_target = np.array([True, True, True])
    sr, n = random_save_rate(labels, on_target)
    assert math.isclose(sr, 1.0 / 3.0)
    assert n == 3


def test_random_save_rate_all_off_target() -> None:
    """All off-target → save rate is 1.0 regardless of labels."""
    labels = np.array([0] * 5, dtype=np.int64)
    on_target = np.array([False] * 5)
    sr, n = random_save_rate(labels, on_target)
    assert sr == 1.0
    assert n == 5


def test_random_save_rate_balanced_labels() -> None:
    """Balanced labels (33% L, 33% C, 33% R), all on-target → 1/3."""
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int64)
    on_target = np.ones(9, dtype=bool)
    sr, _ = random_save_rate(labels, on_target)
    assert math.isclose(sr, 1.0 / 3.0)


def test_random_save_rate_matches_p_eight() -> None:
    """A specific mix: 4 L, 4 C, 4 R, all on-target. P(dive matches)
    is the mean of the per-class fractions weighted by 1/3, which is
    always 1/3 when all three classes are observed (regardless of
    class balance). Save rate = 1/3."""
    labels = np.array([0] * 4 + [1] * 4 + [2] * 4, dtype=np.int64)
    on_target = np.ones(12, dtype=bool)
    sr, _ = random_save_rate(labels, on_target)
    assert math.isclose(sr, 1.0 / 3.0)


def test_random_save_rate_with_missing_class() -> None:
    """When one class is absent (e.g. no R kicks in the data), the
    per-class fraction for R is 0, so the mean drops. 12 L kicks, all
    on-target → mean = (1 + 0 + 0) / 3 = 1/3 (still 1/3 since the
    missing class contributes 0 to the mean)."""
    labels = np.array([0] * 12, dtype=np.int64)
    on_target = np.ones(12, dtype=bool)
    sr, _ = random_save_rate(labels, on_target)
    # mean of (1, 0, 0) = 1/3
    assert math.isclose(sr, 1.0 / 3.0)


# ---------------------------------------------------------------------------
# last_side_save_rate
# ---------------------------------------------------------------------------


def test_last_side_save_rate_dives_per_row_last_side() -> None:
    """For each row, the dive is the row's pre-kick `last_side` field.
    The prior fallback is L when last_side is "".
    """
    rows = [
        _make_row("L", last_side="L", kicker_id=1),
        _make_row("L", last_side="L", kicker_id=1),
        _make_row("R", last_side="R", kicker_id=2),
        _make_row("L", last_side="", kicker_id=3),
    ]
    sr, n = last_side_save_rate(rows)
    # Row 0: dive L, label L, match → save
    # Row 1: dive L, label L, match → save
    # Row 2: dive R, label R, match → save
    # Row 3: dive L (prior fallback), label L, match → save
    # 4/4 = 1.0
    assert sr == 1.0
    assert n == 4


def test_last_side_save_rate_empty() -> None:
    """Empty input returns (None, 0)."""
    sr, n = last_side_save_rate([])
    assert sr is None
    assert n == 0


# ---------------------------------------------------------------------------
# actual_keeper_save_rate
# ---------------------------------------------------------------------------


def test_actual_keeper_save_rate_is_none() -> None:
    """The FotMob data path doesn't carry the keeper's dive
    direction; the baseline is `None` for v1."""
    rows = [_make_row("L"), _make_row("R")]
    sr, n = actual_keeper_save_rate(rows)
    assert sr is None
    assert n == len(rows)


# ---------------------------------------------------------------------------
# log_loss / accuracy
# ---------------------------------------------------------------------------


def test_log_loss_perfect_prediction() -> None:
    """Predicting [1, 0, 0] when the label is L → log loss ≈ 0."""
    probs = np.array([[1.0, 0.0, 0.0]])
    labels = np.array([0])
    val = log_loss(probs, labels)
    # log(1 - 1e-15) ≈ -1e-15, so -log(...) ≈ 1e-15.
    assert abs(val) < 1e-10


def test_log_loss_half_prediction() -> None:
    """Predicting [0.5, 0.3, 0.2] when the label is L → log loss = ln 2."""
    probs = np.array([[0.5, 0.3, 0.2]])
    labels = np.array([0])
    assert math.isclose(log_loss(probs, labels), math.log(2.0))


def test_log_loss_uniform_prediction() -> None:
    """Predicting [1/3, 1/3, 1/3] for any label → log loss = ln 3."""
    probs = np.tile(np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]), (5, 1))
    labels = np.array([0, 1, 2, 0, 1], dtype=np.int64)
    assert math.isclose(log_loss(probs, labels), math.log(3.0))


def test_log_loss_clips_epsilon() -> None:
    """A prediction of 0.0 on the true class is clipped to 1e-15 to
    avoid -inf (the function returns a finite value)."""
    probs = np.array([[0.0, 1.0, 0.0]])
    labels = np.array([0])
    val = log_loss(probs, labels)
    assert math.isfinite(val)


def test_accuracy_top1() -> None:
    probs = np.array(
        [
            [0.7, 0.2, 0.1],  # argmax = 0, label = 0 → match
            [0.1, 0.2, 0.7],  # argmax = 2, label = 0 → miss
        ]
    )
    labels = np.array([0, 0])
    assert accuracy(probs, labels) == 0.5


def test_accuracy_empty() -> None:
    assert accuracy(np.empty((0, 3)), np.empty(0, dtype=np.int64)) == 0.0


# ---------------------------------------------------------------------------
# brier_multiclass
# ---------------------------------------------------------------------------


def test_brier_multiclass_perfect_prediction() -> None:
    """A one-hot prediction on the true class → Brier = 0."""
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    labels = np.array([0, 1, 2], dtype=np.int64)
    assert math.isclose(brier_multiclass(probs, labels), 0.0)


def test_brier_multiclass_worst_prediction() -> None:
    """A one-hot prediction on a wrong class → Brier = 2 (per row)."""
    probs = np.array([[1.0, 0.0, 0.0]])  # predicts L
    labels = np.array([2], dtype=np.int64)  # truth R
    # (1-0)^2 + (0-0)^2 + (0-1)^2 = 1 + 0 + 1 = 2
    assert math.isclose(brier_multiclass(probs, labels), 2.0)


def test_brier_multiclass_uniform_equals_two_thirds() -> None:
    """The uniform predictor (1/3, 1/3, 1/3) has Brier = 2/3
    regardless of the label distribution."""
    probs = np.tile(np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]), (5, 1))
    labels = np.array([0, 1, 2, 0, 1], dtype=np.int64)
    assert math.isclose(brier_multiclass(probs, labels), 2.0 / 3.0)


def test_brier_multiclass_half_prediction() -> None:
    """A specific row: P = (0.5, 0.3, 0.2), label = L (0).
    Brier for that row = 0.5^2 + 0.3^2 + 0.2^2 = 0.25 + 0.09 + 0.04 = 0.38."""
    probs = np.array([[0.5, 0.3, 0.2]])
    labels = np.array([0], dtype=np.int64)
    assert math.isclose(brier_multiclass(probs, labels), 0.38)


def test_brier_multiclass_empty() -> None:
    """Empty input returns 0.0 without error."""
    assert brier_multiclass(np.empty((0, 3)), np.empty(0, dtype=np.int64)) == 0.0


def test_brier_multiclass_independent_of_label_distribution() -> None:
    """The uniform predictor's Brier is 2/3 for any label distribution
    (including a single-class holdout), because the per-row squared
    error against one-hot is always 1/9 + 1/9 + 4/9 = 6/9 = 2/3."""
    probs = np.tile(np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]), (10, 1))
    for label in (0, 1, 2):
        labels = np.full(10, label, dtype=np.int64)
        assert math.isclose(brier_multiclass(probs, labels), 2.0 / 3.0)


# ---------------------------------------------------------------------------
# ece
# ---------------------------------------------------------------------------


def test_ece_perfect_calibration() -> None:
    """A perfectly calibrated predictor: the max prob = 1.0 for the
    predicted class, and the argmax is always correct → ECE = 0."""
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    labels = np.array([0, 1, 2], dtype=np.int64)
    assert math.isclose(ece(probs, labels, n_bins=10), 0.0)


def test_ece_uniform_uniform_labels() -> None:
    """Uniform probs + uniform labels: every row has the same
    confidence (1/3) and every row is correct (argmax 0 = label 0 in
    a 1/3-L/1/3-C/1/3-R holdout). ECE = 0 for balanced labels."""
    probs = np.tile(np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]), (6, 1))
    labels = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    assert math.isclose(ece(probs, labels, n_bins=10), 0.0)


def test_ece_known_value_for_miscalibrated() -> None:
    """A small case: 4 rows, model is over-confident (always predicts
    0.9 on the predicted class) and is wrong on 1 of 4 rows. With
    n_bins=10, all 4 rows fall in bin 9 (confidence 0.9). Bin
    accuracy = 3/4, bin confidence = 0.9, ECE = |0.75 - 0.9| = 0.15."""
    probs = np.array(
        [
            [0.9, 0.05, 0.05],  # argmax 0, label 0 → correct
            [0.9, 0.05, 0.05],  # argmax 0, label 0 → correct
            [0.9, 0.05, 0.05],  # argmax 0, label 0 → correct
            [0.05, 0.9, 0.05],  # argmax 1, label 0 → WRONG
        ]
    )
    labels = np.array([0, 0, 0, 0], dtype=np.int64)
    assert math.isclose(ece(probs, labels, n_bins=10), 0.15)


def test_ece_empty() -> None:
    """Empty input returns 0.0 without error."""
    assert ece(np.empty((0, 3)), np.empty(0, dtype=np.int64)) == 0.0


def test_ece_ignores_empty_bins() -> None:
    """ECE skips empty bins. A 4-row holdout with predictions spread
    across two bins: the empty bins contribute 0, so the metric
    reduces to a weighted average of the populated bins."""
    probs = np.array(
        [
            [0.05, 0.05, 0.9],  # bin 0 (conf 0.9) → wrong (label 0)
            [0.9, 0.05, 0.05],  # bin 0 (conf 0.9) → right (label 0)
        ]
    )
    labels = np.array([0, 0], dtype=np.int64)
    val = ece(probs, labels, n_bins=10)
    # Both rows in bin 9 (0.9-1.0). acc=0.5, conf=0.9 → 1.0 * |0.5 - 0.9| = 0.4
    assert math.isclose(val, 0.4)


# ---------------------------------------------------------------------------
# CalibrationReport / CalibrationMetrics roundtrip
# ---------------------------------------------------------------------------


def test_calibration_metrics_roundtrip() -> None:
    """The CalibrationMetrics dataclass roundtrips through asdict()."""
    m = CalibrationMetrics(brier=0.123, ece=0.456, n_bins=10)
    payload = asdict(m)
    assert payload == {"brier": 0.123, "ece": 0.456, "n_bins": 10}
    assert CalibrationMetrics(**payload) == m


def test_calibration_report_roundtrip() -> None:
    """The CalibrationReport serialises with `baseline` as `None` when
    the metrics report has no baseline classifier."""
    model = CalibrationMetrics(brier=0.986, ece=0.436, n_bins=10)
    baseline = CalibrationMetrics(brier=0.652, ece=0.063, n_bins=10)
    random = CalibrationMetrics(brier=0.667, ece=0.063, n_bins=10)
    rep = CalibrationReport(model=model, baseline=baseline, random=random)
    assert rep.baseline is not None
    payload = {
        "model": asdict(rep.model),
        "baseline": asdict(rep.baseline),
        "random": asdict(rep.random),
    }
    assert payload["model"] == {"brier": 0.986, "ece": 0.436, "n_bins": 10}
    assert payload["random"] == {"brier": 0.667, "ece": 0.063, "n_bins": 10}
    # Roundtrip through the metrics report's from_dict helper.
    metrics_payload = {
        "model": {"name": "m", "log_loss": 1.0, "accuracy": 0.5, "save_rate": 0.5, "n_kicks": 4},
        "random_baseline": {
            "name": "r",
            "log_loss": 1.1,
            "accuracy": 0.33,
            "save_rate": 0.4,
            "n_kicks": 4,
        },
        "kicker_most_frequent_baseline": {
            "name": "k",
            "log_loss": None,
            "accuracy": None,
            "save_rate": 0.4,
            "n_kicks": 4,
        },
        "actual_keeper_baseline": {
            "name": "a",
            "log_loss": None,
            "accuracy": None,
            "save_rate": None,
            "n_kicks": 4,
        },
        "n_train": 151,
        "n_holdout": 4,
        "holdout_cutoff_date": "2026-01-01",
        "calibration": payload,
    }
    report = MetricsReport.from_dict(metrics_payload)
    assert report.calibration is not None
    assert report.calibration.model.brier == 0.986
    assert report.calibration.model.ece == 0.436
    assert report.calibration.baseline is not None
    assert report.calibration.random.brier == 0.667


def test_calibration_report_to_dict_includes_block() -> None:
    """to_dict emits a `calibration` block when set, with `baseline`
    as `None` when the metrics report has no baseline classifier."""
    probs = np.array([[0.5, 0.3, 0.2]] * 4)
    report = evaluate_predictions(probs, [_make_row("L")] * 4)
    assert report.calibration is not None
    payload = report.to_dict()
    assert "calibration" in payload
    assert payload["calibration"]["model"]["brier"] >= 0
    assert payload["calibration"]["model"]["n_bins"] == 10
    assert payload["calibration"]["random"]["brier"] >= 0
    # No baseline classifier was provided → `calibration.baseline`
    # serialises as `null` (not absent).
    assert payload["calibration"]["baseline"] is None


def test_metrics_report_from_dict_handles_missing_calibration() -> None:
    """Backward compat: a metrics report that pre-dates Issue #43
    roundtrips with `calibration=None`."""
    payload = {
        "model": {"name": "m", "log_loss": 1.0, "accuracy": 0.5, "save_rate": 0.5, "n_kicks": 4},
        "random_baseline": {
            "name": "r",
            "log_loss": 1.1,
            "accuracy": 0.33,
            "save_rate": 0.4,
            "n_kicks": 4,
        },
        "kicker_most_frequent_baseline": {
            "name": "k",
            "log_loss": None,
            "accuracy": None,
            "save_rate": 0.4,
            "n_kicks": 4,
        },
        "actual_keeper_baseline": {
            "name": "a",
            "log_loss": None,
            "accuracy": None,
            "save_rate": None,
            "n_kicks": 4,
        },
        "n_train": 151,
        "n_holdout": 4,
        "holdout_cutoff_date": "2026-01-01",
    }
    report = MetricsReport.from_dict(payload)
    assert report.calibration is None


# ---------------------------------------------------------------------------
# evaluate_predictions
# ---------------------------------------------------------------------------


def test_evaluate_predictions_returns_full_report() -> None:
    """The report has all four sections + the split metadata."""
    probs = np.array([[0.5, 0.3, 0.2]] * 4)
    holdout = [
        _make_row("L"),
        _make_row("L"),
        _make_row("L"),
        _make_row("C"),
    ]
    report = evaluate_predictions(probs, holdout)
    assert isinstance(report, MetricsReport)
    assert isinstance(report.model, BaselineMetrics)
    assert isinstance(report.random_baseline, BaselineMetrics)
    assert isinstance(report.kicker_most_frequent_baseline, BaselineMetrics)
    assert isinstance(report.actual_keeper_baseline, BaselineMetrics)
    assert report.n_holdout == 4
    assert report.model.n_kicks == 4
    assert report.random_baseline.n_kicks == 4
    # random log_loss is ln(3) — uniform prior.
    assert report.random_baseline.log_loss is not None
    assert math.isclose(report.random_baseline.log_loss, math.log(3.0))
    # kicker_most_frequent and actual_keeper have no log_loss / accuracy.
    assert report.kicker_most_frequent_baseline.log_loss is None
    assert report.kicker_most_frequent_baseline.accuracy is None
    assert report.kicker_most_frequent_baseline.name == "last_side"
    assert report.actual_keeper_baseline.log_loss is None
    assert report.actual_keeper_baseline.save_rate is None
    # Issue #43: the calibration block is populated for a non-empty
    # holdout. Model + random are always present; baseline is None
    # when no `baseline_probs` is passed.
    assert report.calibration is not None
    assert report.calibration.model.n_bins == 10
    assert report.calibration.random.n_bins == 10
    assert report.calibration.baseline is None


def test_evaluate_predictions_includes_baseline_calibration() -> None:
    """When `baseline_probs` is provided, the calibration block has
    a `baseline` entry."""
    probs = np.array([[0.5, 0.3, 0.2]] * 4)
    baseline_probs = np.array([[0.4, 0.3, 0.3]] * 4)
    holdout = [_make_row("L"), _make_row("L"), _make_row("L"), _make_row("C")]
    report = evaluate_predictions(probs, holdout, baseline_probs=baseline_probs)
    assert report.calibration is not None
    assert report.calibration.baseline is not None
    assert report.calibration.baseline.n_bins == 10
    assert report.calibration.baseline.brier >= 0
    assert 0 <= report.calibration.baseline.ece <= 1


def test_evaluate_predictions_empty() -> None:
    """Empty input produces a report with zero kicks and None metrics."""
    report = evaluate_predictions(np.empty((0, 3)), [])
    assert report.n_holdout == 0
    assert report.model.n_kicks == 0
    assert report.model.log_loss is None
    # Empty holdout: calibration is undefined and the block is None.
    assert report.calibration is None


def test_evaluate_predictions_to_dict() -> None:
    """The to_dict serialisation is JSON-friendly (None → null, no
    NaN literals)."""
    probs = np.array([[0.5, 0.3, 0.2]])
    report = evaluate_predictions(probs, [_make_row("L")])
    payload = report.to_dict()
    serialised = json.dumps(payload)
    assert "null" in serialised  # None values are preserved
    assert "NaN" not in serialised  # no NaN literals


# ---------------------------------------------------------------------------
# write_metrics_json
# ---------------------------------------------------------------------------


def test_write_metrics_json_roundtrip(tmp_path: Path) -> None:
    """The metrics report roundtrips to JSON and back without loss."""
    probs = np.array([[0.5, 0.3, 0.2]] * 3)
    holdout = [_make_row("L"), _make_row("R"), _make_row("C")]
    report = evaluate_predictions(probs, holdout)
    out = tmp_path / "metrics.json"
    write_metrics_json(out, report)
    with out.open(encoding="utf-8") as f:
        payload = json.load(f)
    assert set(payload.keys()) >= {
        "model",
        "random_baseline",
        "kicker_most_frequent_baseline",
        "actual_keeper_baseline",
        "n_holdout",
    }
    assert payload["model"]["n_kicks"] == 3


# ---------------------------------------------------------------------------
# Live smoke test
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not Artifacts().metrics.exists(),
    reason="output/metrics.json not present (run the slice first)",
)
def test_live_metrics_json_shape() -> None:
    """The live `output/metrics.json` has the expected sections and
    the random baseline's log loss is `ln(3)`.

    The model-section log loss was below random in slice #7
    (baseline, logreg) and is above random in slice #8 (LightGBM
    with conservative defaults — see progress.txt for the
    trade-off). The test does not pin a model log loss; it pins
    the random baseline's value (a property of the uniform
    distribution, not the model).

    Issue #43: the calibration block is in the live metrics, with
    `model`, `baseline`, and `random` sub-entries, each carrying a
    `brier` and `ece` and an `n_bins` of 10. The random baseline's
    Brier is the closed-form 2/3 (uniform probs on 3 classes).
    """
    with Artifacts().metrics.open(encoding="utf-8") as f:
        payload = json.load(f)
    assert "model" in payload
    assert "random_baseline" in payload
    assert "kicker_most_frequent_baseline" in payload
    assert "actual_keeper_baseline" in payload
    assert "n_train" in payload
    assert "n_holdout" in payload
    assert "holdout_cutoff_date" in payload
    assert math.isclose(payload["random_baseline"]["log_loss"], math.log(3.0))
    assert payload["actual_keeper_baseline"]["save_rate"] is None
    # Calibration block (Issue #43).
    assert "calibration" in payload
    for key in ("model", "baseline", "random"):
        sub = payload["calibration"][key]
        if sub is None:
            continue
        assert "brier" in sub
        assert "ece" in sub
        assert sub["n_bins"] == 10
    assert math.isclose(payload["calibration"]["random"]["brier"], 2.0 / 3.0)
    # Cross-validation block (Issue #45).
    assert "cv" in payload
    assert payload["cv"]["group_by"] == "tournament_name"
    assert len(payload["cv"]["folds"]) >= 1
    for fold in payload["cv"]["folds"]:
        assert "name" in fold
        assert fold["n_holdout"] > 0
        assert "save_rate" in fold
        assert "log_loss" in fold
        assert "accuracy" in fold
    assert "aggregate" in payload["cv"]
    assert "se_save_rate" in payload["cv"]["aggregate"]
    assert payload["cv"]["aggregate"]["n_total"] >= 1


# ---------------------------------------------------------------------------
# Cross-validation (Issue #45)
# ---------------------------------------------------------------------------


class _PerfectClassifier:
    """A toy classifier that returns a constant prediction per row.

    `cross_validate` only needs `fit(matrix)` to return an object
    with `predict_proba(X) -> (n, 3)`. The toy classifier ignores
    `X` and returns a constant `(n, 3)` array based on the
    `mode` argument. Use this to test the `cross_validate` plumbing
    (per-fold fit + score + aggregate) without depending on
    LightGBM.
    """

    def __init__(self, mode: str = "uniform") -> None:
        self.mode = mode

    def fit(self, matrix) -> _PerfectClassifier:  # noqa: ARG002
        return self

    def predict_proba(self, X) -> np.ndarray:  # noqa: ARG002
        n = len(X)
        if self.mode == "uniform":
            return np.full((n, 3), 1.0 / 3.0)
        if self.mode == "always_L":
            out = np.zeros((n, 3))
            out[:, 0] = 1.0
            return out
        if self.mode == "always_C":
            out = np.zeros((n, 3))
            out[:, 1] = 1.0
            return out
        if self.mode == "always_R":
            out = np.zeros((n, 3))
            out[:, 2] = 1.0
            return out
        raise ValueError(f"unknown mode: {self.mode}")


def _make_uniform_factory(mode: str = "uniform"):
    """Build a model factory that returns a constant `_PerfectClassifier`."""

    def factory(matrix):  # noqa: ARG001
        return _PerfectClassifier(mode=mode).fit(matrix)

    return factory


def test_cross_validate_empty_input() -> None:
    """An empty `rows` list returns an empty report with zero metrics."""
    report = cross_validate(_make_uniform_factory(), [], group_by="tournament_name")
    assert isinstance(report, CVReport)
    assert report.folds == ()
    assert report.n_total == 0
    assert report.aggregate_save_rate == 0.0
    assert report.skipped == {}
    assert report.group_by == "tournament_name"


def test_cross_validate_three_groups_one_fold_each() -> None:
    """3 groups, 1 fold each, uniform model. The uniform predictor
    always dives L (the lowest-index side on a tie), so each
    fold's save rate is the per-fold fraction of L-or-off-target
    kicks (deterministic for a given fold's label distribution)."""
    rows = [
        _make_row("L", tournament_name="A", match_id=1, kick_number=1),
        _make_row("C", tournament_name="A", match_id=1, kick_number=2),
        _make_row("L", tournament_name="B", match_id=2, kick_number=1),
        _make_row("R", tournament_name="B", match_id=2, kick_number=2),
        _make_row("C", tournament_name="C", match_id=3, kick_number=1),
        _make_row("R", tournament_name="C", match_id=3, kick_number=2),
    ]
    report = cross_validate(_make_uniform_factory("uniform"), rows)
    assert len(report.folds) == 3
    assert report.n_total == 6
    # The uniform predictor dives L on every row. So the save rate
    # for each fold is the fold's fraction of L kicks. Group A has
    # 1 L / 2 → 0.5. Group B has 1 L / 2 → 0.5. Group C has 0 L / 2 → 0.0.
    expected_per_fold = {"A": 0.5, "B": 0.5, "C": 0.0}
    for fold in report.folds:
        assert math.isclose(fold.save_rate, expected_per_fold[fold.name], abs_tol=1e-9)
    expected_agg = sum(f.save_rate * f.n_holdout for f in report.folds) / 6
    assert math.isclose(report.aggregate_save_rate, expected_agg)


def test_cross_validate_aggregate_save_rate_weighted_correctly() -> None:
    """The aggregate save rate is the n_holdout-weighted mean across folds.

    4 rows per group, 3 groups. The fold's save rate is whatever
    the model emits (here uniform). The aggregate is the simple
    n_holdout-weighted mean (each fold has 4 rows, so the
    aggregate is the simple mean of the 3 fold save rates).
    """
    rows = []
    for t in ("X", "Y", "Z"):
        for i in range(4):
            rows.append(
                _make_row("L", tournament_name=t, match_id=hash(t) % 1000 + i, kick_number=1)
            )
    report = cross_validate(_make_uniform_factory("uniform"), rows)
    assert len(report.folds) == 3
    assert report.n_total == 12
    expected = sum(f.save_rate for f in report.folds) / 3
    assert math.isclose(report.aggregate_save_rate, expected, abs_tol=1e-9)


def test_cross_validate_se_is_binomial_on_weighted_total() -> None:
    """The aggregate SE is the binomial SE on the aggregate total:
    `sqrt(p * (1 - p) / n_total)` where `p` is the aggregate save
    rate. The test uses an "always L" model so every fold's save
    rate is deterministic (the dive is always L, the save rate
    equals the per-fold fraction of L-or-off-target kicks)."""
    rows = []
    for t in ("G1", "G2", "G3", "G4"):
        for label in ("L", "L", "C", "R", "R"):  # 2 L, 1 C, 2 R per fold
            rows.append(
                _make_row(
                    label, tournament_name=t, match_id=hash(t) % 1000 + len(rows), kick_number=1
                )
            )
    report = cross_validate(_make_uniform_factory("always_L"), rows)
    assert report.n_total == 20
    p = report.aggregate_save_rate
    expected_se = math.sqrt(p * (1 - p) / 20)
    assert math.isclose(report.se_save_rate, expected_se, abs_tol=1e-9)


def test_cross_validate_min_fold_size_skips_small_folds() -> None:
    """Folds with fewer than `min_fold_size` rows are skipped and
    recorded in the `skipped` dict."""
    rows = [
        _make_row("L", tournament_name="big", match_id=1, kick_number=1),
        _make_row("R", tournament_name="big", match_id=1, kick_number=2),
        _make_row("C", tournament_name="big", match_id=1, kick_number=3),
        _make_row("L", tournament_name="small", match_id=2, kick_number=1),
    ]
    report = cross_validate(_make_uniform_factory("uniform"), rows, min_fold_size=2)
    assert len(report.folds) == 1
    assert report.folds[0].name == "big"
    assert report.n_total == 3
    assert report.skipped == {"small": 1}


def test_cross_validate_min_fold_size_zero_is_rejected() -> None:
    """`min_fold_size` must be >= 1 (a 0-row fold has no signal to score)."""
    with pytest.raises(ValueError, match="min_fold_size"):
        cross_validate(_make_uniform_factory(), [], min_fold_size=0)


def test_cross_validate_skipped_fold_kept_when_only_empty_folds() -> None:
    """If every fold is below `min_fold_size`, the report has 0 folds
    but the `skipped` dict records the dropped groups."""
    rows = [
        _make_row("L", tournament_name="a", match_id=1, kick_number=1),
        _make_row("R", tournament_name="b", match_id=2, kick_number=1),
    ]
    report = cross_validate(_make_uniform_factory("uniform"), rows, min_fold_size=2)
    assert report.folds == ()
    assert report.n_total == 0
    assert set(report.skipped) == {"a", "b"}
    assert report.skipped["a"] == 1
    assert report.skipped["b"] == 1


def test_cross_validate_fold_order_is_deterministic() -> None:
    """Folds are sorted by `n_holdout` descending (ties broken by name)
    so the report is reproducible across runs."""
    rows = [
        _make_row("L", tournament_name="z", match_id=1, kick_number=1),
        _make_row("L", tournament_name="a", match_id=2, kick_number=1),
        _make_row("L", tournament_name="m", match_id=3, kick_number=1),
        _make_row("L", tournament_name="m", match_id=3, kick_number=2),
    ]
    report = cross_validate(_make_uniform_factory("uniform"), rows)
    names = [f.name for f in report.folds]
    assert names == ["m", "a", "z"]


def test_cross_validate_deterministic_for_same_input() -> None:
    """Two runs with the same `rows` and factory produce the same report."""
    rows = [
        _make_row("L", tournament_name="A", match_id=1, kick_number=1),
        _make_row("R", tournament_name="A", match_id=1, kick_number=2),
        _make_row("C", tournament_name="B", match_id=2, kick_number=1),
        _make_row("L", tournament_name="B", match_id=2, kick_number=2),
    ]
    r1 = cross_validate(_make_uniform_factory("always_L"), rows)
    r2 = cross_validate(_make_uniform_factory("always_L"), rows)
    assert r1.aggregate_save_rate == r2.aggregate_save_rate
    assert r1.aggregate_log_loss == r2.aggregate_log_loss
    assert r1.aggregate_accuracy == r2.aggregate_accuracy
    assert r1.se_save_rate == r2.se_save_rate
    assert [(f.name, f.save_rate) for f in r1.folds] == [(f.name, f.save_rate) for f in r2.folds]


# ---------------------------------------------------------------------------
# CVReport / CVFold roundtrip
# ---------------------------------------------------------------------------


def test_cv_fold_roundtrip() -> None:
    """A `CVFold` roundtrips through `asdict` (the per-fold payload)."""
    fold = CVFold(
        name="X",
        n_train=100,
        n_holdout=20,
        save_rate=0.5,
        log_loss=1.0,
        accuracy=0.4,
        random_save_rate=0.45,
    )
    payload = asdict(fold)
    assert payload == {
        "name": "X",
        "n_train": 100,
        "n_holdout": 20,
        "save_rate": 0.5,
        "log_loss": 1.0,
        "accuracy": 0.4,
        "random_save_rate": 0.45,
    }
    assert CVFold(**payload) == fold


def test_cv_report_to_from_dict_roundtrip() -> None:
    """A `CVReport` roundtrips through `_cv_to_dict` / `_cv_from_dict`
    without loss. The `skipped` dict is preserved as a plain dict."""
    fold = CVFold(
        name="World Cup",
        n_train=171,
        n_holdout=8,
        save_rate=0.375,
        log_loss=1.2,
        accuracy=0.4,
        random_save_rate=0.42,
    )
    report = CVReport(
        folds=(fold,),
        aggregate_save_rate=0.375,
        aggregate_log_loss=1.2,
        aggregate_accuracy=0.4,
        n_total=8,
        se_save_rate=math.sqrt(0.375 * 0.625 / 8),
        group_by="tournament_name",
        skipped={"Asian Cup": 0},
    )
    payload = _cv_to_dict(report)
    assert payload["group_by"] == "tournament_name"
    assert payload["skipped"] == {"Asian Cup": 0}
    assert payload["aggregate"]["n_total"] == 8
    assert len(payload["folds"]) == 1
    restored = _cv_from_dict(payload)
    assert restored == report


def test_cv_report_from_dict_default_skipped() -> None:
    """A `cv` payload without a `skipped` key (older LOTO reports)
    roundtrips with `skipped={}`."""
    payload = {
        "folds": [],
        "aggregate": {
            "save_rate": 0.0,
            "log_loss": 0.0,
            "accuracy": 0.0,
            "n_total": 0,
            "se_save_rate": 0.0,
        },
        "group_by": "tournament_name",
    }
    report = _cv_from_dict(payload)
    assert report.skipped == {}


def test_cv_report_from_dict_default_group_by() -> None:
    """A `cv` payload without a `group_by` key defaults to
    `tournament_name` (the v3 default)."""
    payload = {"folds": [], "aggregate": {}}
    report = _cv_from_dict(payload)
    assert report.group_by == "tournament_name"


# ---------------------------------------------------------------------------
# MetricsReport.cv roundtrip + backward compat (Issue #45)
# ---------------------------------------------------------------------------


def _make_minimal_metrics_payload() -> dict:
    """Build a minimal `MetricsReport` payload (pre-#45) for the
    roundtrip tests below. The payload omits `cv` so the backward-
    compat path can be exercised."""
    return {
        "model": {"name": "m", "log_loss": 1.0, "accuracy": 0.5, "save_rate": 0.5, "n_kicks": 4},
        "random_baseline": {
            "name": "r",
            "log_loss": 1.1,
            "accuracy": 0.33,
            "save_rate": 0.4,
            "n_kicks": 4,
        },
        "kicker_most_frequent_baseline": {
            "name": "k",
            "log_loss": None,
            "accuracy": None,
            "save_rate": 0.4,
            "n_kicks": 4,
        },
        "actual_keeper_baseline": {
            "name": "a",
            "log_loss": None,
            "accuracy": None,
            "save_rate": None,
            "n_kicks": 4,
        },
        "n_train": 4,
        "n_holdout": 4,
        "holdout_cutoff_date": "2026-01-01",
    }


def test_metrics_report_to_dict_includes_cv_block() -> None:
    """A `MetricsReport` with a `cv` block serialises the CV under
    the `cv` key."""
    fold = CVFold("X", 4, 4, 0.5, 1.0, 0.4, 0.4)
    cv = CVReport(
        folds=(fold,),
        aggregate_save_rate=0.5,
        aggregate_log_loss=1.0,
        aggregate_accuracy=0.4,
        n_total=4,
        se_save_rate=math.sqrt(0.5 * 0.5 / 4),
        group_by="tournament_name",
        skipped={},
    )
    probs = np.array([[0.5, 0.3, 0.2]] * 4)
    report = evaluate_predictions(probs, [_make_row("L")] * 4)
    report = type(report)(
        model=report.model,
        random_baseline=report.random_baseline,
        kicker_most_frequent_baseline=report.kicker_most_frequent_baseline,
        actual_keeper_baseline=report.actual_keeper_baseline,
        n_train=report.n_train,
        n_holdout=report.n_holdout,
        holdout_cutoff_date=report.holdout_cutoff_date,
        baseline=report.baseline,
        calibration=report.calibration,
        cv=cv,
        extras=report.extras,
    )
    payload = report.to_dict()
    assert "cv" in payload
    assert payload["cv"]["group_by"] == "tournament_name"
    assert len(payload["cv"]["folds"]) == 1


def test_metrics_report_from_dict_handles_missing_cv() -> None:
    """Backward compat: a metrics report that pre-dates Issue #45
    (no `cv` key) roundtrips with `cv=None`."""
    payload = _make_minimal_metrics_payload()
    report = MetricsReport.from_dict(payload)
    assert report.cv is None


def test_metrics_report_from_dict_roundtrips_cv_block() -> None:
    """A `cv` block roundtrips through `from_dict` with all fields
    preserved (folds, aggregate, group_by, skipped)."""
    fold = CVFold("Y", 10, 5, 0.6, 0.9, 0.5, 0.42)
    cv = CVReport(
        folds=(fold,),
        aggregate_save_rate=0.6,
        aggregate_log_loss=0.9,
        aggregate_accuracy=0.5,
        n_total=5,
        se_save_rate=0.2,
        group_by="tournament_name",
        skipped={"Asian Cup": 0},
    )
    payload = _make_minimal_metrics_payload()
    payload["cv"] = _cv_to_dict(cv)
    report = MetricsReport.from_dict(payload)
    assert report.cv is not None
    assert report.cv == cv


# ---------------------------------------------------------------------------
# Artifacts.cv_metrics roundtrip (Issue #45)
# ---------------------------------------------------------------------------


def test_artifacts_cv_metrics_roundtrip(tmp_path: Path) -> None:
    """`Artifacts.read_cv` / `Artifacts.write_cv` roundtrip a `CVReport`
    through the `cv_metrics.json` artifact without loss."""
    from twelveyards.artifacts import Artifacts

    art = Artifacts(root=tmp_path, cache_dir=tmp_path / "cache")
    fold = CVFold("Test", 4, 4, 0.5, 1.0, 0.4, 0.42)
    cv = CVReport(
        folds=(fold,),
        aggregate_save_rate=0.5,
        aggregate_log_loss=1.0,
        aggregate_accuracy=0.4,
        n_total=4,
        se_save_rate=0.25,
        group_by="tournament_name",
        skipped={},
    )
    art.write_cv(cv)
    assert art.cv_metrics.exists()
    assert art.cv_metrics.parent == tmp_path
    restored = art.read_cv()
    assert restored == cv
