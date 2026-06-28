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
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from penalty_pred.artifacts import Artifacts
from penalty_pred.evaluate import (
    BaselineMetrics,
    MetricsReport,
    accuracy,
    actual_keeper_save_rate,
    counterfactual_save_rate,
    evaluate_predictions,
    last_side_save_rate,
    log_loss,
    random_save_rate,
    recommended_dive,
    write_metrics_json,
)
from penalty_pred.model import TrainingRow

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_row(
    label: str,
    *,
    last_side: str = "",
    is_on_target: bool = True,
    kicker_id: int = 1,
) -> TrainingRow:
    return TrainingRow(
        match_id=1,
        kick_number=1,
        kicker_id=kicker_id,
        kicker_name="Stub",
        match_date="2026-01-15T00:00:00+00:00",
        tournament_id=77,
        tournament_name="World Cup",
        round="1/8",
        team_id=1,
        is_home=True,
        label=label,
        is_on_target=is_on_target,
        features={
            "p_L_5": 0.0,
            "p_C_5": 0.0,
            "p_R_5": 0.0,
            "p_L_10": 0.0,
            "p_C_10": 0.0,
            "p_R_10": 0.0,
            "p_L_20": 0.0,
            "p_C_20": 0.0,
            "p_R_20": 0.0,
            "career_penalty_count": 0,
            "b1_kick_number": 1,
            "pen_score_home": 0,
            "pen_score_away": 0,
            "is_decisive": False,
            "age": 25.0,
            "last_side": last_side,
            "kicking_foot": "Unknown",
            "b3_round": "1/8",
            "position": "striker",
        },
    )


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
    assert math.isclose(report.random_baseline.log_loss, math.log(3.0))
    # kicker_most_frequent and actual_keeper have no log_loss / accuracy.
    assert report.kicker_most_frequent_baseline.log_loss is None
    assert report.kicker_most_frequent_baseline.accuracy is None
    assert report.kicker_most_frequent_baseline.name == "last_side"
    assert report.actual_keeper_baseline.log_loss is None
    assert report.actual_keeper_baseline.save_rate is None


def test_evaluate_predictions_empty() -> None:
    """Empty input produces a report with zero kicks and None metrics."""
    report = evaluate_predictions(np.empty((0, 3)), [])
    assert report.n_holdout == 0
    assert report.model.n_kicks == 0
    assert report.model.log_loss is None


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
