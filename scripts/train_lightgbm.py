"""Train the LightGBM classifier and write its artifacts.

Slice #8 (Issue #24): the LightGBM model. The script reads
`output/training_table.jsonl` (slice #6), splits the rows temporally
(train: pre-2026; holdout: 2026+), fits a LightGBM multiclass
classifier on the training fold, evaluates on the holdout, and
writes:

- `output/metrics.json` — log loss, accuracy, and counterfactual
  save rate for the LightGBM model, the logreg baseline (trained on
  the same training fold for an apples-to-apples comparison), and
  three fixed baselines (random, kicker-most-frequent, actual keeper).
- `output/lightgbm.pkl` — the LightGBM model fitted on the same
  151-row pre-2026 training fold that produced the metrics. The
  artifact and the metrics describe the SAME model (Issue #40: the
  previous "retrain on all rows" recipe produced an artifact whose
  in-sample holdout save rate was 0.107, not the 0.464 the card
  advertised).

The script is re-runnable: same inputs + same random seed → identical
output. The LightGBM hyperparameters and the holdout cutoff are
CLI-overridable; the PRD specifies conservative defaults
(`num_leaves=31`, `learning_rate=0.05`, `n_estimators=500`,
`min_child_samples=20`) and "no aggressive tuning in v1".
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from penalty_pred.artifacts import Artifacts
from penalty_pred.evaluate import evaluate_predictions
from penalty_pred.model import (
    CATEGORICAL_FEATURES,
    CLASSES,
    FEATURE_COLUMNS,
    HOLDOUT_CUTOFF_DATE,
    LIGHTGBM_DEFAULTS,
    LOGREG_DEFAULTS,
    NUMERIC_FEATURES,
    RANDOM_SEED,
    PredictProba,
    build_feature_matrix,
    fit_lightgbm,
    fit_logistic_regression,
    is_on_target_by_key,
    load_training_table,
    temporal_split,
)


def _train_logreg_on_fold(
    train_matrix,
    *,
    C: float,
    class_weight: str,
) -> PredictProba:
    """Train the logreg baseline on the same training fold as the
    LightGBM, so the report's `baseline` section is a fair comparison.

    The slice reuses the same params the baseline slice (#23) used by
    default (`C=0.005`, `class_weight="balanced"`) — those are the
    params pinned in `LOGREG_DEFAULTS`. The slice doesn't re-load the
    baseline.pkl from disk because we want the baseline to be trained
    on exactly the same training fold the LightGBM is trained on (in
    case the cutoff is reparameterised via `--holdout-cutoff`).
    """
    params: dict[str, object] = {"C": C, "class_weight": class_weight}
    return fit_logistic_regression(train_matrix, params=params)


def main() -> int:
    art = Artifacts()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-table",
        type=Path,
        default=art.training_table,
        help=f"Path to training_table.jsonl (default: {art.training_table}).",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=art.lightgbm_model,
        help=f"Path to write the frozen LightGBM artifact (default: {art.lightgbm_model}).",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=art.metrics,
        help=f"Path to write the metrics JSON (default: {art.metrics}).",
    )
    parser.add_argument(
        "--holdout-cutoff",
        default=HOLDOUT_CUTOFF_DATE,
        help=f"ISO 8601 cutoff for the temporal split (default: {HOLDOUT_CUTOFF_DATE}).",
    )
    parser.add_argument(
        "--num-leaves",
        type=int,
        default=LIGHTGBM_DEFAULTS["num_leaves"],
        help=f"LightGBM num_leaves (default: {LIGHTGBM_DEFAULTS['num_leaves']}).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=LIGHTGBM_DEFAULTS["learning_rate"],
        help=f"LightGBM learning_rate (default: {LIGHTGBM_DEFAULTS['learning_rate']}).",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=LIGHTGBM_DEFAULTS["n_estimators"],
        help=f"LightGBM n_estimators (default: {LIGHTGBM_DEFAULTS['n_estimators']}).",
    )
    parser.add_argument(
        "--min-child-samples",
        type=int,
        default=LIGHTGBM_DEFAULTS["min_child_samples"],
        help=(f"LightGBM min_child_samples (default: {LIGHTGBM_DEFAULTS['min_child_samples']})."),
    )
    parser.add_argument(
        "--C",
        type=float,
        default=LOGREG_DEFAULTS["C"],
        help=f"Logreg inverse regularisation strength (default: {LOGREG_DEFAULTS['C']}).",
    )
    parser.add_argument(
        "--class-weight",
        default=LOGREG_DEFAULTS.get("class_weight", "none"),
        choices=("balanced", "none"),
        help=(
            "Logreg class weighting strategy. `balanced` is the v1 "
            "default — it pulls the logreg's log loss under the "
            "random baseline on the 2026 holdout (slice #23 AC)."
        ),
    )
    args = parser.parse_args()

    if not args.training_table.exists():
        print(f"error: {args.training_table} not found", file=sys.stderr)
        return 1

    # The on-target flag is no longer a sibling-reach: we read the
    # shootout kicks through the artifacts adapter and pass the lookup
    # to `load_training_table` explicitly.
    shootout_kicks = art.read_shootout_kicks()
    rows = load_training_table(
        args.training_table,
        is_on_target_by_key=is_on_target_by_key(shootout_kicks),
    )
    train_rows, holdout_rows = temporal_split(rows, cutoff_date=args.holdout_cutoff)
    print(
        f"Loaded {len(rows)} rows from {args.training_table}; "
        f"train={len(train_rows)} (pre-{args.holdout_cutoff}), "
        f"holdout={len(holdout_rows)} ({args.holdout_cutoff}+)."
    )

    train_matrix = build_feature_matrix(train_rows)
    holdout_matrix = build_feature_matrix(holdout_rows)

    lgb_params: dict[str, object] = {
        "num_leaves": args.num_leaves,
        "learning_rate": args.learning_rate,
        "n_estimators": args.n_estimators,
        "min_child_samples": args.min_child_samples,
    }
    lgb = fit_lightgbm(
        train_matrix,
        params=lgb_params,
        random_state=RANDOM_SEED,
    )
    # Issue #40: `lgb` is the model we just scored on the holdout and
    # wrote to `metrics.json`. Freeze THIS model as the deployment
    # artifact — never a second fit. A second `fit_lightgbm` call on
    # the same data with the same seed can drift a kick or two (LightGBM
    # has minor internal non-determinism from parallel histogram
    # construction), so the artifact and the metrics would describe
    # *almost* the same model, not the same one.
    frozen = lgb
    print(
        f"Fitted LightGBM on {len(train_rows)} rows (params={lgb_params}, "
        f"class_weight=inverse_frequency)."
    )

    baseline = _train_logreg_on_fold(
        train_matrix,
        C=args.C,
        class_weight=args.class_weight,
    )
    print(
        f"Fitted logreg baseline on {len(train_rows)} rows "
        f"(C={args.C}, class_weight={args.class_weight!r})."
    )

    lgb_probs = np.asarray(lgb.predict_proba(holdout_matrix.X))
    baseline_probs = np.asarray(baseline.predict_proba(holdout_matrix.X))
    report = evaluate_predictions(lgb_probs, holdout_rows, baseline_probs=baseline_probs)
    report = replace(
        report,
        n_train=len(train_rows),
        holdout_cutoff_date=args.holdout_cutoff,
    )
    # Issue #45: preserve the LOTO CV block from a prior
    # `scripts/evaluate_cv.py` run, so re-running the train slice
    # doesn't silently drop the cross-validation report. The CV is
    # not recomputed here — `evaluate_cv.py` owns that — but the
    # artifact is the same per (model, data, recipe) so a re-train
    # does not invalidate the existing CV.
    cv_block: object | None = None
    if art.cv_metrics.exists():
        cv_block = art.read_cv()
    report = replace(report, cv=cv_block)
    report = replace(
        report,
        extras={
            "model_kind": "lightgbm",
            "classes": list(CLASSES),
            "feature_columns": list(FEATURE_COLUMNS),
            "params": {
                "lightgbm": lgb_params,
                "logreg": {"C": args.C, "class_weight": args.class_weight},
            },
        },
    )

    art.write_metrics(report, path=args.metrics_output)
    cal = report.calibration
    assert cal is not None  # evaluate_predictions always sets it for non-empty holdouts
    baseline_metrics = report.baseline
    cal_baseline = cal.baseline
    assert baseline_metrics is not None  # train_lightgbm always fits a baseline on the same fold
    assert cal_baseline is not None  # the calibration report mirrors the optional baseline
    print(
        f"Wrote {args.metrics_output} (held-out metrics).\n"
        f"  model (lightgbm): log_loss={report.model.log_loss:.3f} "
        f"acc={report.model.accuracy:.3f} save_rate={report.model.save_rate:.3f}\n"
        f"  baseline (logreg): log_loss={baseline_metrics.log_loss:.3f} "
        f"acc={baseline_metrics.accuracy:.3f} save_rate={baseline_metrics.save_rate:.3f}\n"
        f"  random:           log_loss={report.random_baseline.log_loss:.3f} "
        f"acc={report.random_baseline.accuracy:.3f} "
        f"save_rate={report.random_baseline.save_rate:.3f}\n"
        f"  kmf:              save_rate={report.kicker_most_frequent_baseline.save_rate}\n"
        f"  keeper:           save_rate={report.actual_keeper_baseline.save_rate}\n"
        f"  calibration:      model  brier={cal.model.brier:.3f} ece={cal.model.ece:.3f}\n"
        f"                    base   brier={cal_baseline.brier:.3f} ece={cal_baseline.ece:.3f}\n"
        f"                    rndm   brier={cal.random.brier:.3f} ece={cal.random.ece:.3f}"
    )

    # Issue #40: freeze the deployment artifact on the same 151-row
    # training fold that produced the metrics, so the artifact and
    # metrics describe the same model. The previous "retrain on all
    # rows" recipe produced an in-sample artifact whose holdout save
    # rate was 0.107, not the 0.464 the card advertised.
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    art.write_model(
        frozen,
        list(FEATURE_COLUMNS),
        "lightgbm",
        params={
            **lgb_params,
            "class_weight": "inverse_frequency",
            "random_state": RANDOM_SEED,
            "n_train_rows": len(train_rows),
        },
        path=args.model_output,
    )
    print(
        f"\nFroze LightGBM on {len(train_rows)} rows (pre-{args.holdout_cutoff}) → {args.model_output}.\n"
        f"  categorical_features={list(CATEGORICAL_FEATURES)}\n"
        f"  numeric_features={list(NUMERIC_FEATURES)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
