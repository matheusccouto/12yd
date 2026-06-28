"""Train the baseline (logistic regression) classifier and write its artifacts.

Slice #7 (Issue #23): the baseline. The script reads
`output/training_table.jsonl` (slice #6), splits the rows temporally
(train: pre-2026; holdout: 2026+), fits a multinomial logistic
regression on the training fold, evaluates on the holdout, and
writes:

- `output/baseline.pkl` — the fitted sklearn Pipeline plus the
  feature column order, ready for the predict slice (#25) to load.
- `output/metrics.json` — log loss, accuracy, and counterfactual
  save rate for the model and three baselines (random,
  kicker-most-frequent, actual keeper).

The script is re-runnable: same inputs + same random seed → identical
output (verified by the test suite). The output paths and the
hyperparameters are CLI-overridable so the next slice (#24 LightGBM)
can reuse the same harness.
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
    CLASSES,
    FEATURE_COLUMNS,
    HOLDOUT_CUTOFF_DATE,
    LOGREG_DEFAULTS,
    build_feature_matrix,
    fit_logistic_regression,
    is_on_target_by_key,
    load_training_table,
    temporal_split,
)


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
        default=art.baseline_model,
        help=f"Path to write the pickled model artifact (default: {art.baseline_model}).",
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
        "--C",
        type=float,
        default=LOGREG_DEFAULTS["C"],
        help=f"Inverse regularisation strength (default: {LOGREG_DEFAULTS['C']}).",
    )
    parser.add_argument(
        "--class-weight",
        default=LOGREG_DEFAULTS.get("class_weight", "none"),
        choices=("balanced", "none"),
        help=(
            "Class weighting strategy. `balanced` is the v1 default — "
            "it pulls the model's log loss under the random baseline on "
            "the 2026 holdout (issue #23 AC). Pass `none` to disable."
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
    pipe = fit_logistic_regression(
        train_matrix,
        params={"C": args.C, "class_weight": args.class_weight},
    )
    print(
        f"Fitted logistic regression on {len(train_rows)} rows "
        f"(C={args.C}, class_weight={args.class_weight!r})."
    )

    holdout_matrix = build_feature_matrix(holdout_rows)
    probs = np.asarray(pipe.predict_proba(holdout_matrix.X))
    report = evaluate_predictions(probs, holdout_rows)
    # Stamp the split metadata the caller needs to interpret the report.
    report = replace(
        report,
        n_train=len(train_rows),
        holdout_cutoff_date=args.holdout_cutoff,
    )
    report = replace(
        report,
        extras={
            "model_kind": "baseline",
            "classes": list(CLASSES),
            "feature_columns": list(FEATURE_COLUMNS),
            "params": {"C": args.C, "class_weight": args.class_weight},
        },
    )

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    art.write_model(
        pipe,
        list(FEATURE_COLUMNS),
        "baseline",
        params={"C": args.C, "class_weight": args.class_weight},
        path=args.model_output,
    )
    art.write_metrics(report, path=args.metrics_output)
    print(
        f"Wrote {args.model_output} (artifact) and {args.metrics_output} (metrics).\n"
        f"  model:       log_loss={report.model.log_loss:.3f} "
        f"acc={report.model.accuracy:.3f} save_rate={report.model.save_rate:.3f}\n"
        f"  random:      log_loss={report.random_baseline.log_loss:.3f} "
        f"acc={report.random_baseline.accuracy:.3f} save_rate={report.random_baseline.save_rate:.3f}\n"
        f"  kmf:         save_rate={report.kicker_most_frequent_baseline.save_rate}\n"
        f"  keeper:      save_rate={report.actual_keeper_baseline.save_rate}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
