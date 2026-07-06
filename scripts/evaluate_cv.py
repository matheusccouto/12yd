"""Leave-one-group-out cross-validation for the penalty shootout classifier.

Issue #45: the published `model/metrics.json` reports a single 2026+
holdout. The v3 holdout was 28 rows (the 2026 WC kicks); at n=28 the
standard error on save rate is ~0.09, so the difference between the
model's 0.464 and the random baseline's 0.405 is well within the
noise. The v4 holdout (Issue #51) is 226 rows; at n=226 the SE on
save rate is ~0.032, so the v4 model's 0.345 vs random's 0.437 is a
reliable 2.9-SE gap. A leave-one-tournament-out cross-validation
gives 6 folds (one per `tournament_name`) on the v3 dataset for an
aggregate holdout of ~150-180 kicks with an aggregate SE of ~0.04;
the v4 dataset (8 folds, one per `tournament_name` across 8
national-team + club tournaments) brings the aggregate to 437 rows
with an aggregate SE of 0.022 — a 6x tighter claim.

The script reads `output/training_table.jsonl`, runs a
leave-one-tournament-out CV for both the LightGBM (slice #8) and the
logreg baseline (slice #7), and writes:

- `output/cv_metrics.json` — the standalone CV artifact with
  per-fold metrics, the aggregate summary, and the `group_by`
  attribute used.
- `output/metrics.json` — the existing metrics report, with the
  LightGBM's CV block added to the `cv` field (so the published
  card has a CV section without re-running the train slice).

The model is not retrained: the CV fits a fresh model per fold, but
the deployment artifact (`output/lightgbm.pkl`) is unchanged. The
script is deterministic: same inputs + same random seed + same
grouping → same output.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from penalty_pred.artifacts import Artifacts
from penalty_pred.evaluate import cross_validate
from penalty_pred.model import fit_lightgbm as _fit_lightgbm
from penalty_pred.model import fit_logistic_regression as _fit_logreg
from penalty_pred.model import (
    is_on_target_by_key,
    load_training_table,
)


def _lightgbm_factory(matrix):
    """Fit the LightGBM on `matrix` and return the fitted wrapper.

    The factory shape is `matrix -> fitted_model`; `cross_validate`
    does the per-fold orchestration. The LightGBM hyperparameters
    come from `LIGHTGBM_DEFAULTS` in `model.py`; the inverse-frequency
    class weights are computed from the training fold by
    `fit_lightgbm` (so each LOTO fold gets weights appropriate to
    its training rows, not the full table).
    """
    return _fit_lightgbm(matrix)


def _logreg_factory(matrix):
    """Fit the logreg baseline on `matrix` and return the fitted pipeline."""
    return _fit_logreg(matrix)


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
        "--cv-output",
        type=Path,
        default=art.cv_metrics,
        help=f"Path to write the standalone CV artifact (default: {art.cv_metrics}).",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=art.metrics,
        help=(
            "Path to the existing metrics.json to update with the "
            f"CV block (default: {art.metrics}). Skipped if absent."
        ),
    )
    parser.add_argument(
        "--group-by",
        default="tournament_name",
        help=(
            "Row attribute to group on for LOTO CV (default: "
            "'tournament_name'). Any attribute on TrainingRow works."
        ),
    )
    parser.add_argument(
        "--min-fold-size",
        type=int,
        default=1,
        help=(
            "Skip folds with fewer than this many rows (default: 1). "
            "Asian Cup has 0 rows in the current data; the default "
            "skips it from the fold list and records it in `skipped`."
        ),
    )
    args = parser.parse_args()

    if not args.training_table.exists():
        print(f"error: {args.training_table} not found", file=sys.stderr)
        return 1

    shootout_kicks = art.read_shootout_kicks()
    rows = load_training_table(
        args.training_table,
        is_on_target_by_key=is_on_target_by_key(shootout_kicks),
    )
    print(
        f"Loaded {len(rows)} rows from {args.training_table}; "
        f"group_by={args.group_by!r}, min_fold_size={args.min_fold_size}."
    )

    lgb_cv = cross_validate(
        _lightgbm_factory,
        rows,
        group_by=args.group_by,
        min_fold_size=args.min_fold_size,
    )
    logreg_cv = cross_validate(
        _logreg_factory,
        rows,
        group_by=args.group_by,
        min_fold_size=args.min_fold_size,
    )

    art.write_cv(lgb_cv, path=args.cv_output)
    print(f"\nWrote {args.cv_output} (lightgbm LOTO CV).")
    print(
        f"  folds: {len(lgb_cv.folds)}, aggregate: "
        f"save_rate={lgb_cv.aggregate_save_rate:.3f} "
        f"(se={lgb_cv.se_save_rate:.3f}) "
        f"log_loss={lgb_cv.aggregate_log_loss:.3f} "
        f"accuracy={lgb_cv.aggregate_accuracy:.3f} "
        f"n_total={lgb_cv.n_total}"
    )
    if lgb_cv.skipped:
        print(f"  skipped: {lgb_cv.skipped}")

    if args.metrics_output.exists():
        existing = art.read_metrics(path=args.metrics_output)
        updated = replace(existing, cv=lgb_cv)
        art.write_metrics(updated, path=args.metrics_output)
        print(f"Updated {args.metrics_output} with the CV block.")
    else:
        print(
            f"note: {args.metrics_output} not found; "
            "skipping the metrics.json update. Run scripts/train_lightgbm.py "
            "first to seed the report."
        )

    print("\nLogreg baseline LOTO CV (for context):")
    print(
        f"  folds: {len(logreg_cv.folds)}, aggregate: "
        f"save_rate={logreg_cv.aggregate_save_rate:.3f} "
        f"log_loss={logreg_cv.aggregate_log_loss:.3f} "
        f"accuracy={logreg_cv.aggregate_accuracy:.3f} "
        f"n_total={logreg_cv.n_total}"
    )

    print("\nPer-fold (lightgbm, sorted largest first):")
    print(
        f"  {'tournament':<48s} {'n_train':>8s} {'n_holdout':>10s} {'save':>7s} {'random':>7s} {'log_loss':>9s} {'acc':>6s}"
    )
    for fold in lgb_cv.folds:
        print(
            f"  {fold.name:<48s} {fold.n_train:>8d} {fold.n_holdout:>10d} "
            f"{fold.save_rate:>7.3f} {fold.random_save_rate:>7.3f} "
            f"{fold.log_loss:>9.3f} {fold.accuracy:>6.3f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
