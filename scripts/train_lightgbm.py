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
- `output/lightgbm.pkl` — the LightGBM model retrained on ALL
  `training_table.jsonl` rows (no holdout) — the frozen deployment
  artifact the predict slice (#25) will load.

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

from penalty_pred.evaluate import evaluate_predictions, write_metrics_json
from penalty_pred.model import (
    CATEGORICAL_FEATURES,
    CLASSES,
    FEATURE_COLUMNS,
    HOLDOUT_CUTOFF_DATE,
    LIGHTGBM_DEFAULTS,
    LOGREG_DEFAULTS,
    NUMERIC_FEATURES,
    RANDOM_SEED,
    build_feature_matrix,
    fit_lightgbm,
    fit_logistic_regression,
    load_training_table,
    predict_proba,
    save_artifact,
    temporal_split,
)

# Default artifact paths — consistent with slices #2, #3, #5, #6, #7.
DEFAULT_TRAINING_TABLE_PATH: Path = Path("output/training_table.jsonl")
DEFAULT_MODEL_PATH: Path = Path("output/lightgbm.pkl")
DEFAULT_METRICS_PATH: Path = Path("output/metrics.json")


def _train_logreg_on_fold(
    train_matrix,
    *,
    C: float,
    class_weight: str,
) -> object:
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-table",
        type=Path,
        default=DEFAULT_TRAINING_TABLE_PATH,
        help="Path to training_table.jsonl (default: output/training_table.jsonl).",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to write the frozen LightGBM artifact (default: output/lightgbm.pkl).",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help="Path to write the metrics JSON (default: output/metrics.json).",
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

    rows = load_training_table(args.training_table)
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

    lgb_probs = predict_proba(lgb, holdout_matrix)
    baseline_probs = predict_proba(baseline, holdout_matrix)
    report = evaluate_predictions(lgb_probs, holdout_rows, baseline_probs=baseline_probs)
    report = replace(
        report,
        n_train=len(train_rows),
        holdout_cutoff_date=args.holdout_cutoff,
    )
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

    write_metrics_json(args.metrics_output, report)
    print(
        f"Wrote {args.metrics_output} (held-out metrics).\n"
        f"  model (lightgbm): log_loss={report.model.log_loss:.3f} "
        f"acc={report.model.accuracy:.3f} save_rate={report.model.save_rate:.3f}\n"
        f"  baseline (logreg): log_loss={report.baseline.log_loss:.3f} "
        f"acc={report.baseline.accuracy:.3f} save_rate={report.baseline.save_rate:.3f}\n"
        f"  random:           log_loss={report.random_baseline.log_loss:.3f} "
        f"acc={report.random_baseline.accuracy:.3f} "
        f"save_rate={report.random_baseline.save_rate:.3f}\n"
        f"  kmf:              save_rate={report.kicker_most_frequent_baseline.save_rate}\n"
        f"  keeper:           save_rate={report.actual_keeper_baseline.save_rate}"
    )

    full_matrix = build_feature_matrix(rows)
    frozen = fit_lightgbm(
        full_matrix,
        params=lgb_params,
        random_state=RANDOM_SEED,
    )
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    save_artifact(
        args.model_output,
        frozen,
        FEATURE_COLUMNS,
        model_kind="lightgbm",
        params={
            **lgb_params,
            "class_weight": "inverse_frequency",
            "random_state": RANDOM_SEED,
            "n_train_rows": len(rows),
        },
    )
    print(
        f"\nFroze LightGBM on all {len(rows)} rows → {args.model_output}.\n"
        f"  categorical_features={list(CATEGORICAL_FEATURES)}\n"
        f"  numeric_features={list(NUMERIC_FEATURES)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
