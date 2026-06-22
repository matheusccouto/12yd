"""Predict shootout kick side for the 2026 World Cup roster.

Slice #9 (Issue #25): the predict slice. Reads the frozen LightGBM
artifact, the WC roster, and the per-kicker penalty history, and writes
one row per WC player with P(L), P(C), P(R).

The slice is re-parameterisable for a different roster (e.g. a
knockout-round subset) via the `--roster` flag without code changes.
The `--target-date` flag controls the prediction date; the default is
`today_utc() + 1 day` so the lookback window always includes all of
today's penalties. Re-runs with the same inputs are idempotent (the
FotMob cache is persistent, the model artifact is frozen, and the
target_date is derived from a date-level function).
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from penalty_pred.client import FotMobClient
from penalty_pred.config import DEFAULT_CACHE_DIR, today_utc
from penalty_pred.features import fetcher_from_client, load_player_history
from penalty_pred.model import load_artifact
from penalty_pred.predict import (
    load_roster,
    predict_roster,
    write_predictions_jsonl,
)

# Default artifact paths — consistent with slices #2, #3, #5, #6, #7, #8.
DEFAULT_ROSTER_PATH: Path = Path("output/wc2026_roster.jsonl")
DEFAULT_PLAYER_HISTORY_PATH: Path = Path("output/player_history.jsonl")
DEFAULT_MODEL_PATH: Path = Path("output/lightgbm.pkl")
DEFAULT_OUTPUT_PATH: Path = Path("output/predictions.jsonl")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roster",
        type=Path,
        default=DEFAULT_ROSTER_PATH,
        help="Path to wc2026_roster.jsonl (default: output/wc2026_roster.jsonl).",
    )
    parser.add_argument(
        "--player-history",
        type=Path,
        default=DEFAULT_PLAYER_HISTORY_PATH,
        help="Path to player_history.jsonl (default: output/player_history.jsonl).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the frozen model artifact (default: output/lightgbm.pkl).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to write predictions.jsonl (default: output/predictions.jsonl).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(DEFAULT_CACHE_DIR),
        help="Persistent disk cache directory for player page fetches.",
    )
    parser.add_argument(
        "--target-date",
        default="",
        help=(
            "ISO 8601 date (YYYY-MM-DD) for the prediction target. The "
            "feature builder filters the kicker's history to before this "
            "date. Default: today UTC + 1 day (includes all of today's "
            "penalties). For a future shootout, set this to the shootout "
            "date; the lookback window is `[target_date - lookback_years, "
            "target_date]` (see `player_history.compute_lookback_window`)."
        ),
    )
    args = parser.parse_args()

    if not args.roster.exists():
        print(f"error: {args.roster} not found", file=sys.stderr)
        return 1
    if not args.player_history.exists():
        print(f"error: {args.player_history} not found", file=sys.stderr)
        return 1
    if not args.model.exists():
        print(f"error: {args.model} not found", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Target date: default to (today + 1 day) so all of today's penalties
    # are included. The +1 day puts the strict `<` filter's cutoff at
    # midnight tomorrow, which is unambiguously in the future.
    if args.target_date:
        target = date.fromisoformat(args.target_date)
    else:
        target = today_utc() + timedelta(days=1)
    target_iso = datetime(target.year, target.month, target.day, tzinfo=UTC).isoformat()

    # Load
    roster = load_roster(args.roster)
    history = load_player_history(args.player_history)
    art = load_artifact(args.model)
    model = art["model"]
    print(
        f"Loaded {len(roster)} players from {args.roster}; "
        f"{len(history)} unique kickers in {args.player_history}; "
        f"model_kind={art['model_kind']}; target_date={target.isoformat()}."
    )

    # Metadata fetcher (cache-warm from the player-history slice; the
    # 1243-player roster was fully fetched then). A cold cache would
    # take ~3h; on a warm cache, the in-process cost is dominated by
    # JSON parsing.
    client = FotMobClient(cache_dir=args.cache_dir)
    metadata_fetcher = fetcher_from_client(client)

    # Predict
    predictions = predict_roster(model, roster, history, metadata_fetcher, target_iso)

    # Write
    n = write_predictions_jsonl(args.output, predictions)
    n_with_history = sum(1 for r in predictions if r.kicking_foot != "Unknown")
    n_no_history = n - n_with_history
    print(
        f"Wrote {n} predictions to {args.output}.\n"
        f"  With penalty history: {n_with_history}/{n}\n"
        f"  No penalty history:   {n_no_history}/{n} (model sees the prior)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
