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

from penalty_pred.artifacts import Artifacts
from penalty_pred.client import FotMobClient
from penalty_pred.config import today_utc
from penalty_pred.features import fetcher_from_client, load_player_history
from penalty_pred.model import load_artifact
from penalty_pred.predict import count_kickers_with_history, load_roster, predict_roster


def main() -> int:
    art = Artifacts()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roster",
        type=Path,
        default=art.roster,
        help=f"Path to wc2026_roster.jsonl (default: {art.roster}).",
    )
    parser.add_argument(
        "--player-history",
        type=Path,
        default=art.player_history,
        help=f"Path to player_history.jsonl (default: {art.player_history}).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=art.lightgbm_model,
        help=f"Path to the frozen model artifact (default: {art.lightgbm_model}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=art.predictions,
        help=f"Path to write predictions.jsonl (default: {art.predictions}).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=art.cache_dir,
        help=f"Persistent disk cache directory for player page fetches (default: {art.cache_dir}).",
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
    art_model = load_artifact(args.model)
    model = art_model["model"]
    print(
        f"Loaded {len(roster)} players from {args.roster}; "
        f"{len(history)} unique kickers in {args.player_history}; "
        f"model_kind={art_model['model_kind']}; target_date={target.isoformat()}."
    )

    # Metadata fetcher (cache-warm from the player-history slice; the
    # 1247-player WC 2026 roster was fully fetched then). A cold cache
    # would take ~15-20 min with the v4 parallel orchestrator
    # (`--max-workers 12`); on a warm cache, the in-process cost is
    # dominated by JSON parsing.
    client = FotMobClient(cache_dir=args.cache_dir)
    metadata_fetcher = fetcher_from_client(client)

    # Predict
    predictions = predict_roster(model, roster, history, metadata_fetcher, target_iso)

    # Write
    n = art.write_predictions(predictions, path=args.output)
    # v3 (Issue #36): `kicking_foot` is now the declared preferred foot
    # (one of left/right/both/""), never the v2 `"Unknown"` sentinel, so
    # the old `r.kicking_foot != "Unknown"` check was always True.
    # Count from the `player_history` dict instead — a kicker has
    # history iff the dict has a non-empty list for their `player_id`.
    # The JSONL groups by `kicker_id`, and `load_player_history` reads
    # only non-empty lists, so key existence ≡ `len > 0` for this data
    # layer (the helper's `len > 0` is a defensive guard).
    n_with_history = count_kickers_with_history(roster, history)
    n_no_history = n - n_with_history
    print(
        f"Wrote {n} predictions to {args.output}.\n"
        f"  With penalty history: {n_with_history}/{n}\n"
        f"  No penalty history:   {n_no_history}/{n} (model sees the prior)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
