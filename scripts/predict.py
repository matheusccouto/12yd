"""Predict penalty shootout sides for the WC 2026 roster.

PRD-v5: TabPFN classifier on player-only features. Reads
wc2026_roster.jsonl and player_history.jsonl from data/, fits TabPFN on
the training set, predicts all roster rows in one batched call, writes
predictions.jsonl.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from twelveyards.artifacts import Artifacts
from twelveyards.config import today_utc
from twelveyards.player_history import PlayerMetadata, extract_player_metadata, fetch_player_data
from twelveyards.predict import load_player_history, load_roster, predict_and_write


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
        "--output",
        type=Path,
        default=art.predictions,
        help=f"Path to write predictions.jsonl (default: {art.predictions}).",
    )
    parser.add_argument(
        "--target-date",
        default="",
        help="ISO 8601 date (YYYY-MM-DD) for the prediction target. Default: today.",
    )
    args = parser.parse_args()

    if not args.roster.exists():
        print(f"error: {args.roster} not found", file=sys.stderr)
        return 1
    if not args.player_history.exists():
        print(f"error: {args.player_history} not found", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)

    target = date.fromisoformat(args.target_date) if args.target_date else today_utc()

    roster = load_roster(args.roster)
    player_history = load_player_history(args.player_history)

    metadata_by_id: dict[int, PlayerMetadata] = {}
    for kicker_id in {*player_history.keys(), *(p.player_id for p in roster)}:
        try:
            payload = fetch_player_data(
                art.fotmob_client(), kicker_id,
            )
            metadata = extract_player_metadata(payload)
            if metadata is not None:
                metadata_by_id[kicker_id] = metadata
        except Exception:
            pass

    rows = predict_and_write(
        roster,
        player_history,
        metadata_by_id,
        output_path=args.output,
        target_date=target,
    )

    n_no_history = sum(1 for r in rows if r.total_penalties == 0)
    print(
        f"Wrote {len(rows)} predictions to {args.output}.\n"
        f"  With penalty history: {len(rows) - n_no_history}/{len(rows)}\n"
        f"  No penalty history:   {n_no_history}/{len(rows)} (model sees the prior)",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
