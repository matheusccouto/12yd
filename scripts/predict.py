"""Fit TabPFN and write per-roster-kicker predictions to data/predictions.jsonl."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from twelveyards.artifacts import Artifacts
from twelveyards.config import today_utc
from twelveyards.pipeline import predict


def main() -> int:
    """Fit TabPFN on player history and write predictions."""
    art = Artifacts()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", type=Path, default=art.roster)
    parser.add_argument("--player-history", type=Path, default=art.player_history)
    parser.add_argument("--output", type=Path, default=art.predictions)
    parser.add_argument("--target-date", default="")
    args = parser.parse_args()

    if not args.roster.exists():
        return 1
    if not args.player_history.exists():
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)

    target = date.fromisoformat(args.target_date) if args.target_date else today_utc()

    _n_preds, _n_no_history = predict(
        art.fotmob_client(), args.roster,
        args.player_history, args.output, target_date=target,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
