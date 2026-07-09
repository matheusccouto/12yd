"""Fetch one kicker's penalty history and write it to JSONL.

Slice #4 (Issue #20): the per-kicker penalty history fetcher. The default
test case is Lionel Messi (FotMob id 30981) with target date 2022-12-18
(the 2022 WC Final). Re-running is a no-op thanks to the persistent
FotMob cache; the script is idempotent (overwrites the JSONL with the
same content).

The scraper is the two-level data graph in action:
  - Initial Set: one player (the player_id we pass in).
  - Derived History: every penalty the player took in the lookback window,
    drawn from per-match shotmaps across the (team, season) stints in
    the player's career history.
No further fetches originate from the Derived History — a scraper that
fans out from there is a bug, not a feature.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from twelveyards.artifacts import Artifacts
from twelveyards.client import FotMobClient
from twelveyards.player_history import fetch_player_penalty_history

# Default test case (PRD story 7): Lionel Messi, target date 2022-12-18.
DEFAULT_PLAYER_ID: int = 30981
DEFAULT_PLAYER_SLUG: str = "lionel-messi"
DEFAULT_TARGET_DATE: str = "2022-12-18"


def main() -> int:
    art = Artifacts()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--player-id",
        type=int,
        default=DEFAULT_PLAYER_ID,
        help="FotMob player id (default: 30981 = Lionel Messi).",
    )
    parser.add_argument(
        "--player-slug",
        default=DEFAULT_PLAYER_SLUG,
        help="URL-friendly player name (default: 'lionel-messi').",
    )
    parser.add_argument(
        "--target-date",
        default=DEFAULT_TARGET_DATE,
        help="ISO 8601 date (YYYY-MM-DD) for the lookback window end "
        "(default: 2022-12-18, the 2022 WC Final).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=art.player_history,
        help=f"Path to write the JSONL artifact (default: {art.player_history}).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=art.cache_dir,
        help=f"Persistent disk cache directory (default: {art.cache_dir}).",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    client = FotMobClient(cache_dir=args.cache_dir)
    target = date.fromisoformat(args.target_date)
    rows = fetch_player_penalty_history(
        client,
        player_id=args.player_id,
        player_slug=args.player_slug,
        target_date=target,
    )
    n = Artifacts().write_player_history(rows, path=args.output)
    print(f"Wrote {n} penalty rows to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
