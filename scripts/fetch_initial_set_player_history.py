"""Fetch penalty history for every WC 2026 roster player and write to JSONL."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from twelveyards.artifacts import Artifacts
from twelveyards.config import LOOKBACK_WINDOW_YEARS, SCRAPE_FLOOR, today_utc
from twelveyards.fotmob.client import FotMobClient
from twelveyards.pipeline import fetch_and_write_initial_set


def main() -> int:
    art = Artifacts()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", type=Path, default=art.roster)
    parser.add_argument("--output", type=Path, default=art.player_history)
    parser.add_argument("--missing", type=Path, default=art.missing_history)
    parser.add_argument("--cache-dir", type=Path, default=art.cache_dir)
    parser.add_argument("--target-date", default=None)
    parser.add_argument("--lookback-years", type=int, default=LOOKBACK_WINDOW_YEARS)
    parser.add_argument("--history-floor", default=SCRAPE_FLOOR.isoformat())
    parser.add_argument("--max-workers", type=int, default=12)
    args = parser.parse_args()

    if not args.roster.exists():
        print(f"error: {args.roster} not found", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.missing.parent.mkdir(parents=True, exist_ok=True)

    target = date.fromisoformat(args.target_date) if args.target_date else today_utc()
    history_floor = date.fromisoformat(args.history_floor)
    client = FotMobClient(cache_dir=args.cache_dir)

    total, n_rows, n_missing, n_errored = fetch_and_write_initial_set(
        client, args.roster, args.output, args.missing,
        target_date=target, lookback_years=args.lookback_years,
        history_floor=history_floor, max_workers=args.max_workers,
    )
    pct = 100.0 * (total - n_missing) / total if total else 0.0
    print(
        f"Wrote {n_rows} penalty rows to {args.output} across {total - n_missing}/{total} "
        f"kickers ({pct:.1f}%); {n_missing} kickers have zero penalty rows "
        f"(see {args.missing}).",
    )
    if n_errored:
        print(f"  ({n_errored} kicker fetches errored; reported in {args.missing}.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
