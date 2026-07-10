"""Fetch every WC 2026 roster player's penalty history and write it to JSONL.

PRD-v5: The Initial Set is the WC 2026 roster only (no shootout kicks).
Reads wc2026_roster.jsonl, fans out per-kicker fetches, writes
player_history.jsonl and missing_history.jsonl.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

from twelveyards.artifacts import Artifacts
from twelveyards.client import FotMobClient
from twelveyards.config import LOOKBACK_WINDOW_YEARS, SCRAPE_FLOOR, today_utc
from twelveyards.initial_set import (
    MissingKicker,
    fetch_all_initial_set_penalty_history_parallel,
    iter_initial_set_kickers,
)


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
        "--output",
        type=Path,
        default=art.player_history,
        help=f"Path to write the per-kicker JSONL artifact (default: {art.player_history}).",
    )
    parser.add_argument(
        "--missing",
        type=Path,
        default=art.missing_history,
        help=f"Path to write the missing-kicker JSONL artifact (default: {art.missing_history}).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=art.cache_dir,
        help=f"Path to the FotMob HTTP cache (default: {art.cache_dir}).",
    )
    parser.add_argument(
        "--target-date",
        default=None,
        help="Upper bound of the lookback window in ISO 8601 (default: today UTC).",
    )
    parser.add_argument(
        "--lookback-years",
        type=int,
        default=LOOKBACK_WINDOW_YEARS,
        help=f"Years of history to look back per kicker (default: {LOOKBACK_WINDOW_YEARS}).",
    )
    parser.add_argument(
        "--history-floor",
        default=SCRAPE_FLOOR.isoformat(),
        help=f"Hard lower bound on history dates, ISO 8601 (default: {SCRAPE_FLOOR.isoformat()}).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=12,
        help="Max concurrent per-kicker fetches (default: 12).",
    )
    args = parser.parse_args()

    if not args.roster.exists():
        print(f"error: {args.roster} not found", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.missing.parent.mkdir(parents=True, exist_ok=True)

    target = date.fromisoformat(args.target_date) if args.target_date else today_utc()
    history_floor = date.fromisoformat(args.history_floor)
    client = FotMobClient(cache_dir=args.cache_dir)

    art = Artifacts()
    roster = art.read_roster(path=args.roster)
    initial_set = list(iter_initial_set_kickers(roster))
    print(
        f"Initial Set: {len(initial_set)} unique kickers from {args.roster}",
    )

    art = Artifacts()
    n_rows_written = 0
    results: list = []
    progress_every = 25
    t0 = time.monotonic()
    with args.output.open("w", encoding="utf-8") as out_f:
        for i, result in enumerate(
            fetch_all_initial_set_penalty_history_parallel(
                client,
                initial_set,
                target_date=target,
                lookback_years=args.lookback_years,
                history_floor=history_floor,
                max_workers=args.max_workers,
            ),
            start=1,
        ):
            results.append(result)
            for row in result.rows:
                out_f.write(art.serialize_row(row))
                out_f.write("\n")
                n_rows_written += 1
            out_f.flush()
            if i % progress_every == 0 or i == len(initial_set):
                elapsed = time.monotonic() - t0
                n_missing = sum(1 for r in results if not r.rows)
                n_err = sum(1 for r in results if r.error)
                print(
                    f"  [{i}/{len(initial_set)}] elapsed={elapsed:.0f}s "
                    f"rows={n_rows_written} missing={n_missing} errors={n_err}",
                    flush=True,
                )

    missing = [
        MissingKicker(
            player_id=r.kicker.player_id,
            player_name=r.kicker.player_name,
            team_id=r.kicker.team_id,
            team_name=r.kicker.team_name,
        )
        for r in results
        if not r.rows
    ]
    art.write_missing_history(missing, path=args.missing)

    n_with_rows = len(results) - len(missing)
    pct = 100.0 * n_with_rows / len(results) if results else 0.0
    n_errored = sum(1 for r in results if r.error)
    print(
        f"Wrote {n_rows_written} penalty rows to {args.output} across {n_with_rows}/{len(results)} "
        f"kickers ({pct:.1f}%); {len(missing)} kickers have zero penalty rows in the lookback "
        f"window (see {args.missing}).",
    )
    if n_errored:
        print(f"  ({n_errored} kicker fetches errored; reported in {args.missing}.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
