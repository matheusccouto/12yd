"""Fetch every Initial Set Kicker's penalty history and write it to JSONL.

Slice #5 (Issue #21): extend slice #4 to the union of Training and Prediction
Initial Sets. The Training Initial Set is read from
`output/shootout_kicks.jsonl` (slice #2, Issue #19); the Prediction Initial
Set is read from `output/wc2026_roster.jsonl` (slice #3, Issue #18). The
two sets are deduplicated by `player_id` (training first, roster second),
and the per-kicker fetcher from slice #4 (Issue #20) is fanned out across
the deduped union.

The script writes two artifacts:
  - `output/player_history.jsonl` — every penalty kick from every kicker
    in the Initial Set, in the same schema as slice #4.
  - `output/missing_history.jsonl` — the kickers (with `player_id`,
    `player_name`, `team_id`, `team_name`) that yielded zero penalty rows
    in the lookback window. Downstream slices (#22 features, #25
    predictions) decide whether to skip them or use a prior-based fallback.

Re-runs are cache-hit-dominated: the per-kicker fetcher serves 304 responses
from disk and the BuildId is process-global. The script is idempotent: a
second run with the same inputs produces byte-identical output.

The lookback window ends "now" (config: `today_utc()`) so the
`player_history.jsonl` covers every penalty the player took up to the
present — the same window the feature builder (#22) will use to compute
A1/A2/A3/A4 for any future target kick.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

from penalty_pred.artifacts import Artifacts
from penalty_pred.client import FotMobClient
from penalty_pred.config import HISTORY_FLOOR, LOOKBACK_WINDOW_YEARS, today_utc
from penalty_pred.initial_set import (
    MissingKicker,
    fetch_all_initial_set_penalty_history_parallel,
    iter_initial_set_kickers,
)


def main() -> int:
    art = Artifacts()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shootout-kicks",
        type=Path,
        default=art.shootout_kicks,
        help=f"Path to shootout_kicks.jsonl (default: {art.shootout_kicks}).",
    )
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
        help=(f"Path to write the missing-kicker JSONL artifact (default: {art.missing_history})."),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=art.cache_dir,
        help=f"Persistent disk cache directory (default: {art.cache_dir}).",
    )
    parser.add_argument(
        "--target-date",
        default="",
        help="ISO 8601 date (YYYY-MM-DD) for the lookback window end. "
        "Default: today's date in UTC (per-kicker fetcher's lookback "
        "extends to the present).",
    )
    parser.add_argument(
        "--lookback-years",
        type=int,
        default=LOOKBACK_WINDOW_YEARS,
        help=f"Years of history to look back per kicker (default: {LOOKBACK_WINDOW_YEARS}).",
    )
    parser.add_argument(
        "--history-floor",
        default=HISTORY_FLOOR.isoformat(),
        help=f"Hard lower bound on history dates, ISO 8601 (default: {HISTORY_FLOOR.isoformat()}).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=12,
        help="Max concurrent per-kicker fetches (default: 12). The cold-cache "
        "all-Initial-Set run drops from ~3h serial to ~15-20 min parallel at "
        "this value. Set to 1 to disable parallelism.",
    )
    args = parser.parse_args()

    if not args.shootout_kicks.exists():
        print(f"error: {args.shootout_kicks} not found", file=sys.stderr)
        return 1
    if not args.roster.exists():
        print(f"error: {args.roster} not found", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.missing.parent.mkdir(parents=True, exist_ok=True)

    target = date.fromisoformat(args.target_date) if args.target_date else today_utc()
    history_floor = date.fromisoformat(args.history_floor)
    client = FotMobClient(cache_dir=args.cache_dir)

    # Read the JSONL once into typed collections; the Initial Set
    # assembly operates on the rows, not on the disk. This puts the
    # JSONL re-parse in `Artifacts` (one place) and lets the per-kicker
    # orchestrator take `Iterable[InitialSetKicker]`.
    art = Artifacts()
    shootout_kicks = art.read_shootout_kicks(path=args.shootout_kicks)
    roster = art.read_roster(path=args.roster)
    initial_set = list(iter_initial_set_kickers(shootout_kicks, roster))
    print(
        f"Initial Set: {len(initial_set)} unique kickers "
        f"({args.shootout_kicks} + {args.roster}, deduped by player_id)"
    )

    art = Artifacts()
    # Stream results to disk as we go. A full run on a cold cache takes
    # ~3h for 1327 kickers (or ~15-20 min with `--max-workers 12`); we don't
    # want to lose that to a Ctrl-C or process kill. The rows are written
    # in initial-set order; the missing list is rewritten at the end from
    # the in-memory result cache.
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
        f"window (see {args.missing})."
    )
    if n_errored:
        print(f"  ({n_errored} kicker fetches errored; reported in {args.missing}.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
