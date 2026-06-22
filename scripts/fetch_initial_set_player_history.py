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
import json
import sys
import time
from dataclasses import asdict
from datetime import date
from pathlib import Path

from penalty_pred.client import FotMobClient
from penalty_pred.config import DEFAULT_CACHE_DIR, today_utc
from penalty_pred.player_history import (
    MissingKicker,
    fetch_all_initial_set_penalty_history,
    iter_initial_set_kickers,
    write_missing_jsonl,
)

# Default artifact paths — consistent with slices #2 and #3.
DEFAULT_SHOOTOUT_KICKS_PATH: Path = Path("output/shootout_kicks.jsonl")
DEFAULT_ROSTER_PATH: Path = Path("output/wc2026_roster.jsonl")
DEFAULT_OUTPUT_PATH: Path = Path("output/player_history.jsonl")
DEFAULT_MISSING_PATH: Path = Path("output/missing_history.jsonl")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shootout-kicks",
        type=Path,
        default=DEFAULT_SHOOTOUT_KICKS_PATH,
        help="Path to shootout_kicks.jsonl (default: output/shootout_kicks.jsonl).",
    )
    parser.add_argument(
        "--roster",
        type=Path,
        default=DEFAULT_ROSTER_PATH,
        help="Path to wc2026_roster.jsonl (default: output/wc2026_roster.jsonl).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to write the per-kicker JSONL artifact (default: output/player_history.jsonl).",
    )
    parser.add_argument(
        "--missing",
        type=Path,
        default=DEFAULT_MISSING_PATH,
        help="Path to write the missing-kicker JSONL artifact "
        "(default: output/missing_history.jsonl).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(DEFAULT_CACHE_DIR),
        help="Persistent disk cache directory.",
    )
    parser.add_argument(
        "--target-date",
        default="",
        help="ISO 8601 date (YYYY-MM-DD) for the lookback window end. "
        "Default: today's date in UTC (per-kicker fetcher's lookback "
        "extends to the present).",
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
    client = FotMobClient(cache_dir=args.cache_dir)

    initial_set = list(iter_initial_set_kickers(args.shootout_kicks, args.roster))
    print(
        f"Initial Set: {len(initial_set)} unique kickers "
        f"({args.shootout_kicks} + {args.roster}, deduped by player_id)"
    )

    # Stream results to disk as we go. A full run on a cold cache takes
    # ~3h for 1327 kickers; we don't want to lose that to a Ctrl-C or
    # process kill. The rows are written in initial-set order; the missing
    # list is rewritten at the end from the in-memory result cache.
    n_rows_written = 0
    results: list = []
    progress_every = 25
    t0 = time.monotonic()
    with args.output.open("w", encoding="utf-8") as out_f:
        for i, result in enumerate(
            fetch_all_initial_set_penalty_history(client, initial_set, target_date=target),
            start=1,
        ):
            results.append(result)
            for row in result.rows:
                out_f.write(json.dumps(asdict(row), ensure_ascii=False))
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
    write_missing_jsonl(args.missing, missing)

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
