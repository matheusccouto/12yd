"""Build `output/training_table.jsonl` from the shootout kicks and per-kicker history.

Slice #6 (Issue #22): the feature builder. For every row in
`output/shootout_kicks.jsonl`, joins the corresponding rows from
`output/player_history.jsonl` (filtered to before the target kick's
match date) and the kicker's player page (cached on disk; no fresh
network needed for the in-Initial-Set kickers) to produce one
`TrainingTableRow` per target kick. The output is the supervised
training table the model slice consumes.

Re-runs are idempotent: the input JSONLs and the FotMob cache
determine the output byte-for-byte. The script is re-parameterisable
on the input paths and the output path so the model slice can rebuild
the table from a different JSONL set without code changes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from penalty_pred.artifacts import Artifacts
from penalty_pred.client import FotMobClient
from penalty_pred.features import (
    build_training_table,
    fetcher_from_client,
    load_player_history,
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
        "--player-history",
        type=Path,
        default=art.player_history,
        help=f"Path to player_history.jsonl (default: {art.player_history}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=art.training_table,
        help=f"Path to write training_table.jsonl (default: {art.training_table}).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=art.cache_dir,
        help=f"Persistent disk cache directory for player page fetches (default: {art.cache_dir}).",
    )
    args = parser.parse_args()

    if not args.shootout_kicks.exists():
        print(f"error: {args.shootout_kicks} not found", file=sys.stderr)
        return 1
    if not args.player_history.exists():
        print(f"error: {args.player_history} not found", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)

    shootout_kicks = art.read_shootout_kicks(path=args.shootout_kicks)
    player_history = load_player_history(args.player_history)
    print(
        f"Read {len(shootout_kicks)} target kicks from {args.shootout_kicks}; "
        f"{len(player_history)} unique kickers in {args.player_history}."
    )

    client = FotMobClient(cache_dir=args.cache_dir)
    metadata_fetcher = fetcher_from_client(client)
    rows = build_training_table(shootout_kicks, player_history, metadata_fetcher)
    n = art.write_training_table(rows, path=args.output)
    print(f"Wrote {n} feature rows to {args.output}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
