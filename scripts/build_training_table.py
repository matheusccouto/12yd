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

from penalty_pred.client import FotMobClient
from penalty_pred.config import DEFAULT_CACHE_DIR
from penalty_pred.features import (
    build_training_table,
    fetcher_from_client,
    load_player_history,
    write_jsonl,
)
from penalty_pred.shootouts import read_jsonl as read_shootout_kicks_jsonl

# Default artifact paths — consistent with slices #2, #3, and #5.
DEFAULT_SHOOTOUT_KICKS_PATH: Path = Path("output/shootout_kicks.jsonl")
DEFAULT_PLAYER_HISTORY_PATH: Path = Path("output/player_history.jsonl")
DEFAULT_OUTPUT_PATH: Path = Path("output/training_table.jsonl")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shootout-kicks",
        type=Path,
        default=DEFAULT_SHOOTOUT_KICKS_PATH,
        help="Path to shootout_kicks.jsonl (default: output/shootout_kicks.jsonl).",
    )
    parser.add_argument(
        "--player-history",
        type=Path,
        default=DEFAULT_PLAYER_HISTORY_PATH,
        help="Path to player_history.jsonl (default: output/player_history.jsonl).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to write training_table.jsonl (default: output/training_table.jsonl).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(DEFAULT_CACHE_DIR),
        help="Persistent disk cache directory for player page fetches.",
    )
    args = parser.parse_args()

    if not args.shootout_kicks.exists():
        print(f"error: {args.shootout_kicks} not found", file=sys.stderr)
        return 1
    if not args.player_history.exists():
        print(f"error: {args.player_history} not found", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)

    shootout_kicks = read_shootout_kicks_jsonl(args.shootout_kicks)
    player_history = load_player_history(args.player_history)
    print(
        f"Read {len(shootout_kicks)} target kicks from {args.shootout_kicks}; "
        f"{len(player_history)} unique kickers in {args.player_history}."
    )

    client = FotMobClient(cache_dir=args.cache_dir)
    metadata_fetcher = fetcher_from_client(client)
    rows = build_training_table(shootout_kicks, player_history, metadata_fetcher)
    n = write_jsonl(args.output, rows)
    print(f"Wrote {n} feature rows to {args.output}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
