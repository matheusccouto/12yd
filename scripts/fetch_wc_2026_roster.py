"""Fetch the 2026 World Cup squad and write to data/wc2026_roster.jsonl."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from twelveyards.artifacts import Artifacts
from twelveyards.fotmob.client import FotMobClient
from twelveyards.pipeline import fetch_and_write_roster


def main() -> int:
    art = Artifacts()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=art.roster,
        help=f"JSONL output path (default: {art.roster}).",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    client = FotMobClient()
    n = fetch_and_write_roster(client, args.output)
    print(f"Wrote {n} unique players to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
