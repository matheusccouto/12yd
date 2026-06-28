"""Fetch the 2026 FIFA World Cup squad list and write it to JSONL.

Slice #3 (Issue #18): the 2026 WC is in progress as of 2026-06-22
(group stage, no shootouts yet), so we scrape the squad list now and
the penalty history for every squad player (slice #5), so the dashboard
can produce predictions the moment a knockout round is imminent.

The slice:

1. Fetches the WC 2026 league fixtures (FotMob leagueId 77, season 2026).
2. Fetches each match's `__next/data` JSON.
3. Extracts the registered players from `pageProps.content.lineup.
   {homeTeam,awayTeam}.{starters,subs}`.
4. Stamps `team_id` and `team_name` from the match fixture (NOT from
   the player's `primaryTeamId`, which is the club side).
5. Deduplicates by `player_id` (a player may appear across multiple
   group-stage matches with the same team_id — we keep the first).
6. Writes the unique squad list to `output/wc2026_roster.jsonl`.

Re-runs are cache-hit-dominated: the FotMob client serves 304 responses
from disk and the BuildId is process-global. The script is idempotent:
re-running overwrites the JSONL with the same content.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from penalty_pred.artifacts import Artifacts
from penalty_pred.client import FotMobClient
from penalty_pred.rosters import fetch_wc_2026_roster
from penalty_pred.tournaments import WC_2026_LEAGUE, WC_2026_SEASON


def main() -> int:
    art = Artifacts()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=art.roster,
        help=f"Path to write the JSONL artifact (default: {art.roster}).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=art.cache_dir,
        help=f"Persistent disk cache directory (default: {art.cache_dir}).",
    )
    parser.add_argument(
        "--league-id",
        type=int,
        default=WC_2026_LEAGUE.league_id,
        help=f"FotMob league id (default: {WC_2026_LEAGUE.league_id} = World Cup).",
    )
    parser.add_argument(
        "--slug",
        default=WC_2026_LEAGUE.slug,
        help=f"League SEO slug (default: {WC_2026_LEAGUE.slug!r}).",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=WC_2026_SEASON,
        help=f"FotMob season year (default: {WC_2026_SEASON}).",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    client = FotMobClient(cache_dir=args.cache_dir)

    # Reuse the rosters module's league abstraction indirectly: we accept
    # CLI overrides so the script can be re-parameterised for a different
    # tournament (e.g. Euro 2024) without code changes.
    from penalty_pred.leagues import League

    league = League(args.league_id, args.slug, "")
    rows = fetch_wc_2026_roster(client, league, args.season)
    n = Artifacts().write_roster(rows, path=args.output)
    print(f"Wrote {n} unique players to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
