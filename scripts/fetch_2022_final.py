"""Fetch the 2022 FIFA World Cup Final (match 3370572) and write its 8 shootout kicks.

This is the foundational scraper slice (Issue #17): it exercises the HTTP client
(ETag + gzip + disk cache), the BuildId discovery, the two-segment match route,
and the shootout kick extractor — end-to-end on one match.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from penalty_pred.artifacts import Artifacts
from penalty_pred.client import FotMobClient
from penalty_pred.shootouts import extract_shootout_kicks

# 2022 FIFA World Cup Final: Argentina vs France, matchId 3370572.
# Slug segments are taken from the match pageUrl; seo/h2h are stable for the lifetime
# of the slug.
DEFAULT_SEO = "argentina-vs-france"
DEFAULT_H2H = "1hox8a"
DEFAULT_MATCH_ID = 3370572


def main() -> int:
    art = Artifacts()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=art.shootout_kicks,
        help=f"Path to write the JSONL artifact (default: {art.shootout_kicks}).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=art.cache_dir,
        help=f"Persistent disk cache directory (default: {art.cache_dir}).",
    )
    parser.add_argument("--match-id", type=int, default=DEFAULT_MATCH_ID)
    parser.add_argument("--seo", default=DEFAULT_SEO)
    parser.add_argument("--h2h", default=DEFAULT_H2H)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    client = FotMobClient(cache_dir=args.cache_dir)
    data = client.get(f"matches/{args.seo}/{args.h2h}")
    kicks = extract_shootout_kicks(data)
    n = Artifacts().write_shootout_kicks(kicks, path=args.output)
    print(f"Wrote {n} shootout kicks to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
