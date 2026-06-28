"""Fetch every shootout in the 6 in-scope tournaments between 2021-01-01 and today.

Slice #2 (Issue #19): the orchestrator that drives the per-match extractor
across all (league, season) pairs in the current Prediction Window. The
script:

1. Iterates (league_id, season) pairs from `LEAGUE_SEASONS_PREDICT_WINDOW`.
2. Fetches each league's season fixtures, filters to `penalties_short`.
3. Fetches each shootout match's full JSON, runs `extract_shootout_kicks`.
4. Writes a single `shootout_kicks.jsonl` containing every kick.
5. Loads the RSSSF penaltiestour page, counts in-window shootouts, and writes
   `discrepancies.json` if the count diverges from what the scraper found.

Re-runs are cache-hit-dominated: the FotMob client serves 304 responses from
disk and the BuildId is process-global. The script is idempotent: re-running
overwrites the JSONL with the same content.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from penalty_pred.artifacts import Artifacts
from penalty_pred.client import FotMobClient
from penalty_pred.rsssf import load_rsssf_html, parse_rsssf_html
from penalty_pred.shootouts import (
    fetch_all_shootout_kicks_with_skips,
    fetch_all_shootout_match_refs,
)
from penalty_pred.tournaments import LEAGUE_SEASONS_PREDICT_WINDOW
from penalty_pred.validate import validate_shootout_count

# The RSSSF page is the verification oracle (PRD: "RSSSF is a verification
# oracle, never a data source"). Saved next to this script as a fallback if
# the live page is unreachable; the CLI prefers the live page.
DEFAULT_RSSSF_URL: str = "https://www.rsssf.org/miscellaneous/penaltiestour.html"
DEFAULT_RSSSF_FIXTURE: Path = Path("docs/samples/rsssf_penaltiestour.html")


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
        "--discrepancies",
        type=Path,
        default=art.discrepancies,
        help=(
            "Path to write the discrepancies file if the RSSSF count diverges "
            f"(default: {art.discrepancies})."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=art.cache_dir,
        help=f"Persistent disk cache directory (default: {art.cache_dir}).",
    )
    parser.add_argument(
        "--rsssf-fixture",
        type=Path,
        default=DEFAULT_RSSSF_FIXTURE,
        help="Path to a saved RSSSF HTML page (used as a fallback if the live page is unreachable).",
    )
    parser.add_argument(
        "--skip-rsssf",
        action="store_true",
        help="Skip the RSSSF completeness check. Useful for offline re-runs.",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    client = FotMobClient(cache_dir=args.cache_dir)

    # 1. Discover the shootout match references (one fetch per (league, season)).
    refs = fetch_all_shootout_match_refs(client, LEAGUE_SEASONS_PREDICT_WINDOW)
    print(f"Discovered {len(refs)} shootout matches across all in-scope tournaments")

    # 2. Fetch each match and extract its kicks. The orchestrator returns
    # per-match results so we can surface skipped (stale-URL) and
    # processed-but-no-kicks matches in the discrepancies file.
    results = fetch_all_shootout_kicks_with_skips(client, refs)
    all_kicks = [k for r in results for k in r.kicks]
    skipped = [r.ref for r in results if r.skipped]
    no_kicks = [r.ref for r in results if r.no_kicks]
    n_kicks = Artifacts().write_shootout_kicks(all_kicks, path=args.output)
    print(
        f"Wrote {n_kicks} shootout kicks to {args.output} "
        f"({len(skipped)} skipped due to stale (seo, h2h) hashes, "
        f"{len(no_kicks)} processed with no shootout kicks in the shotmap)"
    )

    # 3. Run the RSSSF completeness check.
    if args.skip_rsssf:
        return 0

    rsssf_html = _load_rsssf_html(args.rsssf_fixture)
    rsssf_shootouts = parse_rsssf_html(rsssf_html)
    report = validate_shootout_count(
        args.output,
        rsssf_shootouts,
        LEAGUE_SEASONS_PREDICT_WINDOW,
        discrepancies_path=args.discrepancies,
        skipped_refs=skipped,
        no_kicks_refs=no_kicks,
    )
    explained = len(skipped) + len(no_kicks)
    unexplained = -report.delta - explained
    if unexplained == 0:
        status = (
            f"OK (divergence fully explained by {len(skipped)} skipped + "
            f"{len(no_kicks)} no-kicks matches; see {args.discrepancies})"
        )
    else:
        status = f"DIVERGED (unexplained delta = {unexplained})"
    print(f"RSSSF check: actual={report.actual} expected={report.expected} ({status})")
    return 0 if unexplained == 0 else 1


def _load_rsssf_html(fixture_path: Path) -> str:
    """Load the RSSSF page, preferring a saved fixture over the live URL.

    The RSSSF page is a static reference; we don't want to depend on its
    availability at every run. If a saved fixture is present, use it.
    Otherwise, fetch the live page once and cache it.
    """
    if fixture_path.exists():
        return load_rsssf_html(fixture_path)
    # Fallback: fetch the live page. We do this through httpx directly to
    # avoid coupling to the FotMob client (RSSSF is not a FotMob endpoint).
    import httpx

    headers = {"User-Agent": "Mozilla/5.0 (compatible; penalty-pred)"}
    with httpx.Client(timeout=15.0, follow_redirects=True) as http:
        response = http.get(DEFAULT_RSSSF_URL, headers=headers)
    response.raise_for_status()
    return response.content.decode("latin-1")


if __name__ == "__main__":
    sys.exit(main())
