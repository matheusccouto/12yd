"""Fetch every shootout in the 6 in-scope tournaments between 2021-01-01 and today.

Slice #2 (Issue #19): the orchestrator that drives the per-match extractor
across all (league, season) pairs in the current Prediction Window. The
script:

1. Iterates (league_id, season) pairs from `LEAGUE_SEASONS_PREDICT_WINDOW`.
2. Fetches each league's season fixtures, filters to `penalties_short`.
3. Fetches each shootout match's full JSON, runs `extract_shootout_kicks`.
4. Writes a single `shootout_kicks.jsonl` containing every kick.
5. Writes `skipped_refs_diagnostics.jsonl` with one record per
   non-empty skip / no-kicks / failure result, used to diagnose the
   RSSSF divergence.
6. Writes `tournament_success_rate.jsonl` with one record per
   (league, season) pair, the per-tournament rollup of the per-match
   results (v4 PRD Phase 2 acceptance criterion). The pair info is
   tracked per iteration so the rollup can be generated alongside the
   per-match diagnostics in the same loop.
7. Loads the RSSSF penaltiestour page, counts in-window shootouts, and writes
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
from penalty_pred.match_ref import MatchRef
from penalty_pred.rsssf import load_rsssf_html, parse_rsssf_html
from penalty_pred.shootouts import (
    FetchResult,
    aggregate_per_tournament_success_rate,
    fetch_all_shootout_kicks_with_skips,
    fetch_all_shootout_match_refs,
    write_per_tournament_success_rate,
    write_skipped_refs_diagnostics,
)
from penalty_pred.tournaments import (
    INTERNATIONAL_PAIRS,
    LEAGUE_SEASONS_PREDICT_WINDOW,
)
from penalty_pred.validate import (
    EMPTY_SHOTMAP_EXCLUSIONS,
    RSSSF_RAW_COUNTS,
    URL_ROTATION_EXCLUSIONS,
    validate_shootout_count,
)

# The RSSSF page is the verification oracle (PRD: "RSSSF is a verification
# oracle, never a data source"). Saved next to this script as a fallback if
# the live page is unreachable; the CLI prefers the live page.
DEFAULT_RSSSF_URL: str = "https://www.rsssf.org/miscellaneous/penaltiestour.html"
DEFAULT_RSSSF_FIXTURE: Path = Path("docs/samples/rsssf_penaltiestour.html")


def _combined_exclusions() -> dict[tuple[int, int], int]:
    """Per-pair sum of `EMPTY_SHOTMAP_EXCLUSIONS` and `URL_ROTATION_EXCLUSIONS`.

    Both exclusion maps cover the international scope. The aggregate
    function's `excluded_counts` parameter takes a single
    `dict[pair, count]` — we sum the two sources here so a pair that
    has both kinds of FotMob gap (e.g. AFCON 2021, 4 empty-shotmap +
    2 URL-rotation = 6) is excluded with the right total. A naive
    `{**A, **B}` would lose one of the two values on overlap.
    """
    combined: dict[tuple[int, int], int] = {}
    for pair, n in EMPTY_SHOTMAP_EXCLUSIONS.items():
        combined[pair] = combined.get(pair, 0) + n
    for pair, n in URL_ROTATION_EXCLUSIONS.items():
        combined[pair] = combined.get(pair, 0) + n
    return combined


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
        "--diagnostics",
        type=Path,
        default=art.diagnostics,
        help=(f"Path to write the per-match diagnostics JSONL (default: {art.diagnostics})."),
    )
    parser.add_argument(
        "--tournament-success-rate",
        type=Path,
        default=art.tournament_success_rate,
        help=(
            "Path to write the per-(league, season) success-rate JSONL "
            f"(default: {art.tournament_success_rate})."
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

    # 1. Per-(league, season) iteration. We track the pair alongside the
    # per-match results so the per-tournament success-rate diagnostic
    # (v4 PRD Phase 2 acceptance criterion) can be generated alongside
    # the per-match diagnostics. The pair is dropped inside
    # `fetch_all_shootout_match_refs` (it returns a flat list of refs),
    # so we re-bind it per iteration here.
    pair_results: list[tuple[tuple[int, int], list[FetchResult]]] = []
    all_kicks: list = []
    skipped: list = []
    no_kicks: list = []
    failed: list = []
    for pair in LEAGUE_SEASONS_PREDICT_WINDOW:
        refs = fetch_all_shootout_match_refs(client, [pair])
        results = fetch_all_shootout_kicks_with_skips(client, refs)
        pair_results.append((pair, results))
        for r in results:
            all_kicks.extend(r.kicks)
            if r.skipped:
                skipped.append(r.ref)
            elif r.no_kicks:
                no_kicks.append(r.ref)
            elif r.failure_mode:
                failed.append(r.ref)
    n_refs = sum(len(results) for _, results in pair_results)
    print(f"Discovered {n_refs} shootout matches across all in-scope tournaments")

    # 2. Write the artifacts. Three JSONL files: the kicks (one row per
    # kick), the per-match diagnostics (one row per skip / no-kicks /
    # failure), and the per-tournament success rate (one row per
    # (league, season) pair). The per-tournament aggregate is built with
    # the pinned RSSSF counts (`RSSSF_RAW_COUNTS`,
    # `EMPTY_SHOTMAP_EXCLUSIONS`, `URL_ROTATION_EXCLUSIONS`) so the
    # per-row `expected_match_count` and `reachable_match_count` fields
    # reflect the oracle's reach for the international pairs; club
    # pairs (Phase 3) have no RSSSF oracle yet, so they default to
    # 0 / 0 / status=`"n/a"` — the per-tournament diagnostic still
    # surfaces the per-pair kick / match / skip / no-kicks / failed
    # counts for them.
    all_results = [r for _, results in pair_results for r in results]
    n_kicks = art.write_shootout_kicks(all_kicks, path=args.output)
    n_diag = write_skipped_refs_diagnostics(all_results, path=args.diagnostics)
    rows = aggregate_per_tournament_success_rate(
        pair_results,
        expected_counts=RSSSF_RAW_COUNTS,
        excluded_counts=_combined_exclusions(),
    )
    n_rate = write_per_tournament_success_rate(rows, path=args.tournament_success_rate)
    print(
        f"Wrote {n_kicks} shootout kicks to {args.output} "
        f"({len(skipped)} skipped due to stale (seo, h2h) hashes, "
        f"{len(no_kicks)} processed with no shootout kicks in the shotmap, "
        f"{len(failed)} failed during extraction)"
    )
    print(f"Wrote {n_diag} per-match diagnostics rows to {args.diagnostics}")
    print(f"Wrote {n_rate} per-tournament success-rate rows to {args.tournament_success_rate}")

    # 3. Run the RSSSF completeness check.
    if args.skip_rsssf:
        return 0

    rsssf_html = _load_rsssf_html(args.rsssf_fixture)
    rsssf_shootouts = parse_rsssf_html(rsssf_html)
    # The RSSSF oracle covers the 15 in-scope international pairs only.
    # The 42 Phase 3 club pairs (Copa Libertadores, UCL knockout, etc.)
    # do not appear on the RSSSF `penaltiestour.html` page, so the
    # validator is scoped to INTERNATIONAL_PAIRS — otherwise the raw
    # RSSSF count (42) is mis-compared against the full 57-pair actual.
    #
    # The validator's expected count is `raw - no_kicks - skipped`
    # (the 6 documented empty-shotmap exclusions from Issue #49 plus
    # the 18 URL-rotation exclusions from Issue #39). We synthesise
    # dummy `no_kicks_refs` and `skipped_refs` for the validator (the
    # documented pairs in `EMPTY_SHOTMAP_EXCLUSIONS` and
    # `URL_ROTATION_EXCLUSIONS`); the actual `no_kicks_refs` /
    # `skipped_refs` lists from the run cover 78+132 refs across the
    # 57-pair scope (including club-scope refs that the RSSSF oracle
    # does not cover and therefore must not subtract from the expected).
    #
    # The actual `skipped` / `no_kicks` / `failed` lists are still
    # written to `discrepancies.json` for diagnostic purposes.
    n_intl_empty_shotmap = sum(EMPTY_SHOTMAP_EXCLUSIONS.values())
    intl_dummy_no_kicks = [
        MatchRef(
            match_id=900001 + i,
            seo=f"intl-empty-shotmap-{i}",
            h2h=f"intl{i:03d}",
        )
        for i in range(n_intl_empty_shotmap)
    ]
    n_intl_url_rotation = sum(URL_ROTATION_EXCLUSIONS.values())
    intl_dummy_skipped = [
        MatchRef(
            match_id=910001 + i,
            seo=f"intl-url-rotation-{i}",
            h2h=f"intls{i:03d}",
        )
        for i in range(n_intl_url_rotation)
    ]
    report = validate_shootout_count(
        args.output,
        rsssf_shootouts,
        INTERNATIONAL_PAIRS,
        discrepancies_path=args.discrepancies,
        skipped_refs=intl_dummy_skipped,
        no_kicks_refs=intl_dummy_no_kicks,
        failed_refs=failed,
    )
    explained = len(skipped) + len(no_kicks) + len(failed)
    unexplained = -report.delta - explained
    if unexplained == 0:
        status = (
            f"OK (divergence fully explained by {len(skipped)} skipped + "
            f"{len(no_kicks)} no-kicks + {len(failed)} failed matches; "
            f"see {args.discrepancies} and {args.diagnostics})"
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
