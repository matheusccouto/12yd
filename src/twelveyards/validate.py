"""Scraped-data validators. Issue #19: shootout count vs. RSSSF oracle.

The validator reads a `shootout_kicks.jsonl` produced by the all-shootouts
slice and checks the count of distinct shootout matches against an expected
count (typically the count of in-scope shootouts on the RSSSF page). Any
discrepancy is written to `discrepancies.json` for investigation.

PRD: "RSSSF is a verification oracle, never a data source. We assert our
scraper finds the same count of shootouts RSSSF lists, and we investigate
discrepancies."
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from .artifacts import Artifacts
from .match_ref import MatchRef
from .rsssf import RSSSFShootout, count_shootouts_by_pairs
from .shootouts import ShootoutKick

# Per-(league_id, season) raw RSSSF shootout count, pinned to the
# `docs/samples/rsssf_penaltiestour.html` snapshot (the
# `?RSSSF_PENALTIES_TOUR_SNAPSHOT=docs/samples/rsssf_penaltiestour.html`
# fixture). The international scope's 15 pairs sum to 42 raw shootouts.
# Club pairs (Phase 3) are not in this map — the RSSSF page does not
# list club shootouts, and the per-tournament success-rate diagnostic
# for club pairs uses 0 as the raw count (the
# `aggregate_per_tournament_success_rate` default).
RSSSF_RAW_COUNTS: dict[tuple[int, int], int] = {
    (77, 2022): 5,  # World Cup 2022
    (77, 2026): 0,  # World Cup 2026 (in progress; RSSSF snapshot is stale)
    (50, 2020): 4,  # Euro 2020 (held 2021)
    (50, 2024): 3,  # Euro 2024
    (44, 2021): 3,  # Copa América 2021
    (44, 2024): 4,  # Copa América 2024
    (289, 2021): 6,  # AFCON 2021 (raw RSSSF)
    (289, 2023): 5,  # AFCON 2023
    (289, 2025): 3,  # AFCON 2025
    (298, 2021): 0,  # Gold Cup 2021 (no shootouts in window)
    (298, 2023): 2,  # Gold Cup 2023
    (298, 2025): 3,  # Gold Cup 2025
    (290, 2021): 0,  # Asian Cup 2021 (no shootouts in window)
    (290, 2023): 4,  # Asian Cup 2023 (raw RSSSF)
    (290, 2025): 0,  # Asian Cup 2025 (no shootouts in window)
}

# Per-pair count of empty-shotmap cases that are excluded from the
# scraper-reachable count (Issue #49: 6 documented FotMob data gaps;
# 4 in AFCON 2021, 2 in Asian Cup 2023). The total is 6. Pairs not
# in this map have 0 exclusions.
EMPTY_SHOTMAP_EXCLUSIONS: dict[tuple[int, int], int] = {
    (289, 2021): 4,  # AFCON 2021: 4 documented cases
    (290, 2023): 2,  # Asian Cup 2023: 2 documented cases
}

# Per-pair count of URL-rotation cases that are excluded from the
# scraper-reachable count (Issue #39: 18 documented FotMob URL
# rotation failures; see `data/url_rotation_wall.md`). The 18
# refs are spread across 9 pairs (1 in WC 2022, 3 in Euro 2020,
# 3 in Copa América 2021, 3 in Copa América 2024, 2 in AFCON 2021,
# 1 in AFCON 2023, 2 in Gold Cup 2023, 1 in Gold Cup 2025, 2 in
# Asian Cup 2023). The total is 18. Pairs not in this map have 0
# exclusions. The `stale_hash` rows in `skipped_refs_diagnostics.jsonl`
# match this count one-for-one (Issue #39 acceptance criterion).
URL_ROTATION_EXCLUSIONS: dict[tuple[int, int], int] = {
    (77, 2022): 1,  # World Cup 2022: 1 documented
    (50, 2020): 3,  # Euro 2020: 3 documented
    (44, 2021): 3,  # Copa América 2021: 3 documented
    (44, 2024): 3,  # Copa América 2024: 3 documented
    (289, 2021): 2,  # AFCON 2021: 2 documented (in addition to 4 empty-shotmap)
    (289, 2023): 1,  # AFCON 2023: 1 documented
    (298, 2023): 2,  # Gold Cup 2023: 2 documented
    (298, 2025): 1,  # Gold Cup 2025: 1 documented
    (290, 2023): 2,  # Asian Cup 2023: 2 documented (in addition to 2 empty-shotmap)
}

# Scraper-reachable shootout count per in-scope (league_id, season) pair.
# Derived from `RSSSF_RAW_COUNTS - EMPTY_SHOTMAP_EXCLUSIONS -
# URL_ROTATION_EXCLUSIONS`. The sum is 18 — the v4 PRD Phase 2
# success criterion after the 6 empty-shotmap and 18 URL-rotation
# exclusions (the 5-strategy URL-rotation wall is documented as
# the stop condition in Issue #39 / `data/url_rotation_wall.md`).
EXPECTED_SHOUTOUT_COUNTS: dict[tuple[int, int], int] = {
    pair: (
        RSSSF_RAW_COUNTS[pair]
        - EMPTY_SHOTMAP_EXCLUSIONS.get(pair, 0)
        - URL_ROTATION_EXCLUSIONS.get(pair, 0)
    )
    for pair in RSSSF_RAW_COUNTS
}


@dataclass(frozen=True)
class ShootoutCountReport:
    """The result of one `validate_shootout_count` call.

    `actual` is the count of distinct shootout matches in the JSONL.
    `expected` is the count the validator asserts against — the raw
    RSSSF count minus the documented `no_kicks_refs` (Issue #49: 42 - 6
    = 36 for the current 15-pair in-scope scope). `raw_expected` is
    the raw RSSSF oracle count, before the `no_kicks_refs` adjustment;
    it is included for debugging and for tests that exercise the raw
    oracle. `match` is `actual == expected`. `actual_pairs` is the sorted
    list of (tournament, year) pairs observed in the JSONL, used for
    debugging. `skipped_refs` is the list of match refs whose (seo, h2h)
    hash was stale (FotMob reuses hashes and pointed us at a different
    match). `no_kicks_refs` are match refs where the matchId was correct
    but `extract_shootout_kicks` returned no kicks. `failed_refs` are
    match refs where the extractor raised (the orchestrator caught the
    exception and reported `failure_mode` on the `FetchResult`; the
    validator only needs the refs, the `failure_mode` is in
    `skipped_refs_diagnostics.jsonl`). When the slice reports any of
    these, the actual count is bounded by
    `raw_expected - len(skipped_refs) - len(failed_refs)`.
    """

    actual: int
    expected: int
    match: bool
    actual_pairs: list[tuple[str, int]]
    skipped_refs: list[MatchRef] = field(default_factory=list)
    no_kicks_refs: list[MatchRef] = field(default_factory=list)
    failed_refs: list[MatchRef] = field(default_factory=list)
    raw_expected: int = 0

    @property
    def delta(self) -> int:
        """`actual - expected` (negative = under-count)."""
        return self.actual - self.expected


def _tournament_year_pairs(
    shootout_kicks: Iterable[ShootoutKick],
) -> list[tuple[str, int]]:
    """Return the sorted (tournament_name, match_year) of every distinct match.

    `tournament_name` is the FotMob `tournament_name` field. `match_year`
    is the calendar year parsed from the ISO 8601 `match_date`. The
    validator surfaces these so the discrepancies file shows which
    (tournament, year) tuples the scraper actually fetched.
    """
    pairs: set[tuple[str, int]] = set()
    for kick in shootout_kicks:
        date_str = kick.match_date or ""
        year = int(date_str[:4]) if len(date_str) >= 4 else 0
        pairs.add((kick.tournament_name, year))
    return sorted(pairs)


def _distinct_match_ids(shootout_kicks: Iterable[ShootoutKick]) -> set[int]:
    """Return the set of distinct `match_id`s across the input."""
    return {k.match_id for k in shootout_kicks}


def validate_shootout_count(
    jsonl_path: Path,
    rsssf_shootouts: list[RSSSFShootout],
    league_seasons: Iterable[tuple[int, int]],
    discrepancies_path: Path | None = None,
    skipped_refs: Iterable[MatchRef] = (),
    no_kicks_refs: Iterable[MatchRef] = (),
    failed_refs: Iterable[MatchRef] = (),
    artifacts: Artifacts | None = None,
) -> ShootoutCountReport:
    """Compare the count of shootout matches in the JSONL to the RSSSF count.

    Returns a `ShootoutCountReport` regardless of whether a discrepancy
    exists. If `discrepancies_path` is provided AND the counts differ, the
    full report is serialised there as JSON. Re-runs are idempotent: if the
    counts match, the discrepancies file is left alone.

    `skipped_refs` are match refs the orchestrator could not fetch (stale
    (seo, h2h) hashes from the URL-rotation wall — Issue #39). `no_kicks_refs`
    are match refs where the matchId was correct but `extract_shootout_kicks`
    returned no kicks (FotMob has `penaltyShootoutEvents` but the shotmap is
    empty for these matches — Issue #49). `failed_refs` are match refs where
    `extract_shootout_kicks` raised an exception; the failure mode is
    recorded in `skipped_refs_diagnostics.jsonl` by the slice script. All
    three lists are included in the discrepancies file for debugging.

    The expected count is the raw RSSSF count minus the number of
    `no_kicks_refs` (Issue #49) and `skipped_refs` (Issue #39): the
    6 documented empty-shotmap cases plus the 18 documented URL-rotation
    cases are accepted as FotMob data gaps, so the validator's reachable
    count is 18 = 42 - 6 - 18. A caller that does not pass `no_kicks_refs`
    or `skipped_refs` (e.g. a unit test that exercises the raw oracle)
    gets the raw count (42) as the expected; the script that drives the
    full orchestrator always passes both lists and gets 18.

    The JSONL is read through the data layer's reader (`Artifacts.read_shootout_kicks`)
    so the validator stops re-parsing the file format with its own
    ad-hoc readers. `artifacts` is overridable for tests that need a
    custom root; the default reads from `Path("output")`.
    """
    art = artifacts or Artifacts()
    shootout_kicks = art.read_shootout_kicks(jsonl_path)
    actual_ids = _distinct_match_ids(shootout_kicks)
    actual = len(actual_ids)
    skipped_list = list(skipped_refs)
    no_kicks_list = list(no_kicks_refs)
    failed_list = list(failed_refs)
    raw_expected = count_shootouts_by_pairs(rsssf_shootouts, league_seasons)
    # Issue #49 + Issue #39: the 6 documented empty-shotmap cases plus
    # the 18 documented URL-rotation cases are accepted as FotMob data
    # gaps; the validator's expected count is the raw RSSSF count minus
    # both exclusions.
    expected = raw_expected - len(no_kicks_list) - len(skipped_list)
    actual_pairs = _tournament_year_pairs(shootout_kicks)
    report = ShootoutCountReport(
        actual=actual,
        expected=expected,
        match=actual == expected,
        actual_pairs=actual_pairs,
        skipped_refs=skipped_list,
        no_kicks_refs=no_kicks_list,
        failed_refs=failed_list,
        raw_expected=raw_expected,
    )
    if discrepancies_path is not None and not report.match:
        discrepancies_path.parent.mkdir(parents=True, exist_ok=True)
        with discrepancies_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "actual_shootout_count": report.actual,
                    "expected_shootout_count": report.expected,
                    "raw_expected_shootout_count": report.raw_expected,
                    "delta": report.delta,
                    "actual_pairs": [{"tournament": t, "year": y} for t, y in report.actual_pairs],
                    "skipped_refs": [_ref_payload(r) for r in skipped_list],
                    "no_kicks_refs": [_ref_payload(r) for r in no_kicks_list],
                    "failed_refs": [_ref_payload(r) for r in failed_list],
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
    return report


def _ref_payload(r: MatchRef) -> dict[str, object]:
    """Serialise a MatchRef for the discrepancies file."""
    return {
        "match_id": r.match_id,
        "home": r.home_team_name,
        "away": r.away_team_name,
        "round": r.round_name,
        "match_date": r.match_date,
    }
