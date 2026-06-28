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


@dataclass(frozen=True)
class ShootoutCountReport:
    """The result of one `validate_shootout_count` call.

    `actual` is the count of distinct shootout matches in the JSONL.
    `expected` is the count RSSSF lists for the in-scope (league_id, season)
    pairs. `match` is `actual == expected`. `actual_pairs` is the sorted
    list of (tournament, year) pairs observed in the JSONL, used for
    debugging. `skipped_refs` is the list of match refs whose (seo, h2h)
    hash was stale (FotMob reuses hashes and pointed us at a different
    match). When the slice reports skipped refs, the actual count is
    bounded by `expected - len(skipped_refs)`.
    """

    actual: int
    expected: int
    match: bool
    actual_pairs: list[tuple[str, int]]
    skipped_refs: list[MatchRef] = field(default_factory=list)

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
    artifacts: Artifacts | None = None,
) -> ShootoutCountReport:
    """Compare the count of shootout matches in the JSONL to the RSSSF count.

    Returns a `ShootoutCountReport` regardless of whether a discrepancy
    exists. If `discrepancies_path` is provided AND the counts differ, the
    full report is serialised there as JSON. Re-runs are idempotent: if the
    counts match, the discrepancies file is left alone.

    `skipped_refs` are match refs the orchestrator could not fetch (stale
    (seo, h2h) hashes). `no_kicks_refs` are match refs where the matchId
    was correct but `extract_shootout_kicks` returned no kicks (FotMob has
    `penaltyShootoutEvents` but the shotmap is empty for these matches).
    Both lists are included in the discrepancies file for debugging.

    The JSONL is read through the data layer's reader (`Artifacts.read_shootout_kicks`)
    so the validator stops re-parsing the file format with its own
    ad-hoc readers. `artifacts` is overridable for tests that need a
    custom root; the default reads from `Path("output")`.
    """
    art = artifacts or Artifacts()
    shootout_kicks = art.read_shootout_kicks(jsonl_path)
    actual_ids = _distinct_match_ids(shootout_kicks)
    actual = len(actual_ids)
    expected = count_shootouts_by_pairs(rsssf_shootouts, league_seasons)
    actual_pairs = _tournament_year_pairs(shootout_kicks)
    skipped_list = list(skipped_refs)
    no_kicks_list = list(no_kicks_refs)
    report = ShootoutCountReport(
        actual=actual,
        expected=expected,
        match=actual == expected,
        actual_pairs=actual_pairs,
        skipped_refs=skipped_list,
    )
    if discrepancies_path is not None and not report.match:
        discrepancies_path.parent.mkdir(parents=True, exist_ok=True)
        with discrepancies_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "actual_shootout_count": report.actual,
                    "expected_shootout_count": report.expected,
                    "delta": report.delta,
                    "actual_pairs": [{"tournament": t, "year": y} for t, y in report.actual_pairs],
                    "skipped_refs": [_ref_payload(r) for r in skipped_list],
                    "no_kicks_refs": [_ref_payload(r) for r in no_kicks_list],
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
