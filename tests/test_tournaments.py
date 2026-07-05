"""Tests for the tournament scope (`src/penalty_pred/tournaments.py`).

PRD: the in-scope tournaments live in `LEAGUE_SEASONS_PREDICT_WINDOW` and
the RSSSF page is the verification oracle. The scope is a *superset* of
the RSSSF data: some in-scope pairs (e.g. Gold Cup 2021, Asian Cup 2021,
Asian Cup 2025) contribute zero shootouts; the WC 2026 is in progress and
may or may not be on the saved RSSSF snapshot. A regression that drops a
(league, season) from the scope or adds a non-existent one should fail
here, not silently produce an empty training set on the next pipeline run.

The test pins the per-pair RSSSF count via `RSSSF_RAW_COUNTS` (the raw
oracle count) and the per-pair scraper-reachable count via
`EXPECTED_SHOOTOUT_COUNTS` (the raw count minus the 6 documented
empty-shotmap cases). The two diverge by 6 for the AFCON 2021 and Asian
Cup 2023 pairs; the divergence is documented in
`data/empty_shotmap_documentation.md` (Issue #49).

The full 15-pair scope totals **42 shootouts on the RSSSF snapshot** at
`docs/samples/rsssf_penaltiestour.html`, but the scraper-reachable
count is **36**: 6 of the 42 are FotMob data gaps (an empty
`pageProps.content.shotmap.shots` array) and are documented in the
empty-shotmap file.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from penalty_pred.rsssf import (
    RSSSFShootout,
    count_shootouts_by_pairs,
    load_rsssf_html,
    parse_rsssf_html,
)
from penalty_pred.tournaments import (
    LEAGUE_SEASONS_PREDICT_WINDOW,
    RSSSF_TO_LEAGUE_NAME,
    WC_2026_LEAGUE,
    WC_2026_SEASON,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RSSSF_FIXTURE = REPO_ROOT / "docs" / "samples" / "rsssf_penaltiestour.html"
EMPTY_SHOTMAP_DOC = REPO_ROOT / "data" / "empty_shotmap_documentation.md"

# Raw RSSSF shootout count per in-scope (league_id, season) pair — the
# count the RSSSF oracle reports for the in-scope (league, season) pair.
# The 4 zero-count pairs (Gold Cup 2021, Asian Cup 2021, Asian Cup 2025,
# WC 2026) are intentionally in scope — they're documented in
# `tournaments.py` with comments explaining the gap (the pair has zero
# shootouts, not a missing entry).
#
# Issue #49: the AFCON 2021 and Asian Cup 2023 pairs have raw counts
# (6 and 4) that are higher than the scraper can reach (2 and 2). The
# 4 + 2 = 6 unreachable cases are documented in
# `data/empty_shotmap_documentation.md`. The test pins the raw RSSSF
# count; the per-pair reachable count is derived as
# `RSSSF_RAW_COUNTS[pair] - EXCLUDED_EMPTY_SHOTMAP[pair]`.
RSSSF_RAW_COUNTS: dict[tuple[int, int], int] = {
    (77, 2022): 5,   # World Cup 2022
    (77, 2026): 0,   # World Cup 2026 (in progress; RSSSF snapshot is stale)
    (50, 2020): 4,   # Euro 2020 (held 2021)
    (50, 2024): 3,   # Euro 2024
    (44, 2021): 3,   # Copa América 2021
    (44, 2024): 4,   # Copa América 2024
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
# scraper-reachable count (Issue #49). The total is 6 (4 + 2 across
# the two affected pairs). Pairs not in this map have 0 exclusions.
EXCLUDED_EMPTY_SHOTMAP: dict[tuple[int, int], int] = {
    (289, 2021): 4,  # AFCON 2021: 4 documented cases
    (290, 2023): 2,  # Asian Cup 2023: 2 documented cases
}

# Scraper-reachable shootout count per in-scope (league_id, season) pair.
# Derived from `RSSSF_RAW_COUNTS - EXCLUDED_EMPTY_SHOTMAP`. This is the
# count the `validate_shootout_count` test pins the scraper against.
# The sum is 36, the scraper-reachable total.
EXPECTED_SHOOTOUT_COUNTS: dict[tuple[int, int], int] = {
    pair: RSSSF_RAW_COUNTS[pair] - EXCLUDED_EMPTY_SHOTMAP.get(pair, 0)
    for pair in RSSSF_RAW_COUNTS
}


@pytest.fixture(scope="module")
def rsssf_shootouts() -> list[RSSSFShootout]:
    if not RSSSF_FIXTURE.exists():
        pytest.skip(f"RSSSF fixture not present at {RSSSF_FIXTURE}")
    return parse_rsssf_html(load_rsssf_html(RSSSF_FIXTURE))


# --- LEAGUE_SEASONS_PREDICT_WINDOW is the source of truth ------------------


def test_scope_has_fifteen_pairs() -> None:
    """The scope is the 15 (league, season) pairs across the 6 in-scope
    tournaments (3× WC, 2× Euro, 2× Copa, 3× AFCON, 3× Gold Cup, 3× Asian
    Cup — minus 1 for WC 2026 = 1 entry because the tournament is in
    progress)."""
    assert len(LEAGUE_SEASONS_PREDICT_WINDOW) == 15


def test_scope_covers_all_six_in_scope_leagues() -> None:
    """Every league in `LEAGUES` is represented in the scope."""
    from penalty_pred.leagues import LEAGUES

    scope_leagues = {lid for lid, _ in LEAGUE_SEASONS_PREDICT_WINDOW}
    assert scope_leagues == {league.league_id for league in LEAGUES}


def test_scope_excludes_extended_leagues() -> None:
    """Leagues in `EXTENDED_LEAGUES` (e.g. LaLiga, Champions League) are
    NOT in the shootout scope — those are for the player-history
    fetcher only."""
    from penalty_pred.leagues import EXTENDED_LEAGUES

    scope_leagues = {lid for lid, _ in LEAGUE_SEASONS_PREDICT_WINDOW}
    extended_ids = {league.league_id for league in EXTENDED_LEAGUES}
    assert scope_leagues.isdisjoint(extended_ids)


# --- The per-pair coverage: each (league, season) matches the RSSSF oracle


@pytest.mark.parametrize(
    ("league_id", "season"),
    list(LEAGUE_SEASONS_PREDICT_WINDOW),
    ids=[f"{lid}-{s}" for lid, s in LEAGUE_SEASONS_PREDICT_WINDOW],
)
def test_each_in_scope_pair_matches_rsssf_count(
    rsssf_shootouts: list[object], league_id: int, season: int
) -> None:
    """Each in-scope (league, season) matches the RSSSF oracle's
    **raw** shootout count (which can be 0 for legitimately empty pairs).

    A regression that drops a (league, season) from the scope, adds a
    non-existent one, or changes the in-scope coverage without
    updating the expected count is caught here. If the RSSSF
    snapshot is refreshed and the per-pair raw counts change, the
    `RSSSF_RAW_COUNTS` map must be updated to match. The reachable
    count (`EXPECTED_SHOOTOUT_COUNTS`) is derived; a change in
    `EXCLUDED_EMPTY_SHOTMAP` (e.g. new documentation, or a Phase 3
    source closing the gap) is the separate signal.
    """
    expected = RSSSF_RAW_COUNTS[(league_id, season)]
    actual = count_shootouts_by_pairs(rsssf_shootouts, [(league_id, season)])
    assert actual == expected, (
        f"in-scope pair ({league_id}, {season}) has {actual} shootouts on the "
        f"RSSSF page, expected {expected} (raw); update RSSSF_RAW_COUNTS if "
        f"the snapshot was refreshed"
    )


def test_total_in_scope_count_is_42(rsssf_shootouts: list[object]) -> None:
    """The full 15-pair scope totals 42 shootouts on the RSSSF page
    (the raw oracle count, before the 6 empty-shotmap exclusions).

    This is the round-trip assertion that the per-pair map is complete:
    if a new tournament is added, the expected count and the validation
    logic both need to be updated. The raw RSSSF count is 42; the
    scraper-reachable count is 36 (see `test_total_in_scope_count_is_36`).
    """
    n = count_shootouts_by_pairs(rsssf_shootouts, LEAGUE_SEASONS_PREDICT_WINDOW)
    assert n == 42


def test_total_in_scope_count_is_36(rsssf_shootouts: list[object]) -> None:
    """The scraper-reachable count is 36 = 42 RSSSF - 6 empty-shotmap cases.

    `validate_shootout_count` pins the validator against the reachable
    count (36), not the raw RSSSF count (42). The 6 unreachable cases
    are the FotMob data gaps documented in
    `data/empty_shotmap_documentation.md` (Issue #49). If a future
    Phase 3 source (Issue #51) recovers the 6 missing shootouts, this
    assertion updates to 42 and the per-pair map in
    `EXPECTED_SHOOTOUT_COUNTS` updates accordingly.
    """
    raw = count_shootouts_by_pairs(rsssf_shootouts, LEAGUE_SEASONS_PREDICT_WINDOW)
    assert raw == 42
    documented = _documented_empty_shotmap_count()
    reachable = raw - documented
    assert reachable == 36


def test_expected_counts_sum_to_36() -> None:
    """`EXPECTED_SHOOTOUT_COUNTS` sums to 36 (the scraper-reachable count).

    A drift between the per-pair map and the reachable total is a
    test bug, not a code bug, but it must be caught here so the
    per-pair test cannot silently pass on a missing pair.
    """
    assert sum(EXPECTED_SHOOTOUT_COUNTS.values()) == 36


def test_excluded_empty_shotmap_count_is_six() -> None:
    """Issue #49: the per-pair empty-shotmap exclusions total 6 cases
    (4 in AFCON 2021 + 2 in Asian Cup 2023). A drift between the
    exclusions map and the documentation file is caught here."""
    assert sum(EXCLUDED_EMPTY_SHOTMAP.values()) == 6
    assert _documented_empty_shotmap_count() == 6


# --- Empty-shotmap documentation (Issue #49) ------------------------------


_CASE_RE = re.compile(r"^###\s+\d+\.\s+(?P<heading>.+?)\s*$", re.MULTILINE)
_SCREENSHOT_RE = re.compile(
    r"^\s*-\s+\*\*screenshot_path:\*\*\s+`(?P<path>[^`]+)`\s*$",
    re.MULTILINE,
)
_EXPLANATION_RE = re.compile(
    r"^\s*-\s+\*\*explanation:\*\*\s+(?P<text>.+?)\s*$",
    re.MULTILINE,
)


def _parse_empty_shotmap_doc(path: Path) -> list[dict[str, str]]:
    """Parse `data/empty_shotmap_documentation.md` into a list of cases.

    The documentation has a fixed format: a `### N. <heading>` per case,
    followed by bulleted fields (matchId, URL pattern, screenshot_path,
    explanation). Returns one record per case with `heading`,
    `screenshot_path`, and `explanation` fields. A case without a
    `screenshot_path` or `explanation` line is returned with an empty
    string for that field — the assertions downstream are responsible
    for the non-empty check.
    """
    text = path.read_text(encoding="utf-8")
    cases: list[dict[str, str]] = []
    headings = list(_CASE_RE.finditer(text))
    for i, match in enumerate(headings):
        start = match.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        body = text[start:end]
        screenshot = _SCREENSHOT_RE.search(body)
        explanation = _EXPLANATION_RE.search(body)
        cases.append(
            {
                "heading": match.group("heading").strip(),
                "screenshot_path": screenshot.group("path").strip() if screenshot else "",
                "explanation": explanation.group("text").strip() if explanation else "",
            }
        )
    return cases


def _documented_empty_shotmap_count() -> int:
    """Return the number of cases documented in the empty-shotmap file."""
    if not EMPTY_SHOTMAP_DOC.exists():
        return 0
    return len(_parse_empty_shotmap_doc(EMPTY_SHOTMAP_DOC))


def test_empty_shotmap_documentation_file_exists() -> None:
    """Issue #49: `data/empty_shotmap_documentation.md` exists and is
    non-empty (the 6 FotMob data gaps are documented)."""
    assert EMPTY_SHOTMAP_DOC.exists(), (
        f"empty-shotmap documentation missing at {EMPTY_SHOTMAP_DOC}; "
        "create the file with one record per FotMob data gap"
    )
    assert EMPTY_SHOTMAP_DOC.stat().st_size > 500, (
        f"empty-shotmap documentation at {EMPTY_SHOTMAP_DOC} is too short "
        f"({EMPTY_SHOTMAP_DOC.stat().st_size} bytes); expected > 500 bytes"
    )


def test_empty_shotmap_documentation_has_six_cases() -> None:
    """Issue #49: the documentation has exactly 6 records — one per
    FotMob data gap (4 in AFCON 2021 + 2 in Asian Cup 2023)."""
    cases = _parse_empty_shotmap_doc(EMPTY_SHOTMAP_DOC)
    assert len(cases) == 6, (
        f"expected 6 empty-shotmap cases, got {len(cases)}: "
        f"{[c['heading'] for c in cases]}"
    )


def test_empty_shotmap_documentation_every_case_has_screenshot_path() -> None:
    """Issue #49: every record has a non-empty `screenshot_path` (the
    path where a screenshot of the live FotMob match page's empty
    shotmap block should be saved)."""
    cases = _parse_empty_shotmap_doc(EMPTY_SHOTMAP_DOC)
    missing = [c["heading"] for c in cases if not c["screenshot_path"]]
    assert not missing, f"cases missing screenshot_path: {missing}"


def test_empty_shotmap_documentation_every_case_has_explanation() -> None:
    """Issue #49: every record has a non-empty `explanation` (a one-line
    note on why the shotmap is empty)."""
    cases = _parse_empty_shotmap_doc(EMPTY_SHOTMAP_DOC)
    missing = [c["heading"] for c in cases if not c["explanation"]]
    assert not missing, f"cases missing explanation: {missing}"


def test_empty_shotmap_documentation_covers_the_six_pairs() -> None:
    """Issue #49: the 6 cases cover the 4 AFCON 2021 and 2 Asian Cup 2023
    empty-shotmap shootouts listed in the issue. The RSSSF lines for
    these are pinned in `docs/samples/rsssf_penaltiestour.html`."""
    cases = _parse_empty_shotmap_doc(EMPTY_SHOTMAP_DOC)
    headings = " | ".join(c["heading"] for c in cases).lower()
    # 4 AFCON 2021 cases
    for team_pair in (
        "burkina faso vs gabon",
        "mali vs equatorial guinea",
        "cameroon vs egypt",
        "burkina faso vs cameroon",
    ):
        assert team_pair in headings, f"missing AFCON 2021 case: {team_pair}"
    # 2 Asian Cup 2023 cases
    for team_pair in (
        "tajikistan vs united arab emirates",
        "saudi arabia vs south korea",
    ):
        assert team_pair in headings, f"missing Asian Cup 2023 case: {team_pair}"


def test_empty_shotmap_documentation_documented_count_matches_per_pair() -> None:
    """Issue #49: the number of documented empty-shotmap cases equals the
    per-pair exclusion map (4 in AFCON 2021 + 2 in Asian Cup 2023 = 6).
    A drift between the exclusions map and the documentation file is
    caught here."""
    documented_gap = sum(EXCLUDED_EMPTY_SHOTMAP.values())
    assert documented_gap == 6
    assert _documented_empty_shotmap_count() == documented_gap


def test_expected_counts_match_scope() -> None:
    """`EXPECTED_SHOOTOUT_COUNTS` covers every in-scope (league, season)
    exactly. A drift between the map and the scope is a test bug,
    not a code bug, but it must be caught here so the per-pair test
    cannot silently pass on a missing pair.
    """
    assert set(EXPECTED_SHOOTOUT_COUNTS.keys()) == set(LEAGUE_SEASONS_PREDICT_WINDOW)


# --- RSSSF_TO_LEAGUE_NAME covers the in-scope headings --------------------


def test_rsssf_heading_map_covers_six_in_scope_tournaments() -> None:
    """The heading map has exactly the 6 in-scope RSSSF headings; any
    out-of-scope heading (e.g. the Confederations Cup) is intentionally
    absent."""
    assert set(RSSSF_TO_LEAGUE_NAME.keys()) == {
        "World Cup",
        "European Nations' Cup",
        "Copa América",
        "African Nations Cup",
        "Gold Cup",
        "Asian Nations Cup",
    }


def test_rsssf_heading_values_match_league_names() -> None:
    """Each heading value is the FotMob league name (matches `LEAGUES`)."""
    from penalty_pred.leagues import LEAGUES

    fotmob_names = {league.name for league in LEAGUES}
    assert set(RSSSF_TO_LEAGUE_NAME.values()) == fotmob_names


# --- WC 2026 module-level constants ---------------------------------------


def test_wc_2026_league_is_world_cup() -> None:
    """The WC 2026 league is league 77 (the FotMob World Cup)."""
    assert WC_2026_LEAGUE.league_id == 77
    assert WC_2026_LEAGUE.name == "World Cup"


def test_wc_2026_season_is_2026() -> None:
    """The WC 2026 season is 2026 — the FotMob `?season=` value."""
    assert WC_2026_SEASON == 2026
    # And the (WC 2026 league, 2026) pair is in the scope.
    assert (WC_2026_LEAGUE.league_id, WC_2026_SEASON) in LEAGUE_SEASONS_PREDICT_WINDOW
