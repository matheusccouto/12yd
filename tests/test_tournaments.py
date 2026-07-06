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

import json
import re
from pathlib import Path
from typing import Any

import pytest

from penalty_pred.rsssf import (
    RSSSFShootout,
    count_shootouts_by_pairs,
    load_rsssf_html,
    parse_rsssf_html,
)
from penalty_pred.tournaments import (
    CLUB_PAIRS,
    INTERNATIONAL_PAIRS,
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
# scraper-reachable count (Issue #49). The total is 6 (4 + 2 across
# the two affected pairs). Pairs not in this map have 0 exclusions.
EXCLUDED_EMPTY_SHOTMAP: dict[tuple[int, int], int] = {
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
EXCLUDED_URL_ROTATION: dict[tuple[int, int], int] = {
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
# Derived from `RSSSF_RAW_COUNTS - EXCLUDED_EMPTY_SHOTMAP -
# EXCLUDED_URL_ROTATION`. This is the count the
# `validate_shootout_count` test pins the scraper against. The sum
# is 18 — the v4 PRD Phase 2 success criterion after the 6
# empty-shotmap and 18 URL-rotation exclusions (the 5-strategy
# URL-rotation wall is documented as the stop condition in
# Issue #39 / `data/url_rotation_wall.md`).
EXPECTED_SHOOTOUT_COUNTS: dict[tuple[int, int], int] = {
    pair: (
        RSSSF_RAW_COUNTS[pair]
        - EXCLUDED_EMPTY_SHOTMAP.get(pair, 0)
        - EXCLUDED_URL_ROTATION.get(pair, 0)
    )
    for pair in RSSSF_RAW_COUNTS
}


@pytest.fixture(scope="module")
def rsssf_shootouts() -> list[RSSSFShootout]:
    if not RSSSF_FIXTURE.exists():
        pytest.skip(f"RSSSF fixture not present at {RSSSF_FIXTURE}")
    return parse_rsssf_html(load_rsssf_html(RSSSF_FIXTURE))


# --- LEAGUE_SEASONS_PREDICT_WINDOW is the source of truth ------------------


def test_scope_has_fifty_seven_pairs() -> None:
    """The scope is the 57 (league, season) pairs across the 6
    in-scope international + 7 in-scope club tournaments (Phase 3,
    Issue #51).

    15 international pairs (3× WC, 2× Euro, 2× Copa, 3× AFCON, 3× Gold
    Cup, 3× Asian Cup — minus 1 for WC 2026 = 1 entry because the
    tournament is in progress) + 42 club pairs (7 leagues × 6 seasons
    2021–2026).
    """
    assert len(LEAGUE_SEASONS_PREDICT_WINDOW) == 57


def test_scope_covers_all_thirteen_in_scope_leagues() -> None:
    """Every league in `LEAGUES` (6 international) and `CLUB_LEAGUES`
    (7 club) is represented in the scope (Phase 3, Issue #51).

    The 12 `domestic_only` extended leagues (LaLiga, Premier League,
    etc.) are NOT in the scope — they are player-history only.
    """
    from penalty_pred.leagues import CLUB_LEAGUES, LEAGUES

    scope_leagues = {lid for lid, _ in LEAGUE_SEASONS_PREDICT_WINDOW}
    expected = {league.league_id for league in LEAGUES} | {
        league.league_id for league in CLUB_LEAGUES
    }
    assert scope_leagues == expected


def test_scope_excludes_domestic_only_extended_leagues() -> None:
    """Leagues in `EXTENDED_LEAGUES` (e.g. LaLiga, Ligue 1) are NOT
    in the shootout scope — those are for the player-history fetcher
    only.

    Phase 3 (Issue #51): the 7 in-scope club leagues (Copa
    Libertadores, Champions League, FA Cup, Coupe de France, DFB-
    Pokal, Coppa Italia, Copa del Rey) have moved from
    `EXTENDED_LEAGUES` to a separate `CLUB_LEAGUES` tuple with
    `kind="club"`. The 12 remaining `EXTENDED_LEAGUES` are
    `kind="domestic_only"` and remain out of scope.
    """
    from penalty_pred.leagues import EXTENDED_LEAGUES

    scope_leagues = {lid for lid, _ in LEAGUE_SEASONS_PREDICT_WINDOW}
    extended_ids = {league.league_id for league in EXTENDED_LEAGUES}
    assert scope_leagues.isdisjoint(extended_ids)


# --- The per-pair coverage: each (league, season) matches the RSSSF oracle


@pytest.mark.parametrize(
    ("league_id", "season"),
    list(INTERNATIONAL_PAIRS),
    ids=[f"{lid}-{s}" for lid, s in INTERNATIONAL_PAIRS],
)
def test_each_in_scope_pair_matches_rsssf_count(
    rsssf_shootouts: list[RSSSFShootout], league_id: int, season: int
) -> None:
    """Each in-scope international (league, season) matches the RSSSF
    oracle's **raw** shootout count (which can be 0 for legitimately
    empty pairs).

    A regression that drops a (league, season) from the scope, adds a
    non-existent one, or changes the in-scope coverage without
    updating the expected count is caught here. If the RSSSF
    snapshot is refreshed and the per-pair raw counts change, the
    `RSSSF_RAW_COUNTS` map must be updated to match. The reachable
    count (`EXPECTED_SHOOTOUT_COUNTS`) is derived; a change in
    `EXCLUDED_EMPTY_SHOTMAP` (e.g. new documentation, or a Phase 3
    source closing the gap) is the separate signal.

    Phase 3 (Issue #51): the parametrized list is now
    `INTERNATIONAL_PAIRS` (the 15 international pairs), not the full
    `LEAGUE_SEASONS_PREDICT_WINDOW` (57 pairs). The 42 new club
    pairs are in scope but the current RSSSF snapshot has no club
    oracle — a future Phase 4 ADR-driven decision will fill in
    `CLUB_RAW_COUNTS` (a per-(club league, season) count) and the
    parametrized list can extend.
    """
    expected = RSSSF_RAW_COUNTS[(league_id, season)]
    actual = count_shootouts_by_pairs(rsssf_shootouts, [(league_id, season)])
    assert actual == expected, (
        f"in-scope pair ({league_id}, {season}) has {actual} shootouts on the "
        f"RSSSF page, expected {expected} (raw); update RSSSF_RAW_COUNTS if "
        f"the snapshot was refreshed"
    )


def test_club_pairs_count_is_42() -> None:
    """Phase 3 (Issue #51): the 7 in-scope club leagues each have 6
    FotMob seasons (2021–2026), totalling 42 club (league, season)
    pairs. The 7 leagues are: Copa Libertadores, Champions League,
    FA Cup, Coupe de France, DFB-Pokal, Coppa Italia, Copa del Rey.
    """
    assert len(CLUB_PAIRS) == 42
    # Each club league has 6 seasons (2021–2026)
    league_ids = {lid for lid, _ in CLUB_PAIRS}
    assert len(league_ids) == 7
    for lid in league_ids:
        league_seasons = [s for pair_lid, s in CLUB_PAIRS if pair_lid == lid]
        assert sorted(league_seasons) == [2021, 2022, 2023, 2024, 2025, 2026], (
            f"club league {lid} has seasons {league_seasons}, expected 2021-2026"
        )


def test_club_pairs_have_no_rsssf_coverage(
    rsssf_shootouts: list[RSSSFShootout],
) -> None:
    """The current RSSSF snapshot has no club shootout coverage —
    the 42 new club pairs all return 0 from the oracle.

    This is a forward-looking pin: a future Phase 4 ADR-driven
    decision (RSSSF per-cup detail scraping, or a non-RSSSF oracle)
    can fill in the per-(club league, season) counts. Until then,
    the scraper-reachable count for club pairs is 0 (no validation
    can pass against an absent oracle).
    """
    n = count_shootouts_by_pairs(rsssf_shootouts, CLUB_PAIRS)
    assert n == 0, (
        f"RSSSF snapshot reported {n} club shootouts, expected 0 "
        f"(no club oracle yet — see Issue #51 + Phase 4 ADR)"
    )


def test_total_in_scope_count_is_42(rsssf_shootouts: list[RSSSFShootout]) -> None:
    """The 15-pair international scope totals 42 shootouts on the
    RSSSF page (the raw oracle count, before the 6 empty-shotmap
    exclusions).

    This is the round-trip assertion that the per-pair map is complete:
    if a new tournament is added, the expected count and the validation
    logic both need to be updated. The raw RSSSF count is 42; the
    scraper-reachable count is 36 (see `test_total_in_scope_count_is_36`).
    The 42 new club pairs contribute 0 (no RSSSF coverage; see
    `test_club_pairs_have_no_rsssf_coverage`).

    Phase 3 (Issue #51): the test is restricted to the
    international pairs (the per-pair oracle counts live in
    `RSSSF_RAW_COUNTS`).
    """
    n = count_shootouts_by_pairs(rsssf_shootouts, INTERNATIONAL_PAIRS)
    assert n == 42


def test_total_in_scope_count_is_36(rsssf_shootouts: list[RSSSFShootout]) -> None:
    """The scraper-reachable count is 36 = 42 RSSSF - 6 empty-shotmap cases.

    `validate_shootout_count` pins the validator against the reachable
    count (36), not the raw RSSSF count (42). The 6 unreachable cases
    are the FotMob data gaps documented in
    `data/empty_shotmap_documentation.md` (Issue #49). If a future
    Phase 3 source (Issue #51) recovers the 6 missing shootouts, this
    assertion updates to 42 and the per-pair map in
    `EXPECTED_SHOOTOUT_COUNTS` updates accordingly.

    Phase 3 (Issue #51): the test is restricted to the international
    pairs (the per-pair oracle counts live in `RSSSF_RAW_COUNTS`).
    """
    raw = count_shootouts_by_pairs(rsssf_shootouts, INTERNATIONAL_PAIRS)
    assert raw == 42
    documented = _documented_empty_shotmap_count()
    reachable = raw - documented
    assert reachable == 36


def test_expected_counts_sum_to_eighteen() -> None:
    """`EXPECTED_SHOOTOUT_COUNTS` sums to 18 (the scraper-reachable count
    after the 6 empty-shotmap and 18 URL-rotation exclusions).

    A drift between the per-pair map and the reachable total is a
    test bug, not a code bug, but it must be caught here so the
    per-pair test cannot silently pass on a missing pair. The v4
    PRD §"Phase 2" originally set this to 36 (the 6-exclusion
    reachable count) before the URL-rotation wall was documented in
    Issue #39 as a stop condition; the 18-exclusion reachable count
    is the v4 Phase 2 close-out.
    """
    assert sum(EXPECTED_SHOOTOUT_COUNTS.values()) == 18


def test_excluded_empty_shotmap_count_is_six() -> None:
    """Issue #49: the per-pair empty-shotmap exclusions total 6 cases
    (4 in AFCON 2021 + 2 in Asian Cup 2023). A drift between the
    exclusions map and the documentation file is caught here."""
    assert sum(EXCLUDED_EMPTY_SHOTMAP.values()) == 6
    assert _documented_empty_shotmap_count() == 6


def test_excluded_url_rotation_count_is_eighteen() -> None:
    """Issue #39: the per-pair URL-rotation exclusions total 18 cases
    (1 WC 2022 + 3 Euro 2020 + 3 Copa América 2021 + 3 Copa
    América 2024 + 2 AFCON 2021 + 1 AFCON 2023 + 2 Gold Cup 2023 +
    1 Gold Cup 2025 + 2 Asian Cup 2023). A drift between the
    exclusions map and the URL-rotation wall documentation is
    caught here."""
    assert sum(EXCLUDED_URL_ROTATION.values()) == 18


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
        f"expected 6 empty-shotmap cases, got {len(cases)}: {[c['heading'] for c in cases]}"
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

    Phase 3 (Issue #51): `EXPECTED_SHOOTOUT_COUNTS` is the per-pair
    map for the **international** scope only (the pairs in
    `RSSSF_RAW_COUNTS`). The 42 new club pairs are in
    `LEAGUE_SEASONS_PREDICT_WINDOW` but have no `RSSSF_RAW_COUNTS`
    entry yet (the club oracle is a Phase 4 ADR-driven decision);
    `EXPECTED_SHOOTOUT_COUNTS` therefore covers the 15 international
    pairs, not the 57 total scope.
    """
    assert set(EXPECTED_SHOOTOUT_COUNTS.keys()) == set(INTERNATIONAL_PAIRS)


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


# --- Phase 3 schema additions (Issue #51) ---------------------------------


def test_tournament_kind_lookup_international() -> None:
    """The 6 in-scope international tournaments all map to
    `"international"` in the `TOURNAMENT_KIND_BY_LEAGUE_ID` lookup.

    The lookup is the source of truth for the `tournament_kind`
    attribute on `TrainingRow` and `PredictionRow`. The international
    tournaments keep the existing 6 IDs.
    """
    from penalty_pred.tournaments import TOURNAMENT_KIND_BY_LEAGUE_ID

    expected_international = {77, 50, 44, 298, 290, 289}
    for league_id in expected_international:
        assert TOURNAMENT_KIND_BY_LEAGUE_ID[league_id] == "international", (
            f"league {league_id} should be 'international', "
            f"got {TOURNAMENT_KIND_BY_LEAGUE_ID[league_id]!r}"
        )


def test_tournament_kind_lookup_club() -> None:
    """The 7 in-scope club tournaments all map to `"club"` in the
    `TOURNAMENT_KIND_BY_LEAGUE_ID` lookup.

    Per the Phase 3 ADR (`docs/adr/0004-phase-3-data-source.md`):
    the 7 in-scope club leagues (Copa Libertadores, Champions
    League, FA Cup, Coupe de France, DFB-Pokal, Coppa Italia, Copa
    del Rey) carry the `"club"` kind; the model treats this as
    metadata, not a model input.
    """
    from penalty_pred.tournaments import TOURNAMENT_KIND_BY_LEAGUE_ID

    expected_clubs = {41, 42, 125, 132, 133, 137, 138}
    for league_id in expected_clubs:
        assert TOURNAMENT_KIND_BY_LEAGUE_ID[league_id] == "club", (
            f"league {league_id} should be 'club', got {TOURNAMENT_KIND_BY_LEAGUE_ID[league_id]!r}"
        )


def test_tournament_kind_lookup_excludes_domestic_only() -> None:
    """The `TOURNAMENT_KIND_BY_LEAGUE_ID` lookup only contains
    in-scope leagues (international + club). The 12
    `domestic_only` extended leagues (LaLiga, Premier League, etc.)
    are NOT in the lookup — they are player-history only and
    never appear in the shootout scope.
    """
    from penalty_pred.leagues import EXTENDED_LEAGUES
    from penalty_pred.tournaments import TOURNAMENT_KIND_BY_LEAGUE_ID

    for league in EXTENDED_LEAGUES:
        assert league.league_id not in TOURNAMENT_KIND_BY_LEAGUE_ID, (
            f"domestic_only league {league.league_id} ({league.name}) "
            "should not be in the tournament_kind lookup"
        )


def test_club_leagues_tuple() -> None:
    """`CLUB_LEAGUES` is the 7-tournament table of in-scope club
    shootouts (Phase 3, Issue #51). Each entry has `kind="club"`.
    The 7 league IDs match the Phase 3 ADR's per-tournament list.
    """
    from penalty_pred.leagues import CLUB_LEAGUES

    assert len(CLUB_LEAGUES) == 7
    for league in CLUB_LEAGUES:
        assert league.kind == "club", (
            f"club league {league.league_id} ({league.name}) has kind={league.kind!r}, "
            "expected 'club'"
        )
    assert {league.league_id for league in CLUB_LEAGUES} == {
        41,
        42,
        125,
        132,
        133,
        137,
        138,
    }


def test_rsssf_to_club_league_name_covers_seven_tournaments() -> None:
    """`RSSSF_TO_CLUB_LEAGUE_NAME` is the 7-heading map for the
    forward-looking Phase 3 club oracle. The current RSSSF snapshot
    has no club shootout coverage (see `test_club_pairs_have_no_rsssf_coverage`),
    but the map is the shape for a future Phase 4 ingest.
    """
    from penalty_pred.tournaments import RSSSF_TO_CLUB_LEAGUE_NAME

    assert set(RSSSF_TO_CLUB_LEAGUE_NAME.keys()) == {
        "Copa Libertadores",
        "UEFA Champions League",
        "FA Cup",
        "Coupe de France",
        "DFB-Pokal",
        "Coppa Italia",
        "Copa del Rey",
    }


def test_rsssf_to_club_league_name_values_match_club_league_names() -> None:
    """Each value in `RSSSF_TO_CLUB_LEAGUE_NAME` is the FotMob
    league name (matches `CLUB_LEAGUES`). A drift between the
    map and the league table is a code bug.
    """
    from penalty_pred.leagues import CLUB_LEAGUES
    from penalty_pred.tournaments import RSSSF_TO_CLUB_LEAGUE_NAME

    fotmob_names = {league.name for league in CLUB_LEAGUES}
    assert set(RSSSF_TO_CLUB_LEAGUE_NAME.values()) == fotmob_names


# --- Phase 3 source ADR (Issue #50) ---------------------------------------

PHASE_3_ADR = REPO_ROOT / "docs" / "adr" / "0004-phase-3-data-source.md"


def test_phase_3_adr_exists() -> None:
    """Issue #50: the Phase 3 data source ADR is checked into
    `docs/adr/0004-phase-3-data-source.md` and is non-empty. A drift
    (missing file, empty file) is caught here so the implementation
    in Issue #51 cannot ship without a recorded source decision.
    """
    assert PHASE_3_ADR.exists(), (
        f"Phase 3 source ADR missing at {PHASE_3_ADR}; create the file "
        "with the source decision rationale (Issue #50)"
    )
    assert PHASE_3_ADR.stat().st_size > 500, (
        f"Phase 3 source ADR at {PHASE_3_ADR} is too short "
        f"({PHASE_3_ADR.stat().st_size} bytes); expected > 500 bytes"
    )


def test_phase_3_adr_cross_references_empty_shotmap_documentation() -> None:
    """Issue #50: the ADR cross-references the 6 empty-shotmap FotMob
    data gaps (Issue #49) and the data gap analysis. The cross-ref
    pins the rationale for choosing club shootouts over a non-FotMob
    source: the 6 cases stay documented as FotMob data gaps; closing
    them is deferred to a future Phase 4 ADR.
    """
    text = PHASE_3_ADR.read_text(encoding="utf-8")
    assert "empty_shotmap" in text.lower() or "issue #49" in text.lower(), (
        "Phase 3 ADR must cross-reference Issue #49 / the empty-shotmap "
        "documentation; the 6 FotMob data gaps are the reason the ADR "
        "rejects a non-FotMob source for Phase 3"
    )


def test_phase_3_adr_cross_references_model_review() -> None:
    """Issue #50: the ADR cross-references `docs/model-review.md` —
    Topic 1.4 (the 86.6% no-history prediction rows) or Topic 5 (the
    LOTO CV statistical-power analysis) — to anchor the data gap
    framing. The model review is the source of the "more data is
    the path to statistical power" claim.
    """
    text = PHASE_3_ADR.read_text(encoding="utf-8")
    assert "model-review" in text.lower() or "model review" in text.lower(), (
        "Phase 3 ADR must cross-reference docs/model-review.md; the "
        "data gap framing comes from Topic 1.4 (no-history) and "
        "Topic 5 (LOTO CV statistical power)"
    )


def test_phase_3_adr_mentions_all_three_candidate_sources() -> None:
    """Issue #50: the ADR's rationale section names each candidate
    source — FotMob club leagues, StatsBomb Open Data, RSSSF detail
    pages — with at least one paragraph of trade-offs. A future
    reader who lands on the ADR must see all three candidates and
    the rejection rationale for StatsBomb + RSSSF, not just the
    FotMob-club decision.
    """
    text = PHASE_3_ADR.read_text(encoding="utf-8")
    for candidate in ("fotmob", "statsbomb", "rsssf"):
        assert candidate in text.lower(), (
            f"Phase 3 ADR must mention the {candidate} candidate source; "
            "the rationale section enumerates the three candidates with "
            "trade-offs for each"
        )


def test_phase_3_adr_records_decision_and_per_tournament_handling() -> None:
    """Issue #50: the ADR records the decision (club Shootout Kicks
    via FotMob), the why (schema identical, no new client, ~360-row
    target), and the per-tournament handling (Copa Libertadores, UCL
    knockout, domestic cup finals; new `LEAGUE_SEASONS_PREDICT_WINDOW`
    entries; new `RSSSF_TO_LEAGUE_NAME` headings). A future
    implementer should be able to read the ADR and know what to
    register in `leagues.py` + `tournaments.py`.
    """
    text = PHASE_3_ADR.read_text(encoding="utf-8")
    for tournament in (
        "copa libertadores",
        "champions league",
        "fa cup",
        "coupe de france",
        "dfb-pokal",
        "coppa italia",
        "copa del rey",
    ):
        assert tournament in text.lower(), (
            f"Phase 3 ADR must list {tournament!r} as a per-tournament "
            "entry; the per-tournament handling is a per-ADR acceptance "
            "criterion"
        )


def test_phase_3_adr_documents_schema_change() -> None:
    """Issue #50: the ADR documents the schema change — a new
    `tournament_kind` attribute ∈ {`international`, `club}` on
    `TrainingRow` and the unchanged 17-feature model input. The
    attribute is metadata, not a model input.
    """
    text = PHASE_3_ADR.read_text(encoding="utf-8")
    assert "tournament_kind" in text.lower(), (
        "Phase 3 ADR must document the new `tournament_kind` attribute "
        "on `TrainingRow`; the schema-change section is a per-ADR "
        "acceptance criterion"
    )
    assert "international" in text.lower() and "club" in text.lower(), (
        "Phase 3 ADR must enumerate the two `tournament_kind` values "
        "(`international`, `club`); the attribute domain is a per-ADR "
        "acceptance criterion"
    )


def test_phase_3_adr_documents_loto_cv_grouping() -> None:
    """Issue #50: the ADR documents the LOTO CV grouping strategy —
    the existing per-`tournament_name` fold unit carries over, the
    new club tournaments become additional folds, and the
    `tournament_kind` attribute is a per-row analysis axis (not a
    fold unit). The strategy section is a per-ADR acceptance
    criterion.
    """
    text = PHASE_3_ADR.read_text(encoding="utf-8")
    assert "loto" in text.lower() or "leave-one-tournament" in text.lower(), (
        "Phase 3 ADR must document the LOTO CV grouping strategy; the "
        "strategy section is a per-ADR acceptance criterion"
    )
    # The decision is per-tournament-name fold (not per-kind or per-source).
    assert "tournament_name" in text.lower() or "tournament name" in text.lower(), (
        "Phase 3 ADR must document the fold unit; the per-tournament-name "
        "fold carries over from the v3 LOTO CV (Issue #45)"
    )


# --- URL rotation wall (Issue #39) -----------------------------------------

URL_ROTATION_WALL = REPO_ROOT / "data" / "url_rotation_wall.md"


def test_url_rotation_wall_exists() -> None:
    """Issue #39: the URL rotation wall documentation
    (`data/url_rotation_wall.md`) exists and is non-empty. The 5
    URL-lookup strategies are listed with the failure mode for each,
    the stop condition is reached, and a future maintainer reading
    the file can find the rationale for the wall without re-doing the
    5 strategies."""
    assert URL_ROTATION_WALL.exists(), (
        f"URL rotation wall documentation missing at {URL_ROTATION_WALL}; "
        "create the file with the 5 URL-lookup strategies, the failure "
        "mode for each, and the stop condition"
    )
    assert URL_ROTATION_WALL.stat().st_size > 1000, (
        f"URL rotation wall at {URL_ROTATION_WALL} is too short "
        f"({URL_ROTATION_WALL.stat().st_size} bytes); expected > 1000 bytes"
    )


def test_url_rotation_wall_documents_five_strategies() -> None:
    """Issue #39: the wall documents all 5 URL-lookup strategies. The
    v4 PRD Phase 2 step 2 enumerates 4 strategies tried in the issue
    body + 1 bounded attempt (the FotMob public page search); all 5
    must be listed with the failure mode for each."""
    text = URL_ROTATION_WALL.read_text(encoding="utf-8").lower()
    for keyword in (
        "public page search",
        "per-team fixture list",
        "direct match data api",
        "match-page anchor",
        "5th",
    ):
        assert keyword in text, (
            f"URL rotation wall must document the {keyword!r} strategy; "
            "the 5 strategies are the stop-condition evidence"
        )


def test_url_rotation_wall_documents_stop_condition() -> None:
    """Issue #39: the wall reaches the v4 PRD Phase 2 step 2 stop
    condition. The text 'wall' appears in the title and the body
    asserts the 5 strategies are the bound on in-FotMob options."""
    text = URL_ROTATION_WALL.read_text(encoding="utf-8").lower()
    assert "wall" in text, (
        "URL rotation wall must include the term 'wall' (the v4 PRD's "
        "stop condition is 'document the wall and stop')"
    )
    assert "stop" in text or "enough" in text, (
        "URL rotation wall must assert the stop condition (the v4 PRD "
        "Phase 2 step 2: 'document the wall and stop')"
    )


def test_url_rotation_wall_lists_18_stale_hash_refs() -> None:
    """Issue #39: the wall documents the per-ref diagnosis. The 18
    original `match_id` values are listed in the table (the live
    `matchId` is logged per ref, and the public `resolved_url` is
    in the JSONL)."""
    text = URL_ROTATION_WALL.read_text(encoding="utf-8")
    for mid in (
        3370565,
        2767865,
        2767870,
        2767869,
        3231662,
        3231660,
        3231664,
        4407868,
        4407869,
        4407870,
        3705434,
        3705509,
        4353245,
        4211901,
        4211904,
        4772526,
        4394637,
        4394643,
    ):
        assert str(mid) in text, (
            f"URL rotation wall must list the stale-hash ref {mid}; "
            "the per-ref diagnosis is the Phase 2 step 1 acceptance "
            "criterion"
        )


def test_url_rotation_wall_cross_references_phase_3_adr() -> None:
    """Issue #39: the wall cross-references the Phase 3 ADR. The
    path forward is Phase 3 (issue #51, club Shootout Kicks via
    FotMob); the 18 stale-hash refs and the 6 empty-shotmap cases
    are deferred to a future Phase 4 ADR. The cross-ref pins the
    boundary between this wall and the Phase 3 work."""
    text = URL_ROTATION_WALL.read_text(encoding="utf-8")
    assert "phase 3" in text.lower() or "issue #51" in text.lower(), (
        "URL rotation wall must cross-reference Phase 3 (Issue #51); "
        "the wall is the stop condition for in-FotMob attempts, and "
        "Phase 3 is the path forward"
    )
    assert "0004" in text or "phase-3-data-source" in text or "issue #50" in text, (
        "URL rotation wall must cross-reference the Phase 3 ADR "
        "(`docs/adr/0004-phase-3-data-source.md`); the wall defers "
        "the 6 empty-shotmap + 18 stale-hash cases to a future Phase 4 "
        "ADR-driven decision"
    )


# --- Per-tournament success-rate diagnostic (v4 PRD Phase 2) -------------

# Path to the on-disk per-tournament success-rate artifact. The file
# is gitignored (`data/*.jsonl`, `output/*.jsonl` patterns) so the
# test skips if the slice pipeline hasn't been re-run. The default
# location is the same path `scripts/fetch_all_shootouts.py` writes
# to, with `Artifacts(root="output")` as the source of truth.
TOURNAMENT_SUCCESS_RATE_PATH: Path = Path("output/tournament_success_rate.jsonl")


def _load_tournament_success_rate() -> list[dict[str, Any]] | None:
    """Read the per-tournament success-rate JSONL, or `None` if absent."""
    if not TOURNAMENT_SUCCESS_RATE_PATH.exists():
        return None
    out: list[dict[str, Any]] = []
    for line in TOURNAMENT_SUCCESS_RATE_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


@pytest.mark.parametrize(
    ("league_id", "season"),
    list(INTERNATIONAL_PAIRS),
    ids=[f"{lid}-{s}" for lid, s in INTERNATIONAL_PAIRS],
)
def test_tournament_success_rate_covers_every_in_scope_pair(league_id: int, season: int) -> None:
    """v4 PRD Phase 2 acceptance criterion: every in-scope
    (league, season) has a row in `tournament_success_rate.jsonl`,
    and the row's `match_count` agrees with the scraper-reachable
    count in `EXPECTED_SHOOTOUT_COUNTS` (so a future regression
    that drops a pair is caught at the artifact).

    The test skips if the artifact is absent (the slice pipeline
    hasn't been re-run). The unit test in `test_shootouts_pipeline.py`
    covers the aggregation logic without a disk artifact; this
    integration test pins the on-disk shape.
    """
    rows = _load_tournament_success_rate()
    if rows is None:
        pytest.skip(
            f"{TOURNAMENT_SUCCESS_RATE_PATH} not present; "
            "run `scripts/fetch_all_shootouts.py` to generate"
        )
    expected_count = EXPECTED_SHOOTOUT_COUNTS[(league_id, season)]
    pair_rows = [r for r in rows if r["league_id"] == league_id and r["season"] == season]
    assert len(pair_rows) == 1, (
        f"pair ({league_id}, {season}) has {len(pair_rows)} rows in the "
        f"artifact, expected exactly 1"
    )
    row = pair_rows[0]
    assert row["expected_match_count"] == (RSSSF_RAW_COUNTS[(league_id, season)]), (
        f"raw expected count drift on ({league_id}, {season})"
    )
    assert row["reachable_match_count"] == expected_count, (
        f"reachable count drift on ({league_id}, {season}): "
        f"artifact={row['reachable_match_count']}, "
        f"EXPECTED_SHOOTOUT_COUNTS={expected_count}"
    )
    if expected_count == 0:
        # The pair legitimately has no reachable shootouts (either
        # because the RSSSF oracle reports 0, or because the
        # exclusions eat the full raw count). The rollup marks
        # this as `"n/a"` (no coverage expected). The
        # `kick_count` / `match_count` may be 0 OR may be non-zero:
        # the WC 2026 pair is the canonical case where the RSSSF
        # snapshot is stale (raw=0 because the tournament was in
        # progress at snapshot time) but the live FotMob fixtures
        # return real shootout matches. The per-tournament success
        # rate correctly keeps `status="n/a"` (the oracle has no
        # reach to assert against) but the per-row `match_count` /
        # `kick_count` are the live counts.
        assert row["status"] == "n/a"
    else:
        # The pair has a reachable count > 0. The artifact's
        # `match_count` must equal the reachable count (the
        # scraper reached every match) — a `partial` or
        # `missing` row would be a regression.
        assert row["match_count"] == expected_count, (
            f"pair ({league_id}, {season}) has match_count="
            f"{row['match_count']}, expected {expected_count} "
            f"(reachable); status={row['status']!r}"
        )
        assert row["status"] == "ok", (
            f"pair ({league_id}, {season}) status is {row['status']!r}, "
            "expected 'ok' (a fresh re-run with the wall reached should "
            "find every reachable match)"
        )
        assert row["kick_count"] > 0, (
            f"pair ({league_id}, {season}) has kick_count="
            f"{row['kick_count']}, expected > 0 (every reachable match "
            "should have at least one shootout kick)"
        )
