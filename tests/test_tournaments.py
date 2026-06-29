"""Tests for the tournament scope (`src/penalty_pred/tournaments.py`).

PRD: the in-scope tournaments live in `LEAGUE_SEASONS_PREDICT_WINDOW` and
the RSSSF page is the verification oracle. The scope is a *superset* of
the RSSSF data: some in-scope pairs (e.g. Gold Cup 2021, Asian Cup 2021,
Asian Cup 2025) contribute zero shootouts; the WC 2026 is in progress and
may or may not be on the saved RSSSF snapshot. A regression that drops a
(league, season) from the scope or adds a non-existent one should fail
here, not silently produce an empty training set on the next pipeline run.

The test pins the per-pair RSSSF count via `EXPECTED_SHOOTOUT_COUNTS`.
Any drift between the in-scope scope and the RSSSF oracle (a
re-snapshot of the page, a season newly added, a season removed) is
caught here. The full 15-pair scope totals 42 shootouts on the
RSSSF snapshot at `docs/samples/rsssf_penaltiestour.html`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from penalty_pred.rsssf import (
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

# Expected RSSSF shootout count per in-scope (league_id, season) pair.
# The 4 zero-count pairs (Gold Cup 2021, Asian Cup 2021, Asian Cup 2025,
# WC 2026) are intentionally in scope — they're documented in
# `tournaments.py` with comments explaining the gap. The test pins the
# current snapshot; a future agent that adds or removes a pair must
# update this map and the corresponding test in `test_shootouts_pipeline.py`.
EXPECTED_SHOOTOUT_COUNTS: dict[tuple[int, int], int] = {
    (77, 2022): 5,   # World Cup 2022
    (77, 2026): 0,   # World Cup 2026 (in progress; RSSSF snapshot is stale)
    (50, 2020): 4,   # Euro 2020 (held 2021)
    (50, 2024): 3,   # Euro 2024
    (44, 2021): 3,   # Copa América 2021
    (44, 2024): 4,   # Copa América 2024
    (289, 2021): 6,  # AFCON 2021
    (289, 2023): 5,  # AFCON 2023
    (289, 2025): 3,  # AFCON 2025
    (298, 2021): 0,  # Gold Cup 2021 (no shootouts in window)
    (298, 2023): 2,  # Gold Cup 2023
    (298, 2025): 3,  # Gold Cup 2025
    (290, 2021): 0,  # Asian Cup 2021 (no shootouts in window)
    (290, 2023): 4,  # Asian Cup 2023
    (290, 2025): 0,  # Asian Cup 2025 (no shootouts in window)
}


@pytest.fixture(scope="module")
def rsssf_shootouts() -> list[object]:
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
    shootout count (which can be 0 for legitimately empty pairs).

    A regression that drops a (league, season) from the scope, adds a
    non-existent one, or changes the in-scope coverage without
    updating the expected count is caught here. If the RSSSF
    snapshot is refreshed and the per-pair counts change, the
    expected-count map must be updated to match.
    """
    expected = EXPECTED_SHOOTOUT_COUNTS[(league_id, season)]
    actual = count_shootouts_by_pairs(rsssf_shootouts, [(league_id, season)])
    assert actual == expected, (
        f"in-scope pair ({league_id}, {season}) has {actual} shootouts on the "
        f"RSSSF page, expected {expected}; update EXPECTED_SHOOTOUT_COUNTS if "
        f"the snapshot was refreshed"
    )


def test_total_in_scope_count_is_42(rsssf_shootouts: list[object]) -> None:
    """The full 15-pair scope totals 42 shootouts on the RSSSF page.

    This is the round-trip assertion that drives `validate_shootout_count`:
    the scraper must find 42 distinct shootout matches or the run
    surfaces a discrepancy. If a new tournament is added, the
    expected count and the validation logic both need to be updated.
    """
    n = count_shootouts_by_pairs(rsssf_shootouts, LEAGUE_SEASONS_PREDICT_WINDOW)
    assert n == 42


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
