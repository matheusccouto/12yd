"""Tests for the RSSSF parser. Slice #2 (Issue #19) uses the parser as a
verification oracle; these tests pin the parser to the saved RSSSF page and
the known in-window count (42).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from penalty_pred.rsssf import (
    count_shootouts_by_pairs,
    load_rsssf_html,
    parse_rsssf_html,
)
from penalty_pred.tournaments import LEAGUE_SEASONS_PREDICT_WINDOW, RSSSF_TO_LEAGUE_NAME

REPO_ROOT = Path(__file__).resolve().parent.parent
RSSSF_FIXTURE = REPO_ROOT / "docs" / "samples" / "rsssf_penaltiestour.html"


@pytest.fixture(scope="module")
def rsssf_html() -> str:
    if not RSSSF_FIXTURE.exists():
        pytest.skip(f"RSSSF fixture not present at {RSSSF_FIXTURE}")
    return load_rsssf_html(RSSSF_FIXTURE)


@pytest.fixture(scope="module")
def all_shootouts(rsssf_html: str) -> list[object]:
    return parse_rsssf_html(rsssf_html)


# --- parse_rsssf_html -------------------------------------------------------


def test_load_handles_latin1(rsssf_html: str) -> None:
    """The page is latin-1, not UTF-8. Loader must round-trip the é in 'é'."""
    assert "é" in rsssf_html
    # The fixture must contain "Copa América" to verify the encoding was right.
    assert "Copa Am" in rsssf_html


def test_parse_finds_all_six_tournaments(all_shootouts: list[object]) -> None:
    tournaments = {s.tournament for s in all_shootouts}  # type: ignore[attr-defined]
    assert tournaments == set(RSSSF_TO_LEAGUE_NAME.values())


def test_parse_skips_confederations_cup(all_shootouts: list[object]) -> None:
    """The Confederations Cup section is on the page but out of scope."""
    raw_texts = {s.raw for s in all_shootouts}  # type: ignore[attr-defined]
    # Sanity: there are Confederations Cup entries in the source page
    fixture = RSSSF_FIXTURE.read_text(encoding="latin-1")
    assert "Confederations Cup" in fixture
    # None of the parsed records are from the Confederations Cup.
    assert not any("Confederations Cup" in r for r in raw_texts)


def test_parse_returns_record_per_shootout(rsssf_html: str) -> None:
    """The page lists 185 shootouts across the 6 in-scope tournaments."""
    all_shootouts = parse_rsssf_html(rsssf_html)
    assert len(all_shootouts) == 185


def test_parse_world_cup_count(all_shootouts: list[object]) -> None:
    """Sanity: 35 World Cup shootouts on the page (1982–2022)."""
    n = sum(1 for s in all_shootouts if s.tournament == "World Cup")  # type: ignore[attr-defined]
    assert n == 35


def test_parse_year_and_round_extracted(all_shootouts: list[object]) -> None:
    """The 2022 World Cup Final is the last WC entry."""
    final = next(
        s
        for s in all_shootouts
        if s.tournament == "World Cup" and s.year == 2022 and s.round_label == "F"
    )  # type: ignore[attr-defined]
    assert "Argentina" in final.raw  # type: ignore[attr-defined]
    assert "France" in final.raw


def test_parse_handles_unicode_round_label(all_shootouts: list[object]) -> None:
    """The '3/4' (third-place playoff) round label is preserved."""
    s3_4 = [s for s in all_shootouts if s.round_label == "3/4"]  # type: ignore[attr-defined]
    assert len(s3_4) > 0


def test_parse_year_column_is_int(all_shootouts: list[object]) -> None:
    for s in all_shootouts:  # type: ignore[attr-defined]
        assert isinstance(s.year, int)  # type: ignore[attr-defined]
        assert 1900 <= s.year <= 2100


# --- count_shootouts_by_pairs -----------------------------------------------


def test_count_by_pairs_predict_window_is_42(all_shootouts: list[object]) -> None:
    """PRD: 42 in-window shootouts (5 WC 2022 + 7 Euro (4+3) + 7 Copa (3+4) +
    14 AFCON (6+5+3) + 5 Gold Cup (0+2+3) + 4 Asian Cup (0+4+0) = 42)."""
    n = count_shootouts_by_pairs(all_shootouts, LEAGUE_SEASONS_PREDICT_WINDOW)
    assert n == 42


def test_count_by_pairs_excludes_out_of_window(all_shootouts: list[object]) -> None:
    """A smaller window (only WC 2022 + Euro 2024) gives a smaller count."""
    pairs = [
        (77, 2022),  # World Cup 2022
        (50, 2024),  # Euro 2024
    ]
    n = count_shootouts_by_pairs(all_shootouts, pairs)
    assert n == 5 + 3  # 5 WC 2022 + 3 Euro 2024


def test_count_by_pairs_empty_set() -> None:
    assert count_shootouts_by_pairs(parse_rsssf_html(""), []) == 0


def test_count_by_pairs_excludes_unrelated_tournaments(all_shootouts: list[object]) -> None:
    """Asking for a league that exists but has no shootouts in the window
    returns 0. Gold Cup 2021 has 0 shootouts."""
    n = count_shootouts_by_pairs(all_shootouts, [(298, 2021)])
    assert n == 0
