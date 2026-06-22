"""Tests for `parse_page_url` and `extract_shootout_match_fixtures`.

Slice #2 (Issue #19). The pageUrl parser is the foundation for turning
season-fixture entries into per-match fetchers; the filter is the entry
point for the orchestrator. Both are pure functions — no network — and
exercised against a saved WC 2022 slim fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from penalty_pred.leagues import LEAGUE_BY_ID
from penalty_pred.shootouts import (
    ShootoutMatchRef,
    extract_shootout_match_fixtures,
    parse_page_url,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SLIM_FIXTURE = REPO_ROOT / "docs" / "samples" / "league_wc_2022_slim.json"


@pytest.fixture(scope="module")
def slim_season() -> list[dict[str, object]]:
    if not SLIM_FIXTURE.exists():
        pytest.skip(f"Slim fixture not present at {SLIM_FIXTURE}")
    data = json.loads(SLIM_FIXTURE.read_text())
    return list(data["pageProps"]["fixtures"]["allMatches"])  # type: ignore[index]


# --- parse_page_url ---------------------------------------------------------


@pytest.mark.parametrize(
    ("page_url", "expected"),
    [
        ("/matches/japan-vs-croatia/2cq9vk#3370555", (3370555, "japan-vs-croatia", "2cq9vk")),
        ("/matches/argentina-vs-france/1hox8a#3370572", (3370572, "argentina-vs-france", "1hox8a")),
        (
            "/matches/argentina-vs-netherlands/1hklvd#3370566",
            (3370566, "argentina-vs-netherlands", "1hklvd"),
        ),
        (
            "/matches/switzerland-vs-spain/1hr85f#2767865",
            (2767865, "switzerland-vs-spain", "1hr85f"),
        ),
    ],
)
def test_parse_page_url_extracts_id_seo_h2h(page_url: str, expected: tuple[int, str, str]) -> None:
    assert parse_page_url(page_url) == expected


def test_parse_page_url_no_anchor_raises() -> None:
    with pytest.raises(ValueError, match="anchor"):
        parse_page_url("/matches/argentina-vs-france/1hox8a")


def test_parse_page_url_bad_path_raises() -> None:
    with pytest.raises(ValueError, match="did not match"):
        parse_page_url("/something/else/1hox8a#3370572")


def test_parse_page_url_no_anchor_in_path_raises() -> None:
    with pytest.raises(ValueError, match="anchor"):
        parse_page_url("just-a-string")


# --- extract_shootout_match_fixtures ----------------------------------------


def test_filter_returns_only_penalties_short(slim_season: list[dict[str, object]]) -> None:
    refs = extract_shootout_match_fixtures(slim_season)
    assert len(refs) == 3
    for ref in refs:
        assert isinstance(ref, ShootoutMatchRef)


def test_filter_preserves_fixture_order(slim_season: list[dict[str, object]]) -> None:
    refs = extract_shootout_match_fixtures(slim_season)
    ids = [r.match_id for r in refs]
    # The 3 shootouts in our slim fixture are at indices 0, 1, 2 (3 shootouts
    # + 2 non-shootouts appended at the end). The exact match_ids come from
    # the WC 2022 fixture list.
    assert ids == [3370555, 3370556, 3370565]


def test_filter_populates_seo_h2h_from_pageurl(slim_season: list[dict[str, object]]) -> None:
    refs = extract_shootout_match_fixtures(slim_season)
    seo_h2h = [(r.seo, r.h2h) for r in refs]
    assert seo_h2h[0] == ("japan-vs-croatia", "2cq9vk")
    assert seo_h2h[1] == ("morocco-vs-spain", "1e6edp")
    assert seo_h2h[2] == ("brazil-vs-croatia", "2swyz6")


def test_filter_skips_non_shootout_fixtures(slim_season: list[dict[str, object]]) -> None:
    """The slim fixture has 2 non-shootout entries appended for the filter test."""
    # Sanity: 2 non-shootouts in the input.
    non_shootouts = [
        m for m in slim_season if m["status"]["reason"]["shortKey"] != "penalties_short"
    ]  # type: ignore[index]
    assert len(non_shootouts) == 2
    # The filter drops them all.
    assert len(extract_shootout_match_fixtures(slim_season)) == 3


def test_filter_handles_empty_input() -> None:
    assert extract_shootout_match_fixtures([]) == []


def test_from_fixture_round_label_falls_back_to_round() -> None:
    """Some fixtures carry `round` only (e.g. older tournaments)."""
    fixture = {
        "pageUrl": "/matches/argentina-vs-france/1hox8a#3370572",
        "round": "F",
        "home": {"name": "Argentina"},
        "away": {"name": "France"},
        "status": {"utcTime": "2022-12-18T15:00:00Z", "scoreStr": "3 - 3"},
    }
    ref = ShootoutMatchRef.from_fixture(fixture)
    assert ref.round_name == "F"


def test_from_fixture_team_names() -> None:
    fixture = {
        "pageUrl": "/matches/argentina-vs-france/1hox8a#3370572",
        "roundName": "Final",
        "home": {"name": "Argentina"},
        "away": {"name": "France"},
        "status": {"utcTime": "2022-12-18T15:00:00Z", "scoreStr": "3 - 3"},
    }
    ref = ShootoutMatchRef.from_fixture(fixture)
    assert ref.home_name == "Argentina"
    assert ref.away_name == "France"
    assert ref.match_date == "2022-12-18T15:00:00Z"
    assert ref.score_str == "3 - 3"


# --- LEAGUE_SEASONS_PREDICT_WINDOW constant ---------------------------------


def test_league_seasons_constant_has_all_six_tournaments() -> None:
    from penalty_pred.shootouts import LEAGUE_SEASONS_PREDICT_WINDOW

    leagues = {lid for lid, _ in LEAGUE_SEASONS_PREDICT_WINDOW}
    assert leagues == {league.league_id for league in LEAGUE_BY_ID.values()}


def test_league_seasons_constant_covers_predict_window() -> None:
    """The 15 (league, season) pairs cover all tournaments with shootouts
    between 2021-01-01 and today (2026-06-22)."""
    from penalty_pred.shootouts import LEAGUE_SEASONS_PREDICT_WINDOW

    expected = {
        (77, 2022),  # World Cup 2022
        (77, 2026),  # World Cup 2026 (in progress)
        (50, 2020),  # Euro 2020 (held 2021)
        (50, 2024),  # Euro 2024
        (44, 2021),  # Copa América 2021
        (44, 2024),  # Copa América 2024
        (289, 2021),
        (289, 2023),
        (289, 2025),  # AFCON × 3
        (298, 2021),
        (298, 2023),
        (298, 2025),  # Gold Cup × 3
        (290, 2021),
        (290, 2023),
        (290, 2025),  # Asian Cup × 3
    }
    assert set(LEAGUE_SEASONS_PREDICT_WINDOW) == expected


# --- predict_window_bounds --------------------------------------------------


def test_predict_window_bounds_returns_pair() -> None:
    from datetime import UTC, datetime

    from penalty_pred.shootouts import predict_window_bounds

    start, end = predict_window_bounds()
    assert isinstance(start, datetime)
    assert isinstance(end, datetime)
    assert start.tzinfo == UTC
    assert end.tzinfo == UTC
    assert start <= end


# --- fetch_all_shootout_kicks_with_skips -----------------------------------


def test_skips_match_when_response_id_differs(
    tmp_path: Path, monkeypatch, sample_2022_final: dict[str, object]
) -> None:
    """A (seo, h2h) that resolves to a different matchId in the response
    (e.g. FotMob has reused the hash) is reported as `skipped=True`."""
    from penalty_pred.client import FotMobClient
    from penalty_pred.shootouts import (
        ShootoutMatchRef,
        fetch_all_shootout_kicks_with_skips,
    )

    # The sample is the 2022 final (matchId 3370572). Pretend the ref's
    # matchId is different from what the response says.
    fake_ref = ShootoutMatchRef(
        match_id=999999,  # not the real id
        seo="argentina-vs-france",
        h2h="1hox8a",
        round_name="Final",
        home_name="Argentina",
        away_name="France",
        match_date="2022-12-18T15:00:00Z",
        score_str="3 - 3",
    )
    client = FotMobClient(cache_dir=tmp_path)
    results = fetch_all_shootout_kicks_with_skips(client, [fake_ref])
    assert len(results) == 1
    assert results[0].skipped is True
    assert results[0].kicks == []


def test_processes_match_when_response_id_matches(
    tmp_path: Path, monkeypatch, sample_2022_final: dict[str, object]
) -> None:
    """A ref whose (seo, h2h) resolves to the right matchId is processed."""
    from penalty_pred.client import FotMobClient
    from penalty_pred.shootouts import (
        ShootoutMatchRef,
        fetch_all_shootout_kicks_with_skips,
    )

    real_ref = ShootoutMatchRef(
        match_id=3370572,
        seo="argentina-vs-france",
        h2h="1hox8a",
        round_name="Final",
        home_name="Argentina",
        away_name="France",
        match_date="2022-12-18T15:00:00Z",
        score_str="3 - 3",
    )
    client = FotMobClient(cache_dir=tmp_path)
    results = fetch_all_shootout_kicks_with_skips(client, [real_ref])
    assert len(results) == 1
    assert results[0].skipped is False
    assert len(results[0].kicks) == 8


def test_fetch_all_kicks_simple_still_works(
    tmp_path: Path, sample_2022_final: dict[str, object]
) -> None:
    """`fetch_all_shootout_kicks` keeps the simple (kicks-only) API for tests."""
    from penalty_pred.client import FotMobClient
    from penalty_pred.shootouts import (
        ShootoutMatchRef,
        fetch_all_shootout_kicks,
    )

    real_ref = ShootoutMatchRef(
        match_id=3370572,
        seo="argentina-vs-france",
        h2h="1hox8a",
        round_name="Final",
        home_name="Argentina",
        away_name="France",
        match_date="2022-12-18T15:00:00Z",
        score_str="3 - 3",
    )
    client = FotMobClient(cache_dir=tmp_path)
    kicks = list(fetch_all_shootout_kicks(client, [real_ref]))
    assert len(kicks) == 8


def test_fetch_all_kicks_skips_stale_match_silently(
    tmp_path: Path, sample_2022_final: dict[str, object]
) -> None:
    """`fetch_all_shootout_kicks` silently skips stale-URL matches."""
    from penalty_pred.client import FotMobClient
    from penalty_pred.shootouts import (
        ShootoutMatchRef,
        fetch_all_shootout_kicks,
    )

    fake_ref = ShootoutMatchRef(
        match_id=999999,
        seo="argentina-vs-france",
        h2h="1hox8a",
        round_name="Final",
        home_name="Argentina",
        away_name="France",
        match_date="2022-12-18T15:00:00Z",
        score_str="3 - 3",
    )
    client = FotMobClient(cache_dir=tmp_path)
    kicks = list(fetch_all_shootout_kicks(client, [fake_ref]))
    assert kicks == []  # silently skipped


def test_marks_match_as_no_kicks_when_shotmap_empty(
    tmp_path: Path, sample_2022_final: dict[str, object], monkeypatch
) -> None:
    """A match with the right matchId but an empty shootout shotmap is
    reported as `no_kicks=True` (not `skipped`)."""
    # The sample is the 2022 final — kick out the shootout shots so the
    # shotmap has 0 entries with period == "PenaltyShootout".
    import copy

    from penalty_pred.client import FotMobClient
    from penalty_pred.shootouts import (
        ShootoutMatchRef,
        fetch_all_shootout_kicks_with_skips,
    )

    data = copy.deepcopy(sample_2022_final)
    shots = data["pageProps"]["content"]["shotmap"]["shots"]
    data["pageProps"]["content"]["shotmap"]["shots"] = [
        s for s in shots if s.get("period") != "PenaltyShootout"
    ]

    # Stub the client to return our modified data for this ref.
    real_ref = ShootoutMatchRef(
        match_id=3370572,
        seo="argentina-vs-france",
        h2h="1hox8a",
        round_name="Final",
        home_name="Argentina",
        away_name="France",
        match_date="2022-12-18T15:00:00Z",
        score_str="3 - 3",
    )

    from penalty_pred import client as client_module

    def fake_get(self, path: str, params: dict | None = None) -> object:
        return data

    monkeypatch.setattr(client_module.FotMobClient, "get", fake_get)
    client = FotMobClient(cache_dir=tmp_path)
    results = fetch_all_shootout_kicks_with_skips(client, [real_ref])
    assert len(results) == 1
    assert results[0].skipped is False
    assert results[0].no_kicks is True
    assert results[0].kicks == []
