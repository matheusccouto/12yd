"""Tests for the per-kicker penalty history fetcher (slice #4, Issue #20; refined #36).

The tests cover three layers:

1. **Pure helpers**: `season_name_to_year`, `compute_lookback_window`,
   `extract_player_metadata`, `iter_career_season_entries`,
   `iter_team_season_lookups`, `filter_fixtures_by_team`. No network.
   v3 (Issue #36) added `_preferred_foot` (reads from
   `pageProps.data.playerInformation[]`) and the corresponding
   `PlayerMetadata.preferred_foot` field; the helper tests pin the
   four shapes the function must handle.

2. **Per-match extraction**: `extract_player_penalties_from_match` on
   the 2022 WC Final (cached at `docs/samples/match_3370572.json.gz`).
   The final has 2 Messi penalty shots — one in-match and one in the
   shootout — both of which we should recover.

3. **Orchestration**: `fetch_player_penalty_history` against a stubbed
   `FotMobClient` that returns canned data. The stub mirrors the live
   data graph (player page → league/season fixtures → per-match
   details) so the test exercises the same code paths the live run
   takes, with no network.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from penalty_pred.artifacts import Artifacts
from penalty_pred.client import FotMobClient
from penalty_pred.player_history import (
    PlayerMetadata,
    PlayerPenalty,
    TeamSeasonLookup,
    compute_lookback_window,
    extract_player_metadata,
    extract_player_penalties_from_match,
    fetch_player_penalty_history,
    filter_fixtures_by_team,
    iter_career_season_entries,
    iter_team_season_lookups,
    season_name_to_year,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_MATCH_PATH = REPO_ROOT / "docs" / "samples" / "match_3370572.json.gz"
SAMPLE_PLAYER_PATH = REPO_ROOT / "docs" / "samples" / "player_30981_messi.json.gz"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sample_2022_final() -> Mapping[str, object]:
    """The 2022 WC Final (Argentina vs France) match JSON, off the disk cache."""
    if not SAMPLE_MATCH_PATH.exists():
        pytest.skip(f"Sample match not present at {SAMPLE_MATCH_PATH}")
    return json.loads(gzip.decompress(SAMPLE_MATCH_PATH.read_bytes()))


@pytest.fixture(scope="module")
def sample_messi_player() -> Mapping[str, object]:
    """The Messi player page JSON, off the disk cache.

    The sample was fetched on 2026-06-22 from the live FotMob API and
    saved to `docs/samples/player_30981_messi.json.gz`. It is used
    here as a fixture so the per-player parsing helpers can be tested
    offline.
    """
    if not SAMPLE_PLAYER_PATH.exists():
        pytest.skip(f"Sample player not present at {SAMPLE_PLAYER_PATH}")
    return json.loads(gzip.decompress(SAMPLE_PLAYER_PATH.read_bytes()))


def _stub_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    player_page: Mapping[str, object] | None = None,
    league_fixtures: Mapping[tuple[int, int], list[dict[str, object]]] | None = None,
    matches: list[Mapping[str, object]] | None = None,
) -> tuple[Path, list[str]]:
    """Patch the FotMobClient to return canned data per URL.

    Returns (cache_dir, urls_seen). The stub builds the URL the way
    `FotMobClient.get` does, so the stub can match on the path +
    params. The BuildId is pinned to "stub-build" by skipping the
    live discovery.

    `league_fixtures` is keyed by `(league_id, season_year)`. `matches`
    is a list of canned match payloads — the stub returns the next
    payload from the list for each `matches/...` request, in order.
    This is enough for the orchestrator's per-match tests since the
    test controls both the fixture and the call sequence.
    """
    cache_dir = Path("/tmp/opencode/player_history_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    urls_seen: list[str] = []
    match_iter = iter(matches or [])

    def fake_get(self: FotMobClient, path: str, params: dict | None = None) -> Any:
        urls_seen.append(f"{path}?{params or {}}")
        if path.startswith("players/"):
            if player_page is None:
                msg = f"stub: no player_page configured for {path!r}"
                raise AssertionError(msg)
            return {"pageProps": {"data": player_page}}
        if path.startswith("leagues/"):
            # leagues/{leagueId}/overview/{slug}
            parts = path.split("/")
            league_id = int(parts[1])
            season_year = int((params or {}).get("season", "0"))
            fixtures = (league_fixtures or {}).get((league_id, season_year))
            if fixtures is None:
                msg = f"stub: no league_fixtures for league={league_id} season={season_year}"
                raise AssertionError(msg)
            return {"pageProps": {"fixtures": {"allMatches": fixtures}}}
        if path.startswith("matches/"):
            try:
                return next(match_iter)
            except StopIteration:
                msg = f"stub: no more matches configured for {path!r}"
                raise AssertionError(msg) from None
        msg = f"stub: unknown path {path!r}"
        raise AssertionError(msg)

    # Pin the BuildId to avoid hitting the homepage.
    from penalty_pred import client as client_module

    monkeypatch.setattr(client_module, "_discover_build_id", lambda c: "stub-build")
    monkeypatch.setattr(client_module.FotMobClient, "get", fake_get)
    return cache_dir, urls_seen


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("season_name", "expected_year"),
    [
        ("2022", 2022),  # calendar-year league
        ("2020/2021", 2020),  # split-year league
        ("2022 Qatar", 2022),  # tournament year with location suffix
        ("2018 Russia", 2018),
        ("2014 Brazil", 2014),
        ("2009 UAE", 2009),
    ],
)
def test_season_name_to_year(season_name: str, expected_year: int) -> None:
    assert season_name_to_year(season_name) == expected_year


def test_season_name_to_year_raises_on_garbage() -> None:
    with pytest.raises(ValueError, match="Cannot extract year"):
        season_name_to_year("not a season")


def test_compute_lookback_window_with_floor() -> None:
    """For a 2022-12-18 target with 5y lookback and 2016-01-01 floor, the
    floor doesn't kick in (5y back is 2017-12-18, which is after 2016-01-01)."""
    start, end = compute_lookback_window(date(2022, 12, 18), 5, date(2016, 1, 1))
    assert start == date(2017, 12, 18)
    assert end == date(2022, 12, 18)


def test_compute_lookback_window_floor_kicks_in() -> None:
    """For a 2020-12-18 target with 5y back, naive start is 2015-12-18,
    before the 2016-01-01 floor — the floor wins."""
    start, end = compute_lookback_window(date(2020, 12, 18), 5, date(2016, 1, 1))
    assert start == date(2016, 1, 1)
    assert end == date(2020, 12, 18)


def test_compute_lookback_window_zero_lookback() -> None:
    """lookback_years=0 yields a zero-width window at the target date."""
    start, end = compute_lookback_window(date(2022, 12, 18), 0, date(2016, 1, 1))
    assert start == date(2022, 12, 18)
    assert end == date(2022, 12, 18)


# ---------------------------------------------------------------------------
# Player metadata
# ---------------------------------------------------------------------------


def test_extract_player_metadata_messi(sample_messi_player: Mapping[str, object]) -> None:
    """The Messi fixture has player_id=30981, position='striker', birth 1987-06-24,
    preferred_foot='left' (v3: read from playerInformation[])."""
    md = extract_player_metadata(sample_messi_player)
    assert isinstance(md, PlayerMetadata)
    assert md.player_id == 30981
    assert md.player_name == "Lionel Messi"
    assert md.position_key == "striker"
    assert md.birth_date == "1987-06-24"
    assert md.preferred_foot == "left"


def test_extract_player_metadata_reads_preferred_foot() -> None:
    """v3 (Issue #36): the A3 feature is the declared preferred foot,
    read from `pageProps.data.playerInformation[]` (the cached
    player-page JSON the scraper already fetches).

    Tests the four shapes we need to handle:
    - `value.key` in {"left", "right", "both"} → returned
    - `value.key` unknown → "" (defensive)
    - entry missing entirely → "" (defensive)
    - other entries (height, shirt, etc.) ignored
    """
    from penalty_pred.player_history import _preferred_foot

    # Standard "left" via the cache's value.key shape.
    assert (
        _preferred_foot(
            [{"translationKey": "preferred_foot", "value": {"key": "left", "fallback": "Left"}}]
        )
        == "left"
    )
    # "right" and "both" round-trip.
    assert (
        _preferred_foot(
            [{"translationKey": "preferred_foot", "value": {"key": "right", "fallback": "Right"}}]
        )
        == "right"
    )
    assert (
        _preferred_foot(
            [{"translationKey": "preferred_foot", "value": {"key": "both", "fallback": "Both"}}]
        )
        == "both"
    )
    # Unknown key → "" (defensive; the model treats it as missing).
    assert (
        _preferred_foot(
            [{"translationKey": "preferred_foot", "value": {"key": "switch", "fallback": "?"}}]
        )
        == ""
    )
    # No preferred_foot entry → "".
    assert (
        _preferred_foot([{"translationKey": "height_sentencecase", "value": {"numberValue": 180}}])
        == ""
    )
    # Empty list → "".
    assert _preferred_foot([]) == ""


def test_extract_player_metadata_handles_missing_preferred_foot() -> None:
    """A player page with no `playerInformation[]` (or with the
    `preferred_foot` entry missing) yields `preferred_foot=""` (not
    a crash). The other C-group fields (position, birth date) are
    unaffected."""
    md = extract_player_metadata(
        {
            "pageProps": {
                "data": {
                    "id": 42,
                    "name": "Test",
                    "positionDescription": {"primaryPosition": {"key": "midfielder"}},
                    "birthDate": {"utcTime": "1990-01-01T00:00:00.000Z"},
                    "playerInformation": [
                        {"translationKey": "height_sentencecase", "value": {"numberValue": 180}},
                    ],
                }
            }
        }
    )
    assert md.player_id == 42
    assert md.position_key == "midfielder"
    assert md.birth_date == "1990-01-01"
    assert md.preferred_foot == ""


def test_extract_player_metadata_missing_position() -> None:
    """A player with no positionDescription falls back to empty string."""
    md = extract_player_metadata(
        {"pageProps": {"data": {"id": 42, "name": "Test", "birthDate": None}}}
    )
    assert md.player_id == 42
    assert md.player_name == "Test"
    assert md.position_key == ""
    assert md.birth_date == ""


def test_extract_player_metadata_handles_malformed_birthdate() -> None:
    """A malformed birthDate falls back to empty string (not a crash)."""
    md = extract_player_metadata(
        {"pageProps": {"data": {"id": 42, "name": "Test", "birthDate": {"utcTime": "garbage"}}}}
    )
    assert md.birth_date == ""


# ---------------------------------------------------------------------------
# Career history traversal
# ---------------------------------------------------------------------------


def test_iter_career_season_entries_yields_senior_and_national(
    sample_messi_player: Mapping[str, object],
) -> None:
    """Messi's career history has 23 senior + 22 national team season entries = 45 total."""
    entries = list(iter_career_season_entries(sample_messi_player))
    assert len(entries) == 45
    # The first few entries are the most recent senior stints (Inter Miami, PSG).
    # We don't pin exact indices because FotMob may reorder; we just check
    # that the buckets are both present.
    seasons = {e["seasonName"] for e in entries}
    assert "2026" in seasons  # current Inter Miami / Argentina season
    assert "2022/2023" in seasons  # PSG


def test_iter_career_season_entries_skips_youth_bucket() -> None:
    """A player page with a `careerItems.youth` bucket has those entries skipped.

    PRD: "skip `careerItems.youth`". We construct a synthetic page that
    has youth entries with a unique sentinel teamId, and assert the
    sentinel never appears in the output.
    """
    payload = {
        "pageProps": {
            "data": {
                "id": 1,
                "name": "X",
                "careerHistory": {
                    "careerItems": {
                        "senior": {
                            "seasonEntries": [
                                {
                                    "seasonName": "2020",
                                    "teamId": 100,
                                    "tournamentStats": [],
                                }
                            ]
                        },
                        "national team": {"seasonEntries": []},
                        "youth": {
                            "seasonEntries": [
                                {
                                    "seasonName": "2018",
                                    "teamId": 999,
                                    "tournamentStats": [],
                                }
                            ]
                        },
                    }
                },
            }
        }
    }
    entries = list(iter_career_season_entries(payload))
    team_ids = {int(e.get("teamId") or 0) for e in entries}
    assert 999 not in team_ids
    assert 100 in team_ids


def test_iter_team_season_lookups_yields_one_per_tournament(
    sample_messi_player: Mapping[str, object],
) -> None:
    """Each season entry has multiple tournamentStats; each becomes a TeamSeasonLookup.
    Lookups without a `leagueId` (e.g. CONMEBOL Qualifiers) are skipped — we
    can't form a FotMob URL without one.
    """
    entries = list(iter_career_season_entries(sample_messi_player))
    lookups = list(iter_team_season_lookups(entries))
    assert len(lookups) > 0
    for lu in lookups:
        assert isinstance(lu, TeamSeasonLookup)
        assert lu.team_id > 0
        assert lu.tournament_stat.get("leagueId")  # every yielded lookup has one


def test_iter_team_season_lookups_skips_stats_without_league_id() -> None:
    """A `tournamentStat` with no `leagueId` is skipped."""
    entries = [
        {
            "seasonName": "2026",
            "teamId": 100,
            "tournamentStats": [
                {"leagueId": 42, "leagueName": "Has ID", "seasonName": "2026"},
                {"leagueName": "No ID", "seasonName": "2026"},  # no leagueId
            ],
        }
    ]
    lookups = list(iter_team_season_lookups(entries))
    assert len(lookups) == 1
    assert lookups[0].tournament_stat["leagueName"] == "Has ID"


# ---------------------------------------------------------------------------
# filter_fixtures_by_team
# ---------------------------------------------------------------------------


def _fixture(
    match_id: int, home_id: int, away_id: int, home_name: str = "Home", away_name: str = "Away"
) -> dict[str, object]:
    return {
        "id": str(match_id),
        "pageUrl": f"/matches/{home_name.lower()}-vs-{away_name.lower()}/{match_id:06x}#{match_id}",
        "home": {"id": str(home_id), "name": home_name},
        "away": {"id": str(away_id), "name": away_name},
        "status": {
            "utcTime": "2022-12-18T15:00:00Z",
            "scoreStr": "0 - 0",
            "reason": {"shortKey": "fulltime_short"},
        },
    }


def test_filter_fixtures_by_team_keeps_home_and_away() -> None:
    """A fixture is kept if the team is home OR away."""
    f1 = _fixture(1, home_id=100, away_id=200)
    f2 = _fixture(2, home_id=300, away_id=100)  # team 100 is away
    f3 = _fixture(3, home_id=300, away_id=400)  # team 100 not involved
    kept = list(filter_fixtures_by_team([f1, f2, f3], team_id=100))
    assert [f["id"] for f in kept] == ["1", "2"]


def test_filter_fixtures_by_team_drops_unknown_team() -> None:
    """Fixtures with no home/away id (e.g. friendly metadata placeholders) are dropped."""
    bad = {
        "id": "5",
        "pageUrl": "/matches/x/y#5",
        "home": {"id": "", "name": "?"},
        "away": {"id": "0", "name": "?"},
        "status": {"utcTime": "2022-01-01T00:00:00Z"},
    }
    good = _fixture(6, home_id=100, away_id=200)
    kept = list(filter_fixtures_by_team([bad, good], team_id=100))
    assert [f["id"] for f in kept] == ["6"]


# ---------------------------------------------------------------------------
# Per-match extraction
# ---------------------------------------------------------------------------


def test_extract_penalties_from_2022_final_returns_two_messi_kicks(
    sample_2022_final: Mapping[str, object],
) -> None:
    """The 2022 final has 2 Messi penalty shots: one in-match (23') and
    one shootout (kick 2). Both should be returned."""
    rows = extract_player_penalties_from_match(
        sample_2022_final,
        player_id=30981,
        team_id=6706,
        league_id=77,  # WC
        league_name="World Cup",
    )
    assert len(rows) == 2
    for row in rows:
        assert isinstance(row, PlayerPenalty)
        assert row.kicker_id == 30981
        assert row.match_id == 3370572
        assert 0.0 <= row.x <= 2.0
        assert row.side in {"L", "C", "R"}
        assert row.outcome in {"Goal", "Saved", "Missed"}
        assert row.shot_type in {"RightFoot", "LeftFoot"}
        assert row.is_home is True  # Argentina (team 6706) was home
        assert row.league_id == 77


def test_extract_penalties_from_2022_final_includes_in_match_and_shootout(
    sample_2022_final: Mapping[str, object],
) -> None:
    """The two Messi kicks span two distinct `period` values. We don't
    assert on the period here (it isn't carried in the row), but the row
    count of 2 covers both the in-match penalty (23') and the shootout kick."""
    rows = extract_player_penalties_from_match(sample_2022_final, 30981, 6706, 77, "World Cup")
    assert len(rows) == 2


def test_extract_penalties_returns_empty_for_non_kicker(
    sample_2022_final: Mapping[str, object],
) -> None:
    """A player who didn't take any penalty in the match yields no rows."""
    rows = extract_player_penalties_from_match(
        sample_2022_final,
        player_id=999999,  # nobody
        team_id=6706,
        league_id=77,
        league_name="World Cup",
    )
    assert rows == []


def test_extract_penalties_uses_match_league_name_when_present(
    sample_2022_final: Mapping[str, object],
) -> None:
    """The `league_name` parameter is a fallback; the match JSON's
    `general.leagueName` wins when present."""
    rows = extract_player_penalties_from_match(
        sample_2022_final,
        player_id=30981,
        team_id=6706,
        league_id=77,
        league_name="WRONG FALLBACK",
    )
    assert all(r.league_name == "World Cup" for r in rows)


# ---------------------------------------------------------------------------
# Artifacts.write_player_history roundtrip
# ---------------------------------------------------------------------------


def test_write_player_history_roundtrip(tmp_path: Path) -> None:
    """Writing then reading a JSONL file gives the same rows back."""
    rows = [
        PlayerPenalty(
            kicker_id=30981,
            match_id=3370572,
            match_date="2022-12-18T15:00:00+00:00",
            league_id=77,
            league_name="World Cup",
            team_id=6706,
            is_home=True,
            x=0.5,
            side="L",
            is_on_target=True,
            outcome="Goal",
            shot_type="RightFoot",
        )
    ]
    out = tmp_path / "x.jsonl"
    art = Artifacts(root=tmp_path)
    n = art.write_player_history(rows, path=out)
    assert n == 1
    with out.open() as f:
        for line in f:
            row = json.loads(line)
            assert row["kicker_id"] == 30981
            assert row["side"] == "L"
            assert row["x"] == 0.5


# ---------------------------------------------------------------------------
# Orchestration (stubbed FotMobClient)
# ---------------------------------------------------------------------------


def _build_minimal_player_page(player_id: int) -> dict[str, object]:
    """Build a minimal player page with a single (team, season) entry.

    The page declares a one-year stint on team 100, playing in league 42
    in season 2020/2021. The orchestrator's only required output is
    the per-match penalty shots in the canned match payload; everything
    else is plumbing.
    """
    return {
        "id": player_id,
        "name": "Stub Player",
        "birthDate": {"utcTime": "1990-01-01T00:00:00.000Z", "timezone": "UTC"},
        "positionDescription": {"primaryPosition": {"key": "striker", "label": "Striker"}},
        "careerHistory": {
            "careerItems": {
                "senior": {
                    "teamEntries": [],
                    "seasonEntries": [
                        {
                            "seasonName": "2020/2021",
                            "teamId": 100,
                            "team": "Stub Team",
                            "tournamentStats": [
                                {
                                    "leagueId": 42,
                                    "leagueName": "Stub League",
                                    "seasonName": "2020/2021",
                                }
                            ],
                        }
                    ],
                },
                "national team": {"teamEntries": [], "seasonEntries": []},
            }
        },
    }


def _build_minimal_match(
    match_id: int,
    *,
    utc_time: str = "2020-06-15T19:00:00Z",
    home_id: int = 100,
    away_id: int = 200,
    player_id: int = 30981,
    include_penalty: bool = True,
    shot_team_id: int | None = None,
) -> dict[str, object]:
    """Build a minimal match JSON with optional penalty shot by the player."""
    shots: list[dict[str, object]] = []
    if include_penalty:
        shots.append(
            {
                "id": 1,
                "eventType": "Goal",
                "teamId": shot_team_id if shot_team_id is not None else home_id,
                "playerId": player_id,
                "playerName": "Stub Player",
                "situation": "Penalty",
                "period": "FirstHalf",
                "isOnTarget": True,
                "shotType": "RightFoot",
                "onGoalShot": {"x": 0.5, "y": 0.1, "zoomRatio": 1},
            }
        )
    return {
        "pageProps": {
            "general": {
                "matchId": match_id,
                "matchTimeUTC": utc_time,
                "leagueId": 42,
                "leagueName": "Stub League",
            },
            "header": {
                "teams": [
                    {"id": home_id, "name": "Home Team", "score": 1},
                    {"id": away_id, "name": "Away Team", "score": 0},
                ]
            },
            "content": {"shotmap": {"shots": shots}},
        }
    }


def test_orchestrator_yields_penalty_from_canned_match(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """End-to-end: the stub client serves a player page + one league
    season + one match. The orchestrator should yield that one penalty."""
    player_id = 30981
    match_id = 99999
    _cache_dir, _urls = _stub_client(
        monkeypatch,
        player_page=_build_minimal_player_page(player_id),
        league_fixtures={
            (42, 2020): [
                {
                    "id": str(match_id),
                    "pageUrl": f"/matches/home-vs-away/abcdef#{match_id}",
                    "home": {"id": "100", "name": "Home Team"},
                    "away": {"id": "200", "name": "Away Team"},
                    "status": {
                        "utcTime": "2020-06-15T19:00:00Z",
                        "scoreStr": "1 - 0",
                        "reason": {"shortKey": "fulltime_short"},
                    },
                }
            ]
        },
        matches=[_build_minimal_match(match_id, player_id=player_id)],
    )
    client = FotMobClient(cache_dir=tmp_path / "stub")
    rows = list(
        fetch_player_penalty_history(
            client,
            player_id=player_id,
            player_slug="stub-player",
            target_date=date(2022, 12, 18),
        )
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.kicker_id == player_id
    assert row.match_id == match_id
    assert row.x == 0.5
    assert row.side == "L"
    assert row.outcome == "Goal"
    assert row.shot_type == "RightFoot"
    assert row.is_home is True


def test_orchestrator_skips_match_outside_lookback_window(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A match dated before the lookback window is silently skipped
    (we never even fetch the match JSON)."""
    player_id = 30981
    match_id_early = 11111
    match_id_in_window = 22222
    _cache_dir, urls_seen = _stub_client(
        monkeypatch,
        player_page=_build_minimal_player_page(player_id),
        league_fixtures={
            (42, 2020): [
                {
                    "id": str(match_id_early),
                    "pageUrl": f"/matches/early-vs-x/aaa111#{match_id_early}",
                    "home": {"id": "100", "name": "Home"},
                    "away": {"id": "200", "name": "Away"},
                    "status": {
                        "utcTime": "2015-01-01T00:00:00Z",  # before floor
                        "scoreStr": "1 - 0",
                    },
                },
                {
                    "id": str(match_id_in_window),
                    "pageUrl": f"/matches/in-vs-window/bbb222#{match_id_in_window}",
                    "home": {"id": "100", "name": "Home"},
                    "away": {"id": "200", "name": "Away"},
                    "status": {
                        "utcTime": "2020-06-15T19:00:00Z",  # in window
                        "scoreStr": "1 - 0",
                    },
                },
            ]
        },
        matches=[
            _build_minimal_match(match_id_in_window, player_id=player_id),
        ],
    )
    client = FotMobClient(cache_dir=tmp_path / "stub")
    rows = list(
        fetch_player_penalty_history(
            client,
            player_id=player_id,
            player_slug="stub-player",
            target_date=date(2022, 12, 18),
        )
    )
    # Only the in-window match yields a row; the early match is skipped.
    assert len(rows) == 1
    assert rows[0].match_id == match_id_in_window
    # The early match's URL is never requested (the orchestrator filters
    # by date before fetching per-match).
    match_urls = [u for u in urls_seen if u.startswith("matches/")]
    assert len(match_urls) == 1
    assert "bbb222" in match_urls[0]


def test_orchestrator_skips_stale_url_match(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A (seo, h2h) hash that points to a different matchId in the
    response is skipped silently, like in the shootout pipeline."""
    player_id = 30981
    stale_match_id = 33333
    real_match_id = 44444
    _cache_dir, _urls = _stub_client(
        monkeypatch,
        player_page=_build_minimal_player_page(player_id),
        league_fixtures={
            (42, 2020): [
                {
                    "id": str(stale_match_id),
                    "pageUrl": f"/matches/x-vs-y/cccccc#{stale_match_id}",
                    "home": {"id": "100", "name": "Home"},
                    "away": {"id": "200", "name": "Away"},
                    "status": {
                        "utcTime": "2020-06-15T19:00:00Z",
                        "scoreStr": "1 - 0",
                    },
                }
            ]
        },
        # The canned payload is for a DIFFERENT matchId — simulates FotMob
        # reusing a hash and pointing us at a newer match.
        matches=[_build_minimal_match(real_match_id, player_id=player_id)],
    )
    client = FotMobClient(cache_dir=tmp_path / "stub")
    rows = list(
        fetch_player_penalty_history(
            client,
            player_id=player_id,
            player_slug="stub-player",
            target_date=date(2022, 12, 18),
        )
    )
    # The stale match is silently skipped; no row is yielded.
    assert rows == []


def test_orchestrator_dedupes_team_season_lookups(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Two season entries that map to the same (team, league, season)
    are deduped to a single fixture fetch."""
    player_id = 30981
    fetch_count = {"n": 0}
    match_id = 55555

    def fake_get(self: FotMobClient, path: str, params: dict | None = None) -> Any:
        if path.startswith("players/"):
            return {
                "pageProps": {
                    "data": {
                        "id": player_id,
                        "name": "Stub Player",
                        "birthDate": None,
                        "positionDescription": {
                            "primaryPosition": {"key": "striker", "label": "Striker"}
                        },
                        "careerHistory": {
                            "careerItems": {
                                "senior": {
                                    "teamEntries": [],
                                    "seasonEntries": [
                                        # Two season entries for the same
                                        # (team, league, season) — should
                                        # dedupe to one fixture fetch.
                                        {
                                            "seasonName": "2020/2021",
                                            "teamId": 100,
                                            "team": "T",
                                            "tournamentStats": [
                                                {
                                                    "leagueId": 42,
                                                    "leagueName": "L",
                                                    "seasonName": "2020/2021",
                                                }
                                            ],
                                        },
                                        {
                                            "seasonName": "2020/2021",
                                            "teamId": 100,
                                            "team": "T",
                                            "tournamentStats": [
                                                {
                                                    "leagueId": 42,
                                                    "leagueName": "L",
                                                    "seasonName": "2020/2021",
                                                }
                                            ],
                                        },
                                    ],
                                },
                                "national team": {
                                    "teamEntries": [],
                                    "seasonEntries": [],
                                },
                            }
                        },
                    }
                }
            }
        if path.startswith("leagues/"):
            fetch_count["n"] += 1
            return {
                "pageProps": {
                    "fixtures": {
                        "allMatches": [
                            {
                                "id": str(match_id),
                                "pageUrl": f"/matches/x-vs-y/dddddd#{match_id}",
                                "home": {"id": "100", "name": "H"},
                                "away": {"id": "200", "name": "A"},
                                "status": {
                                    "utcTime": "2020-06-15T19:00:00Z",
                                    "scoreStr": "1 - 0",
                                },
                            }
                        ]
                    }
                }
            }
        if path.startswith("matches/"):
            return _build_minimal_match(match_id, player_id=player_id)
        msg = f"unknown path: {path}"
        raise AssertionError(msg)

    from penalty_pred import client as client_module

    monkeypatch.setattr(client_module, "_discover_build_id", lambda c: "stub-build")
    monkeypatch.setattr(client_module.FotMobClient, "get", fake_get)
    client = FotMobClient(cache_dir=tmp_path / "stub")
    list(
        fetch_player_penalty_history(
            client,
            player_id=player_id,
            player_slug="stub-player",
            target_date=date(2022, 12, 18),
        )
    )
    # Two duplicate (team, league, season) lookups → one fixture fetch.
    assert fetch_count["n"] == 1
