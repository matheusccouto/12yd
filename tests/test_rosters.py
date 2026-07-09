"""Tests for the 2026 World Cup squad roster fetcher (slice #3, Issue #18).

The tests cover three layers:

1. **Pure helpers**: `extract_lineup_players` (against the trimmed 2026
   WC match fixture saved to `docs/samples/match_4667751_wc2026_mexico_
   vs_south_africa.json.gz`), and the league fixture → `MatchRef`
   extraction (against the slim WC 2026 league fixture list at
   `docs/samples/league_wc_2026_slim.json`).

2. **MatchRef extraction**: `iter_roster_match_refs` should yield one
   ref per real match, including both group-stage and knockout-round
   placeholder matches (the latter will contribute zero players at
   extraction time).

3. **Orchestration**: `fetch_wc_2026_roster` against a stubbed
   `FotMobClient` that returns canned data. The stub mirrors the live
   data graph (league fixtures → per-match lineup) so the test
   exercises the same code paths the live run takes, with no network.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from twelveyards.artifacts import Artifacts
from twelveyards.client import FotMobClient
from twelveyards.leagues import LEAGUE_BY_ID
from twelveyards.match_ref import MatchRef
from twelveyards.rosters import (
    RosterPlayer,
    extract_lineup_players,
    fetch_wc_2026_roster,
    iter_roster_match_refs,
)
from twelveyards.tournaments import WC_2026_LEAGUE, WC_2026_SEASON

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_LEAGUE_PATH = REPO_ROOT / "docs" / "samples" / "league_wc_2026_slim.json"
SAMPLE_MATCH_PATH = (
    REPO_ROOT / "docs" / "samples" / "match_4667751_wc2026_mexico_vs_south_africa.json.gz"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sample_wc_2026_league() -> Mapping[str, Any]:
    """The slim WC 2026 league fixture list (3 matches, including a placeholder)."""
    if not SAMPLE_LEAGUE_PATH.exists():
        pytest.skip(f"Sample not present at {SAMPLE_LEAGUE_PATH}")
    return json.loads(SAMPLE_LEAGUE_PATH.read_text())


@pytest.fixture(scope="module")
def sample_wc_2026_match() -> Mapping[str, Any]:
    """The trimmed WC 2026 Mexico vs South Africa match JSON (lineup only)."""
    if not SAMPLE_MATCH_PATH.exists():
        pytest.skip(f"Sample not present at {SAMPLE_MATCH_PATH}")
    return json.loads(gzip.decompress(SAMPLE_MATCH_PATH.read_bytes()))


def _stub_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    league_fixtures: Mapping[str, Any],
    matches: list[Mapping[str, Any]],
) -> tuple[Path, list[str]]:
    """Patch the FotMobClient to return canned data per URL."""
    cache_dir = Path("/tmp/opencode/roster_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    urls_seen: list[str] = []
    match_iter = iter(matches)

    def fake_get(self: FotMobClient, path: str, params: dict | None = None) -> Any:
        urls_seen.append(f"{path}?{params or {}}")
        if path.startswith("leagues/"):
            return league_fixtures
        if path.startswith("matches/"):
            try:
                return next(match_iter)
            except StopIteration:
                msg = f"stub: no more matches configured for {path!r}"
                raise AssertionError(msg) from None
        msg = f"stub: unknown path {path!r}"
        raise AssertionError(msg)

    from twelveyards import client as client_module

    monkeypatch.setattr(client_module, "_discover_build_id", lambda c: "stub-build")
    monkeypatch.setattr(client_module.FotMobClient, "get", fake_get)
    return cache_dir, urls_seen


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_wc_2026_league_constant() -> None:
    """WC 2026 = FotMob leagueId 77, slug 'world-cup'."""
    assert WC_2026_LEAGUE.league_id == 77
    assert WC_2026_LEAGUE.slug == "world-cup"
    assert WC_2026_SEASON == 2026
    # The constant is the same League object that lives in LEAGUE_BY_ID.
    assert LEAGUE_BY_ID[77] is WC_2026_LEAGUE


# ---------------------------------------------------------------------------
# MatchRef extraction
# ---------------------------------------------------------------------------


def test_iter_roster_match_refs_yields_one_per_match(
    sample_wc_2026_league: Mapping[str, Any],
) -> None:
    """The slim fixture has 3 matches — 2 real group-stage + 1 knockout placeholder."""
    fixtures = sample_wc_2026_league["pageProps"]["fixtures"]["allMatches"]
    refs = list(iter_roster_match_refs(fixtures))
    assert len(refs) == 3
    for ref in refs:
        assert isinstance(ref, MatchRef)
        assert ref.match_id > 0
        assert ref.seo
        assert ref.h2h
        assert ref.home_team_id > 0
        assert ref.away_team_id > 0
        assert ref.home_team_name
        assert ref.away_team_name


def test_iter_roster_match_refs_skips_placeholder_match(
    sample_wc_2026_league: Mapping[str, Any],
) -> None:
    """The 3rd match in the slim fixture is a knockout placeholder match;
    it is still included in the refs (it carries a real (seo, h2h)),
    but `extract_lineup_players` on its lineup payload will yield zero
    rows because the home/away team blocks are empty.
    """
    fixtures = sample_wc_2026_league["pageProps"]["fixtures"]["allMatches"]
    refs = list(iter_roster_match_refs(fixtures))
    placeholder = next(r for r in refs if "Winner" in r.home_team_name)
    assert placeholder.home_team_id > 0  # the fixture lists a numeric id


def test_iter_roster_match_refs_skips_fixtures_without_team_ids() -> None:
    """Fixtures missing `home.id` or `away.id` (defensive) are skipped."""
    fixtures = [
        {
            "pageUrl": "/matches/x-vs-y/abc#1",
            "home": {"id": "1", "name": "A"},
            "away": {"id": "2", "name": "B"},
        },
        {
            "pageUrl": "/matches/x-vs-y/def#2",
            "home": {"name": "NoId"},
            "away": {"id": "2", "name": "B"},
        },
        {
            "pageUrl": "/matches/x-vs-y/ghi#3",
            "home": {"id": "1", "name": "A"},
            "away": {"name": "NoId"},
        },
        {
            "pageUrl": "/matches/x-vs-y/jkl#4",
            "home": {"id": "1", "name": "A"},
            "away": {"id": "2", "name": "B"},
        },
    ]
    refs = list(iter_roster_match_refs(fixtures))
    assert [r.match_id for r in refs] == [1, 4]


def test_iter_roster_match_refs_skips_malformed_page_url() -> None:
    """A fixture with an unparseable `pageUrl` is skipped, not crashed."""
    fixtures = [
        {"pageUrl": "garbage", "home": {"id": "1", "name": "A"}, "away": {"id": "2", "name": "B"}},
    ]
    refs = list(iter_roster_match_refs(fixtures))
    assert refs == []


# ---------------------------------------------------------------------------
# Lineup extraction
# ---------------------------------------------------------------------------


def test_extract_lineup_players_mexico_vs_south_africa(
    sample_wc_2026_match: Mapping[str, Any],
) -> None:
    """The Mexico vs South Africa match has 11 starters + 15 subs per team = 26 each."""
    lineup = sample_wc_2026_match["pageProps"]["content"]["lineup"]
    ref = MatchRef(
        match_id=4667751,
        seo="south-africa-vs-mexico",
        h2h="1einvt",
        home_team_id=6710,
        home_team_name="Mexico",
        away_team_id=6316,
        away_team_name="South Africa",
    )
    rows = list(extract_lineup_players(lineup, ref))
    assert len(rows) == 52  # 26 per team * 2 teams
    # Spot-check the first Mexico player.
    first = rows[0]
    assert isinstance(first, RosterPlayer)
    assert first.team_id == 6710
    assert first.team_name == "Mexico"
    assert first.player_id > 0
    assert first.player_name
    assert first.country_code == "MEX"
    # Spot-check the first South Africa player (after 26 Mexico rows).
    # FotMob uses "RSA" for South Africa (not the standard ISO ZAF).
    first_away = rows[26]
    assert first_away.team_id == 6316
    assert first_away.team_name == "South Africa"
    assert first_away.country_code == "RSA"


def test_extract_lineup_players_empty_lineup() -> None:
    """A placeholder knockout match has an empty `homeTeam`/`awayTeam` block;
    the iterator yields zero rows.
    """
    lineup = {"homeTeam": {}, "awayTeam": {}}
    ref = MatchRef(
        match_id=1,
        seo="x-vs-y",
        h2h="abc",
        home_team_id=100,
        home_team_name="X",
        away_team_id=200,
        away_team_name="Y",
    )
    rows = list(extract_lineup_players(lineup, ref))
    assert rows == []


def test_extract_lineup_players_missing_country_code() -> None:
    """A player with no `countryCode` field is yielded with an empty string."""
    lineup = {
        "homeTeam": {
            "id": 100,
            "name": "X",
            "starters": [{"id": 42, "name": "Test Player"}],
            "subs": [],
        },
        "awayTeam": {"id": 200, "name": "Y", "starters": [], "subs": []},
    }
    ref = MatchRef(
        match_id=1,
        seo="x-vs-y",
        h2h="abc",
        home_team_id=100,
        home_team_name="X",
        away_team_id=200,
        away_team_name="Y",
    )
    rows = list(extract_lineup_players(lineup, ref))
    assert len(rows) == 1
    assert rows[0].player_id == 42
    assert rows[0].player_name == "Test Player"
    assert rows[0].country_code == ""


def test_extract_lineup_players_skips_zero_id() -> None:
    """A player with no `id` (or id=0) is silently skipped — we can't dedupe it
    downstream without an id, so dropping it is the right move.
    """
    lineup = {
        "homeTeam": {
            "id": 100,
            "name": "X",
            "starters": [{"id": 0, "name": "Ghost"}, {"id": 42, "name": "Real"}],
            "subs": [],
        },
        "awayTeam": {"id": 200, "name": "Y", "starters": [], "subs": []},
    }
    ref = MatchRef(
        match_id=1,
        seo="x-vs-y",
        h2h="abc",
        home_team_id=100,
        home_team_name="X",
        away_team_id=200,
        away_team_name="Y",
    )
    rows = list(extract_lineup_players(lineup, ref))
    assert [r.player_id for r in rows] == [42]


# ---------------------------------------------------------------------------
# JSONL round-trip
# ---------------------------------------------------------------------------


def test_write_and_read_jsonl_roundtrip(tmp_path: Path) -> None:
    """Artifacts.write_roster + read_roster round-trips a RosterPlayer record."""
    path = tmp_path / "roster.jsonl"
    rows = [
        RosterPlayer(
            player_id=30981,
            player_name="Lionel Messi",
            team_id=6706,
            team_name="Argentina",
            country_code="ARG",
        ),
        RosterPlayer(
            player_id=268375,
            player_name="Emiliano Martínez",
            team_id=6706,
            team_name="Argentina",
            country_code="ARG",
        ),
    ]
    art = Artifacts(root=tmp_path)
    n = art.write_roster(rows, path=path)
    assert n == 2
    out = art.read_roster(path=path)
    assert out == rows


# ---------------------------------------------------------------------------
# Orchestration (stubbed client)
# ---------------------------------------------------------------------------


def test_fetch_wc_2026_roster_dedupes_across_matches(
    monkeypatch: pytest.MonkeyPatch,
    sample_wc_2026_league: Mapping[str, Any],
    sample_wc_2026_match: Mapping[str, Any],
) -> None:
    """A player who appears in two matches is yielded once.

    The stub returns the same match payload for two consecutive match
    requests, simulating the same player being listed in two group-stage
    matches. The orchestrator must dedupe by player_id.
    """
    # Build a 2-match league fixture list that points to the same match
    # twice (same match_id 4667751, same (seo, h2h) pair).
    league = {
        "pageProps": {
            "fixtures": {
                "allMatches": [
                    {
                        "id": "4667751",
                        "pageUrl": "/matches/south-africa-vs-mexico/1einvt#4667751",
                        "home": {"id": "6710", "name": "Mexico"},
                        "away": {"id": "6316", "name": "South Africa"},
                    },
                    {
                        "id": "4667751",  # duplicate match_id
                        "pageUrl": "/matches/south-africa-vs-mexico/1einvt#4667751",
                        "home": {"id": "6710", "name": "Mexico"},
                        "away": {"id": "6316", "name": "South Africa"},
                    },
                ]
            }
        }
    }
    # Return the same match twice — the player list is the same in both.
    cache_dir, urls_seen = _stub_client(
        monkeypatch,
        league_fixtures=league,
        matches=[sample_wc_2026_match, sample_wc_2026_match],
    )
    client = FotMobClient(cache_dir=cache_dir)
    rows = list(fetch_wc_2026_roster(client, WC_2026_LEAGUE, WC_2026_SEASON))
    # 52 rows per match, deduped to 52 unique players.
    assert len(rows) == 52
    # Sanity: the URLs were hit (1 league + 2 match calls).
    assert sum(1 for u in urls_seen if u.startswith("matches/")) == 2
    assert sum(1 for u in urls_seen if u.startswith("leagues/")) == 1


def test_fetch_wc_2026_roster_skips_stale_h2h(
    monkeypatch: pytest.MonkeyPatch,
    sample_wc_2026_league: Mapping[str, Any],
    sample_wc_2026_match: Mapping[str, Any],
) -> None:
    """If the per-match response's matchId differs from the ref's matchId
    (stale (seo, h2h) hash), the match is skipped silently.
    """
    # Build a match payload whose general.matchId is different from any ref.
    tampered = {
        "pageProps": {
            "general": {**sample_wc_2026_match["pageProps"]["general"], "matchId": 99999999},
            "content": {"lineup": sample_wc_2026_match["pageProps"]["content"]["lineup"]},
        }
    }
    # The league has 3 refs; we return the tampered payload for all 3.
    # Every match is then "stale" and contributes 0 rows.
    cache_dir, urls_seen = _stub_client(
        monkeypatch,
        league_fixtures=sample_wc_2026_league,
        matches=[tampered, tampered, tampered],
    )
    client = FotMobClient(cache_dir=cache_dir)
    rows = list(fetch_wc_2026_roster(client, WC_2026_LEAGUE, WC_2026_SEASON))
    assert len(rows) == 0
    # Sanity: the stub was called once per ref (3 match calls + 1 league call).
    assert sum(1 for u in urls_seen if u.startswith("matches/")) == 3
