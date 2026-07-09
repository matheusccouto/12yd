"""Tests for the Initial Set assembly and per-kicker orchestration.

The tests cover three layers:

1. **Pure helpers**: `iter_initial_set_kickers` dedup, ordering, and
   enrichment. No network. The typed iterables are sourced from JSONL
   through the same `Artifacts` adapter the slice script uses, so the
   test surface follows the production surface.

2. **Orchestration**: `fetch_all_initial_set_penalty_history` against a
   stubbed `FotMobClient` that returns canned per-kicker data. Verifies
   the per-kicker yield, the error-capture (a single bad kicker must
   not abort the run), and the empty input case.

3. **JSONL helpers**: `write_missing_history` roundtrip and the
   `InitialSetKicker` / `MissingKicker` / `InitialSetFetchResult`
   dataclass shapes.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from twelveyards.artifacts import Artifacts
from twelveyards.client import FotMobClient
from twelveyards.initial_set import (
    InitialSetFetchResult,
    InitialSetKicker,
    MissingKicker,
    fetch_all_initial_set_penalty_history,
    iter_initial_set_kickers,
)
from twelveyards.player_history import fetch_player_data
from twelveyards.rosters import RosterPlayer
from twelveyards.shootouts import ShootoutKick
from tests._factories import (
    MISSING_KICKER_FIELDS,
    PLAYER_PENALTY_FIELDS,
    make_roster_player,
    make_shootout_kick,
)

# ---------------------------------------------------------------------------
# Typed row builders (the iter_initial_set_kickers tests)
# ---------------------------------------------------------------------------


def _shootout_kick(
    kicker_id: int,
    kicker_name: str = "Stub",
    team_id: int = 0,
    match_id: int = 1,
) -> ShootoutKick:
    """A `ShootoutKick` carrying the fields `iter_initial_set_kickers`
    reads. The factory's defaults cover the rest."""
    return make_shootout_kick(
        match_id=match_id,
        kick_number=1,
        kicker_id=kicker_id,
        kicker_name=kicker_name,
        team_id=team_id,
    )


def _roster_player(
    player_id: int,
    player_name: str = "Stub",
    team_id: int = 0,
    team_name: str = "",
) -> RosterPlayer:
    return make_roster_player(
        player_id=player_id,
        player_name=player_name,
        team_id=team_id,
        team_name=team_name,
    )


# ---------------------------------------------------------------------------
# iter_initial_set_kickers
# ---------------------------------------------------------------------------


def test_iter_initial_set_kickers_dedupes_by_player_id() -> None:
    """Training kickers are emitted first, then roster-only kickers.
    Kickers present in both sets keep the roster record (because the
    roster row has the team_name)."""
    shootout_kicks = [
        _shootout_kick(kicker_id=100, kicker_name="Alpha", team_id=1),
        _shootout_kick(kicker_id=200, kicker_name="Bravo", team_id=1),
        # Duplicate kicker_id 100 within the training set itself — the
        # same player can take multiple kicks in a single shootout.
        # Dedupe is by player_id, so this second row is dropped.
        _shootout_kick(kicker_id=100, kicker_name="Alpha", team_id=1),
    ]
    roster = [
        # Bravo is in both — roster row wins (has team_name).
        _roster_player(200, "Bravo", team_id=1, team_name="Argentina"),
        # Charlie is roster-only.
        _roster_player(300, "Charlie", team_id=2, team_name="Brazil"),
    ]

    kickers = list(iter_initial_set_kickers(shootout_kicks, roster))
    assert len(kickers) == 3
    assert [k.player_id for k in kickers] == [100, 200, 300]
    # Training-only row (Alpha) has no team_name.
    assert kickers[0].player_name == "Alpha"
    assert kickers[0].team_id == 1
    assert kickers[0].team_name == ""
    # Bravo is in both — roster row wins, so team_name is populated.
    assert kickers[1].player_name == "Bravo"
    assert kickers[1].team_name == "Argentina"
    # Charlie is roster-only.
    assert kickers[2].player_name == "Charlie"
    assert kickers[2].team_name == "Brazil"


def test_iter_initial_set_kickers_handles_empty_inputs() -> None:
    """Empty inputs yield no kickers."""
    assert list(iter_initial_set_kickers([], [])) == []


def test_iter_initial_set_kickers_training_only() -> None:
    """An empty roster still yields the training kickers."""
    shootout_kicks = [_shootout_kick(kicker_id=100, kicker_name="Alpha", team_id=1)]
    kickers = list(iter_initial_set_kickers(shootout_kicks, []))
    assert len(kickers) == 1
    assert kickers[0].player_id == 100


def test_iter_initial_set_kickers_roster_only() -> None:
    """Empty training data still yields the roster kickers."""
    roster = [_roster_player(300, "Charlie", team_id=2, team_name="Brazil")]
    kickers = list(iter_initial_set_kickers([], roster))
    assert len(kickers) == 1
    assert kickers[0].player_id == 300
    assert kickers[0].team_name == "Brazil"


def test_iter_initial_set_kickers_reads_through_artifacts(tmp_path: Path) -> None:
    """The slice's production path (Artifacts.read_* + iter_initial_set_kickers)
    is exercised end-to-end against a JSONL fixture. The test pins the
    seam at the data layer, not at the per-field construction site."""
    art = Artifacts(root=tmp_path)
    art.write_shootout_kicks(
        [
            _shootout_kick(kicker_id=100, kicker_name="Alpha", team_id=1),
            _shootout_kick(kicker_id=200, kicker_name="Bravo", team_id=1),
        ],
    )
    art.write_roster(
        [
            _roster_player(200, "Bravo", team_id=1, team_name="Argentina"),
            _roster_player(300, "Charlie", team_id=2, team_name="Brazil"),
        ],
    )
    kickers = list(
        iter_initial_set_kickers(
            art.read_shootout_kicks(),
            art.read_roster(),
        )
    )
    assert [k.player_id for k in kickers] == [100, 200, 300]
    assert kickers[1].team_name == "Argentina"
    assert kickers[2].team_name == "Brazil"


# ---------------------------------------------------------------------------
# fetch_player_data with no slug
# ---------------------------------------------------------------------------


def _stub_player_only(monkeypatch: pytest.MonkeyPatch, player_page: Mapping[str, object]) -> None:
    """Patch FotMobClient.get to return the player page for any path
    starting with `players/`. Other paths raise (caller should not
    need them)."""
    from twelveyards import client as client_module

    def fake_get(self: FotMobClient, path: str, params: dict | None = None) -> Any:
        if path.startswith("players/"):
            return {"pageProps": {"data": player_page}}
        msg = f"stub: unexpected path {path!r}"
        raise AssertionError(msg)

    monkeypatch.setattr(client_module, "_discover_build_id", lambda c: "stub-build")
    monkeypatch.setattr(client_module.FotMobClient, "get", fake_get)


def test_fetch_player_data_with_empty_slug_uses_no_slug_url(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`fetch_player_data(client, id)` (empty slug) calls `players/{id}`,
    not `players/{id}/` (with a trailing slash) and not
    `players/{id}/{slug}`. FotMob does not use the slug for routing;
    the player_id is the authoritative key."""
    seen_paths: list[str] = []
    from twelveyards import client as client_module

    def fake_get(self: FotMobClient, path: str, params: dict | None = None) -> Any:
        seen_paths.append(path)
        return {"pageProps": {"data": {"id": 42, "name": "Stub"}}}

    monkeypatch.setattr(client_module, "_discover_build_id", lambda c: "stub-build")
    monkeypatch.setattr(client_module.FotMobClient, "get", fake_get)
    client = FotMobClient(cache_dir=tmp_path / "stub")
    payload = fetch_player_data(client, player_id=42, slug="")
    assert seen_paths == ["players/42"]
    assert payload["pageProps"]["data"]["id"] == 42


def test_fetch_player_data_with_slug_keeps_url_with_slug(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`fetch_player_data(client, id, "lionel-messi")` calls
    `players/42/lionel-messi`. Slug is preserved for cache stability
    when the caller knows it."""
    seen_paths: list[str] = []
    from twelveyards import client as client_module

    def fake_get(self: FotMobClient, path: str, params: dict | None = None) -> Any:
        seen_paths.append(path)
        return {"pageProps": {"data": {"id": 42, "name": "Stub"}}}

    monkeypatch.setattr(client_module, "_discover_build_id", lambda c: "stub-build")
    monkeypatch.setattr(client_module.FotMobClient, "get", fake_get)
    client = FotMobClient(cache_dir=tmp_path / "stub")
    fetch_player_data(client, player_id=42, slug="lionel-messi")
    assert seen_paths == ["players/42/lionel-messi"]


# ---------------------------------------------------------------------------
# fetch_all_initial_set_penalty_history (stubbed FotMobClient)
# ---------------------------------------------------------------------------


def _stub_two_kickers(
    monkeypatch: pytest.MonkeyPatch,
    *,
    kicker_with_penalty: int,
    kicker_with_no_penalty: int,
) -> tuple[Path, list[str]]:
    """Patch FotMobClient to return per-kicker stub payloads.

    The kicker_with_penalty player has one (team, season) lookup with one
    fixture with one in-match penalty. The kicker_with_no_penalty player
    has one (team, season) lookup with one fixture with NO penalty shots.
    """
    cache_dir = Path("/tmp/opencode/initial_set_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    urls_seen: list[str] = []
    match_iter_log: list[int] = []  # ordered log of match requests by match_id

    def make_player_page(pid: int) -> dict[str, object]:
        return {
            "id": pid,
            "name": f"Player {pid}",
            "careerHistory": {
                "careerItems": {
                    "senior": {
                        "teamEntries": [],
                        "seasonEntries": [
                            {
                                "seasonName": "2022/2023",
                                "teamId": 100,
                                "team": "T",
                                "tournamentStats": [
                                    {
                                        "leagueId": 42,
                                        "leagueName": "L",
                                        "seasonName": "2022/2023",
                                    }
                                ],
                            }
                        ],
                    },
                    "national team": {"teamEntries": [], "seasonEntries": []},
                }
            },
        }

    def fake_get(self: FotMobClient, path: str, params: dict | None = None) -> Any:
        urls_seen.append(f"{path}?{params or {}}")
        if path.startswith("players/"):
            parts = path.split("/")
            pid = int(parts[1])
            return {"pageProps": {"data": make_player_page(pid)}}
        if path.startswith("leagues/"):
            return {
                "pageProps": {
                    "fixtures": {
                        "allMatches": [
                            {
                                "id": "9001",
                                "pageUrl": "/matches/x-vs-y/abcdef#9001",
                                "home": {"id": "100", "name": "Home"},
                                "away": {"id": "200", "name": "Away"},
                                "status": {
                                    "utcTime": "2023-01-15T19:00:00Z",
                                    "scoreStr": "1 - 0",
                                },
                            }
                        ]
                    }
                }
            }
        if path.startswith("matches/"):
            match_iter_log.append(9001)
            shots: list[dict[str, object]] = []
            if kicker_with_penalty:
                shots.append(
                    {
                        "id": 1,
                        "eventType": "Goal",
                        "teamId": 100,
                        "playerId": kicker_with_penalty,
                        "playerName": f"Player {kicker_with_penalty}",
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
                        "matchId": 9001,
                        "matchTimeUTC": "Mon, Jan 15, 2024, 19:00 UTC",
                        "leagueId": 42,
                        "leagueName": "L",
                    },
                    "header": {
                        "teams": [
                            {"id": 100, "name": "Home", "score": 1},
                            {"id": 200, "name": "Away", "score": 0},
                        ]
                    },
                    "content": {"shotmap": {"shots": shots}},
                }
            }
        msg = f"stub: unknown path {path!r}"
        raise AssertionError(msg)

    from twelveyards import client as client_module

    monkeypatch.setattr(client_module, "_discover_build_id", lambda c: "stub-build")
    monkeypatch.setattr(client_module.FotMobClient, "get", fake_get)
    return cache_dir, urls_seen


def test_fetch_all_initial_set_yields_one_result_per_kicker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The orchestrator yields one `InitialSetFetchResult` per input
    Kicker, in input order, even when the per-kicker fetch yields zero
    rows for some kickers."""
    _cache_dir, _urls = _stub_two_kickers(
        monkeypatch, kicker_with_penalty=100, kicker_with_no_penalty=200
    )
    client = FotMobClient(cache_dir=tmp_path / "stub")
    initial_set = [
        InitialSetKicker(player_id=100, player_name="Alpha", team_id=1, team_name="A"),
        InitialSetKicker(player_id=200, player_name="Bravo", team_id=1, team_name="B"),
    ]
    results = list(
        fetch_all_initial_set_penalty_history(client, initial_set, target_date=date(2024, 1, 1))
    )
    assert len(results) == 2
    assert [r.kicker.player_id for r in results] == [100, 200]
    # The first kicker has 1 penalty, the second has 0.
    assert len(results[0].rows) == 1
    assert results[0].rows[0].kicker_id == 100
    assert results[1].rows == []
    assert results[0].error is None
    assert results[1].error is None


def test_fetch_all_initial_set_continues_on_per_kicker_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A per-kicker fetch error is captured in `error`; the run
    continues to the next kicker. The errored kicker is reported as
    having zero rows."""
    cache_dir = Path("/tmp/opencode/initial_set_error_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    def fake_get(self: FotMobClient, path: str, params: dict | None = None) -> Any:
        if path == "players/100":
            raise RuntimeError("simulated FotMob 5xx for kicker 100")
        if path == "players/200":
            return {
                "pageProps": {
                    "data": {
                        "id": 200,
                        "name": "Player 200",
                        "careerHistory": {
                            "careerItems": {
                                "senior": {"teamEntries": [], "seasonEntries": []},
                                "national team": {
                                    "teamEntries": [],
                                    "seasonEntries": [],
                                },
                            }
                        },
                    }
                }
            }
        msg = f"stub: unknown path {path!r}"
        raise AssertionError(msg)

    from twelveyards import client as client_module

    monkeypatch.setattr(client_module, "_discover_build_id", lambda c: "stub-build")
    monkeypatch.setattr(client_module.FotMobClient, "get", fake_get)
    client = FotMobClient(cache_dir=tmp_path / "stub")
    initial_set = [
        InitialSetKicker(player_id=100, player_name="Errored", team_id=1, team_name="A"),
        InitialSetKicker(player_id=200, player_name="Empty", team_id=1, team_name="B"),
    ]
    results = list(fetch_all_initial_set_penalty_history(client, initial_set))
    assert len(results) == 2
    # First kicker errored — captured, no rows.
    assert results[0].error is not None
    assert "simulated FotMob 5xx" in results[0].error
    assert results[0].rows == []
    # Second kicker succeeded with zero rows.
    assert results[1].error is None
    assert results[1].rows == []


def test_fetch_all_initial_set_empty_input(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An empty Initial Set yields no results (no FotMob calls)."""
    from twelveyards import client as client_module

    urls_called: list[str] = []

    def fake_get(self: FotMobClient, path: str, params: dict | None = None) -> Any:
        urls_called.append(path)
        msg = "should not be called on empty input"
        raise AssertionError(msg)

    monkeypatch.setattr(client_module, "_discover_build_id", lambda c: "stub-build")
    monkeypatch.setattr(client_module.FotMobClient, "get", fake_get)
    client = FotMobClient(cache_dir=tmp_path / "stub")
    results = list(fetch_all_initial_set_penalty_history(client, []))
    assert results == []
    assert urls_called == []


# ---------------------------------------------------------------------------
# JSONL roundtrip
# ---------------------------------------------------------------------------


def test_write_missing_jsonl_roundtrip(tmp_path: Path) -> None:
    """Writing then reading a missing-kicker JSONL gives the same rows back."""
    rows = [
        MissingKicker(
            player_id=42,
            player_name="No History",
            team_id=1,
            team_name="Argentina",
        ),
        MissingKicker(player_id=43, player_name="Also Empty", team_id=2, team_name="Brazil"),
    ]
    out = tmp_path / "missing.jsonl"
    art = Artifacts(root=tmp_path)
    n = art.write_missing_history(rows, path=out)
    assert n == 2
    with out.open() as f:
        loaded = [json.loads(line) for line in f if line.strip()]
    assert loaded[0]["player_id"] == 42
    assert loaded[0]["player_name"] == "No History"
    assert loaded[0]["team_id"] == 1
    assert loaded[1]["player_id"] == 43
    assert loaded[1]["team_name"] == "Brazil"


# ---------------------------------------------------------------------------
# Dataclass shapes
# ---------------------------------------------------------------------------


def test_initial_set_kicker_fields() -> None:
    """InitialSetKicker has player_id, player_name, team_id, team_name."""
    k = InitialSetKicker(player_id=100, player_name="Alpha", team_id=1, team_name="Argentina")
    assert k.player_id == 100
    assert k.player_name == "Alpha"
    assert k.team_id == 1
    assert k.team_name == "Argentina"


def test_initial_set_fetch_result_default_error() -> None:
    """`InitialSetFetchResult.error` defaults to None (not set)."""
    kicker = InitialSetKicker(player_id=100, player_name="Alpha", team_id=1, team_name="Argentina")
    r = InitialSetFetchResult(kicker=kicker, rows=[])
    assert r.error is None


# ---------------------------------------------------------------------------
# Script CLI: --lookback-years / --history-floor / --target-date
# ---------------------------------------------------------------------------


def test_fetch_initial_set_script_accepts_reparameterisation_flags() -> None:
    """The slice script exposes --lookback-years and --history-floor so the
    Lookback Window can be changed without touching code (issue #21 AC:
    "re-parameterised for a different lookback_window without code changes").
    Run the script with `--help` and assert the flags are present and have
    sensible defaults — this is a static check, no FotMob calls.
    """
    import subprocess
    import sys

    from twelveyards.config import HISTORY_FLOOR, LOOKBACK_WINDOW_YEARS

    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "scripts/fetch_initial_set_player_history.py", "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0
    out = result.stdout
    assert "--lookback-years" in out
    assert "--history-floor" in out
    assert "--target-date" in out
    # Defaults are surfaced in --help so the user can discover them.
    assert str(LOOKBACK_WINDOW_YEARS) in out
    assert HISTORY_FLOOR.isoformat() in out


# ---------------------------------------------------------------------------
# JSONL schema smoke test (issue #21 AC: rows + missing list)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (
        Artifacts().player_history.exists()
        and Artifacts().missing_history.exists()
        and Artifacts().shootout_kicks.exists()
        and Artifacts().roster.exists()
    ),
    reason="output/ JSONL artifacts not present (run the slice first)",
)
def test_player_history_jsonl_schema_smoke() -> None:
    """Smoke test against the live output: every row in
    `output/player_history.jsonl` has the `PlayerPenalty` schema (derived
    from `dataclasses.fields(PlayerPenalty)` in `tests._factories`),
    every `x` is in [0, 2], every `side` is in {L, C, R}, every
    `outcome` is in {Goal, Saved, Missed}, and the missing list has
    the `MissingKicker` schema.
    """
    art = Artifacts()
    n_rows = 0
    kickers_with_rows: set[int] = set()
    with art.player_history.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            assert PLAYER_PENALTY_FIELDS <= set(row.keys()), (
                f"row missing fields: {PLAYER_PENALTY_FIELDS - set(row.keys())}"
            )
            assert 0.0 <= float(row["x"]) <= 2.0
            assert row["side"] in {"L", "C", "R"}
            assert row["outcome"] in {"Goal", "Saved", "Missed"}
            n_rows += 1
            kickers_with_rows.add(int(row["kicker_id"]))
    assert n_rows > 0

    n_missing = 0
    missing_ids: set[int] = set()
    with art.missing_history.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            assert MISSING_KICKER_FIELDS <= set(row.keys())
            missing_ids.add(int(row["player_id"]))
            n_missing += 1
    # Sanity: no kicker should appear in both the row list and the missing list.
    assert kickers_with_rows.isdisjoint(missing_ids)

    # Sanity: most Training Kickers (from `shootout_kicks.jsonl`) have at
    # least one row in `player_history.jsonl`. A small uncovered subset is
    # expected when the lookback window doesn't include any of the kicker's
    # penalty matches (e.g. a player whose only shootout was 6+ years ago
    # and who has had no in-match penalties in the window). The model
    # handles these via the prior — the test pins the floor of the
    # coverage, not the strict 100% the v1 era saw.
    training_kicker_ids: set[int] = set()
    with art.shootout_kicks.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            training_kicker_ids.add(int(row["kicker_id"]))
    assert training_kicker_ids, "shootout_kicks.jsonl is empty"
    uncovered_training = training_kicker_ids - kickers_with_rows
    n_training = len(training_kicker_ids)
    n_covered = n_training - len(uncovered_training)
    coverage = n_covered / n_training
    assert coverage >= 0.90, (
        f"only {n_covered}/{n_training} ({coverage:.1%}) training kickers have "
        f"penalty rows; uncovered: {sorted(uncovered_training)[:10]}..."
    )
