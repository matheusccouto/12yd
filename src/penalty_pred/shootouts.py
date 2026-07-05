"""Shootout kick extraction from FotMob match data.

PRD: Shootout kick placement comes from `pageProps.content.shotmap.shots`
filtered to `period == "PenaltyShootout"`, NOT from `penaltyShootoutEvents[*].shotmapEvent`
(the latter is missing for missed/saved kicks). Kicker identity is available on both
paths; we use shotmap as the source of truth and join with `penaltyShootoutEvents`
for the running shootout score.

Slice #2 (Issue #19) adds the league-fixture → shootout-match driver on top of the
per-match extractor. The driver fans out across (league, season) pairs, filters the
season fixture list to shootouts via `status.reason.shortKey == "penalties_short"`,
and reuses `extract_shootout_kicks` for each match.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .client import FotMobClient
from .config import PREDICT_WINDOW_START, today_utc
from .coordinates import side
from .fotmob_parsing import (
    SHOTMAP_EVENT_TYPE_TO_OUTCOME,
    coerce_int,
    parse_match_date,
)
from .leagues import LEAGUE_BY_ID, League
from .match_ref import MatchRef
from .tournaments import LEAGUE_SEASONS_PREDICT_WINDOW

# Short key that marks a match as decided by a penalty shootout (docs/fotmob.md).
SHOOTOUT_SHORT_KEY: str = "penalties_short"


@dataclass(frozen=True)
class ShootoutKick:
    match_id: int
    match_date: str  # ISO 8601 (UTC)
    tournament_id: int
    tournament_name: str
    round: str
    kick_number: int
    kicker_id: int
    kicker_name: str
    team_id: int
    is_home: bool
    x: float  # [0, 2] continuous
    side: str  # "L" | "C" | "R"
    is_on_target: bool
    outcome: str  # "Goal" | "Saved" | "Missed"
    pen_score_before: list[int]
    pen_score_after: list[int]
    match_score_home: int
    match_score_away: int


def extract_shootout_kicks(match: Mapping[str, Any]) -> list[ShootoutKick]:
    """Extract every Shootout Kick from a match's full JSON payload.

    Source: `pageProps.content.shotmap.shots` filtered to `period == "PenaltyShootout"`,
    joined with `pageProps.content.matchFacts.events.penaltyShootoutEvents` on
    `(playerId, isHome)` for the running shootout score.
    """
    page = match["pageProps"]
    content = page["content"]
    general = page["general"]
    header = page["header"]

    match_id = coerce_int(general["matchId"])
    tournament_id = coerce_int(general["leagueId"])
    tournament_name = str(general.get("leagueName", ""))
    round_label = str(general.get("matchRound") or general.get("leagueRoundName") or "")
    match_date = parse_match_date(general.get("matchTimeUTC"))

    # match_score_home / away: full-time (incl. extra time) score, i.e. before the shootout.
    home_team = header["teams"][0]
    away_team = header["teams"][1]
    match_score_home = coerce_int(home_team.get("score"))
    match_score_away = coerce_int(away_team.get("score"))
    home_team_id = coerce_int(home_team.get("id"))

    shots = (content.get("shotmap") or {}).get("shots") or []
    shootout_shots = [s for s in shots if s.get("period") == "PenaltyShootout"]
    if not shootout_shots:
        return []

    events = ((content.get("matchFacts") or {}).get("events") or {}).get(
        "penaltyShootoutEvents"
    ) or []
    # penaltyShootoutEvents is already in shootout order (away, home, away, home, ...).
    pre_by_index, post_by_index = _indexed_scores(events)

    # The shotmap's `isHome` field is unset on every shot; we derive it from
    # the shot's `teamId` vs the match's home/away team ids.
    #
    # The two arrays are 1:1 by INDEX, not by playerId. We pair them by
    # position because FotMob's shotmap and penaltyShootoutEvents occasionally
    # disagree on the kicker's id (e.g. a substitute with a different FotMob
    # record vs the player they replaced). The shotmap is the source of truth
    # for placement (`onGoalShot.x`); the events are the source of truth for
    # the running shootout score. We take playerId/playerName from the
    # shotmap (the actual kicker) and the score from the event at the same
    # index.
    kicks: list[ShootoutKick] = []
    n_shots = len(shootout_shots)
    n_events = len(events)
    if n_shots != n_events:
        msg = f"shootout shotmap has {n_shots} shots but events has {n_events}"
        raise ValueError(msg)
    pre_by_index, post_by_index = _indexed_scores(events)
    for idx, shot in enumerate(shootout_shots, start=1):
        player_id = coerce_int(shot["playerId"])
        team_id = coerce_int(shot.get("teamId"))
        is_home = (team_id == home_team_id) if (team_id and home_team_id) else False
        x = float(shot["onGoalShot"]["x"])
        outcome = SHOTMAP_EVENT_TYPE_TO_OUTCOME.get(
            shot.get("eventType", ""), str(shot.get("eventType", ""))
        )
        kicks.append(
            ShootoutKick(
                match_id=match_id,
                match_date=match_date,
                tournament_id=tournament_id,
                tournament_name=tournament_name,
                round=round_label,
                kick_number=idx,
                kicker_id=player_id,
                kicker_name=str(shot.get("playerName", "")),
                team_id=team_id,
                is_home=is_home,
                x=x,
                side=side(x),
                is_on_target=bool(shot.get("isOnTarget")),
                outcome=outcome,
                pen_score_before=list(pre_by_index[idx - 1]),
                pen_score_after=list(post_by_index[idx - 1]),
                match_score_home=match_score_home,
                match_score_away=match_score_away,
            )
        )
    return kicks


def _indexed_scores(
    events: list[Mapping[str, Any]],
) -> tuple[list[list[int]], list[list[int]]]:
    """Walk `penaltyShootoutEvents` in order and emit pre/post scores per kick.

    Returns two parallel lists (one entry per kick, in shootout order):
    - pre_scores[i]: the running shootout score BEFORE kick i
    - post_scores[i]: the running shootout score AFTER kick i

    `pre_scores[i]` is the score after applying event i-1; `post_scores[i]`
    is the score after applying event i.

    We use the event's `penShootoutScore` as the authoritative post value
    when present (it matches the walked score for Goals; we fall back to
    the walked score when absent).
    """
    pre: list[list[int]] = []
    post: list[list[int]] = []
    score = [0, 0]
    for ev in events:
        ev_type = str(ev.get("type", ""))
        is_home = bool(ev.get("isHome"))
        pre.append(list(score))
        if ev_type == "Goal":
            if is_home:
                score[0] += 1
            else:
                score[1] += 1
        if ev.get("penShootoutScore"):
            post.append(list(ev["penShootoutScore"]))
        else:
            post.append(list(score))
    return pre, post


# ---------------------------------------------------------------------------
# Slice #2: league-fixture → shootout-match driver (Issue #19)
# ---------------------------------------------------------------------------


def fetch_season_fixtures(
    client: FotMobClient, league: League, season: int
) -> list[dict[str, Any]]:
    """Fetch a league's season fixtures. Returns the `pageProps.fixtures.allMatches` list.

    The `overview` tab is the source of truth for season fixtures; it carries
    the same `allMatches` list as the dedicated `fixtures` tab. Pass `season`
    as the year (e.g. 2022 for the 2022 World Cup, 2020 for the Euro 2020 —
    which was actually held in 2021).
    """
    payload = client.get(
        f"leagues/{league.league_id}/overview/{league.slug}",
        params={"season": str(season)},
    )
    fixtures = (payload.get("pageProps") or {}).get("fixtures") or {}
    matches = fixtures.get("allMatches") or []
    return list(matches)


def extract_shootout_match_fixtures(
    fixtures: Iterable[Mapping[str, Any]],
) -> list[MatchRef]:
    """Filter season fixtures to those that ended in a penalty shootout.

    The filter is `status.reason.shortKey == "penalties_short"` (docs/fotmob.md).
    Returns one `MatchRef` per shootout match, in the order the API
    listed them.
    """
    out: list[MatchRef] = []
    for fixture in fixtures:
        status = fixture.get("status") or {}
        reason = status.get("reason") or {}
        if reason.get("shortKey") != SHOOTOUT_SHORT_KEY:
            continue
        ref = MatchRef.from_fixture(fixture)
        if ref is not None:
            out.append(ref)
    return out


def fetch_all_shootout_match_refs(
    client: FotMobClient,
    league_seasons: Iterable[tuple[int, int]] = LEAGUE_SEASONS_PREDICT_WINDOW,
) -> list[MatchRef]:
    """Fetch the season fixtures for each (league_id, season) pair, return all
    shootout match refs. The result is the candidate list of matches to drive
    `extract_shootout_kicks` over.
    """
    refs: list[MatchRef] = []
    for league_id, season in league_seasons:
        league = LEAGUE_BY_ID[league_id]
        fixtures = fetch_season_fixtures(client, league, season)
        refs.extend(extract_shootout_match_fixtures(fixtures))
    return refs


@dataclass(frozen=True)
class FetchResult:
    """The outcome of one match in `fetch_all_shootout_kicks_with_skips`.

    `kicks` is the list of `ShootoutKick` records extracted from the match.
    `skipped` is True iff the (seo, h2h) hash was stale and the response was
    for a different matchId. `no_kicks` is True iff the matchId was correct
    but `extract_shootout_kicks` returned no kicks (e.g. the shotmap is
    empty even though the match is listed as a shootout — a known FotMob
    data quality issue for some AFCON 2021 and Asian Cup 2023 matches).
    `failure_mode` is a non-empty string when the extractor raised an
    exception (e.g. shotmap/events count mismatch, missing keys); the
    string is `f"{ExceptionClass}: {message}"`. A failed match is
    neither `skipped` (the matchId is correct) nor `no_kicks` (we never
    finished extracting); the `kicks` list is empty.

    `live_match_id` and `resolved_url` capture where the (seo, h2h) pair
    actually resolved to when the response's `matchId` differed from the
    ref's `match_id` (a stale-hash skip). `live_match_id` is the
    `pageProps.general.matchId` of the live response; `resolved_url` is
    the absolute FotMob URL the (seo, h2h) pair resolves to (the public
    match page, not the `__next/data` endpoint). Both fields are empty
    for non-skip results.
    """

    ref: MatchRef
    kicks: list[ShootoutKick]
    skipped: bool
    no_kicks: bool = False
    failure_mode: str = ""
    live_match_id: int = 0
    resolved_url: str = ""


def fetch_all_shootout_kicks_with_skips(
    client: FotMobClient,
    match_refs: Iterable[MatchRef],
) -> list[FetchResult]:
    """Yield every ShootoutKick per match, surfacing skipped, no-kicks, and failed matches.

    Each ref is fetched via `client.get` and parsed with
    `extract_shootout_kicks`. Use this when the caller needs to surface
    the matches that were skipped due to a stale (seo, h2h) — the
    JSONL's RSSSF count assertion will fail without this information.
    Iterating one match at a time means a single bad ref does not
    abort the rest of the run.

    `extract_shootout_kicks` can raise `ValueError` (e.g. when the
    shotmap and `penaltyShootoutEvents` counts disagree, or when a
    required JSON key is missing). Such exceptions are caught and
    reported as a `FetchResult` with `failure_mode` set to a short
    `f"{ExceptionClass}: {message}"` string and empty `kicks`. The
    caller is expected to surface these in the diagnostics JSONL.

    For `skipped` results (stale (seo, h2h) hash — the response's
    `matchId` differs from the ref's `match_id`), the `live_match_id`
    and `resolved_url` fields capture where the (seo, h2h) pair
    actually resolved to. The `live_match_id` is the response's
    `pageProps.general.matchId`; the `resolved_url` is the public
    FotMob match page (`https://www.fotmob.com/matches/{seo}/{h2h}`)
    so a maintainer can verify the rotation in a browser. The two
    fields let a future URL-rotation handler answer the v4 PRD's
    Phase 2 step 1 question: are the 18 stale-hash refs concentrated
    (a single mapping table would fix them) or spread (a general
    URL-rotation handler is needed)?
    """
    results: list[FetchResult] = []
    for ref in match_refs:
        data = client.get(f"matches/{ref.seo}/{ref.h2h}")
        page_match_id = coerce_int(data.get("pageProps", {}).get("general", {}).get("matchId"))
        if page_match_id and page_match_id != ref.match_id:
            results.append(
                FetchResult(
                    ref=ref,
                    kicks=[],
                    skipped=True,
                    live_match_id=page_match_id,
                    resolved_url=f"https://www.fotmob.com/matches/{ref.seo}/{ref.h2h}",
                )
            )
            continue
        try:
            kicks = extract_shootout_kicks(data)
        except Exception as exc:  # noqa: BLE001 - we want the message, not a filter
            results.append(
                FetchResult(
                    ref=ref,
                    kicks=[],
                    skipped=False,
                    no_kicks=False,
                    failure_mode=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        results.append(FetchResult(ref=ref, kicks=kicks, skipped=False, no_kicks=not kicks))
    return results


def write_skipped_refs_diagnostics(
    results: Iterable[FetchResult],
    path: Path,
) -> int:
    """Write one JSONL row per non-empty skip / no-kicks / failure result.

    The output is a JSONL file with one record per match that did not
    contribute kicks, used for diagnosing the RSSSF divergence. Each
    row is a dict with the match identity (`match_id`, `home`, `away`,
    `round`, `match_date`) and a `failure_mode` field. The field
    discriminates the three states:

    - `stale_hash` — `(seo, h2h)` resolved to a different matchId
      (`skipped=True`).
    - `empty_shotmap` — matchId was correct but the shotmap had no
      `period == "PenaltyShootout"` entries (`no_kicks=True`).
    - `f"{ExceptionClass}: {message}"` — the extractor raised.

    For `stale_hash` rows, the `live_match_id` and `resolved_url` fields
    capture where the (seo, h2h) pair actually resolved to (v4 PRD
    Phase 2 step 1: "concrete diagnosis" of the 18 URL-rotation
    failures). The two fields let a future URL-rotation handler decide
    between a single mapping table (refs are concentrated on a small
    set of newer matchIds) and a general URL-rotation handler (refs
    are spread across many newer matchIds). The fields are empty
    strings for `empty_shotmap` and extractor-failure rows.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for r in results:
            if r.skipped:
                failure_mode = "stale_hash"
            elif r.failure_mode:
                failure_mode = r.failure_mode
            elif r.no_kicks:
                failure_mode = "empty_shotmap"
            else:
                continue
            row = {
                "match_id": r.ref.match_id,
                "home": r.ref.home_team_name,
                "away": r.ref.away_team_name,
                "round": r.ref.round_name,
                "match_date": r.ref.match_date,
                "failure_mode": failure_mode,
            }
            if r.skipped:
                row["live_match_id"] = r.live_match_id
                row["resolved_url"] = r.resolved_url
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
            count += 1
    return count


def predict_window_bounds() -> tuple[datetime, datetime]:
    """Return the (start, end) datetime bounds of the current Prediction Window.

    `start` is `PREDICT_WINDOW_START` (2021-01-01 today, config-driven).
    `end` is `today_utc()`. Both are floored to midnight UTC for date arithmetic.
    """
    start = datetime.combine(PREDICT_WINDOW_START, datetime.min.time(), tzinfo=UTC)
    end = datetime.combine(today_utc(), datetime.min.time(), tzinfo=UTC)
    return start, end


# --- Per-tournament success-rate diagnostic (v4 PRD Phase 2) ---------------


@dataclass(frozen=True)
class TournamentSuccessRate:
    """One row in the per-tournament success-rate diagnostic.

    The per-(league, season) rollup of a scraper run. The fields
    answer two questions:

    1. **Did the scraper find every match the RSSSF oracle says exists
       in this (league, season)?** — `match_count` vs.
       `reachable_match_count` (raw RSSSF count minus the documented
       empty-shotmap exclusions).
    2. **If not, why (skipped, no_kicks, failed)?** — the three count
       fields surface the per-pair diagnostic breakdown.

    `status` is one of:
    - `"ok"` — `match_count >= reachable_match_count`.
    - `"partial"` — `0 < match_count < reachable_match_count`.
    - `"missing"` — `match_count == 0` (and `reachable_match_count > 0`).
    - `"n/a"` — `expected_match_count == 0` (the pair legitimately has
      zero shootouts, e.g. Gold Cup 2021, Asian Cup 2021, Asian Cup
      2025, WC 2026).

    The dataclass is the on-disk row shape for
    `tournament_success_rate.jsonl`. The aggregate function
    (`aggregate_per_tournament_success_rate`) and the writer
    (`write_per_tournament_success_rate`) live next to it so the
    diagnostic is one import.
    """

    league_id: int
    season: int
    tournament_name: str
    match_count: int
    kick_count: int
    skipped_count: int
    no_kicks_count: int
    failed_count: int
    expected_match_count: int
    reachable_match_count: int
    status: str


def aggregate_per_tournament_success_rate(
    pair_results: Iterable[tuple[tuple[int, int], list[FetchResult]]],
    *,
    expected_counts: Mapping[tuple[int, int], int] | None = None,
    excluded_counts: Mapping[tuple[int, int], int] | None = None,
    tournament_names: Mapping[int, str] | None = None,
) -> list[TournamentSuccessRate]:
    """Roll up per-(league, season) results into a coverage diagnostic.

    The rollup compares the scraper's match count against the RSSSF
    oracle's expected count (minus documented empty-shotmap cases).
    The output is one `TournamentSuccessRate` row per `(league, season)`
    pair in `pair_results` — pairs that are in the scope but not in
    `pair_results` (e.g. a league with zero shootout fixtures) are
    NOT included; the integration test cross-checks the artifact
    against the full in-scope pair list separately.

    Args:
        pair_results: iterable of `((league_id, season), [FetchResult])`.
        expected_counts: per-pair raw RSSSF count, e.g.
            `{(77, 2022): 5, ...}`. Pairs not in the map get 0.
        excluded_counts: per-pair empty-shotmap exclusions, e.g.
            `{(289, 2021): 4, ...}`. The reachable count is
            `expected - excluded`. Pairs not in the map get 0.
        tournament_names: per-league-id display name, e.g.
            `{77: "World Cup"}`. Defaults to `LEAGUE_BY_ID` lookup
            (the union of international + club + extended leagues).
    """
    expected_counts = dict(expected_counts or {})
    excluded_counts = dict(excluded_counts or {})
    if tournament_names is None:
        tournament_names = {lid: league.name for lid, league in LEAGUE_BY_ID.items()}

    rows: list[TournamentSuccessRate] = []
    for pair, results in pair_results:
        league_id, season = pair
        match_count = sum(1 for r in results if r.kicks and not r.skipped)
        kick_count = sum(len(r.kicks) for r in results)
        skipped_count = sum(1 for r in results if r.skipped)
        no_kicks_count = sum(1 for r in results if r.no_kicks)
        failed_count = sum(1 for r in results if r.failure_mode)
        expected = expected_counts.get(pair, 0)
        excluded = excluded_counts.get(pair, 0)
        reachable = max(0, expected - excluded)
        if expected == 0:
            status = "n/a"
        elif match_count >= reachable:
            status = "ok"
        elif match_count > 0:
            status = "partial"
        else:
            status = "missing"
        rows.append(
            TournamentSuccessRate(
                league_id=league_id,
                season=season,
                tournament_name=tournament_names.get(league_id, ""),
                match_count=match_count,
                kick_count=kick_count,
                skipped_count=skipped_count,
                no_kicks_count=no_kicks_count,
                failed_count=failed_count,
                expected_match_count=expected,
                reachable_match_count=reachable,
                status=status,
            )
        )
    return rows


def write_per_tournament_success_rate(
    rows: Iterable[TournamentSuccessRate],
    path: Path,
) -> int:
    """Write one JSONL row per (league, season) pair to `path`.

    The output complements `write_skipped_refs_diagnostics` (which
    surfaces the per-match skip / no-kicks / failure reasons) with a
    per-tournament rollup that answers "did the scraper reach every
    (league, season) the oracle says exists?". The artifact is a
    v4 PRD Phase 2 acceptance criterion: a future maintainer running
    the slice pipeline can spot a regression on a single tournament
    (a `"partial"` or `"missing"` row) without parsing the per-match
    diagnostics file. The test in `test_tournaments.py` pins the
    per-pair coverage against `EXPECTED_SHOOTOUT_COUNTS`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=False))
            f.write("\n")
            count += 1
    return count
