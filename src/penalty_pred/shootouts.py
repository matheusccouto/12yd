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
from collections.abc import Iterable, Iterator, Mapping
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


def fetch_match_data(client: FotMobClient, match_id: int, seo: str, h2h: str) -> Mapping[str, Any]:
    """Fetch the `__next/data` JSON for a single match by seo/h2h slug pair."""
    return client.get(f"matches/{seo}/{h2h}")


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


def write_jsonl(path: Path, rows: Iterable[ShootoutKick]) -> int:
    """Write ShootoutKick records to a JSONL file. Returns the row count written."""
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=False))
            f.write("\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[ShootoutKick]:
    """Read a JSONL file of ShootoutKick records."""
    out: list[ShootoutKick] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(ShootoutKick(**json.loads(line)))
    return out


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


# (league_id, season) pairs that fall inside the current Prediction Window
# (2021-01-01 → today). The seasons are the FotMob `?season=` year values
# (e.g. Euro 2020 is `season=2020` even though it was held in 2021).
# Held in a module-level constant so the orchestrator takes a single parameter
# and the slice is re-parameterisable by editing one line.
LEAGUE_SEASONS_PREDICT_WINDOW: tuple[tuple[int, int], ...] = (
    (LEAGUE_BY_ID[77].league_id, 2022),  # World Cup 2022 (Qatar, Dec 2022)
    (LEAGUE_BY_ID[77].league_id, 2026),  # World Cup 2026 (in progress)
    (LEAGUE_BY_ID[50].league_id, 2020),  # Euro 2020 (held Jun–Jul 2021)
    (LEAGUE_BY_ID[50].league_id, 2024),  # Euro 2024 (Germany)
    (LEAGUE_BY_ID[44].league_id, 2021),  # Copa América 2021
    (LEAGUE_BY_ID[44].league_id, 2024),  # Copa América 2024
    (LEAGUE_BY_ID[289].league_id, 2021),  # AFCON 2021 (held Jan–Feb 2022)
    (LEAGUE_BY_ID[289].league_id, 2023),  # AFCON 2023 (held Jan–Feb 2024)
    (LEAGUE_BY_ID[289].league_id, 2025),  # AFCON 2025 (held Dec 2025–Jan 2026)
    (LEAGUE_BY_ID[298].league_id, 2021),  # Gold Cup 2021 (no shootouts)
    (LEAGUE_BY_ID[298].league_id, 2023),  # Gold Cup 2023
    (LEAGUE_BY_ID[298].league_id, 2025),  # Gold Cup 2025
    (LEAGUE_BY_ID[290].league_id, 2021),  # Asian Cup 2021 (held 2023, no shootouts)
    (LEAGUE_BY_ID[290].league_id, 2023),  # Asian Cup 2023 (held Jan–Feb 2024)
    (LEAGUE_BY_ID[290].league_id, 2025),  # Asian Cup 2025
)


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


def fetch_all_shootout_kicks(
    client: FotMobClient,
    match_refs: Iterable[MatchRef],
) -> Iterator[ShootoutKick]:
    """Yield every ShootoutKick for the given shootout match references.

    Each ref is fetched via `fetch_match_data` and parsed with
    `extract_shootout_kicks`. The yielded kicks are in (match_date, match_id,
    kick_number) order — stable across re-runs because the API lists fixtures
    in a fixed chronological order.

    If a ref's (seo, h2h) is stale — FotMob reuses hashes across matches, and
    an old h2h sometimes points to a newer match — the actual matchId in the
    response will differ from `ref.match_id`. In that case we skip the
    match silently and let the caller discover the gap from the JSONL.
    A `FetchResult` (via `fetch_all_shootout_kicks_with_skips`) is the
    right tool when you need to surface the skipped matches.
    """
    for ref in match_refs:
        data = fetch_match_data(client, ref.match_id, ref.seo, ref.h2h)
        page_match_id = coerce_int(data.get("pageProps", {}).get("general", {}).get("matchId"))
        if page_match_id and page_match_id != ref.match_id:
            # Stale (seo, h2h) hash — skip. See `fetch_all_shootout_kicks_with_skips`.
            continue
        yield from extract_shootout_kicks(data)


@dataclass(frozen=True)
class FetchResult:
    """The outcome of one fetch in `fetch_all_shootout_kicks_with_skips`.

    `kicks` is the list of `ShootoutKick` records extracted from the match.
    `skipped` is True iff the (seo, h2h) hash was stale and the response was
    for a different matchId. `no_kicks` is True iff the matchId was correct
    but `extract_shootout_kicks` returned no kicks (e.g. the shotmap is
    empty even though the match is listed as a shootout — a known FotMob
    data quality issue for some AFCON 2021 and Asian Cup 2023 matches).
    """

    ref: MatchRef
    kicks: list[ShootoutKick]
    skipped: bool
    no_kicks: bool = False


def fetch_all_shootout_kicks_with_skips(
    client: FotMobClient,
    match_refs: Iterable[MatchRef],
) -> list[FetchResult]:
    """Like `fetch_all_shootout_kicks`, but returns per-match results.

    Use this when the caller needs to surface the matches that were
    skipped due to a stale (seo, h2h) — the JSONL's RSSSF count assertion
    will fail without this information. Iterating one match at a time means
    a single bad ref does not abort the rest of the run.
    """
    results: list[FetchResult] = []
    for ref in match_refs:
        data = fetch_match_data(client, ref.match_id, ref.seo, ref.h2h)
        page_match_id = coerce_int(data.get("pageProps", {}).get("general", {}).get("matchId"))
        if page_match_id and page_match_id != ref.match_id:
            results.append(FetchResult(ref=ref, kicks=[], skipped=True))
            continue
        kicks = extract_shootout_kicks(data)
        results.append(FetchResult(ref=ref, kicks=kicks, skipped=False, no_kicks=not kicks))
    return results


def predict_window_bounds() -> tuple[datetime, datetime]:
    """Return the (start, end) datetime bounds of the current Prediction Window.

    `start` is `PREDICT_WINDOW_START` (2021-01-01 today, config-driven).
    `end` is `today_utc()`. Both are floored to midnight UTC for date arithmetic.
    """
    start = datetime.combine(PREDICT_WINDOW_START, datetime.min.time(), tzinfo=UTC)
    end = datetime.combine(today_utc(), datetime.min.time(), tzinfo=UTC)
    return start, end
