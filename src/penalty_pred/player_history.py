"""Per-kicker penalty history fetcher.

PRD: For every Kicker in the union of Training and Prediction Initial Sets,
fetch all their penalty kicks (shootout + in-match) over a 5-year Lookback
Window floored at 2016-01-01. The data graph is two-level: the Initial Set
(per-player lookup) fans out to the Derived History (per-match penalty
shots). No further fetches originate from the Derived History — a scraper
that fans out from there is a bug, not a feature.

The source of truth for the per-kicker lookup is the player page's
`pageProps.data.careerHistory`. We iterate `careerItems.senior` and
`careerItems["national team"]` (skip `careerItems.youth` — out of scope
for v1), and for each (team, season) overlap with the lookback window we
fetch the league's season fixtures, filter to the team's matches, and
extract the player's penalty shots from each match's shotmap.

The shootout penalty shots live in the same `pageProps.content.shotmap.shots`
array as in-match penalties; the discriminator is `situation == "Penalty"`
(present for both) and the period is implicit (a shootout shot has
`period == "PenaltyShootout"`, an in-match shot has `period` in the
`FirstHalf`/`SecondHalf`/... set). We keep both — the per-kicker history
is over every penalty, not just shootout kicks.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

from .client import FotMobClient
from .config import HISTORY_FLOOR, LOOKBACK_WINDOW_YEARS
from .coordinates import side
from .fotmob_parsing import (
    SHOTMAP_EVENT_TYPE_TO_OUTCOME,
    coerce_int,
    parse_match_date,
)
from .leagues import LEAGUE_BY_ID
from .match_ref import parse_page_url


@dataclass(frozen=True)
class PlayerPenalty:
    """One penalty kick (shootout or in-match) by a Kicker in a given match.

    `match_date` is the match start time in ISO 8601 UTC. `x` is the
    goal-mouth coordinate in [0, 2] from the kicker's perspective
    (0 = left post, 1 = center, 2 = right post). `side` is bucketed from
    `x` via the standard thresholds. `outcome` and `shot_type` come from
    the shotmap; `shotType` is the body part (RightFoot / LeftFoot),
    NOT the situation tag (which is always "Penalty" on this path).
    """

    kicker_id: int
    match_id: int
    match_date: str  # ISO 8601 (UTC)
    league_id: int
    league_name: str
    team_id: int
    is_home: bool
    x: float  # [0, 2]
    side: str  # "L" | "C" | "R"
    is_on_target: bool
    outcome: str  # "Goal" | "Saved" | "Missed"
    shot_type: str  # "RightFoot" | "LeftFoot"


@dataclass(frozen=True)
class PlayerMetadata:
    """A subset of the player page that downstream features (C1, C2) need.

    `position_key` is the FotMob position key (e.g. "striker", "centreback")
    from `positionDescription.primaryPosition.key`. `birth_date` is the
    ISO 8601 date (UTC) the player was born, parsed from `birthDate.utcTime`.
    """

    player_id: int
    player_name: str
    position_key: str  # e.g. "striker"
    birth_date: str  # ISO 8601 date (UTC)


# ---------------------------------------------------------------------------
# Player page helpers
# ---------------------------------------------------------------------------


def fetch_player_data(client: FotMobClient, player_id: int, slug: str = "") -> Mapping[str, Any]:
    """Fetch the player page JSON. Returns the full `__next/data` payload.

    The path is `players/{playerId}/{slug}` per docs/fotmob.md. The `slug`
    is the kebab-case player name (e.g. "lionel-messi"); it is part of
    the URL but FotMob does not use it for routing — the playerId is
    authoritative. We accept it as a parameter to keep the URL stable
    for caching, but the all-Initial-Set fan-out (slice #5, Issue #21)
    uses the no-slug form `players/{id}` because we do not have slugs
    for every kicker. Both forms return the same payload.

    The `slug` parameter is kept for callers (and tests) that already
    know it. An empty string yields the no-slug URL.
    """
    if slug:
        return client.get(f"players/{player_id}/{slug}")
    return client.get(f"players/{player_id}")


def extract_player_metadata(player_payload: Mapping[str, Any]) -> PlayerMetadata:
    """Extract the player's name, position, and birth date from the page payload.

    Accepts the full `__next/data` payload (the same shape `fetch_player_data`
    returns). The fields live at `pageProps.data.{id,name,birthDate,positionDescription}`.

    The position comes from `positionDescription.primaryPosition.key`
    (PRD: feature C1). The birth date comes from `birthDate.utcTime`
    (PRD: feature C2). The PRD originally cited `pageProps.data.playerInformation`
    for these, but in the live payload they are top-level on `pageProps.data`.
    We fall back to "" / "Unknown" if either is missing.
    """
    player_data = (player_payload.get("pageProps") or {}).get("data") or {}
    player_id = coerce_int(player_data.get("id"))
    player_name = str(player_data.get("name", ""))
    position_key = _primary_position_key(player_data)
    birth_date = _parse_birth_date(player_data.get("birthDate"))
    return PlayerMetadata(
        player_id=player_id,
        player_name=player_name,
        position_key=position_key,
        birth_date=birth_date,
    )


def _primary_position_key(player_data: Mapping[str, Any]) -> str:
    """Return the player's primary position key (e.g. "striker")."""
    pos_desc = player_data.get("positionDescription") or {}
    primary = pos_desc.get("primaryPosition") or {}
    return str(primary.get("key") or "")


def _parse_birth_date(birth_date: Any) -> str:
    """Parse a FotMob `birthDate` block into an ISO 8601 date string.

    The shape is `{"utcTime": "1987-06-24T00:00:00.000Z", "timezone": "UTC"}`.
    We return just the date part (e.g. "1987-06-24"). If the field is
    missing or malformed, return "".
    """
    if not isinstance(birth_date, Mapping):
        return ""
    utc_time = birth_date.get("utcTime")
    if not utc_time:
        return ""
    text = str(utc_time)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).astimezone(UTC).date().isoformat()
    except ValueError:
        return ""


# ---------------------------------------------------------------------------
# Career history traversal
# ---------------------------------------------------------------------------


def iter_career_season_entries(
    player_payload: Mapping[str, Any],
) -> Iterator[Mapping[str, Any]]:
    """Yield season entries from senior + national team stints (skip youth).

    Accepts the full `__next/data` payload (the same shape `fetch_player_data`
    returns). The career history lives at `pageProps.data.careerHistory`,
    which has `{"careerItems": {"senior": {...}, "national team": {...},
    "youth": {...}}}`. We iterate `senior` and `national team` in that
    order, and yield each entry in `seasonEntries` (a list, in
    reverse-chronological order on FotMob).
    """
    player_data = (player_payload.get("pageProps") or {}).get("data") or {}
    career_history = player_data.get("careerHistory") or {}
    career_items = career_history.get("careerItems") or {}
    for bucket in ("senior", "national team"):
        bucket_data = career_items.get(bucket) or {}
        yield from bucket_data.get("seasonEntries") or []


def season_name_to_year(season_name: str) -> int:
    """Convert a FotMob season name to the year int for the `?season=` URL param.

    Handles the three patterns we see in the wild:
    - "2022" → 2022 (calendar-year leagues like MLS, AFCON, World Cup)
    - "2020/2021" → 2020 (split-year leagues like LaLiga, Ligue 1, UCL)
    - "2022 Qatar" → 2022 (tournament year with a location suffix)
    """
    match = re.match(r"^(\d{4})", season_name)
    if not match:
        msg = f"Cannot extract year from FotMob season name: {season_name!r}"
        raise ValueError(msg)
    return int(match.group(1))


# ---------------------------------------------------------------------------
# Per-team-season traversal
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TeamSeasonLookup:
    """A (teamId, seasonEntry, tournamentStat) triple ready for fixture fetching.

    Each `TeamSeasonLookup` corresponds to ONE FotMob league/season fixture
    fetch. We yield one per (seasonEntry, tournamentStat) pair; downstream
    code calls `fetch_league_season_fixtures` once per lookup.
    """

    team_id: int
    season_entry: Mapping[str, Any]
    tournament_stat: Mapping[str, Any]


def iter_team_season_lookups(
    season_entries: Iterable[Mapping[str, Any]],
) -> Iterator[TeamSeasonLookup]:
    """Yield one TeamSeasonLookup per (seasonEntry, tournamentStat) pair.

    This is the fan-out point: each lookup corresponds to one FotMob
    league/season fixture fetch. We do NOT recurse into per-match data
    here — that's a separate step in the orchestrator.

    Lookups with no `leagueId` (e.g. the CONMEBOL World Cup Qualifiers
    in some seasons, which FotMob tracks by name only) are skipped —
    we cannot form a FotMob URL without a league id.
    """
    for entry in season_entries:
        team_id = coerce_int(entry.get("teamId"))
        if not team_id:
            continue
        for stat in entry.get("tournamentStats") or []:
            league_id = coerce_int(stat.get("leagueId"))
            if not league_id:
                continue
            yield TeamSeasonLookup(
                team_id=team_id,
                season_entry=entry,
                tournament_stat=stat,
            )


def fetch_league_season_fixtures(
    client: FotMobClient,
    league_id: int,
    season_year: int,
) -> list[dict[str, Any]]:
    """Fetch a league's season fixtures. Returns `pageProps.fixtures.allMatches`.

    The `league_id` is the FotMob integer (e.g. 87 for LaLiga, 53 for Ligue 1,
    42 for Champions League). The `season_year` is the FotMob `?season=`
    value — the start year of the season, e.g. 2020 for "2020/2021" LaLiga
    or 2022 for the 2022 World Cup.

    This is the same endpoint `fetch_season_fixtures` uses for the shootout
    pipeline; we duplicate it here so the player-history module has no
    dependency on the shootouts module's internals beyond the URL parser.
    """
    league = LEAGUE_BY_ID.get(league_id)
    if league is None:
        # The player may have played in a league we don't have a slug for
        # (e.g. a national second division). We can't form a URL without
        # the slug, so skip the lookup.
        return []
    payload = client.get(
        f"leagues/{league.league_id}/overview/{league.slug}",
        params={"season": str(season_year)},
    )
    fixtures = (payload.get("pageProps") or {}).get("fixtures") or {}
    return list(fixtures.get("allMatches") or [])


def filter_fixtures_by_team(
    fixtures: Iterable[Mapping[str, Any]],
    team_id: int,
) -> Iterator[Mapping[str, Any]]:
    """Yield season fixtures involving the given team (home OR away).

    The fixture's `home.id` and `away.id` are strings on FotMob; we cast
    to int for the comparison. Matches without a known home/away id
    (e.g. some friendly metadata) are dropped.
    """
    for fixture in fixtures:
        home_id = coerce_int((fixture.get("home") or {}).get("id"))
        away_id = coerce_int((fixture.get("away") or {}).get("id"))
        if team_id and team_id in (home_id, away_id):
            yield fixture


# ---------------------------------------------------------------------------
# Per-match extraction
# ---------------------------------------------------------------------------


def extract_player_penalties_from_match(
    match: Mapping[str, Any],
    player_id: int,
    team_id: int,
    league_id: int,
    league_name: str,
) -> list[PlayerPenalty]:
    """Extract the player's penalty shots from a match's full JSON.

    Source: `pageProps.content.shotmap.shots` filtered to
    `playerId == player_id` AND `situation == "Penalty"`. The `situation`
    tag distinguishes penalty shots from regular play. Shootout kicks
    have `situation == "Penalty"` AND `period == "PenaltyShootout"` —
    both are kept on this path. (The shootout kicker history is the
    same set of players; the per-kicker history is over every penalty,
    not just shootout kicks.)

    `is_home` is derived from the shot's `teamId` vs the match's home
    team id. If the shot's `teamId` is unset (some shootout shots), we
    fall back to the lookup's `team_id` and a home/away guess from the
    shootout order (shootouts alternate away/home/away/...).
    """
    page = match.get("pageProps") or {}
    content = page.get("content") or {}
    header = page.get("header") or {}
    general = page.get("general") or {}

    match_id = coerce_int(general.get("matchId"))
    match_date = parse_match_date(general.get("matchTimeUTC"))
    home_team_id = coerce_int((header.get("teams") or [{}])[0].get("id"))
    league_name_actual = str(general.get("leagueName") or league_name)

    shots = (content.get("shotmap") or {}).get("shots") or []
    out: list[PlayerPenalty] = []
    for shot in shots:
        if coerce_int(shot.get("playerId")) != player_id:
            continue
        if shot.get("situation") != "Penalty":
            continue
        shot_team_id = coerce_int(shot.get("teamId"))
        if not shot_team_id:
            shot_team_id = team_id
        is_home = (shot_team_id == home_team_id) if (shot_team_id and home_team_id) else False
        x = float(shot["onGoalShot"]["x"])
        outcome = SHOTMAP_EVENT_TYPE_TO_OUTCOME.get(
            shot.get("eventType", ""), str(shot.get("eventType", ""))
        )
        shot_type = str(shot.get("shotType", ""))
        out.append(
            PlayerPenalty(
                kicker_id=player_id,
                match_id=match_id,
                match_date=match_date,
                league_id=league_id,
                league_name=league_name_actual,
                team_id=shot_team_id,
                is_home=is_home,
                x=x,
                side=side(x),
                is_on_target=bool(shot.get("isOnTarget")),
                outcome=outcome,
                shot_type=shot_type,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def compute_lookback_window(
    target_date: date,
    lookback_years: int = LOOKBACK_WINDOW_YEARS,
    history_floor: date = HISTORY_FLOOR,
) -> tuple[date, date]:
    """Compute the (start, end) date bounds of the Lookback Window for `target_date`.

    The window is `[target_date - lookback_years, target_date]`, floored at
    `history_floor`. The floor is the hard lower bound: we never look back
    further than the floor, even if `lookback_years` would normally allow it.

    Example: target 2022-12-18, lookback 5y, floor 2016-01-01 →
    `[2017-12-18, 2022-12-18]` (5y back wins; the floor doesn't kick in
    until the target is earlier than 2021-01-01).
    """
    end = target_date
    naive_start = date(target_date.year - lookback_years, target_date.month, target_date.day)
    start = max(naive_start, history_floor)
    return start, end


def fetch_player_penalty_history(
    client: FotMobClient,
    player_id: int,
    player_slug: str = "",
    target_date: date | None = None,
    lookback_years: int = LOOKBACK_WINDOW_YEARS,
    history_floor: date = HISTORY_FLOOR,
) -> Iterator[PlayerPenalty]:
    """Yield every penalty the player took in the Lookback Window.

    Two-level data graph: the player is the Initial Set, the per-match
    penalty shots are the Derived History. No further fetches originate
    from the Derived History — the orchestrator never recurses into
    "who else was on this team?" or "what else was in this match?".

    `player_slug` is optional: FotMob does not use it for routing (the
    `player_id` is authoritative), so the all-Initial-Set fan-out (slice
    #5, Issue #21) calls with `player_slug=""` for every kicker it does
    not have a slug for. The default target date is the current day; the
    slice's default is 2022-12-18 (the 2022 WC Final) so the test case
    is reproducible.
    """
    if target_date is None:
        target_date = date(2022, 12, 18)
    start, end = compute_lookback_window(target_date, lookback_years, history_floor)
    window_start_year = start.year
    window_end_year = end.year

    # 1. Initial Set: the player page.
    player_payload = fetch_player_data(client, player_id, player_slug)
    season_entries = list(iter_career_season_entries(player_payload))

    # 2. Per-team-season fan-out. Each (seasonEntry, tournamentStat) pair
    #    is one fixture fetch.
    lookups = list(iter_team_season_lookups(season_entries))
    seen_lookups: set[tuple[int, int, int, int]] = set()  # dedupe duplicates

    for lookup in lookups:
        season_year = season_name_to_year(str(lookup.tournament_stat.get("seasonName", "")))
        if not (window_start_year <= season_year <= window_end_year):
            continue
        dedupe_key = (
            lookup.team_id,
            coerce_int(lookup.tournament_stat.get("leagueId")),
            season_year,
            player_id,
        )
        if dedupe_key in seen_lookups:
            continue
        seen_lookups.add(dedupe_key)

        league_id = coerce_int(lookup.tournament_stat.get("leagueId"))
        league_name = str(lookup.tournament_stat.get("leagueName", ""))
        fixtures = fetch_league_season_fixtures(client, league_id, season_year)
        for fixture in filter_fixtures_by_team(fixtures, lookup.team_id):
            # 3. Per-match fan-out: fetch the match and extract penalty shots.
            yield from _process_match_fixture(
                client,
                fixture,
                player_id,
                lookup.team_id,
                league_id,
                league_name,
                start,
                end,
            )


def _process_match_fixture(
    client: FotMobClient,
    fixture: Mapping[str, Any],
    player_id: int,
    team_id: int,
    league_id: int,
    league_name: str,
    window_start: date,
    window_end: date,
) -> Iterator[PlayerPenalty]:
    """Fetch one match (if in window) and yield the player's penalty rows.

    Skips matches outside the lookback window by date — the FotMob
    `status.utcTime` is the cheapest check before we burn a per-match
    fetch. Stale-URL matches (where the (seo, h2h) hash points to a
    different matchId in the response) are skipped silently — the same
    behaviour as the shootout pipeline.
    """
    utc_time = str((fixture.get("status") or {}).get("utcTime") or "")
    fixture_date = _parse_fixture_date(utc_time)
    if fixture_date is None:
        return
    if not (window_start <= fixture_date <= window_end):
        return

    page_url = str(fixture.get("pageUrl") or "")
    try:
        match_id, seo, h2h = parse_page_url(page_url)
    except ValueError:
        return
    match = client.get(f"matches/{seo}/{h2h}")
    page_match_id = coerce_int((match.get("pageProps") or {}).get("general", {}).get("matchId"))
    if page_match_id and page_match_id != match_id:
        return  # stale (seo, h2h) hash — skip silently
    yield from extract_player_penalties_from_match(
        match, player_id, team_id, league_id, league_name
    )


# ---------------------------------------------------------------------------
# Initial Set fan-out (slice #5, Issue #21)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InitialSetKicker:
    """A Kicker in the Initial Set, identified by `player_id`.

    The Initial Set is the union of training kickers (read from
    `shootout_kicks.jsonl`) and prediction kickers (read from
    `wc2026_roster.jsonl`). Both sources carry the player's national
    team id (`team_id`); only the roster source carries a human-readable
    `team_name`. Training kickers not in the roster (e.g. retired players
    from earlier tournaments) have `team_name == ""`.

    We do NOT carry the player slug — FotMob does not use the slug for
    routing, only the `player_id` is authoritative, so the per-kicker
    fetcher uses the no-slug URL form.
    """

    player_id: int
    player_name: str
    team_id: int
    team_name: str  # "" for training kickers not in the prediction roster


@dataclass(frozen=True)
class MissingKicker:
    """An Initial Set Kicker with zero penalty rows in the lookback window.

    Written to `missing_history.jsonl` so downstream slices (#22 features,
    #25 predictions) can decide whether to skip them or use a prior-based
    fallback. We carry `team_name` for parity with the roster (some
    downstream views show the player name + team together).
    """

    player_id: int
    player_name: str
    team_id: int
    team_name: str


@dataclass(frozen=True)
class InitialSetFetchResult:
    """The per-kicker outcome of `fetch_all_initial_set_penalty_history`.

    `kicker` is the input Initial Set Kicker. `rows` is the list of
    `PlayerPenalty` records found in the lookback window (possibly empty).
    `error` is the stringified exception when the per-kicker fetch raised
    (e.g. a transient FotMob 5xx); the kicker is reported as missing in
    that case but the run continues. Successful fetches that yielded zero
    rows have `error=None, rows=[]`.
    """

    kicker: InitialSetKicker
    rows: list[PlayerPenalty]
    error: str | None = None


def iter_initial_set_kickers(
    shootout_kicks_path: Path,
    roster_path: Path,
) -> Iterator[InitialSetKicker]:
    """Yield the deduplicated union of training and prediction Kickers.

    Training kickers come from `shootout_kicks.jsonl` (Issue #19). The row
    carries `kicker_id`, `kicker_name`, and `team_id` but not `team_name`
    (the team name is not on the shootout kick row). Prediction kickers
    come from `wc2026_roster.jsonl` (Issue #18) and carry the full
    (player_id, player_name, team_id, team_name) tuple.

    Dedup + enrichment:
    1. Read the roster into a dict by `player_id` (small, fits in memory).
    2. Yield training kickers first, enriched with the roster's
       `team_name` (and `player_name`/`team_id` if the roster has a more
       up-to-date value) when the kicker is in both sets.
    3. Yield roster-only kickers (not in training) at the end.

    The "training first, roster-only last" ordering is what the caller
    needs to tell apart a kicker the model is going to be trained on
    (a shootout taker from a past tournament) from a kicker we only
    know from the WC roster (no shootout kicks yet).

    Two-level data graph preserved: the Initial Set is built from the
    Training Initial Set (shootout kicks) and the Prediction Initial Set
    (WC roster) — neither of which is derived from per-player penalty
    data. The orchestrator never fans out from the Derived History
    (per-kicker penalty rows) back into the Initial Set.
    """
    roster_by_id: dict[int, dict[str, object]] = {}
    with roster_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pid = int(row.get("player_id", 0))
            if pid:
                roster_by_id[pid] = row

    seen: set[int] = set()
    with shootout_kicks_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            kicker_id = int(row.get("kicker_id", 0))
            if not kicker_id or kicker_id in seen:
                continue
            seen.add(kicker_id)
            roster_row = roster_by_id.get(kicker_id) or {}
            yield InitialSetKicker(
                player_id=kicker_id,
                player_name=str(roster_row.get("player_name") or row.get("kicker_name", "")),
                team_id=coerce_int(roster_row.get("team_id") or row.get("team_id")),
                team_name=str(roster_row.get("team_name", "")),
            )

    for player_id, row in roster_by_id.items():
        if player_id in seen:
            continue
        seen.add(player_id)
        yield InitialSetKicker(
            player_id=player_id,
            player_name=str(row.get("player_name", "")),
            team_id=coerce_int(row.get("team_id")),
            team_name=str(row.get("team_name", "")),
        )


def fetch_all_initial_set_penalty_history(
    client: FotMobClient,
    initial_set: Iterable[InitialSetKicker],
    target_date: date | None = None,
    lookback_years: int = LOOKBACK_WINDOW_YEARS,
    history_floor: date = HISTORY_FLOOR,
) -> Iterator[InitialSetFetchResult]:
    """Fan out the per-kicker fetcher across the Initial Set.

    Yields one `InitialSetFetchResult` per kicker, in input order. Per-kicker
    fetch errors (transient FotMob 5xx, malformed player pages, etc.) are
    caught and recorded in `error`; the kicker is reported as missing in
    that case but the run continues. A successful fetch that returned
    zero penalty rows in the window has `error=None, rows=[]`.

    `target_date` defaults to 2022-12-18 (the 2022 WC Final) for the same
    reason as `fetch_player_penalty_history`. The all-Initial-Set slice
    uses `target_date=today_utc()` from `config` so the lookback window
    ends "now" (we want every penalty the player took up to the present,
    not just up to a fixed historical date).

    Two-level data graph preserved: this orchestrator fans out across the
    Initial Set (per-player) only; the per-kicker fetcher fans out
    across per-team-season lookups within the player's career. The Derived
    History (per-match penalty rows) is a leaf in the graph — we never
    recurse from there.
    """
    for kicker in initial_set:
        try:
            rows = list(
                fetch_player_penalty_history(
                    client,
                    player_id=kicker.player_id,
                    target_date=target_date,
                    lookback_years=lookback_years,
                    history_floor=history_floor,
                )
            )
        except Exception as e:  # noqa: BLE001 — boundary: one bad kicker must not abort the run
            yield InitialSetFetchResult(kicker=kicker, rows=[], error=repr(e))
            continue
        yield InitialSetFetchResult(kicker=kicker, rows=rows, error=None)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _parse_fixture_date(value: str) -> date | None:
    """Parse a FotMob `status.utcTime` into a date, or None if malformed.

    The status `utcTime` is ISO 8601 (e.g. "2022-12-18T15:00:00Z") on the
    fixture list, in contrast to the match detail's `matchTimeUTC` which
    is RFC 2822. We accept both forms.
    """
    if not value:
        return None
    text = value
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).astimezone(UTC).date()
    except ValueError:
        pass
    try:
        return parsedate_to_datetime(text).astimezone(UTC).date()
    except (TypeError, ValueError):
        return None
