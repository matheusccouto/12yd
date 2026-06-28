"""Tournament roster fetcher.

Slice #3 (Issue #18): fetch the 2026 World Cup squad list and write it as
JSONL. The slice uses the WC 2026 league fixtures (FotMob leagueId 77,
season 2026) to discover the 104 matches, then fetches each match's
lineup to extract the registered players, deduplicating across matches
to produce the unique squad list (~700-1000 players across 48 teams).

The slice is independent of the shootout pipeline — it reuses the HTTP
client and JSONL writer but produces a different artifact. It feeds the
Prediction Initial Set in slice #5 (Issue #21).

The data source for a player's `(player_id, player_name, country_code,
primary_position_id)` is `pageProps.content.lineup.{homeTeam,awayTeam}.
{starters,subs}` — each entry is one registered player. The national
team id and name come from the match fixture's `home`/`away`, not from
the player's `primaryTeamId`/`primaryTeamName` (the latter is the
player's club, which is not what we want for the WC roster).

Stale-URL matches (where FotMob's (seo, h2h) hash now points to a
different matchId) are skipped silently — the same behaviour as the
shootout and player-history pipelines. Knockout-round matches with
placeholder teams (e.g. "Winner QF 1") have empty lineups and are
skipped at extraction time.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .client import FotMobClient
from .fotmob_parsing import coerce_int
from .leagues import League
from .match_ref import MatchRef
from .shootouts import fetch_season_fixtures
from .tournaments import WC_2026_LEAGUE, WC_2026_SEASON


@dataclass(frozen=True)
class RosterPlayer:
    """One player registered in a tournament squad.

    The (player_id, team_id) pair is unique within a single match's lineup
    (one player plays for one team in one match), but the same player
    may appear across multiple matches — deduplication by player_id is
    the orchestrator's job. We keep `team_id` and `team_name` on the row
    so a player who transferred national teams mid-tournament could be
    detected downstream (uncommon in the WC, but possible in qualifiers).
    """

    player_id: int
    player_name: str
    team_id: int
    team_name: str
    country_code: str  # ISO 3166-1 alpha-3, "" if missing


def fetch_wc_2026_roster(
    client: FotMobClient,
    league: League = WC_2026_LEAGUE,
    season: int = WC_2026_SEASON,
) -> Iterator[RosterPlayer]:
    """Yield every unique player registered in the 2026 World Cup squads.

    Algorithm:

    1. Fetch the league's season fixtures (one HTTP call).
    2. For each match, fetch its `__next/data` JSON (one HTTP call per
       match — the per-match lineup is the source of truth for the squad).
    3. Extract starters + subs from both home and away teams; stamp
       `team_id` and `team_name` from the match fixture.
    4. Deduplicate by `player_id` (a player may appear across multiple
       group-stage matches with the same team_id — we keep the first
       occurrence).

    Knockout-round matches with placeholder teams (e.g. "Winner QF 1",
    "Loser SF 2") carry empty lineups on the live response; those
    contribute zero rows to the output. Stale (seo, h2h) hashes
    (the FotMob hash now points to a different match) are skipped
    silently — the same behaviour as the shootout pipeline.
    """
    seen_player_ids: set[int] = set()
    for player in _iter_roster_players(client, league, season):
        if player.player_id in seen_player_ids:
            continue
        seen_player_ids.add(player.player_id)
        yield player


def _iter_roster_players(
    client: FotMobClient,
    league: League,
    season: int,
) -> Iterator[RosterPlayer]:
    """Yield roster rows from each match, before dedup. Internal helper."""
    fixtures = fetch_season_fixtures(client, league, season)
    for ref in iter_roster_match_refs(fixtures):
        data = client.get(f"matches/{ref.seo}/{ref.h2h}")
        page_match_id = coerce_int((data.get("pageProps") or {}).get("general", {}).get("matchId"))
        if page_match_id and page_match_id != ref.match_id:
            # Stale (seo, h2h) hash — skip silently.
            continue
        # `lineup_payload` is the value at `pageProps.content.lineup`
        # (NOT the full content object — extract_lineup_players expects
        # the lineup dict directly).
        content = (data.get("pageProps") or {}).get("content") or {}
        lineup_payload = content.get("lineup") or {}
        yield from extract_lineup_players(lineup_payload, ref)


def iter_roster_match_refs(
    fixtures: Iterable[Mapping[str, Any]],
) -> Iterator[MatchRef]:
    """Yield a `MatchRef` for every fixture in the season list.

    We do NOT filter on `status.reason.shortKey` — the roster slice
    wants every match, not just shootouts. We do filter out fixtures
    with no `home.id` or `away.id` (defensive; the live payload
    always has them for the WC 2026 league) and entries whose
    `pageUrl` cannot be parsed.
    """
    for fixture in fixtures:
        ref = MatchRef.from_fixture(fixture)
        if ref is None or not (ref.home_team_id and ref.away_team_id):
            continue
        yield ref


def extract_lineup_players(
    lineup_payload: Mapping[str, Any],
    ref: MatchRef,
) -> Iterator[RosterPlayer]:
    """Yield every registered player from one match's lineup payload.

    `lineup_payload` is the value at `pageProps.content.lineup` (NOT the
    full match payload). `ref` carries the match's home/away team ids and
    names — these are the source of truth for the national team the
    player is representing, not the player's `primaryTeamId` (which is
    the club side).

    Players appear as both `starters` (11 per team) and `subs` (up to 15
    per team in the WC 2026 format, for a 26-man squad). We yield every
    entry from both lists. Knockout-round placeholder matches have an
    empty `homeTeam`/`awayTeam` block — the iteration yields zero rows.

    The `countryCode` field is the player's nationality (ISO 3166-1
    alpha-3); it should match the national team they're representing in
    a WC context, but we take the per-player value to be safe.
    """
    for team_id, team_name, key in (
        (ref.home_team_id, ref.home_team_name, "homeTeam"),
        (ref.away_team_id, ref.away_team_name, "awayTeam"),
    ):
        team_block = lineup_payload.get(key) or {}
        if not team_block:
            # Placeholder knockout-round match — skip.
            continue
        for player in (*(team_block.get("starters") or []), *(team_block.get("subs") or [])):
            player_id = coerce_int(player.get("id"))
            if not player_id:
                continue
            yield RosterPlayer(
                player_id=player_id,
                player_name=str(player.get("name", "")),
                team_id=team_id,
                team_name=team_name,
                country_code=str(player.get("countryCode") or ""),
            )


def write_jsonl(path: Path, rows: Iterable[RosterPlayer]) -> int:
    """Write RosterPlayer records to a JSONL file. Returns the row count written."""
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=False))
            f.write("\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[RosterPlayer]:
    """Read a JSONL file of RosterPlayer records."""
    out: list[RosterPlayer] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(RosterPlayer(**json.loads(line)))
    return out
