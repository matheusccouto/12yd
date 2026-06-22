"""Shootout kick extraction from FotMob match data.

PRD: Shootout kick placement comes from `pageProps.content.shotmap.shots`
filtered to `period == "PenaltyShootout"`, NOT from `penaltyShootoutEvents[*].shotmapEvent`
(the latter is missing for missed/saved kicks). Kicker identity is available on both
paths; we use shotmap as the source of truth and join with `penaltyShootoutEvents`
for the running shootout score.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

from .client import FotMobClient
from .coordinates import side


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


# shotmap eventType → canonical outcome label.
_SHOTMAP_EVENT_TYPE_TO_OUTCOME: dict[str, str] = {
    "Goal": "Goal",
    "AttemptSaved": "Saved",
    "Miss": "Missed",
}


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

    match_id = _int(general["matchId"])
    tournament_id = _int(general["leagueId"])
    tournament_name = str(general.get("leagueName", ""))
    round_label = str(general.get("matchRound") or general.get("leagueRoundName") or "")
    match_date = _parse_match_date(general.get("matchTimeUTC"))

    # match_score_home / away: full-time (incl. extra time) score, i.e. before the shootout.
    home_team = header["teams"][0]
    away_team = header["teams"][1]
    match_score_home = _int(home_team.get("score"))
    match_score_away = _int(away_team.get("score"))
    home_team_id = _int(home_team.get("id"))

    shots = (content.get("shotmap") or {}).get("shots") or []
    shootout_shots = [s for s in shots if s.get("period") == "PenaltyShootout"]
    if not shootout_shots:
        return []

    events = ((content.get("matchFacts") or {}).get("events") or {}).get(
        "penaltyShootoutEvents"
    ) or []
    # penaltyShootoutEvents is already in shootout order (away, home, away, home, ...).
    pre_scores, post_scores, _ = _running_scores(events)
    pre_by_player = _flatten_by_player(pre_scores)
    post_by_player = _flatten_by_player(post_scores)

    # The shotmap's `isHome` field is unset on every shot; we derive it from
    # the shot's `teamId` vs the match's home/away team ids.
    kicks: list[ShootoutKick] = []
    for idx, shot in enumerate(shootout_shots, start=1):
        player_id = _int(shot["playerId"])
        team_id = _int(shot.get("teamId"))
        is_home = (team_id == home_team_id) if (team_id and home_team_id) else False
        if player_id not in post_by_player:
            msg = f"shotmap shootout player {player_id} has no matching penaltyShootoutEvent"
            raise ValueError(msg)
        x = float(shot["onGoalShot"]["x"])
        outcome = _SHOTMAP_EVENT_TYPE_TO_OUTCOME.get(
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
                pen_score_before=list(pre_by_player[player_id]),
                pen_score_after=list(post_by_player[player_id]),
                match_score_home=match_score_home,
                match_score_away=match_score_away,
            )
        )
    return kicks


def _running_scores(
    events: list[Mapping[str, Any]],
) -> tuple[
    dict[tuple[int, bool], list[int]],
    dict[tuple[int, bool], list[int]],
    dict[tuple[int, bool], str],
]:
    """Walk `penaltyShootoutEvents` in order and emit pre/post scores per kick.

    Returns three dicts keyed by `(playerId, isHome)`:
    - pre_scores: the running shootout score BEFORE the kick
    - post_scores: the running shootout score AFTER the kick
    - event_type_by_key: the raw event type ("Goal" / "MissedPenalty" / "SavedPenalty")
    """
    pre_scores: dict[tuple[int, bool], list[int]] = {}
    post_scores: dict[tuple[int, bool], list[int]] = {}
    event_type_by_key: dict[tuple[int, bool], str] = {}
    score = [0, 0]
    for ev in events:
        player = ev.get("player") or {}
        player_id = _int(player.get("id"))
        is_home = bool(ev.get("isHome"))
        key = (player_id, is_home)
        pre_scores[key] = list(score)
        ev_type = str(ev.get("type", ""))
        event_type_by_key[key] = ev_type
        if ev_type == "Goal":
            if is_home:
                score[0] += 1
            else:
                score[1] += 1
        # For Goals, penShootoutScore is the post-kick running score; use it as the
        # authoritative post value when present (matches the walked score).
        if ev_type == "Goal" and ev.get("penShootoutScore"):
            post_scores[key] = list(ev["penShootoutScore"])
        else:
            post_scores[key] = list(score)
    return pre_scores, post_scores, event_type_by_key


def _flatten_by_player(scores: dict[tuple[int, bool], list[int]]) -> dict[int, list[int]]:
    """Drop the (playerId, isHome) key's isHome component — we join by playerId alone.

    penaltyShootoutEvents has exactly one event per player, so isHome adds no information.
    """
    out: dict[int, list[int]] = {}
    for (player_id, _is_home), value in scores.items():
        out[player_id] = list(value)
    return out


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


def _int(value: Any) -> int:
    if value is None or value == "":
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _parse_match_date(value: Any) -> str:
    """Coerce a FotMob matchTimeUTC to an ISO 8601 string (UTC, second precision)."""
    if not value:
        return ""
    text = str(value)
    # FotMob returns RFC 2822 dates like "Sun, Dec 18, 2022, 15:00 UTC".
    try:
        return parsedate_to_datetime(text).astimezone(UTC).isoformat()
    except (TypeError, ValueError):
        pass
    # ISO 8601 fallback.
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).astimezone(UTC).isoformat()
    except ValueError:
        return text
