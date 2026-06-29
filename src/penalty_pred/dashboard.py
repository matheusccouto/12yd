"""Streamlit dashboard logic — the library side of the live shootout predictor.

PRD: A single-page Streamlit app on Streamlit Cloud that surfaces live
shootout predictions. At load time, the app fetches the WC 2026 fixture
list from FotMob, filters to upcoming knockout matches (R16, QF, SF, F)
with both teams decided, and lets the user pick a match from a selectbox.
For the selected match, the app loads `lightgbm.pkl` from HF, builds the
9-feature row for each likely kicker on each team (using the match's
actual `round` for the B3 feature), re-scores, and shows a per-kicker
table: name, team, kicking foot, P(L), P(C), P(R), and the recommended
dive (`argmin`).

This module is the dashboard's *library* side: the data loading, the
match filter, the re-score, and the recommended dive. The Streamlit
`app.py` at the repo root is a thin layer on top of these three
functions — the seam is the data, not the UI, so the logic can be
unit-tested without launching Streamlit.

The dashboard's three entry points (the three functions the PRD names):

- `load_upcoming_knockouts(client)` — fetch + filter to upcoming + knockout
  + both teams decided. Returns a list of `MatchContext`.
- `predict_match(roster, history, metadata_fetcher, model, context)` —
  re-score the match's likely kickers (the roster, filtered to the
  match's two teams) with the match's actual round. Returns a list of
  `KickerPrediction`, sorted by `total_penalties` descending then by
  `player_name` for stability.
- `recommended_dive(p_L, p_C, p_R)` — the keeper's optimal pre-kick
  dive, `argmin` over the three probabilities.

The `MatchContext` is a pure value object — it carries the match's
identity (FotMob match id), the two teams (id + name), the kickoff
time, and the round. The round string is the same form FotMob uses on
the fixture list AND the same form the model was trained on
("1/8", "1/4", "1/2", "bronze", "final"), so the re-score is a
literal pass-through — no round-name translation needed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from .client import FotMobClient
from .player_history import PlayerPenalty
from .predict import (
    PredictContext,
    predict_roster_with_context,
)
from .rosters import RosterPlayer

# FotMob round codes for the four knockout rounds we surface in the
# dashboard. "bronze" (3rd-place play-off) is excluded — it doesn't have
# a shootout in practice and would never be selected for a "what side
# should the keeper dive" use case.
#
# These exact strings are what FotMob returns in the fixture's `round`
# field AND what the model was trained on (the training data's
# `ShootoutKick.round` carries the same codes), so the dashboard can
# pass them straight through to `PredictContext(round=...)` without a
# translation layer.
KNOCKOUT_ROUNDS: frozenset[str] = frozenset({"1/8", "1/4", "1/2", "final"})


@dataclass(frozen=True)
class MatchContext:
    """One upcoming knockout match, filtered to both-teams-decided.

    Carries the union of fields the dashboard renders (kickoff, round,
    the two teams) and the fields `predict_match` needs (the team ids
    to filter the roster, the round string for the PredictContext).

    `kickoff_utc` is the parsed datetime (UTC) of the fixture's
    `status.utcTime`. The `round` is the FotMob round code
    ("1/8" / "1/4" / "1/2" / "final") — same form the model was
    trained on.
    """

    match_id: int
    kickoff_utc: datetime
    round: str
    home_team_id: int
    home_team_name: str
    away_team_id: int
    away_team_name: str


@dataclass(frozen=True)
class KickerPrediction:
    """One kicker's per-match prediction, for the dashboard's table.

    Same probabilities as `PredictionRow` (sum to 1.0 within 1e-6), but
    with the `recommended_dive` pre-computed (so the UI doesn't have
    to do the argmin) and the `total_penalties` count included (the
    UI sorts by this column descending — the most-experienced kicker
    at the top of the table). The `kicker_id` / `player_id` is the
    FotMob `playerId`; the UI uses it as a stable row key.
    """

    player_id: int
    player_name: str
    team_id: int
    team_name: str
    kicking_foot: str
    total_penalties: int
    p_L: float
    p_C: float
    p_R: float
    recommended_dive: str  # "L" | "C" | "R"


# ---------------------------------------------------------------------------
# Match filter
# ---------------------------------------------------------------------------


def is_placeholder_team(name: str, team_id: int) -> bool:
    """A team is a placeholder if its name or id indicates an undecided slot.

    FotMob uses two placeholder shapes on the WC 2026 fixture list:
    - "Winner EF 1" / "Loser SF 2" — explicit slot names that name
      the previous round's outcome
    - "Netherlands/Morocco" — two teams joined by a slash, naming the
      two group-stage opponents whose winner will fill the slot

    Both indicate the team is not yet decided; the dashboard hides
    those matches. Real teams always have a non-zero `team_id` (the
    FotMob integer teamId); placeholders may have a non-zero id but
    the name is the authoritative signal. A missing/empty name with
    id 0 is also a placeholder.
    """
    if not name:
        return True
    if team_id == 0:
        return True
    stripped = name.strip()
    if stripped.startswith("Winner") or stripped.startswith("Loser"):
        return True
    if "/" in stripped:
        return True
    return False


def _parse_kickoff_utc(value: str) -> datetime | None:
    """Parse a FotMob `status.utcTime` (ISO 8601 with `Z`) into a UTC datetime.

    Returns `None` when the value is missing or malformed — the caller
    drops the fixture (an unparseable kickoff is the same as no kickoff
    for filter purposes).
    """
    if not value:
        return None
    text = value
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).astimezone(UTC)
    except ValueError:
        return None


def load_upcoming_knockouts(
    client: FotMobClient,
    *,
    now: datetime | None = None,
    league_id: int = 77,
    slug: str = "world-cup",
    season: int = 2026,
) -> list[MatchContext]:
    """Fetch the WC 2026 fixture list, filter to upcoming knockout matches.

    The filter is the conjunction of three conditions:

    1. **Knockout round.** `round` is in `KNOCKOUT_ROUNDS` (R16, QF,
       SF, F). Group-stage matches (round `1`, `2`, `3`, `1/16`) and
       the 3rd-place play-off (`bronze`) are dropped.
    2. **Upcoming.** The fixture's `status.utcTime` is strictly after
       `now` (default: current UTC time). Past matches are dropped so
       the user never picks a match that's already kicked off.
    3. **Both teams decided.** Neither `home` nor `away` is a
       placeholder. The placeholder check is the conjunction of
       name pattern (no "Winner " / "Loser " prefix, no `/`) and
       non-zero `id` (defensive; the live payload always has them for
       real teams).

    The returned list is sorted by `kickoff_utc` ascending so the
    selectbox shows the nearest match first.

    `now` is a parameter (defaulting to `datetime.now(UTC)`) so the
    filter is testable against a fixed clock. The Streamlit app passes
    the default; the test suite passes a `now` that pins the fixtures
    to known upcoming / past states.
    """
    if now is None:
        now = datetime.now(UTC)
    payload = client.get(f"leagues/{league_id}/overview/{slug}", params={"season": str(season)})
    fixtures = (payload.get("pageProps") or {}).get("fixtures") or {}
    all_matches = list(fixtures.get("allMatches") or [])

    out: list[MatchContext] = []
    for f in all_matches:
        round_name = str(f.get("round") or "")
        if round_name not in KNOCKOUT_ROUNDS:
            continue
        kickoff = _parse_kickoff_utc(str((f.get("status") or {}).get("utcTime") or ""))
        if kickoff is None or kickoff <= now:
            continue
        home = f.get("home") or {}
        away = f.get("away") or {}
        home_id = int(home.get("id") or 0)
        away_id = int(away.get("id") or 0)
        home_name = str(home.get("name") or "")
        away_name = str(away.get("name") or "")
        if is_placeholder_team(home_name, home_id):
            continue
        if is_placeholder_team(away_name, away_id):
            continue
        out.append(
            MatchContext(
                match_id=int(f.get("id") or 0),
                kickoff_utc=kickoff,
                round=round_name,
                home_team_id=home_id,
                home_team_name=home_name,
                away_team_id=away_id,
                away_team_name=away_name,
            )
        )
    out.sort(key=lambda m: m.kickoff_utc)
    return out


# ---------------------------------------------------------------------------
# Re-score
# ---------------------------------------------------------------------------


def _roster_for_match(
    roster: Sequence[RosterPlayer],
    home_team_id: int,
    away_team_id: int,
) -> list[RosterPlayer]:
    """Filter the full WC roster to the match's two teams.

    The roster is `output/wc2026_roster.jsonl` — every player in every
    WC 2026 squad. The match is between two of those 48 teams; we
    drop every player whose `team_id` is neither the home nor the
    away team. Defensive: a player with `team_id == 0` (shouldn't
    happen in the live roster, but possible in fixtures) is also
    dropped.
    """
    return [p for p in roster if p.team_id in (home_team_id, away_team_id)]


def _total_penalties(
    player_id: int,
    player_history: Mapping[int, Sequence[PlayerPenalty]],
) -> int:
    """Count the kicker's rows in `player_history` (the A2 feature's source).

    The dashboard sorts by `total_penalties` descending so the
    most-experienced kicker is at the top of the table. The count is
    over the full `player_history` map (no date filter) — the same
    window the model uses, since `predict_roster_with_context` will
    re-filter per row.
    """
    return len(player_history.get(player_id, []))


def predict_match(
    roster: Sequence[RosterPlayer],
    player_history: Mapping[int, Sequence[PlayerPenalty]],
    metadata_fetcher: Any,
    model: Any,
    context: MatchContext,
    *,
    target_date: str | None = None,
) -> list[KickerPrediction]:
    """Re-score the match's likely kickers with the match's actual round.

    Filters the WC roster to the match's two teams, builds a
    `PredictContext(round=context.round)` (the only non-neutral
    override — the B3 feature), calls
    `predict_roster_with_context`, and packages the result into
    `KickerPrediction`s sorted by `total_penalties` descending.

    `target_date` defaults to "tomorrow" (in UTC) so the lookback
    window includes every penalty the kicker has taken to date. The
    dashboard's single-page app always wants today's predictions,
    not a pinned historical date.
    """
    if target_date is None:
        target_date = (datetime.now(UTC) + timedelta(days=1)).date().isoformat()
    match_roster = _roster_for_match(roster, context.home_team_id, context.away_team_id)
    predict_context = PredictContext(round=context.round)
    rows = predict_roster_with_context(
        model,
        match_roster,
        player_history,
        metadata_fetcher,
        target_date,
        predict_context,
    )
    out: list[KickerPrediction] = []
    for r in rows:
        out.append(
            KickerPrediction(
                player_id=r.player_id,
                player_name=r.player_name,
                team_id=r.team_id,
                team_name=r.team_name,
                kicking_foot=r.kicking_foot,
                total_penalties=_total_penalties(r.player_id, player_history),
                p_L=r.p_L,
                p_C=r.p_C,
                p_R=r.p_R,
                recommended_dive=recommended_dive(r.p_L, r.p_C, r.p_R),
            )
        )
    out.sort(key=lambda k: (-k.total_penalties, k.player_name))
    return out


# ---------------------------------------------------------------------------
# Recommended dive
# ---------------------------------------------------------------------------


def recommended_dive(p_L: float, p_C: float, p_R: float) -> str:
    """The keeper's optimal pre-kick dive: `argmin` over the three probabilities.

    The model's policy is uniform-prior-dive: the keeper picks the
    side with the lowest predicted probability of the kicker aiming
    there. The output is one of `"L"`, `"C"`, `"R"`. Ties are broken
    by the documented L→C→R order so the function is deterministic —
    `recommended_dive(0.33, 0.33, 0.34) == "L"` (L and C tie at 0.33,
    R is strictly larger, so L wins by the L→C→R tiebreaker).
    """
    minimum = min(p_L, p_C, p_R)
    for side, value in (("L", p_L), ("C", p_C), ("R", p_R)):
        if value == minimum:
            return side
    return "L"  # unreachable: `min` is one of the three


__all__ = [
    "KickerPrediction",
    "KNOCKOUT_ROUNDS",
    "MatchContext",
    "is_placeholder_team",
    "load_upcoming_knockouts",
    "predict_match",
    "recommended_dive",
]
