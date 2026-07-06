"""Streamlit dashboard logic — the library side of the live shootout predictor.

PRD: A single-page Streamlit app on Streamlit Cloud that surfaces live
shootout predictions. At load time, the app fetches the WC 2026 fixture
list from FotMob, filters to upcoming matches with both teams decided
(any round, including the 48-team format's Round of 32, code `"1/16"`),
and lets the user pick a match from a selectbox. For the selected
match, the app loads `predictions.jsonl` from HF, filters to the
match's two teams, and shows a per-kicker table: name, team, kicking
foot, P(L), P(C), P(R), and the recommended dive (`argmin`).

This module is the dashboard's *library* side: the data loading, the
match filter, the per-kicker view, and the recommended dive. The
Streamlit `app.py` at the repo root is a thin layer on top of these
functions — the seam is the data, not the UI, so the logic can be
unit-tested without launching Streamlit.

The dashboard's entry points (the functions the PRD names):

- `load_upcoming_knockouts(client)` — fetch + filter to upcoming + both
  teams decided. The round is not consulted (any round passes, so the
  selector adapts to whatever knockout stage the tournament is in:
  R32, R16, QF, SF, F). Returns a list of `MatchContext`.
- `predictions_for_match(predictions, context, *, player_history=...)` —
  filter the round-agnostic `predictions.jsonl` to the match's two
  teams and return a list of `KickerPrediction`, sorted by
  `total_penalties` descending (with name as tiebreaker) for a stable
  table order. `total_penalties` is read from `player_history` when
  provided; the v4 card layout uses it to show "N career penalties"
  on each card and to rank the most-experienced kickers first.
- `recommended_dive(p_L, p_C, p_R)` — the keeper's optimal pre-kick
  dive, `argmin` over the three probabilities.
- `opposite_side(side)` — the Kicker's mirror-side (`L↔R`, `C↔C`),
  used by the v4 card layout to render the "GK dive ↔ X" hint in the
  Kicker's PoV.

v3 (Issue #36) collapsed the per-match re-score path: with `b3_round`
dropped from the model schema, every match shows the same per-kicker
probabilities from `predictions.jsonl`. The dashboard now reads the
artifact directly; the round is a display attribute only.

v4 (Issue #48) re-introduces `total_penalties` as a per-kicker field
on `KickerPrediction`, populated from `player_history` so the card
layout can show "N career penalties" on each card and sort by
experience. The card layout itself lives in `app.py`; this module
stays the library seam.

The `MatchContext` is a pure value object — it carries the match's
identity (FotMob match id), the two teams (id + name), the kickoff
time, and the round (kept for display only).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from .client import FotMobClientLike
from .player_history import PlayerPenalty
from .predict import PredictionRow


@dataclass(frozen=True)
class MatchContext:
    """One upcoming match, filtered to both-teams-decided.

    Carries the union of fields the dashboard renders (kickoff, round,
    the two teams) and the fields `predictions_for_match` needs (the
    team ids to filter the predictions to the match's two squads).

    `kickoff_utc` is the parsed datetime (UTC) of the fixture's
    `status.utcTime`. `round` is the FotMob round code (e.g. `"1/16"`
    for Round of 32, `"1/8"` for Round of 16, `"1/4"` for QF, `"1/2"`
    for SF, `"final"` for F) — kept as a display attribute only; the
    model is round-agnostic (v3 dropped B3).
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
    at the top of the table). The `player_id` is the FotMob
    `playerId`; the UI uses it as a stable row key.

    `total_penalties` is the size of the Kicker's penalty history in
    `player_history` (the A-group count). The v4 card layout renders
    it as "N career penalties" on each card; the sort uses it as the
    primary key.
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
    client: FotMobClientLike,
    *,
    now: datetime | None = None,
    league_id: int = 77,
    slug: str = "world-cup",
    season: int = 2026,
) -> list[MatchContext]:
    """Fetch the WC 2026 fixture list, filter to upcoming matches with both teams decided.

    The filter is the conjunction of two conditions:

    1. **Upcoming.** The fixture's `status.utcTime` is strictly after
       `now` (default: current UTC time). Past matches are dropped so
       the user never picks a match that's already kicked off.
    2. **Both teams decided.** Neither `home` nor `away` is a
       placeholder. The placeholder check is the conjunction of
       name pattern (no "Winner " / "Loser " prefix, no `/`) and
       non-zero `id` (defensive; the live payload always has them for
       real teams).

    The round is not consulted — group-stage matches are filtered out
    only if at least one team is a placeholder (group-stage matches
    whose group opponents are joined by `/`, e.g. `"Netherlands/Morocco"`,
    ARE placeholders and are dropped). The 48-team WC's Round of 32
    (FotMob code `"1/16"`) is the first knockout round and passes the
    filter like any other round. The same code works for the 32-team
    WC's Round of 16 (code `"1/8"`), the 24-team format's playoff
    round, etc.

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
# Per-match view (read predictions.jsonl)
# ---------------------------------------------------------------------------


def predictions_for_match(
    predictions: Iterable[PredictionRow],
    context: MatchContext,
    *,
    player_history: Mapping[int, Sequence[PlayerPenalty]] | None = None,
) -> list[KickerPrediction]:
    """Filter the round-agnostic `predictions.jsonl` to the match's two teams.

    v3 (Issue #36): the previous `predict_match` re-score is gone.
    The model is round-agnostic, so the artifact on disk is the
    source of truth for every match. This function is the per-match
    view: filter to `context.home_team_id` / `context.away_team_id`,
    compute `recommended_dive` per kicker, and sort by
    `total_penalties` descending (with name as the tiebreaker for
    stability).

    `total_penalties` is the per-kicker count of the player's
    `player_history` — the A2/A4 source. v4 (Issue #48) reintroduces
    it as a per-kicker field; the v3 dashboard used a constant 0
    placeholder because the v3 layout was a table where the column
    was sortable client-side. The v4 card layout renders it as
    "N career penalties" on each card and uses it as the primary
    sort key (most-experienced kickers at the top).

    When `player_history` is omitted (the historical v3 callers that
    didn't read history), every `total_penalties` is 0 and the sort
    falls back to name (the v3 behaviour). The Streamlit app passes
    the loaded `player_history.jsonl` so the cards are sorted by
    experience.
    """
    match_team_ids = {context.home_team_id, context.away_team_id}
    out: list[KickerPrediction] = []
    for r in predictions:
        if r.team_id not in match_team_ids:
            continue
        if player_history is None:
            total = 0
        else:
            total = len(player_history.get(r.player_id, ()))
        out.append(
            KickerPrediction(
                player_id=r.player_id,
                player_name=r.player_name,
                team_id=r.team_id,
                team_name=r.team_name,
                kicking_foot=r.kicking_foot,
                total_penalties=total,
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

    **Frame pin (Kicker-PoV).** The returned `"L"`, `"C"`, `"R"` is in
    the **Kicker's** point of view — the horizontal half of the goal
    as the Kicker sees it (per `CONTEXT.md`, `Side` is "the horizontal
    half of the goal from the kicker's perspective"). The model
    predicts where the Kicker will aim; `argmin` picks the side the
    Kicker is *least* likely to aim at; the Goalkeeper dives *that*
    side. A viewer reading the recommendation must re-anchor the L/R
    letter to themselves: the L the Kicker sees is the Goalkeeper's
    R. The new v4 card layout (Issue #48) surfaces this with a
    "Kicker will aim: L 55%  ·  GK dive: R ↔" prediction row so the
    re-anchoring is explicit; this function's return value is
    unchanged.

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


# ---------------------------------------------------------------------------
# v4 (Issue #48): opposite-side helper for the card's "GK dive ↔ X" hint.
# ---------------------------------------------------------------------------


def opposite_side(side: str) -> str:
    """The Kicker's mirror-side: `L ↔ R`, `C ↔ C`.

    The v4 card layout surfaces the dive hint as the **opposite** of
    the Kicker's most-likely aim (e.g. WILL AIM = L 55% → GK dive = R
    in the Kicker's PoV). The L/R letters are mirror images — the
    Kicker's L is the Goalkeeper's R, and vice versa. The centre has
    no mirror; the dive hint for a centre-aim is `C` (the Goalkeeper
    has nowhere to dive but stays central).

    The function is total over `"L" | "C" | "R"`. Any other input
    falls through to `side` unchanged — the card renderer treats
    `opposite_side` output as a display label and the dashboard
    always passes the canonical L/C/R letters from
    `recommended_dive` / `argmax`.
    """
    if side == "L":
        return "R"
    if side == "R":
        return "L"
    if side == "C":
        return "C"
    return side


def most_likely_side(p_L: float, p_C: float, p_R: float) -> str:
    """The Kicker's most-likely aim: `argmax` over the three probabilities.

    The v4 card layout's prediction row leads with "Kicker will aim:
    [side] [%]" — the most-likely side, in the Kicker's PoV. Ties
    are broken by the L→C→R order so the result is deterministic
    (`most_likely_side(0.33, 0.33, 0.34) == "R"` — R is the unique
    max).

    This is a *display* helper, not a model policy. The model's
    deployment policy remains `argmin` (see `recommended_dive`); the
    card shows the most-likely aim as the headline and the opposite
    side as the dive hint. When the Kicker's distribution is
    single-sided, the two are equal (opposite(argmax) = argmin); when
    the distribution is flatter they may differ.
    """
    maximum = max(p_L, p_C, p_R)
    for side, value in (("L", p_L), ("C", p_C), ("R", p_R)):
        if value == maximum:
            return side
    return "L"  # unreachable: `max` is one of the three


__all__ = [
    "KickerPrediction",
    "MatchContext",
    "is_placeholder_team",
    "load_upcoming_knockouts",
    "most_likely_side",
    "opposite_side",
    "predictions_for_match",
    "recommended_dive",
]
