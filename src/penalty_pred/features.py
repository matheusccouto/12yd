"""Feature builder for the penalty shootout model.

PRD: For each Shootout Kick (target), build a 9-feature row from the
kicker's penalty history (filtered to before the target kick date) and
the player's metadata (position, birth date). The output is the
`training_table.jsonl` artifact — one row per target kick — that the
model slice (#23, #24) consumes.

The 9 features (with their internal column count):

- **A1** — `P(L), P(C), P(R)` over the last 5, 10, and 20 kicks
  (continuous x bucketed; 9 columns: `p_L_5, p_C_5, p_R_5, ...,
  p_R_20`).
- **A2** — last kick's side ("L" / "C" / "R"; "" when no history).
- **A3** — kicking foot (mode of `shot_type` in history; "RightFoot"
  on tie; "Unknown" when no history).
- **A4** — total career penalty count (before the target kick date).
- **B1** — kick number within the shootout (pass-through from
  `shootout_kicks.jsonl`).
- **B2** — current shootout score (`pen_score_before[0]`,
  `pen_score_before[1]`) plus an `is_decisive` flag.
- **B3** — match round (pass-through from `shootout_kicks.jsonl`).
- **C1** — position (from the player page; "" when metadata is
  missing).
- **C2** — age in years (target_match_date − dateOfBirth; `null` when
  birth date is missing).

For kickers with no penalty history, A1 falls back to the uniform prior
(1/3, 1/3, 1/3), A2 is "", A3 is "Unknown", and A4 is 0 — the same
defaults the model will see at prediction time for the prediction
kickers.

The orchestrator (`build_training_table`) consumes two artifacts:
`shootout_kicks.jsonl` (the targets) and `player_history.jsonl` (the
per-kicker history, keyed by `kicker_id`). It also fetches the
kicker's player page once per unique kicker to recover C1/C2; the
fetch is cache-hit-dominated because the player-history slice
already populated the disk cache for every Initial Set kicker.

Re-runs are idempotent: the same input JSONLs and the same FotMob
cache produce byte-identical output. The orchestrator sorts the
training table by `(match_date, match_id, kick_number)` so the
output order is stable.

Phase 0 (Issue #30): a `PredictionTarget` value object carries only
the 9 model features (no identifiers, no label). `build_features` is
a thin packaging function that takes a `PredictionTarget` plus the
row's identifiers and label, and returns a `TrainingRow`. The
computation (A-group from history, C-group from metadata, is_decisive
from the B-group context) lives in `compute_features` so the same
pipeline serves both the training slice (with real B-group values
from a `ShootoutKick`) and the prediction slice (with neutral
B-group values, no fake `ShootoutKick`).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from .player_history import (
    PlayerMetadata,
    PlayerPenalty,
    extract_player_metadata,
    fetch_player_data,
)

# Prior over (L, C, R) for kickers with no history. The PRD specifies
# "1/3 each" for the missing-history fallback.
PRIOR_PROB: tuple[float, float, float] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

# A1 horizons (PRD: "P(L), P(C), P(R) over the last 5, 10, and 20 kicks").
A1_HORIZONS: tuple[int, ...] = (5, 10, 20)

# Canonical class order. Probabilities are always returned in this order:
# index 0 = P(L), index 1 = P(C), index 2 = P(R). The class strings are
# the literal labels used in `training_table.jsonl`. Lives in
# `features.py` because the unified `TrainingRow`'s `label_index` reads
# from it; the model layer re-imports it from here.
CLASSES: tuple[str, ...] = ("L", "C", "R")


# ---------------------------------------------------------------------------
# Row schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PredictionTarget:
    """The 9 model feature inputs (19 fields) for a single prediction target.

    Carries only what the model needs — no identifiers, no label, no
    `is_on_target`. The data layer's `ShootoutKick` is no longer
    threaded through the feature builder; the prediction slice
    constructs a `PredictionTarget` directly (with neutral B-group
    values) and the feature builder packages it into a `TrainingRow`.

    For the training slice, `compute_features` produces a
    `PredictionTarget` from a `ShootoutKick` + history + metadata. For
    the prediction slice, the same function produces one from
    history + metadata + neutral B-group context — no synthetic
    `ShootoutKick` needed.
    """

    # A1: P(L), P(C), P(R) over the last 5, 10, 20 kicks (chronological, oldest first)
    p_L_5: float
    p_C_5: float
    p_R_5: float
    p_L_10: float
    p_C_10: float
    p_R_10: float
    p_L_20: float
    p_C_20: float
    p_R_20: float
    # A2: last kick's side
    last_side: str
    # A3: kicking foot
    kicking_foot: str
    # A4: total career penalty count (before the target kick date)
    career_penalty_count: int
    # B1: kick number within the shootout
    b1_kick_number: int
    # B2: current shootout score + is_decisive flag
    pen_score_home: int
    pen_score_away: int
    is_decisive: bool
    # B3: match round
    b3_round: str
    # C1: position
    position: str
    # C2: age in years
    age: float  # NaN → None in the JSONL (Python json does not emit NaN).


@dataclass(frozen=True)
class TrainingRow:
    """One training row: a target Shootout Kick plus the 9 features for it.

    The unified row type (Issue #30): replaces both the old
    `TrainingTableRow` (features.py, 30 fields, no `is_on_target`) and
    the old `TrainingRow` (model.py, 13 fields + a `features` dict).
    The 19 model features are individual fields (not a dict) so the
    type checker can pin them, and the row carries its own identifiers
    and label so the on-disk format is self-contained.

    Identifier fields are pass-throughs from `shootout_kicks.jsonl` so
    the row is self-contained. `label` is the side the kicker actually
    took — the supervised target. `is_on_target` is the per-row flag
    for the counterfactual save rate (slice #22 dropped it from the
    JSONL; the model layer joins it back in via
    `is_on_target_by_key(shootout_kicks)`).

    The `is_decisive` flag is the B2 part: whether this kick is one
    whose outcome ends the shootout (either a Goal or a Miss/Missed
    clinches it). The computation lives in `is_decisive_kick`.
    """

    # Identifiers
    match_id: int
    kick_number: int
    kicker_id: int
    kicker_name: str
    match_date: str
    tournament_id: int
    tournament_name: str
    round: str
    team_id: int
    is_home: bool
    # Label
    label: str
    # is_on_target (joined from shootout_kicks.jsonl; True for prediction)
    is_on_target: bool
    # A1: P(L), P(C), P(R) over the last 5, 10, 20 kicks (chronological, oldest first)
    p_L_5: float
    p_C_5: float
    p_R_5: float
    p_L_10: float
    p_C_10: float
    p_R_10: float
    p_L_20: float
    p_C_20: float
    p_R_20: float
    # A2: last kick's side
    last_side: str
    # A3: kicking foot
    kicking_foot: str
    # A4: total career penalty count (before the target kick date)
    career_penalty_count: int
    # B1: kick number (duplicate of `kick_number` for column ordering)
    b1_kick_number: int
    # B2: current shootout score + is_decisive flag
    pen_score_home: int
    pen_score_away: int
    is_decisive: bool
    # B3: match round (duplicate of `round` for column ordering)
    b3_round: str
    # C1: position
    position: str
    # C2: age in years
    age: float  # NaN → None in the JSONL (Python json does not emit NaN).

    @property
    def label_index(self) -> int:
        return CLASSES.index(self.label)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def side_distribution(sides: Sequence[str], n: int) -> tuple[float, float, float]:
    """Return `(p_L, p_C, p_R)` over the last `n` entries of `sides`.

    The input is treated as chronological (oldest first, latest last).
    The "last n" are the n most recent entries. If `len(sides) < n`,
    we use all available entries. Empty input → the uniform prior
    `(1/3, 1/3, 1/3)`.

    The function does not validate the side strings — the caller is
    expected to have already bucketed via `coordinates.side` (every
    value in `sides` is "L", "C", or "R"). An unexpected value would
    not crash; it would simply be excluded from the count.
    """
    if not sides:
        return PRIOR_PROB
    recent = sides[-n:] if n > 0 else []
    total = len(recent)
    if total == 0:
        return PRIOR_PROB
    n_l = sum(1 for s in recent if s == "L")
    n_c = sum(1 for s in recent if s == "C")
    n_r = sum(1 for s in recent if s == "R")
    return (n_l / total, n_c / total, n_r / total)


def mode_kicking_foot(shot_types: Sequence[str]) -> str:
    """Return the mode of `shot_types`, with "RightFoot" as the tiebreaker.

    Per PRD: ties are broken in favour of "RightFoot" because the
    population is right-foot-dominant. Returns "Unknown" for empty
    input. Non-{RightFoot, LeftFoot} values are ignored (e.g. a
    "Header" shot from a deflection would not count).
    """
    relevant = [s for s in shot_types if s in ("RightFoot", "LeftFoot")]
    if not relevant:
        return "Unknown"
    counts = Counter(relevant)
    max_count = max(counts.values())
    candidates = [v for v, c in counts.items() if c == max_count]
    if "RightFoot" in candidates:
        return "RightFoot"
    return candidates[0]


def is_decisive_kick(
    pen_score_home: int,
    pen_score_away: int,
    home_kicks_done: int,
    away_kicks_done: int,
    is_home_kicking: bool,
) -> bool:
    """Whether this kick is "decisive" (its outcome ends the shootout).

    A kick is decisive if scoring OR missing it ends the shootout. The
    shootout ends when one team has clinched (their score exceeds the
    opponent's score + the opponent's remaining kicks) OR has been
    eliminated (their score + their remaining kicks is less than the
    opponent's score).

    The function assumes the standard 5-kick-each format (and sudden
    death as needed). For sudden death kicks, both outcomes leave the
    shootout open (the next pair of kicks matters), so the result is
    always False — the kick is "decisive" only in retrospect, never in
    advance.

    Parameters
    ----------
    pen_score_home, pen_score_away
        The shootout score BEFORE this kick.
    home_kicks_done, away_kicks_done
        How many kicks each team has taken BEFORE this kick (i.e.
        excluding this one).
    is_home_kicking
        True iff the kicking team is the home team.
    """
    # Kicks remaining AFTER this kick. The kicking team's count
    # increments by 1; the other team's count is unchanged. Clamp to 0
    # to be safe with out-of-range inputs (a defensive floor; live
    # data is well-behaved).
    if is_home_kicking:
        home_remaining = max(0, 4 - home_kicks_done)
        away_remaining = max(0, 5 - away_kicks_done)
    else:
        home_remaining = max(0, 5 - home_kicks_done)
        away_remaining = max(0, 4 - away_kicks_done)

    def _shootout_ended(h: int, a: int) -> bool:
        return (h > a + away_remaining) or (a > h + home_remaining)

    if is_home_kicking:
        scored = (pen_score_home + 1, pen_score_away)
        missed = (pen_score_home, pen_score_away)
    else:
        scored = (pen_score_home, pen_score_away + 1)
        missed = (pen_score_home, pen_score_away)

    return _shootout_ended(*scored) or _shootout_ended(*missed)


def age_in_years(birth_date_str: str, target_date_str: str) -> float:
    """Return the age in years (float) at the target date.

    Returns `nan` if `birth_date_str` is empty or malformed. The result
    is the number of completed years (the fractional part is the time
    within the year, but in practice the granularity is whole years
    because the scraper stores match dates as ISO 8601 to the second
    and birth dates as dates).
    """
    if not birth_date_str:
        return math.nan
    try:
        birth = date.fromisoformat(birth_date_str)
    except ValueError:
        return math.nan
    try:
        target = datetime.fromisoformat(target_date_str).astimezone(UTC).date()
    except ValueError:
        return math.nan
    years = target.year - birth.year
    if (target.month, target.day) < (birth.month, birth.day):
        years -= 1
    return float(years)


# ---------------------------------------------------------------------------
# Per-kick feature builder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KickIndex:
    """Pre-computed `kicks_done` counts for one (match_id, kick_number).

    Built by `index_kicks_done` over the full `shootout_kicks.jsonl`.
    The orchestrator looks up the index per target kick to compute the
    `is_decisive` flag.
    """

    home_kicks_done: int
    away_kicks_done: int


@dataclass(frozen=True)
class BGroupContext:
    """The B-group inputs (B1, B2, B3) for a single prediction target.

    Carries the shootout-state information that is known for a training
    target (from the `ShootoutKick`) and is neutral for a prediction
    target. Using a value object instead of a `ShootoutKick` keeps the
    data layer's shape from leaking across the feature-builder seam.

    For a prediction target: `kick_number=1`, `pen_score_before=(0, 0)`,
    `is_home=True` (doesn't matter; both 0 kicks done → is_decisive=False),
    `round=""` (LightGBM treats the empty string as missing).
    """

    kick_number: int
    pen_score_home: int
    pen_score_away: int
    is_home: bool
    round: str

    @classmethod
    def neutral(cls) -> BGroupContext:
        """The neutral B-group context for a prediction target."""
        return cls(
            kick_number=1,
            pen_score_home=0,
            pen_score_away=0,
            is_home=True,
            round="",
        )


def index_kicks_done(shootout_kicks: Iterable) -> dict[tuple[int, int], KickIndex]:
    """Build a `(home_kicks_done, away_kicks_done)` index per kick.

    For each `(match_id, kick_number)` in the input, records how many
    kicks each team had taken BEFORE that kick. The index is computed
    by walking the kicks per-match in `kick_number` order.

    The function is idempotent: re-running on the same input yields
    the same index.
    """
    from .shootouts import ShootoutKick

    by_match: dict[int, list[ShootoutKick]] = {}
    for kick in shootout_kicks:
        by_match.setdefault(kick.match_id, []).append(kick)
    out: dict[tuple[int, int], KickIndex] = {}
    for match_id, kicks in by_match.items():
        kicks.sort(key=lambda k: k.kick_number)
        home_done = 0
        away_done = 0
        for kick in kicks:
            out[(match_id, kick.kick_number)] = KickIndex(
                home_kicks_done=home_done,
                away_kicks_done=away_done,
            )
            if kick.is_home:
                home_done += 1
            else:
                away_done += 1
    return out


def filter_history(
    history: Iterable[PlayerPenalty],
    target_date: str,
) -> list[PlayerPenalty]:
    """Return the player's penalties strictly before `target_date`.

    `target_date` is the ISO 8601 match start time of the target kick
    (e.g. "2022-12-18T15:00:00+00:00"). Penalties with the same
    timestamp as the target are excluded (strict `<`), so the current
    match's shootout kicks are not in the kicker's history for any of
    its own kicks.

    The result is sorted by `match_date` (ascending) so "last n" in
    `side_distribution` and the model's `last_side` feature means the
    most recent n.
    """
    out: list[PlayerPenalty] = []
    for row in history:
        if row.match_date < target_date:
            out.append(row)
    out.sort(key=lambda r: r.match_date)
    return out


def compute_features(
    history: Sequence[PlayerPenalty],
    metadata: PlayerMetadata | None,
    target_date: str,
    b_group: BGroupContext,
    kicks_done: KickIndex,
) -> PredictionTarget:
    """Compute the 19 model features for one prediction target.

    Pure function. `history` is the kicker's full scraped history;
    this function filters it to before `target_date` and computes the
    A-group (side distribution, last_side, kicking_foot, career count).
    `metadata` may be None if the player page could not be fetched
    (then C1 is "" and C2 is NaN). `b_group` carries the B-group
    inputs (kick_number, pen_score, is_home, round); `kicks_done` is
    the pre-computed index for the `is_decisive` flag.

    The function is the shared core of the training slice (where
    `b_group` comes from a real `ShootoutKick`) and the prediction
    slice (where `b_group = BGroupContext.neutral()`). No `ShootoutKick`
    is constructed in the prediction path.
    """
    filtered = filter_history(history, target_date)
    sides = [p.side for p in filtered]
    shot_types = [p.shot_type for p in filtered]

    p_L_5, p_C_5, p_R_5 = side_distribution(sides, 5)
    p_L_10, p_C_10, p_R_10 = side_distribution(sides, 10)
    p_L_20, p_C_20, p_R_20 = side_distribution(sides, 20)

    position = metadata.position_key if metadata is not None else ""
    birth_date = metadata.birth_date if metadata is not None else ""
    age = age_in_years(birth_date, target_date)

    return PredictionTarget(
        p_L_5=p_L_5,
        p_C_5=p_C_5,
        p_R_5=p_R_5,
        p_L_10=p_L_10,
        p_C_10=p_C_10,
        p_R_10=p_R_10,
        p_L_20=p_L_20,
        p_C_20=p_C_20,
        p_R_20=p_R_20,
        last_side=sides[-1] if sides else "",
        kicking_foot=mode_kicking_foot(shot_types),
        career_penalty_count=len(filtered),
        b1_kick_number=b_group.kick_number,
        pen_score_home=b_group.pen_score_home,
        pen_score_away=b_group.pen_score_away,
        is_decisive=is_decisive_kick(
            b_group.pen_score_home,
            b_group.pen_score_away,
            kicks_done.home_kicks_done,
            kicks_done.away_kicks_done,
            b_group.is_home,
        ),
        b3_round=b_group.round,
        position=position,
        age=age,
    )


def build_features(
    features: PredictionTarget,
    *,
    match_id: int,
    kick_number: int,
    kicker_id: int,
    kicker_name: str,
    match_date: str,
    tournament_id: int,
    tournament_name: str,
    round: str,
    team_id: int,
    is_home: bool,
    label: str,
    is_on_target: bool,
) -> TrainingRow:
    """Package a `PredictionTarget` into a `TrainingRow`.

    Thin packaging function (Issue #30): takes the 19 model features
    plus the row's identifiers and label, returns the unified row
    type. The actual feature computation lives in `compute_features`.

    `is_on_target` is the per-row flag for the counterfactual save
    rate. For training rows it comes from the `ShootoutKick` (joined
    by `is_on_target_by_key` in `load_training_table`); for prediction
    rows it is `True` (the dummy value — the model doesn't use it at
    predict time, but the unified type carries it so the on-disk
    schema is self-describing).
    """
    return TrainingRow(
        match_id=match_id,
        kick_number=kick_number,
        kicker_id=kicker_id,
        kicker_name=kicker_name,
        match_date=match_date,
        tournament_id=tournament_id,
        tournament_name=tournament_name,
        round=round,
        team_id=team_id,
        is_home=is_home,
        label=label,
        is_on_target=is_on_target,
        p_L_5=features.p_L_5,
        p_C_5=features.p_C_5,
        p_R_5=features.p_R_5,
        p_L_10=features.p_L_10,
        p_C_10=features.p_C_10,
        p_R_10=features.p_R_10,
        p_L_20=features.p_L_20,
        p_C_20=features.p_C_20,
        p_R_20=features.p_R_20,
        last_side=features.last_side,
        kicking_foot=features.kicking_foot,
        career_penalty_count=features.career_penalty_count,
        b1_kick_number=features.b1_kick_number,
        pen_score_home=features.pen_score_home,
        pen_score_away=features.pen_score_away,
        is_decisive=features.is_decisive,
        b3_round=features.b3_round,
        position=features.position,
        age=features.age,
    )


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def load_player_history(path: Path) -> dict[int, list[PlayerPenalty]]:
    """Load `player_history.jsonl` into a dict keyed by `kicker_id`.

    Each value is the unsorted list of `PlayerPenalty` rows for that
    kicker. The caller is expected to sort by `match_date` after
    filtering to the target date (see `filter_history`).
    """
    import json

    out: dict[int, list[PlayerPenalty]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out.setdefault(int(row["kicker_id"]), []).append(PlayerPenalty(**row))
    return out


MetadataFetcher = Callable[[int], PlayerMetadata | None]


def cached_metadata_fetcher(
    fetch_one: Callable[[int], PlayerMetadata | None],
) -> MetadataFetcher:
    """Wrap a per-id fetcher with a per-process dict cache.

    The training slice sees ~163 unique training kickers; fetching each
    player's page once and caching the result in memory keeps the
    orchestrator's per-kick cost O(1) after the first lookup.
    """
    cache: dict[int, PlayerMetadata | None] = {}

    def _get(player_id: int) -> PlayerMetadata | None:
        if player_id not in cache:
            cache[player_id] = fetch_one(player_id)
        return cache[player_id]

    return _get


def fetcher_from_client(client: Any) -> MetadataFetcher:
    """Build a `MetadataFetcher` from a `FotMobClient`.

    Uses the on-disk cache automatically (the same cache the
    player-history slice populated), so a re-run is a no-op once the
    cache is warm.
    """
    def _fetch(player_id: int) -> PlayerMetadata | None:
        try:
            payload = fetch_player_data(client, player_id)
        except Exception:  # noqa: BLE001 — boundary: a missing player page must not abort the row
            return None
        try:
            return extract_player_metadata(payload)
        except Exception:  # noqa: BLE001 — boundary: a malformed page must not abort the row
            return None

    return cached_metadata_fetcher(_fetch)


def build_training_table(
    shootout_kicks: Sequence,
    player_history: Mapping[int, Sequence[PlayerPenalty]],
    metadata_fetcher: MetadataFetcher,
) -> list[TrainingRow]:
    """Build the full training table from the inputs.

    Returns the rows sorted by `(match_date, match_id, kick_number)`
    for a stable, idempotent output. The function is pure: same input
    → same output.
    """
    from .shootouts import ShootoutKick

    kicks_done_index = index_kicks_done(shootout_kicks)
    rows: list[TrainingRow] = []
    for target in shootout_kicks:
        assert isinstance(target, ShootoutKick)
        history = player_history.get(target.kicker_id, [])
        metadata = metadata_fetcher(target.kicker_id)
        b_group = BGroupContext(
            kick_number=target.kick_number,
            pen_score_home=target.pen_score_before[0],
            pen_score_away=target.pen_score_before[1],
            is_home=target.is_home,
            round=target.round,
        )
        features = compute_features(
            history,
            metadata,
            target.match_date,
            b_group,
            kicks_done_index[(target.match_id, target.kick_number)],
        )
        rows.append(
            build_features(
                features,
                match_id=target.match_id,
                kick_number=target.kick_number,
                kicker_id=target.kicker_id,
                kicker_name=target.kicker_name,
                match_date=target.match_date,
                tournament_id=target.tournament_id,
                tournament_name=target.tournament_name,
                round=target.round,
                team_id=target.team_id,
                is_home=target.is_home,
                label=target.side,
                is_on_target=target.is_on_target,
            )
        )
    rows.sort(key=lambda r: (r.match_date, r.match_id, r.kick_number))
    return rows
