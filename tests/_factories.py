"""Centralised test data factories for the twelveyards test suite.

One builder per row type. The schema constants (`*_FIELDS`) are derived
from `dataclasses.fields(...)` so adding a field to a row type is a
one-line change here, not a five-place hunt across the test files.

The deletion test for this module: every builder must be a real
duplicate of code that was previously inlined (or near-duplicated)
across test files. A builder that is only used by one test should
stay local to that test — pulling it here would create a
"hypothetical seam" that the deletion test rejects.

The factories are deliberately permissive: every field has a
sensible default, and callers override only what they care about. The
defaults are chosen so `make_training_row()`, `make_history_row()`,
etc. return a row that is valid for the data layer's reader
(`Artifacts.read_*`) and for the model layer's matrix builder
(`build_feature_matrix`).

Builders:
- `make_training_row` — the unified training row (features.TrainingRow).
- `make_history_row` — a `PlayerPenalty` (one row of player history).
- `make_metadata` — a `PlayerMetadata` (the per-player C-group data).
- `make_shootout_kick` — a `ShootoutKick` (one shootout kick).
- `make_roster_player` — a `RosterPlayer` (one WC squad entry).
- `make_initial_set_kicker` — an `InitialSetKicker`.
- `make_missing_kicker` — a `MissingKicker`.
- `make_prediction_row` — a `PredictionRow` (one row of predictions.jsonl).

Schema constants (derived from `dataclasses.fields`):
- `TRAINING_ROW_FIELDS`
- `PLAYER_PENALTY_FIELDS`
- `MISSING_KICKER_FIELDS`
- `PREDICTION_ROW_FIELDS`
"""

from __future__ import annotations

from dataclasses import fields as _fields

from twelveyards.features import TrainingRow
from twelveyards.initial_set import InitialSetKicker, MissingKicker
from twelveyards.player_history import PlayerMetadata, PlayerPenalty
from twelveyards.predict import PredictionRow
from twelveyards.rosters import RosterPlayer
from twelveyards.shootouts import ShootoutKick
from twelveyards.tournaments import TournamentKind

# ---------------------------------------------------------------------------
# Schema constants (derived from the dataclasses — adding a field is
# a one-line change in the source, not a five-place hunt in tests).
# ---------------------------------------------------------------------------


TRAINING_ROW_FIELDS: frozenset[str] = frozenset(f.name for f in _fields(TrainingRow))
PLAYER_PENALTY_FIELDS: frozenset[str] = frozenset(f.name for f in _fields(PlayerPenalty))
MISSING_KICKER_FIELDS: frozenset[str] = frozenset(f.name for f in _fields(MissingKicker))
PREDICTION_ROW_FIELDS: frozenset[str] = frozenset(f.name for f in _fields(PredictionRow))


# ---------------------------------------------------------------------------
# TrainingRow (unified row type — the model layer's input)
# ---------------------------------------------------------------------------


def make_training_row(
    label: str = "L",
    *,
    match_id: int = 1,
    kick_number: int = 1,
    kicker_id: int = 1,
    kicker_name: str = "Stub",
    match_date: str = "2024-06-01T00:00:00+00:00",
    tournament_id: int = 77,
    tournament_name: str = "World Cup",
    round: str = "1/8",
    team_id: int = 1,
    is_home: bool = True,
    is_on_target: bool = True,
    # A1 — uniform-ish; tests verify the matrix shape, not the math.
    p_L_5: float | None = None,
    p_C_5: float | None = None,
    p_R_5: float | None = None,
    p_L_10: float | None = None,
    p_C_10: float | None = None,
    p_R_10: float | None = None,
    p_L_20: float | None = None,
    p_C_20: float | None = None,
    p_R_20: float | None = None,
    # A2 / A3 / A4
    last_side: str = "L",
    preferred_foot: str = "right",
    career_penalty_count: int = 5,
    # B1 / B2
    b1_kick_number: int | None = None,
    pen_score_home: int = 0,
    pen_score_away: int = 0,
    is_decisive: bool = False,
    # C1 (C2 was dropped in Issue #41)
    position: str = "striker",
    # Phase 3 (Issue #51)
    tournament_kind: TournamentKind = "international",
) -> TrainingRow:
    """Build a `TrainingRow` with sensible defaults.

    A1 defaults to a one-hot distribution on `label` (so the toy
    dataset is a balanced L/C/R signal). `b1_kick_number` mirrors
    `kick_number` unless overridden. v3 (Issue #36) dropped the
    B3 (`b3_round`) column; the round is now identifier-only.
    v3 (Issue #41) dropped the C2 (`age`) column; the model's
    ablation in `docs/model-review.md` Topic 2.3 showed age
    actively hurt the save rate on the 28-row 2026 holdout.
    Phase 3 (Issue #51) added the `tournament_kind` metadata
    attribute; default is `"international"` (the existing 6
    national-team cup competitions).
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
        p_L_5=1.0 if label == "L" else 0.0 if p_L_5 is None else p_L_5,
        p_C_5=1.0 if label == "C" else 0.0 if p_C_5 is None else p_C_5,
        p_R_5=1.0 if label == "R" else 0.0 if p_R_5 is None else p_R_5,
        p_L_10=1.0 if label == "L" else 0.0 if p_L_10 is None else p_L_10,
        p_C_10=1.0 if label == "C" else 0.0 if p_C_10 is None else p_C_10,
        p_R_10=1.0 if label == "R" else 0.0 if p_R_10 is None else p_R_10,
        p_L_20=1.0 if label == "L" else 0.0 if p_L_20 is None else p_L_20,
        p_C_20=1.0 if label == "C" else 0.0 if p_C_20 is None else p_C_20,
        p_R_20=1.0 if label == "R" else 0.0 if p_R_20 is None else p_R_20,
        last_side=last_side,
        preferred_foot=preferred_foot,
        career_penalty_count=career_penalty_count,
        b1_kick_number=kick_number if b1_kick_number is None else b1_kick_number,
        pen_score_home=pen_score_home,
        pen_score_away=pen_score_away,
        is_decisive=is_decisive,
        position=position,
        tournament_kind=tournament_kind,
    )


# ---------------------------------------------------------------------------
# PlayerPenalty (one row of player_history.jsonl)
# ---------------------------------------------------------------------------


def make_history_row(
    match_id: int = 1,
    match_date: str = "2024-06-01T00:00:00+00:00",
    *,
    side: str = "L",
    shot_type: str = "RightFoot",
    kicker_id: int = 1,
    league_id: int = 77,
    league_name: str = "World Cup",
    team_id: int = 100,
    is_home: bool = True,
    x: float = 0.5,
    is_on_target: bool = True,
    outcome: str = "Goal",
) -> PlayerPenalty:
    """Build a `PlayerPenalty` (one row of `player_history.jsonl`).

    `match_id` and `match_date` are the common positional args
    (most tests are chronology-shaped). The remaining fields are
    keyword-only.
    """
    return PlayerPenalty(
        kicker_id=kicker_id,
        match_id=match_id,
        match_date=match_date,
        league_id=league_id,
        league_name=league_name,
        team_id=team_id,
        is_home=is_home,
        x=x,
        side=side,
        is_on_target=is_on_target,
        outcome=outcome,
        shot_type=shot_type,
    )


# ---------------------------------------------------------------------------
# PlayerMetadata (the per-player C-group data — C1 position, C2 age)
# ---------------------------------------------------------------------------


def make_metadata(
    player_id: int = 1,
    *,
    player_name: str = "Alpha",
    position_key: str = "striker",
    birth_date: str = "1995-01-01",
) -> PlayerMetadata:
    """Build a `PlayerMetadata` (the C1/C2 source for the feature builder)."""
    return PlayerMetadata(
        player_id=player_id,
        player_name=player_name,
        position_key=position_key,
        birth_date=birth_date,
    )


# ---------------------------------------------------------------------------
# ShootoutKick (one row of shootout_kicks.jsonl)
# ---------------------------------------------------------------------------


def make_shootout_kick(
    match_id: int = 1,
    kick_number: int = 1,
    *,
    match_date: str = "2022-12-18T15:00:00+00:00",
    tournament_id: int = 77,
    tournament_name: str = "World Cup",
    round: str = "Final",
    kicker_id: int = 42,
    kicker_name: str = "Stub",
    team_id: int = 100,
    is_home: bool = True,
    x: float = 0.5,
    side: str = "L",
    is_on_target: bool = True,
    outcome: str = "Goal",
    pen_score_before: tuple[int, int] = (0, 0),
    pen_score_after: tuple[int, int] = (1, 0),
    match_score_home: int = 3,
    match_score_away: int = 3,
) -> ShootoutKick:
    """Build a `ShootoutKick` with sensible defaults.

    `match_id` and `kick_number` are the common positional args
    (every test sets both). The remaining fields are keyword-only.

    `pen_score_before` and `pen_score_after` are tuples (immutable)
    in the factory but `ShootoutKick` stores them as `list[int]`
    (the data layer's reader expects a JSON array). The conversion
    is one-line; the test side prefers tuples to keep the call site
    terse.
    """
    return ShootoutKick(
        match_id=match_id,
        match_date=match_date,
        tournament_id=tournament_id,
        tournament_name=tournament_name,
        round=round,
        kick_number=kick_number,
        kicker_id=kicker_id,
        kicker_name=kicker_name,
        team_id=team_id,
        is_home=is_home,
        x=x,
        side=side,
        is_on_target=is_on_target,
        outcome=outcome,
        pen_score_before=list(pen_score_before),
        pen_score_after=list(pen_score_after),
        match_score_home=match_score_home,
        match_score_away=match_score_away,
    )


# ---------------------------------------------------------------------------
# RosterPlayer (one row of wc2026_roster.jsonl)
# ---------------------------------------------------------------------------


def make_roster_player(
    player_id: int = 1,
    *,
    player_name: str = "Alpha",
    team_id: int = 100,
    team_name: str = "Argentina",
    country_code: str = "ARG",
) -> RosterPlayer:
    """Build a `RosterPlayer` (one row of `wc2026_roster.jsonl`)."""
    return RosterPlayer(
        player_id=player_id,
        player_name=player_name,
        team_id=team_id,
        team_name=team_name,
        country_code=country_code,
    )


# ---------------------------------------------------------------------------
# InitialSetKicker / MissingKicker (the deduped Initial Set + the
# no-history list)
# ---------------------------------------------------------------------------


def make_initial_set_kicker(
    player_id: int = 1,
    *,
    player_name: str = "Stub",
    team_id: int = 1,
    team_name: str = "",
) -> InitialSetKicker:
    """Build an `InitialSetKicker` (the deduped Initial Set entry)."""
    return InitialSetKicker(
        player_id=player_id,
        player_name=player_name,
        team_id=team_id,
        team_name=team_name,
    )


def make_missing_kicker(
    player_id: int = 1,
    *,
    player_name: str = "No History",
    team_id: int = 1,
    team_name: str = "Argentina",
) -> MissingKicker:
    """Build a `MissingKicker` (a no-history Initial Set entry)."""
    return MissingKicker(
        player_id=player_id,
        player_name=player_name,
        team_id=team_id,
        team_name=team_name,
    )


# ---------------------------------------------------------------------------
# PredictionRow (one row of predictions.jsonl)
# ---------------------------------------------------------------------------


def make_prediction_row(
    player_id: int = 1,
    *,
    player_name: str = "Alpha",
    team_id: int = 100,
    team_name: str = "Argentina",
    country_code: str = "ARG",
    kicking_foot: str = "RightFoot",
    p_L: float = 0.5,
    p_C: float = 0.25,
    p_R: float = 0.25,
    tournament_kind: TournamentKind = "international",
) -> PredictionRow:
    """Build a `PredictionRow` (one row of `predictions.jsonl`).

    Default probabilities are roughly balanced so a unit test that
    sums them sees 1.0. Phase 3 (Issue #51) added the
    `tournament_kind` metadata attribute; default is
    `"international"` (the WC 2026 roster always predicts
    international).
    """
    return PredictionRow(
        player_id=player_id,
        player_name=player_name,
        team_id=team_id,
        team_name=team_name,
        country_code=country_code,
        kicking_foot=kicking_foot,
        p_L=p_L,
        p_C=p_C,
        p_R=p_R,
        tournament_kind=tournament_kind,
    )
