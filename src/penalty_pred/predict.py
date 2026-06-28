"""Live predictions for the 2026 World Cup roster (slice #9, Issue #25).

PRD: For each player in the 2026 World Cup squads, compute the 9 features
(using the player's penalty history filtered to before the target date) and
run the frozen LightGBM model. Write one row per WC player with the
predicted probabilities P(L), P(C), P(R), the kicking foot, and the
team/country metadata.

The slice is the deliverable artifact the dashboard (separate PRD) will
consume. The model's input is a `TrainingRow` (the same shape the
training slice used); the model's output is a 3-vector of probabilities
in `CLASSES` order (L=0, C=1, R=2).

The feature row for a prediction target uses neutral B-group values:
`kick_number=1`, `pen_score_before=[0, 0]`, `is_decisive=False`,
`round=""`. These are not the values of any real shootout kick (we don't
know which side of the bracket the team will be on), they're the
"nothing-yet" defaults. The B-group values become LightGBM "missing"
markers for the categorical `b3_round=""` (not in the training
categories), and the numeric B fields are 0/False which is the
well-defined neutral state. The A-group (history) and C-group (metadata)
features are the kicker-specific signal the model uses.

Re-runs are idempotent: same roster + same history + same model +
same `target_date` → same predictions. The `target_date` is a CLI flag
with a deterministic default (`today_utc() + 1 day`, so the prediction
window always includes all of today's penalties).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .features import (
    PRIOR_PROB,
    KickIndex,
    TrainingTableRow,
    build_features,
)
from .model import (
    CLASSES,
    FEATURE_COLUMNS,
    TrainingRow,
    predict_proba,
    rows_to_predict_matrix,
)
from .player_history import PlayerMetadata, PlayerPenalty
from .rosters import RosterPlayer
from .shootouts import ShootoutKick


@dataclass(frozen=True)
class PredictionRow:
    """One row in `predictions.jsonl`: a WC player's predicted P(L/C/R).

    Fields:
    - `player_id`, `player_name`, `team_id`, `team_name`: pass-through
      from the roster. `team_id`/`team_name` are the national team the
      player is representing at the WC, NOT the club side.
    - `country_code`: ISO 3166-1 alpha-3 from the roster ("" if FotMob
      did not return one for the player).
    - `kicking_foot`: mode of the player's history `shot_type`
      ("LeftFoot" / "RightFoot" / "Unknown" for no-history kickers).
    - `p_L`, `p_C`, `p_R`: predicted probabilities from the frozen
      LightGBM, in `CLASSES` order. Sum to 1.0 within 1e-6.
    """

    player_id: int
    player_name: str
    team_id: int
    team_name: str
    country_code: str
    kicking_foot: str
    p_L: float
    p_C: float
    p_R: float


# ---------------------------------------------------------------------------
# Feature builder for a single prediction target
# ---------------------------------------------------------------------------


def build_prediction_features(
    kicker: RosterPlayer,
    history: Sequence[PlayerPenalty],
    metadata: PlayerMetadata | None,
    target_date: str,
) -> TrainingTableRow:
    """Build the 9-feature row for one roster player at `target_date`.

    Constructs a synthetic `ShootoutKick` with neutral B1/B2/B3 values
    (kick_number=1, pen_score=0-0, is_decisive=False, round="") so the
    model's B-group features don't leak shootout-state information that
    doesn't exist for a prediction target. `round=""` is not in the
    training categories, so the LightGBM wrapper treats it as missing
    via the categorical coercion (same behaviour as the training slice's
    unseen-categorical handling).

    The A1 features (side distribution over the last 5/10/20 kicks) and
    A2/A3/A4 features come from the player's history, filtered to
    before `target_date`. C1 (position) and C2 (age) come from the
    metadata. For kickers with no history, A1 falls back to the prior
    `(1/3, 1/3, 1/3)`, A2 is "", A3 is "Unknown", A4 is 0.
    """
    target = ShootoutKick(
        match_id=0,  # synthetic
        match_date=target_date,
        tournament_id=77,  # FotMob WC leagueId
        tournament_name="World Cup",
        round="",  # neutral; LightGBM treats as missing
        kick_number=1,  # neutral
        kicker_id=kicker.player_id,
        kicker_name=kicker.player_name,
        team_id=kicker.team_id,
        is_home=True,  # doesn't matter — pen_score=0-0, both 0 kicks done → is_decisive=False
        x=0.0,  # unused (label side, not used at predict time)
        side="L",  # unused
        is_on_target=True,  # unused
        outcome="Goal",  # unused
        pen_score_before=[0, 0],  # neutral
        pen_score_after=[0, 0],  # unused
        match_score_home=0,  # unused
        match_score_away=0,  # unused
    )
    # Both teams have taken 0 kicks; the is_decisive computation in
    # `is_decisive_kick` short-circuits to False for this case.
    kicks_done = KickIndex(home_kicks_done=0, away_kicks_done=0)
    return build_features(target, history, metadata, kicks_done)


def _training_row_from_table_row(row: TrainingTableRow) -> TrainingRow:
    """Convert a `TrainingTableRow` (feature builder output) to a
    `TrainingRow` (model input).

    The model module's `rows_to_predict_matrix` reads from
    `row.features` (a dict), not from the `TrainingTableRow`'s individual
    fields. This helper bridges the two: it extracts the 19 PRD
    features from the `TrainingTableRow` into a dict and wraps it in a
    `TrainingRow` with dummy `label` / `is_on_target` (the model doesn't
    use them at predict time).
    """
    features: dict[str, Any] = {col: getattr(row, col) for col in FEATURE_COLUMNS}
    return TrainingRow(
        match_id=row.match_id,
        kick_number=row.kick_number,
        kicker_id=row.kicker_id,
        kicker_name=row.kicker_name,
        match_date=row.match_date,
        tournament_id=row.tournament_id,
        tournament_name=row.tournament_name,
        round=row.round,
        team_id=row.team_id,
        is_home=row.is_home,
        label="L",  # dummy — unused at predict time
        is_on_target=True,  # dummy — unused at predict time
        features=features,
    )


# ---------------------------------------------------------------------------
# Per-kicker predict
# ---------------------------------------------------------------------------


def predict_kicker(
    model: Any,
    kicker: RosterPlayer,
    history: Sequence[PlayerPenalty],
    metadata: PlayerMetadata | None,
    target_date: str,
) -> PredictionRow:
    """Run the model for one kicker and return the `PredictionRow`.

    Builds the 9-feature row, converts to a `TrainingRow`, runs the
    model, and returns the predicted probabilities. The `kicking_foot`
    is the mode of the player's history `shot_type` — the same value
    the feature row carries (so the slice doesn't re-compute it).
    """
    row = build_prediction_features(kicker, history, metadata, target_date)
    training_row = _training_row_from_table_row(row)
    matrix = rows_to_predict_matrix([training_row])
    probs = predict_proba(model, matrix)
    p_L, p_C, p_R = (float(p) for p in probs[0])
    return PredictionRow(
        player_id=kicker.player_id,
        player_name=kicker.player_name,
        team_id=kicker.team_id,
        team_name=kicker.team_name,
        country_code=kicker.country_code,
        kicking_foot=row.kicking_foot,
        p_L=p_L,
        p_C=p_C,
        p_R=p_R,
    )


# ---------------------------------------------------------------------------
# I/O — the predictions JSONL is the read/write seam for the dashboard.
# `load_player_history` re-exports the features-module reader; `load_roster`
# re-exports the rosters JSONL reader. Both thin wrappers are kept here so
# callers don't have to import four modules to wire the slice.
# ---------------------------------------------------------------------------


def load_player_history(path: Path) -> dict[int, list[PlayerPenalty]]:
    """Load `player_history.jsonl` into a dict keyed by `kicker_id`.

    Re-exported from `features.load_player_history` for the predict
    module's callers. Each value is the unsorted list of
    `PlayerPenalty` rows for that kicker; the caller is expected to
    sort by `match_date` after filtering to the target date.
    """
    from .features import load_player_history as _load_player_history

    return _load_player_history(path)


def load_roster(path: Path) -> list[RosterPlayer]:
    """Load `wc2026_roster.jsonl` into a list of `RosterPlayer`."""
    from .artifacts import Artifacts

    return Artifacts().read_roster(path)


def predict_roster(
    model: Any,
    roster: Sequence[RosterPlayer],
    player_history: Mapping[int, Sequence[PlayerPenalty]],
    metadata_fetcher: Any,
    target_date: str,
) -> list[PredictionRow]:
    """Run the model for every roster player.

    Returns a list of `PredictionRow` in the same order as the input
    `roster`. The function is pure: same inputs → same outputs. Each
    kicker's prediction is independent — a single bad metadata fetch
    (returns None) does not abort the run; the kicker just gets
    `position=""` and `age=NaN`, and the model's categorical `position`
    becomes NaN (LightGBM treats it as missing).

    `metadata_fetcher` is the `MetadataFetcher` callable from
    `features.py` — typically `fetcher_from_client(client)` with the
    on-disk cache populated by the player-history slice. The fetcher
    is process-global; no per-process caching is done here.
    """
    out: list[PredictionRow] = []
    for kicker in roster:
        history = player_history.get(kicker.player_id, [])
        metadata = metadata_fetcher(kicker.player_id)
        out.append(predict_kicker(model, kicker, history, metadata, target_date))
    return out


__all__ = [
    "CLASSES",
    "PRIOR_PROB",
    "PredictionRow",
    "build_prediction_features",
    "load_player_history",
    "load_roster",
    "predict_kicker",
    "predict_roster",
]
