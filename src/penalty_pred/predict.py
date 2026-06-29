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
`kick_number=1`, `pen_score_before=(0, 0)`, `is_decisive=False`,
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

Phase 0 (Issue #30): the predict slice no longer constructs a synthetic
`ShootoutKick`. The feature builder's prediction entry point takes a
`RosterPlayer` + history + metadata + target_date and returns a
`TrainingRow` with neutral B-group values via the shared
`compute_features` / `build_features` path.

Phase 2 (Issue #34): adds `PredictContext` (the B-group override) and
`predict_roster_with_context`. The dashboard's re-score path uses a
context whose `round` is the match's actual round (e.g. "Quarter-finals"
for an R16 match) so the model sees the B3 categorical it was trained
on. The v1 slice's `predict_roster(roster, history, model, target_date)`
is now a thin wrapper that calls `predict_roster_with_context` with the
default (neutral) context — the round-agnostic predictions on disk
(`predictions.jsonl`) are byte-for-byte the v1 result.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .features import (
    CLASSES,
    PRIOR_PROB,
    BGroupContext,
    KickIndex,
    TrainingRow,
    build_features,
    compute_features,
)
from .model import (
    build_feature_matrix,
)
from .player_history import PlayerMetadata, PlayerPenalty
from .rosters import RosterPlayer


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
# B-group context (Phase 2: dashboard re-score)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PredictContext:
    """The B-group override for a single prediction target.

    The default context (`PredictContext()`) is the v1 neutral context:
    `round=""` (LightGBM treats as missing), `kick_number=1`, the
    shootout score 0-0. The dashboard's re-score path uses a non-neutral
    `round` so the model sees the B3 categorical it was trained on
    ("Round of 16", "Quarter-finals", "Semi-finals", "Final").

    The numeric B-group fields stay at the v1 neutral defaults for both
    the v1 slice and the dashboard re-score — a real shootout kick has
    non-zero `kick_number` / `pen_score`, but the model is asked to
    predict the *first* kick of a hypothetical match, and the per-kick
    feature row at the model's input carries the B-group as "kick 1 of
    a 0-0 shootout" for every match. The B3 round is the only field the
    dashboard overrides because it's the only one that varies across
    matches in a way the model was trained on.
    """

    round: str = ""
    kick_number: int = 1
    pen_score_home: int = 0
    pen_score_away: int = 0
    is_home: bool = True

    def to_b_group(self) -> BGroupContext:
        """Materialise the value object as the feature builder's B-group."""
        return BGroupContext(
            kick_number=self.kick_number,
            pen_score_home=self.pen_score_home,
            pen_score_away=self.pen_score_away,
            is_home=self.is_home,
            round=self.round,
        )


# Module-level singleton: the neutral context reused as the default for
# the `context` argument. Using a module-level constant (instead of
# `PredictContext()` inline) avoids the B008 ruff rule that flags
# function calls in argument defaults.
_NEUTRAL_CONTEXT: PredictContext = PredictContext()


# ---------------------------------------------------------------------------
# Feature builder for a single prediction target
# ---------------------------------------------------------------------------


def build_prediction_features(
    kicker: RosterPlayer,
    history: Sequence[PlayerPenalty],
    metadata: PlayerMetadata | None,
    target_date: str,
    context: PredictContext = _NEUTRAL_CONTEXT,
) -> TrainingRow:
    """Build the 9-feature row for one roster player at `target_date`.

    The B-group is taken from `context`. The default context is neutral
    (round="", kick_number=1, pen_score=0-0) so the model's B-group
    features don't leak shootout-state information that doesn't exist
    for a prediction target. The dashboard re-scores with a non-neutral
    `round` (e.g. "Quarter-finals") to get round-specific predictions.

    The A1 features (side distribution over the last 5/10/20 kicks) and
    A2/A3/A4 features come from the player's history, filtered to
    before `target_date`. C1 (position) and C2 (age) come from the
    metadata. For kickers with no history, A1 falls back to the prior
    `(1/3, 1/3, 1/3)`, A2 is "", A3 is "Unknown", A4 is 0.
    """
    features = compute_features(
        history=history,
        metadata=metadata,
        target_date=target_date,
        b_group=context.to_b_group(),
        kicks_done=KickIndex(home_kicks_done=0, away_kicks_done=0),
    )
    return build_features(
        features,
        # Synthetic identifiers — the model doesn't use them at
        # predict time, but the unified row type carries them so the
        # data layer's directory layout is consistent.
        match_id=0,
        kick_number=context.kick_number,
        kicker_id=kicker.player_id,
        kicker_name=kicker.player_name,
        match_date=target_date,
        tournament_id=77,  # FotMob WC leagueId
        tournament_name="World Cup",
        round=context.round,
        team_id=kicker.team_id,
        is_home=context.is_home,
        label="L",  # dummy — unused at predict time
        is_on_target=True,  # dummy — unused at predict time
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
    context: PredictContext = _NEUTRAL_CONTEXT,
) -> PredictionRow:
    """Run the model for one kicker and return the `PredictionRow`.

    Builds the 9-feature row, runs the model, and returns the
    predicted probabilities. The `kicking_foot` is the mode of the
    player's history `shot_type` — the same value the feature row
    carries (so the slice doesn't re-compute it).
    """
    row = build_prediction_features(kicker, history, metadata, target_date, context)
    matrix = build_feature_matrix([row])
    probs = np.asarray(model.predict_proba(matrix.X))
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


# ---------------------------------------------------------------------------
# Per-roster predict (with optional context)
# ---------------------------------------------------------------------------


def predict_roster_with_context(
    model: Any,
    roster: Sequence[RosterPlayer],
    player_history: Mapping[int, Sequence[PlayerPenalty]],
    metadata_fetcher: Any,
    target_date: str,
    context: PredictContext,
) -> list[PredictionRow]:
    """Run the model for every roster player with the given context.

    The full entry point: takes an explicit `context` so the dashboard
    can re-score with the match's actual round. The v1 round-agnostic
    `predictions.jsonl` uses `predict_roster(...)`, which is a thin
    wrapper that calls this with the default (neutral) context.

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
        out.append(predict_kicker(model, kicker, history, metadata, target_date, context))
    return out


def predict_roster(
    model: Any,
    roster: Sequence[RosterPlayer],
    player_history: Mapping[int, Sequence[PlayerPenalty]],
    metadata_fetcher: Any,
    target_date: str,
) -> list[PredictionRow]:
    """Run the model for every roster player with the neutral context.

    Thin wrapper around `predict_roster_with_context` that uses the
    default (neutral) `PredictContext`. The v1 slice writes
    `predictions.jsonl` from this function — the round-agnostic
    predictions on disk are byte-for-byte the v1 result.

    Kept as a separate function (not inlined into the v1 script) so
    the existing v1 callers don't need to change. The dashboard
    re-score path calls `predict_roster_with_context` directly.
    """
    return predict_roster_with_context(
        model,
        roster,
        player_history,
        metadata_fetcher,
        target_date,
        PredictContext(),
    )


__all__ = [
    "CLASSES",
    "PRIOR_PROB",
    "PredictContext",
    "PredictionRow",
    "build_prediction_features",
    "load_player_history",
    "load_roster",
    "predict_kicker",
    "predict_roster",
    "predict_roster_with_context",
]
