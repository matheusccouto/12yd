"""Live predictions for the 2026 World Cup roster (slice #9, Issue #25).

PRD: For each player in the 2026 World Cup squads, compute the 18 features
(using the player's penalty history filtered to before the target date) and
run the frozen LightGBM model. Write one row per WC player with the
predicted probabilities P(L), P(C), P(R), the kicking foot, and the
team/country metadata.

The slice is the deliverable artifact the dashboard consumes. The
model's input is a `TrainingRow` (the same shape the training slice
used); the model's output is a 3-vector of probabilities in `CLASSES`
order (L=0, C=1, R=2).

The feature row for a prediction target uses neutral B-group values:
`kick_number=1`, `pen_score_before=(0, 0)`, `is_decisive=False`. These
are not the values of any real shootout kick (we don't know which
side of the bracket the team will be on), they're the "nothing-yet"
defaults. The numeric B fields are 0/False which is the well-defined
neutral state. The A-group (history) and C-group (metadata) features
are the kicker-specific signal the model uses.

v3 (Issue #36) removed the B3 (`b3_round`) feature and the dashboard
re-score path (`PredictContext` / `predict_roster_with_context`).
The model is round-agnostic; the dashboard reads `predictions.jsonl`
directly.

Re-runs are idempotent: same roster + same history + same model +
same `target_date` → same predictions. The `target_date` is a CLI flag
with a deterministic default (`today_utc() + 1 day`, so the prediction
window always includes all of today's penalties).

Phase 0 (Issue #30): the predict slice no longer constructs a synthetic
`ShootoutKick`. The feature builder's prediction entry point takes a
`RosterPlayer` + history + metadata + target_date and returns a
`TrainingRow` with neutral B-group values via the shared
`compute_features` / `build_features` path.
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
    - `kicking_foot`: the declared preferred foot (v3: pass-through
      from `PlayerMetadata.preferred_foot`, lowercase
      "left" / "right" / "both"). The JSONL column keeps the
      `kicking_foot` name for consumer continuity (the dashboard's
      per-kicker table reads the same column it always did); the
      underlying semantic is now the declared foot, not the
      mode-of-history inference.
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
) -> TrainingRow:
    """Build the 18-feature row for one roster player at `target_date`.

    The B-group is the neutral context (kick_number=1, pen_score=0-0,
    is_home=True) so the model's B-group features don't leak
    shootout-state information that doesn't exist for a prediction
    target. The A1 features (side distribution over the last 5/10/20
    kicks) and A2/A3/A4 features come from the player's history and
    metadata, filtered to before `target_date`. C1 (position) and
    C2 (age) come from the metadata. For kickers with no history, A1
    falls back to the prior `(1/3, 1/3, 1/3)`, A2 is "", and A4 is 0.
    A3 falls through from `metadata.preferred_foot` ("" when metadata
    is missing or the field is absent).
    """
    features = compute_features(
        history=history,
        metadata=metadata,
        target_date=target_date,
        b_group=BGroupContext.neutral(),
        kicks_done=KickIndex(home_kicks_done=0, away_kicks_done=0),
    )
    return build_features(
        features,
        # Synthetic identifiers — the model doesn't use them at
        # predict time, but the unified row type carries them so the
        # data layer's directory layout is consistent.
        match_id=0,
        kick_number=1,
        kicker_id=kicker.player_id,
        kicker_name=kicker.player_name,
        match_date=target_date,
        tournament_id=77,  # FotMob WC leagueId
        tournament_name="World Cup",
        round="",
        team_id=kicker.team_id,
        is_home=True,
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
) -> PredictionRow:
    """Run the model for one kicker and return the `PredictionRow`.

    Builds the 18-feature row, runs the model, and returns the
    predicted probabilities. The `kicking_foot` is the declared
    preferred foot from `PlayerMetadata.preferred_foot` (v3: the same
    value the feature row carries, so the slice doesn't re-read
    metadata).
    """
    row = build_prediction_features(kicker, history, metadata, target_date)
    matrix = build_feature_matrix([row])
    probs = np.asarray(model.predict_proba(matrix.X))
    p_L, p_C, p_R = (float(p) for p in probs[0])
    return PredictionRow(
        player_id=kicker.player_id,
        player_name=kicker.player_name,
        team_id=kicker.team_id,
        team_name=kicker.team_name,
        country_code=kicker.country_code,
        kicking_foot=row.preferred_foot,
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
# Per-roster predict
# ---------------------------------------------------------------------------


def predict_roster(
    model: Any,
    roster: Sequence[RosterPlayer],
    player_history: Mapping[int, Sequence[PlayerPenalty]],
    metadata_fetcher: Any,
    target_date: str,
) -> list[PredictionRow]:
    """Run the model for every roster player with the neutral B-group.

    Returns a list of `PredictionRow` in the same order as the input
    `roster`. The function is pure: same inputs → same outputs. Each
    kicker's prediction is independent — a single bad metadata fetch
    (returns None) does not abort the run; the kicker just gets
    `position=""` and `age=NaN`, and the model's categorical
    `position` becomes NaN (LightGBM treats it as missing).

    `metadata_fetcher` is the `MetadataFetcher` callable from
    `features.py` — typically `fetcher_from_client(client)` with the
    on-disk cache populated by the player-history slice. The fetcher
    is process-global; no per-process caching is done here.

    v3 (Issue #36): the previous `predict_roster_with_context` /
    `PredictContext(round=...)` round-override entry point is gone.
    The model is round-agnostic; the dashboard reads
    `predictions.jsonl` directly.
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
