"""Feature builder: 7-column match-agnostic feature matrix from player penalty history."""

from __future__ import annotations

from datetime import date, timedelta
from typing import TYPE_CHECKING

import numpy as np

from twelveyards.config import LOOKBACK_WINDOW_YEARS, TRAIN_FLOOR

if TYPE_CHECKING:
    from collections.abc import Sequence

    from twelveyards.scraper.player_history import PlayerMetadata, PlayerPenalty

PRIOR_PROB: tuple[float, float, float] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
CLASSES: tuple[str, ...] = ("L", "C", "R")

CATEGORICAL_INDICES: list[int] = [4, 5, 6]
_NUM_FEATURES: int = 7


def side_distribution(sides: Sequence[str], n: int) -> tuple[float, float, float]:
    """Return (p_L, p_C, p_R) for the last `n` sides, or the uniform prior if empty."""
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


def _filter_history_window(
    history: Sequence[PlayerPenalty],
    target_date: date,
    lookback_years: int = LOOKBACK_WINDOW_YEARS,
) -> list[PlayerPenalty]:
    window_start = target_date - timedelta(days=lookback_years * 365)
    out: list[PlayerPenalty] = []
    for row in history:
        row_date = date.fromisoformat(row.match_date[:10])
        if window_start <= row_date < target_date:
            out.append(row)
    out.sort(key=lambda r: r.match_date)
    return out


def _build_categorical_encoder(
    rows: list[dict],
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    side_map: dict[str, int] = {}
    foot_map: dict[str, int] = {}
    pos_map: dict[str, int] = {}
    for r in rows:
        s = str(r.get("last_side", ""))
        f = str(r.get("preferred_foot", ""))
        p = str(r.get("position", ""))
        if s not in side_map:
            side_map[s] = len(side_map)
        if f not in foot_map:
            foot_map[f] = len(foot_map)
        if p not in pos_map:
            pos_map[p] = len(pos_map)
    return side_map, foot_map, pos_map


def compute_features(
    history: Sequence[PlayerPenalty],
    metadata: PlayerMetadata | None,
    target_date: date,
    lookback_years: int = LOOKBACK_WINDOW_YEARS,
) -> dict:
    """
    Compute the 7-element feature dict for a player at a target date.

    Returns p_L, p_C, p_R (A1), last_side (A2), preferred_foot (A3),
    career_penalty_count (A4), and position (C1).
    """
    window_kicks = _filter_history_window(history, target_date, lookback_years)
    sides = [p.side for p in window_kicks]
    total = len(window_kicks)

    p_l, p_c, p_r = side_distribution(sides, total)
    last_side = sides[-1] if sides else ""
    preferred_foot = metadata.preferred_foot if metadata is not None else ""
    position = metadata.position_key if metadata is not None else ""

    return {
        "p_L": p_l,
        "p_C": p_c,
        "p_R": p_r,
        "last_side": last_side,
        "preferred_foot": preferred_foot,
        "career_penalty_count": total,
        "position": position,
    }


def build_training_matrix(
    player_history: dict[int, list[PlayerPenalty]],
    metadata_by_id: dict[int, PlayerMetadata],
    lookback_years: int = LOOKBACK_WINDOW_YEARS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) training matrices from player history.

    Each kick becomes a training row whose features are derived from
    prior kicks within the 5-year window and whose label is the kick's side.
    Only kicks on or after TRAIN_FLOOR are used as training rows.
    """
    rows: list[dict] = []
    labels: list[int] = []

    for player_id, penalties in player_history.items():
        sorted_kicks = sorted(penalties, key=lambda p: p.match_date)
        metadata = metadata_by_id.get(player_id)

        for kick in sorted_kicks:
            kick_date = date.fromisoformat(kick.match_date[:10])
            if kick_date < TRAIN_FLOOR:
                continue

            features = compute_features(sorted_kicks, metadata, kick_date, lookback_years)
            rows.append(features)
            labels.append(CLASSES.index(kick.side))

    X = _features_to_array(rows)  # noqa: N806
    y = np.array(labels, dtype=np.int64)
    return X, y


def build_prediction_matrix(
    roster_player_ids: list[int],
    player_history: dict[int, list[PlayerPenalty]],
    metadata_by_id: dict[int, PlayerMetadata],
    target_date: date | None = None,
    lookback_years: int = LOOKBACK_WINDOW_YEARS,
) -> np.ndarray:
    """
    Build the prediction feature matrix for roster players.

    Each roster player gets one row of features computed from their
    penalty history up to `target_date` (defaults to today UTC).
    """
    if target_date is None:
        target_date = date.today()  # noqa: DTZ011

    rows: list[dict] = []
    for player_id in roster_player_ids:
        history = player_history.get(player_id, [])
        metadata = metadata_by_id.get(player_id)
        features = compute_features(history, metadata, target_date, lookback_years)
        rows.append(features)

    return _features_to_array(rows)


def _features_to_array(rows: list[dict]) -> np.ndarray:
    n = len(rows)
    if n == 0:
        return np.empty((0, _NUM_FEATURES))

    side_map, foot_map, pos_map = _build_categorical_encoder(rows)

    x = np.zeros((n, _NUM_FEATURES), dtype=np.float64)
    for i, r in enumerate(rows):
        x[i, 0] = float(r["p_L"])
        x[i, 1] = float(r["p_C"])
        x[i, 2] = float(r["p_R"])
        x[i, 3] = float(r["career_penalty_count"])
        x[i, 4] = float(side_map.get(str(r["last_side"]), 0))
        x[i, 5] = float(foot_map.get(str(r["preferred_foot"]), 0))
        x[i, 6] = float(pos_map.get(str(r["position"]), 0))

    return x
