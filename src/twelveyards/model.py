"""Feature engineering and prediction pipeline (TabPFN model fitting & prediction)."""

from __future__ import annotations

from datetime import date, timedelta
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel
from tabpfn_client import TabPFNClassifier
from tabpfn_client import init as _tabpfn_init

from twelveyards.config import LOOKBACK_WINDOW_YEARS, TRAIN_FLOOR

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class PlayerPenalty(BaseModel):
    """One penalty kick row (player_history.jsonl)."""

    kicker_id: int
    match_id: int
    match_date: str
    league_id: int
    league_name: str
    team_id: int
    is_home: bool
    x: float
    side: str
    is_on_target: bool
    outcome: str
    shot_type: str


class PlayerMetadata(BaseModel):
    """Per-player metadata."""

    player_id: int
    player_name: str
    position_key: str
    birth_date: str
    preferred_foot: str


class RosterPlayer(BaseModel):
    """One player on the roster."""

    player_id: int
    player_name: str
    team_id: int
    team_name: str
    country_code: str


class PredictionRow(BaseModel):
    """One row of predictions.jsonl."""

    player_id: int
    player_name: str
    short_name: str
    team_id: int
    team_name: str
    country_code: str
    kicking_foot: str
    photo_url: str
    p_L: float  # noqa: N815
    p_C: float  # noqa: N815
    p_R: float  # noqa: N815
    total_penalties: int

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

            features = compute_features(
                sorted_kicks, metadata, kick_date, lookback_years,
            )
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


def _init_tabpfn() -> None:
    """Initialise the TabPFN client (reads TABPFN_TOKEN env var)."""
    _tabpfn_init()


class _TabPFN:
    """Wrapper over TabPFNClassifier enforcing 'fit before predict'."""

    def __init__(
        self,
        *,
        n_estimators: int = 8,
        thinking_mode: bool = False,
        random_state: int = 0,
        categorical_features_indices: list[int] | None = None,
    ) -> None:
        self._n_estimators = n_estimators
        self._thinking_mode = thinking_mode
        self._random_state = random_state
        self._categorical_features_indices = categorical_features_indices
        self._classifier: TabPFNClassifier | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:  # noqa: N803
        self._classifier = TabPFNClassifier(
            n_estimators=self._n_estimators,
            thinking_mode=self._thinking_mode,
            random_state=self._random_state,
            categorical_features_indices=self._categorical_features_indices,
            ignore_pretraining_limits=True,
        )
        self._classifier.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        if self._classifier is None:
            msg = "TabPFN must be fit before predict_proba"
            raise RuntimeError(msg)
        return np.asarray(self._classifier.predict_proba(X))


def load_player_history(path: Path) -> dict[int, list[PlayerPenalty]]:
    """Load player_history.jsonl into a dict keyed by kicker_id."""
    import json  # noqa: PLC0415

    out: dict[int, list[PlayerPenalty]] = {}
    with path.open(encoding="utf-8") as f:
        for raw in f:
            stripped = raw.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            out.setdefault(int(row["kicker_id"]), []).append(PlayerPenalty(**row))
    return out


def predict_and_write(
    roster: Sequence[RosterPlayer],
    player_history: dict[int, list[PlayerPenalty]],
    metadata_by_id: dict[int, PlayerMetadata],
    output_path: Path,
    *,
    target_date: date | None = None,
) -> list[PredictionRow]:
    """Fit TabPFN on training data, predict all roster rows, write predictions.jsonl."""
    _init_tabpfn()
    model = _TabPFN(categorical_features_indices=CATEGORICAL_INDICES)

    X_train, y_train = build_training_matrix(player_history, metadata_by_id)  # noqa: N806
    if len(X_train) > 0:
        model.fit(X_train, y_train)

    roster_ids = [p.player_id for p in roster]
    X_test = build_prediction_matrix(  # noqa: N806
        roster_ids, player_history, metadata_by_id, target_date,
    )

    if len(X_train) > 0 and len(X_test) > 0:
        probs = model.predict_proba(X_test)
    else:
        probs = np.full((len(X_test), 3), 1.0 / 3.0)

    rows: list[PredictionRow] = []
    for i, kicker in enumerate(roster):
        metadata = metadata_by_id.get(kicker.player_id)
        preferred_foot = metadata.preferred_foot if metadata is not None else ""
        total = len(player_history.get(kicker.player_id, []))

        rows.append(
            PredictionRow(
                player_id=kicker.player_id,
                player_name=kicker.player_name,
                short_name=_derive_short_name(kicker.player_name),
                team_id=kicker.team_id,
                team_name=kicker.team_name,
                country_code=kicker.country_code,
                kicking_foot=preferred_foot,
                photo_url=(
                    "https://images.fotmob.com/image_resources/"
                    f"playerimages/{kicker.player_id}.png"
                ),
                p_L=(
                    float(probs[i, CLASSES.index("L")])
                    if len(X_test) > 0
                    else 1.0 / 3.0
                ),
                p_C=(
                    float(probs[i, CLASSES.index("C")])
                    if len(X_test) > 0
                    else 1.0 / 3.0
                ),
                p_R=(
                    float(probs[i, CLASSES.index("R")])
                    if len(X_test) > 0
                    else 1.0 / 3.0
                ),
                total_penalties=total,
            ),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(row.model_dump_json() + "\n")
    return rows


def _derive_short_name(full_name: str) -> str:
    """Return the last word of a full name, or the full name if single-word."""
    parts = full_name.strip().split()
    if len(parts) == 1:
        return parts[0]
    return parts[-1]


__all__ = [
    "CATEGORICAL_INDICES",
    "CLASSES",
    "build_prediction_matrix",
    "build_training_matrix",
    "compute_features",
    "load_player_history",
    "predict_and_write",
    "side_distribution",
]
