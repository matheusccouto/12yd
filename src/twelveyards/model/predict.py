"""TabPFN prediction pipeline: fit on player history, score all roster players."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from tabpfn_client import TabPFNClassifier
from tabpfn_client import init as _tabpfn_init

from twelveyards.artifacts import Artifacts, PredictionRow

from .features import (
    CATEGORICAL_INDICES,
    CLASSES,
    build_prediction_matrix,
    build_training_matrix,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import date
    from pathlib import Path

    from twelveyards.scraper.player_history import PlayerMetadata, PlayerPenalty
    from twelveyards.scraper.rosters import RosterPlayer


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

    from twelveyards.scraper.player_history import PlayerPenalty  # noqa: PLC0415

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
    """Fit TabPFN on training data, predict all roster rows, and write predictions.jsonl."""
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
                p_L=float(probs[i, CLASSES.index("L")]) if len(X_test) > 0 else 1.0 / 3.0,
                p_C=float(probs[i, CLASSES.index("C")]) if len(X_test) > 0 else 1.0 / 3.0,
                p_R=float(probs[i, CLASSES.index("R")]) if len(X_test) > 0 else 1.0 / 3.0,
                total_penalties=total,
            ),
        )

    Artifacts().write_predictions(rows, path=output_path)
    return rows


def _derive_short_name(full_name: str) -> str:
    """Return the last word of a full name, or the full name if single-word."""
    parts = full_name.strip().split()
    if len(parts) == 1:
        return parts[0]
    return parts[-1]
