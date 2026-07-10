"""Live predictions for the 2026 World Cup roster.

PRD-v5: TabPFN classifier on player-only features. Each roster player is
scored once; the same prediction row serves any match. Predictions are
match-agnostic — no opponent or match-context features.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from .features import (
    CATEGORICAL_INDICES,
    CLASSES,
    build_prediction_matrix,
    build_training_matrix,
)
from .player_history import PlayerMetadata, PlayerPenalty
from .rosters import RosterPlayer
from .tabpfn import TabPFN
from .tabpfn import init as tabpfn_init


@dataclass(frozen=True)
class PredictionRow:
    player_id: int
    player_name: str
    short_name: str
    team_id: int
    team_name: str
    country_code: str
    kicking_foot: str
    photo_url: str
    p_L: float
    p_C: float
    p_R: float
    total_penalties: int = 0


def load_player_history(path: Path) -> dict[int, list[PlayerPenalty]]:
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


def load_roster(path: Path) -> list[RosterPlayer]:
    from .artifacts import Artifacts

    return Artifacts().read_roster(path)


def predict_and_write(
    roster: Sequence[RosterPlayer],
    player_history: dict[int, list[PlayerPenalty]],
    metadata_by_id: dict[int, PlayerMetadata],
    output_path: Path,
    *,
    target_date: date | None = None,
) -> list[PredictionRow]:
    tabpfn_init()
    model = TabPFN(categorical_features_indices=CATEGORICAL_INDICES)

    X_train, y_train = build_training_matrix(player_history, metadata_by_id)
    if len(X_train) > 0:
        model.fit(X_train, y_train)

    roster_ids = [p.player_id for p in roster]
    X_test = build_prediction_matrix(
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
                photo_url=f"https://images.fotmob.com/image_resources/playerimages/{kicker.player_id}.png",
                p_L=float(probs[i, CLASSES.index("L")]) if len(X_test) > 0 else 1.0 / 3.0,
                p_C=float(probs[i, CLASSES.index("C")]) if len(X_test) > 0 else 1.0 / 3.0,
                p_R=float(probs[i, CLASSES.index("R")]) if len(X_test) > 0 else 1.0 / 3.0,
                total_penalties=total,
            ),
        )

    from .artifacts import Artifacts

    Artifacts().write_predictions(rows, path=output_path)
    return rows


def _derive_short_name(full_name: str) -> str:
    parts = full_name.strip().split()
    if len(parts) == 1:
        return parts[0]
    return parts[-1]
