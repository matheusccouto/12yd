"""Single seam for the on-disk artifact layout.

PRD-v5: Artifacts live under `data/`. Surviving artifacts: wc2026_roster.jsonl,
player_history.jsonl, predictions.jsonl, and missing_history.jsonl. Dropped
artifacts: shootout_kicks, training_table, model pickles, metrics, cv, diagnostics,
discrepancies, tournament_success_rate.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any, TypeVar

from .client import FotMobClient
from .initial_set import MissingKicker
from .player_history import PlayerPenalty
from .predict import PredictionRow
from .rosters import RosterPlayer

_T = TypeVar("_T")


def _write_jsonl(
    path: Path,
    rows: Iterable[Any],
    *,
    nan_to_null: bool = False,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(_serialize_row(row, nan_to_null=nan_to_null))
            f.write("\n")
            count += 1
    return count


def _serialize_row(row: Any, *, nan_to_null: bool = False) -> str:
    payload = asdict(row)
    return json.dumps(payload, ensure_ascii=False, allow_nan=(not nan_to_null))


def _read_jsonl_of_dataclasses[T](path: Path, cls: type[T]) -> list[T]:
    out: list[T] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(cls(**json.loads(line)))
    return out


class Artifacts:
    def __init__(
        self,
        root: Path = Path("data"),
        cache_dir: Path = Path("data/fotmob_cache"),
    ) -> None:
        self.root = Path(root)
        self.cache_dir = Path(cache_dir)

    # ------------------------------------------------------------------ paths

    @property
    def player_history(self) -> Path:
        return self.root / "player_history.jsonl"

    @property
    def missing_history(self) -> Path:
        return self.root / "missing_history.jsonl"

    @property
    def roster(self) -> Path:
        return self.root / "wc2026_roster.jsonl"

    @property
    def predictions(self) -> Path:
        return self.root / "predictions.jsonl"

    # ------------------------------------------------------------ player_h

    def read_player_history(self, path: Path | None = None) -> list[PlayerPenalty]:
        return _read_jsonl_of_dataclasses(path or self.player_history, PlayerPenalty)

    def write_player_history(self, rows: Iterable[PlayerPenalty], path: Path | None = None) -> int:
        return _write_jsonl(path or self.player_history, rows)

    # ----------------------------------------------------------- missing_h

    def read_missing_history(self, path: Path | None = None) -> list[MissingKicker]:
        return _read_jsonl_of_dataclasses(path or self.missing_history, MissingKicker)

    def write_missing_history(self, rows: Iterable[MissingKicker], path: Path | None = None) -> int:
        return _write_jsonl(path or self.missing_history, rows)

    # ----------------------------------------------------------------- roster

    def read_roster(self, path: Path | None = None) -> list[RosterPlayer]:
        return _read_jsonl_of_dataclasses(path or self.roster, RosterPlayer)

    def write_roster(self, rows: Iterable[RosterPlayer], path: Path | None = None) -> int:
        return _write_jsonl(path or self.roster, rows)

    # ----------------------------------------------------------- predictions

    def read_predictions(self, path: Path | None = None) -> list[PredictionRow]:
        return _read_jsonl_of_dataclasses(path or self.predictions, PredictionRow)

    def write_predictions(self, rows: Iterable[PredictionRow], path: Path | None = None) -> int:
        return _write_jsonl(path or self.predictions, rows)

    # ----------------------------------------------------------- cache factory

    def fotmob_client(self) -> FotMobClient:
        return FotMobClient(cache_dir=self.cache_dir)

    # ---------------------------------------------------- streaming serialise

    def serialize_row(self, row: Any, *, nan_to_null: bool = False) -> str:
        return _serialize_row(row, nan_to_null=nan_to_null)
