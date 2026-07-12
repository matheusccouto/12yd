"""On-disk artifact layout and typed JSONL I/O."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .fotmob.client import FotMobClient
from .scraper.initial_set import MissingKicker
from .scraper.player_history import PlayerPenalty
from .scraper.rosters import RosterPlayer

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass(frozen=True)
class PredictionRow:
    """One row in predictions.jsonl — a roster player's predicted side distribution."""

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
    total_penalties: int = 0


def _write_jsonl(
    path: Path,
    rows: Iterable[object],
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


def _serialize_row(row: object, *, nan_to_null: bool = False) -> str:
    payload = asdict(row)
    return json.dumps(payload, ensure_ascii=False, allow_nan=(not nan_to_null))


def _read_jsonl_of_dataclasses[T](path: Path, cls: type[T]) -> list[T]:
    out: list[T] = []
    with path.open(encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            out.append(cls(**json.loads(line)))
    return out


class Artifacts:
    """Single seam for the on-disk artifact layout under `data/`."""

    def __init__(
        self,
        root: Path = Path("data"),
    ) -> None:
        """Create an Artifacts handle rooted at `root`."""
        self.root = Path(root)

    # ------------------------------------------------------------------ paths

    @property
    def player_history(self) -> Path:
        """Path to player_history.jsonl."""
        return self.root / "player_history.jsonl"

    @property
    def missing_history(self) -> Path:
        """Path to missing_history.jsonl."""
        return self.root / "missing_history.jsonl"

    @property
    def roster(self) -> Path:
        """Path to wc2026_roster.jsonl."""
        return self.root / "wc2026_roster.jsonl"

    @property
    def predictions(self) -> Path:
        """Path to predictions.jsonl."""
        return self.root / "predictions.jsonl"

    # ------------------------------------------------------------ player_h

    def read_player_history(self, path: Path | None = None) -> list[PlayerPenalty]:
        """Read player_history.jsonl into PlayerPenalty rows."""
        return _read_jsonl_of_dataclasses(path or self.player_history, PlayerPenalty)

    def write_player_history(
        self, rows: Iterable[PlayerPenalty], path: Path | None = None,
    ) -> int:
        """Write PlayerPenalty rows to player_history.jsonl."""
        return _write_jsonl(path or self.player_history, rows)

    # ----------------------------------------------------------- missing_h

    def read_missing_history(self, path: Path | None = None) -> list[MissingKicker]:
        """Read missing_history.jsonl into MissingKicker rows."""
        return _read_jsonl_of_dataclasses(path or self.missing_history, MissingKicker)

    def write_missing_history(
        self, rows: Iterable[MissingKicker], path: Path | None = None,
    ) -> int:
        """Write MissingKicker rows to missing_history.jsonl."""
        return _write_jsonl(path or self.missing_history, rows)

    # ----------------------------------------------------------------- roster

    def read_roster(self, path: Path | None = None) -> list[RosterPlayer]:
        """Read wc2026_roster.jsonl into RosterPlayer rows."""
        return _read_jsonl_of_dataclasses(path or self.roster, RosterPlayer)

    def write_roster(
        self, rows: Iterable[RosterPlayer], path: Path | None = None,
    ) -> int:
        """Write RosterPlayer rows to wc2026_roster.jsonl."""
        return _write_jsonl(path or self.roster, rows)

    # ----------------------------------------------------------- predictions

    def read_predictions(self, path: Path | None = None) -> list[PredictionRow]:
        """Read predictions.jsonl into PredictionRow rows."""
        return _read_jsonl_of_dataclasses(path or self.predictions, PredictionRow)

    def write_predictions(
        self, rows: Iterable[PredictionRow], path: Path | None = None,
    ) -> int:
        """Write PredictionRow rows to predictions.jsonl."""
        return _write_jsonl(path or self.predictions, rows)

    # ----------------------------------------------------------- cache factory

    def fotmob_client(self) -> FotMobClient:
        """Return a FotMobClient instance."""
        return FotMobClient()

    def serialize_row(self, row: object, *, nan_to_null: bool = False) -> str:
        """Serialize a dataclass row as a JSON string for streaming writes."""
        return _serialize_row(row, nan_to_null=nan_to_null)
