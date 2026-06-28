"""Single seam for the on-disk artifact layout.

PRD: One adapter that owns the on-disk layout. Replaces the nine
hardcoded `Path("output/...")` defaults in the slice scripts, the five
`write_jsonl` and three `read_jsonl` functions in five modules, the
`validate.py` ad-hoc re-parse, the `load_training_table` sibling-JSONL
reach, and the nine live smoke tests that open `Path("output/...")`
directly. Migration notes:

- The shape of every JSONL is unchanged — the adapter is a re-statement
  of what already exists, not a re-design.
- The added value is that no caller has to know the artifact filenames,
  the artifact directory, the JSONL serialization (NaN → null), the
  pickle format for the model, or the JSON format for the metrics
  report.
- Adding a new artifact is one new path accessor + one read/write pair
  in this module. Renaming an artifact is one rename in this module.

The on-disk layout (relative to `root`):

- `shootout_kicks.jsonl` — 179 target kicks
- `player_history.jsonl` — 745 rows of per-kicker penalty history
- `missing_history.jsonl` — 1063 kickers with zero penalty rows
- `wc2026_roster.jsonl` — 1243 unique players across 48 teams
- `training_table.jsonl` — 179 rows of 9-feature rows
- `predictions.jsonl` — 1243 rows of per-player predictions
- `lightgbm.pkl` — the frozen LightGBM model
- `baseline.pkl` — the baseline logreg model
- `metrics.json` — the held-out metrics report
- `discrepancies.json` — the RSSSF vs. scraper divergence report

The cache directory (separate from `root`):

- `data/fotmob_cache/` — the persistent ETag/gzip disk cache for the
  FotMob HTTP responses.

All read/write methods accept an optional `path` argument so callers
that take a CLI `--output` override can route through the same code
path as the default (e.g. `art.write_predictions(rows, path=args.output)`).
The default is the instance's path accessor.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any, TypeVar

from .client import FotMobClient
from .evaluate import MetricsReport
from .features import TrainingRow
from .model import (
    load_artifact as _model_load_artifact,
)
from .model import (
    save_artifact as _model_save_artifact,
)
from .player_history import MissingKicker, PlayerPenalty
from .predict import PredictionRow
from .rosters import RosterPlayer
from .shootouts import ShootoutKick

_T = TypeVar("_T")


def _write_jsonl(
    path: Path,
    rows: Iterable[Any],
    *,
    nan_to_null: bool = False,
) -> int:
    """Write `rows` to `path` as JSONL (one record per line).

    When `nan_to_null` is True, `NaN` floats in the serialised payload
    are emitted as JSON `null` (strict JSON parsers reject `NaN`). The
    caller is expected to use `dataclasses.asdict` for the row; the
    helper only handles the top-level `NaN` fields (not nested ones).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(_serialize_row(row, nan_to_null=nan_to_null))
            f.write("\n")
            count += 1
    return count


def _serialize_row(row: Any, *, nan_to_null: bool = False) -> str:
    """Serialise a single dataclass row to a JSON string (no trailing newline).

    The streaming-writer scripts (e.g. `fetch_initial_set_player_history.py`)
    need to write one row at a time to keep the disk copy current on a
    long run. The serialisation is the same as `_write_jsonl`'s per-row
    body, lifted into a public helper.
    """
    payload = asdict(row)
    if nan_to_null:
        age = payload.get("age")
        if isinstance(age, float) and math.isnan(age):
            payload["age"] = None
    return json.dumps(payload, ensure_ascii=False, allow_nan=(not nan_to_null))


def _read_jsonl_of_dataclasses[T](path: Path, cls: type[T]) -> list[T]:
    """Read a JSONL file of `cls` records into a list of `cls`.

    Blank lines are skipped. The reconstruction uses `cls(**row)`; the
    caller is expected to have written the JSONL via `dataclasses.asdict`
    of the same class (or an equivalent shape).
    """
    out: list[T] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(cls(**json.loads(line)))
    return out


class Artifacts:
    """The on-disk artifact layout adapter.

    An instance owns the artifact `root` and the FotMob `cache_dir`.
    Path accessors and typed read/write methods cross the same seam.

    >>> art = Artifacts()
    >>> art.shootout_kicks
    PosixPath('output/shootout_kicks.jsonl')
    >>> art.fotmob_client().cache_dir
    PosixPath('data/fotmob_cache')
    """

    def __init__(
        self,
        root: Path = Path("output"),
        cache_dir: Path = Path("data/fotmob_cache"),
    ) -> None:
        self.root = Path(root)
        self.cache_dir = Path(cache_dir)

    # ------------------------------------------------------------------ paths

    @property
    def shootout_kicks(self) -> Path:
        return self.root / "shootout_kicks.jsonl"

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
    def training_table(self) -> Path:
        return self.root / "training_table.jsonl"

    @property
    def predictions(self) -> Path:
        return self.root / "predictions.jsonl"

    @property
    def lightgbm_model(self) -> Path:
        return self.root / "lightgbm.pkl"

    @property
    def baseline_model(self) -> Path:
        return self.root / "baseline.pkl"

    @property
    def metrics(self) -> Path:
        return self.root / "metrics.json"

    @property
    def discrepancies(self) -> Path:
        return self.root / "discrepancies.json"

    # -------------------------------------------------------------- shootouts

    def read_shootout_kicks(self, path: Path | None = None) -> list[ShootoutKick]:
        return _read_jsonl_of_dataclasses(path or self.shootout_kicks, ShootoutKick)

    def write_shootout_kicks(
        self, rows: Iterable[ShootoutKick], path: Path | None = None
    ) -> int:
        return _write_jsonl(path or self.shootout_kicks, rows)

    # ------------------------------------------------------------ player_h

    def read_player_history(self, path: Path | None = None) -> list[PlayerPenalty]:
        return _read_jsonl_of_dataclasses(path or self.player_history, PlayerPenalty)

    def write_player_history(
        self, rows: Iterable[PlayerPenalty], path: Path | None = None
    ) -> int:
        return _write_jsonl(path or self.player_history, rows)

    # ----------------------------------------------------------- missing_h

    def read_missing_history(self, path: Path | None = None) -> list[MissingKicker]:
        return _read_jsonl_of_dataclasses(path or self.missing_history, MissingKicker)

    def write_missing_history(
        self, rows: Iterable[MissingKicker], path: Path | None = None
    ) -> int:
        return _write_jsonl(path or self.missing_history, rows)

    # ----------------------------------------------------------------- roster

    def read_roster(self, path: Path | None = None) -> list[RosterPlayer]:
        return _read_jsonl_of_dataclasses(path or self.roster, RosterPlayer)

    def write_roster(self, rows: Iterable[RosterPlayer], path: Path | None = None) -> int:
        return _write_jsonl(path or self.roster, rows)

    # -------------------------------------------------------- training_table

    def read_training_table(self, path: Path | None = None) -> list[TrainingRow]:
        return _read_jsonl_of_dataclasses(path or self.training_table, TrainingRow)

    def write_training_table(
        self, rows: Iterable[TrainingRow], path: Path | None = None
    ) -> int:
        return _write_jsonl(path or self.training_table, rows, nan_to_null=True)

    # ----------------------------------------------------------- predictions

    def read_predictions(self, path: Path | None = None) -> list[PredictionRow]:
        return _read_jsonl_of_dataclasses(path or self.predictions, PredictionRow)

    def write_predictions(
        self, rows: Iterable[PredictionRow], path: Path | None = None
    ) -> int:
        return _write_jsonl(path or self.predictions, rows)

    # -------------------------------------------------------------- metrics

    def read_metrics(self, path: Path | None = None) -> MetricsReport:
        with (path or self.metrics).open(encoding="utf-8") as f:
            return MetricsReport.from_dict(json.load(f))

    def write_metrics(self, report: MetricsReport, path: Path | None = None) -> int:
        target = path or self.metrics
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        return 1

    # ----------------------------------------------------------------- model

    def read_model(self, path: Path | None = None) -> dict[str, Any]:
        return _model_load_artifact(path or self.lightgbm_model)

    def write_model(
        self,
        model: Any,
        feature_columns: list[str],
        model_kind: str,
        params: dict[str, Any] | None = None,
        path: Path | None = None,
    ) -> int:
        _model_save_artifact(
            path or self.lightgbm_model,
            model,
            feature_columns,
            model_kind,
            params=params,
        )
        return 1

    # ----------------------------------------------------------- cache factory

    def fotmob_client(self) -> FotMobClient:
        """Build a `FotMobClient` rooted at this adapter's cache_dir."""
        return FotMobClient(cache_dir=self.cache_dir)

    # ---------------------------------------------------- streaming serialise

    def serialize_row(self, row: Any, *, nan_to_null: bool = False) -> str:
        """Serialise one dataclass row to a JSON string (no trailing newline).

        Streaming-writer scripts (e.g. the initial-set player history
        slice) want to write one row at a time so the on-disk copy is
        kept current on long runs (a 3h cold-cache run cannot afford to
        lose rows to a Ctrl-C). The serialisation matches the
        `write_*` methods' per-row shape, so the file format is
        unchanged.
        """
        return _serialize_row(row, nan_to_null=nan_to_null)
