"""Initial Set assembly and per-kicker penalty history fetch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from twelveyards.config import LOOKBACK_WINDOW_YEARS, SCRAPE_FLOOR

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from datetime import date

    from .fotmob.client import FotMobClientLike
    from .scraper.player_history import PlayerPenalty
    from .scraper.rosters import RosterPlayer

__all__ = [
    "InitialSetFetchResult",
    "InitialSetKicker",
    "MissingKicker",
    "fetch_all_initial_set_penalty_history_parallel",
    "iter_initial_set_kickers",
]


@dataclass(frozen=True)
class InitialSetKicker:
    """One kicker in the initial set, identified by player + team."""

    player_id: int
    player_name: str
    team_id: int
    team_name: str


@dataclass(frozen=True)
class MissingKicker:
    """A kicker for whom no penalty history was found."""

    player_id: int
    player_name: str
    team_id: int
    team_name: str


@dataclass(frozen=True)
class InitialSetFetchResult:
    """The result of fetching penalty history for one kicker."""

    kicker: InitialSetKicker
    rows: list[PlayerPenalty]
    error: str | None = None


def iter_initial_set_kickers(
    roster: Iterable[RosterPlayer],
) -> Iterator[InitialSetKicker]:
    """Yield an InitialSetKicker for every RosterPlayer."""
    for row in roster:
        yield InitialSetKicker(
            player_id=row.player_id,
            player_name=row.player_name,
            team_id=row.team_id,
            team_name=row.team_name,
        )


def _fetch_one(
    kicker: InitialSetKicker,
    client: FotMobClientLike,
    target_date: date | None,
    lookback_years: int,
    history_floor: date,
) -> Iterator[PlayerPenalty]:
    from .scraper.player_history import fetch_player_penalty_history  # noqa: PLC0415

    yield from fetch_player_penalty_history(
        client,
        player_id=kicker.player_id,
        target_date=target_date,
        lookback_years=lookback_years,
        history_floor=history_floor,
    )


def fetch_all_initial_set_penalty_history_parallel(  # noqa: PLR0913
    client: FotMobClientLike,
    initial_set: Iterable[InitialSetKicker],
    target_date: date | None = None,
    lookback_years: int = LOOKBACK_WINDOW_YEARS,
    history_floor: date = SCRAPE_FLOOR,
    max_workers: int = 12,
) -> Iterator[InitialSetFetchResult]:
    """Yield InitialSetFetchResult for each kicker, using thread-pool parallelism."""
    from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: PLC0415

    initial_list = list(initial_set)
    if not initial_list:
        return

    def _fetch(k: InitialSetKicker) -> InitialSetFetchResult:
        try:
            rows = list(
                _fetch_one(k, client, target_date, lookback_years, history_floor),
            )
        except Exception as e:  # noqa: BLE001
            return InitialSetFetchResult(kicker=k, rows=[], error=repr(e))
        return InitialSetFetchResult(kicker=k, rows=rows, error=None)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_kicker = {ex.submit(_fetch, k): k for k in initial_list}
        for future in as_completed(future_to_kicker):
            kicker = future_to_kicker[future]
            try:
                yield future.result()
            except Exception as e:  # noqa: BLE001
                yield InitialSetFetchResult(kicker=kicker, rows=[], error=repr(e))
