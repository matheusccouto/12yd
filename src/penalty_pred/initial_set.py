"""Initial Set assembly and per-kicker orchestration.

PRD: The Initial Set is the union of Training and Prediction Initial Sets.
Training kickers come from past Shootout Kicks (read from
`shootout_kicks.jsonl`); Prediction kickers come from a Tournament Roster
(read from `wc2026_roster.jsonl`). The two sets are deduplicated by
`player_id` (training first, roster second), and the per-kicker fetcher
from `player_history` is fanned out across the deduped union.

The split from `player_history` keeps each module's responsibility
single-named:
- `player_history` owns the FotMob fan-out (per-player → per-team-season
  → per-match → per-penalty).
- `initial_set` owns the Initial Set assembly (typed-iterable merge,
  per-kicker orchestration, missing-list reporting).

The two-level data graph is enforced at the seam: `iter_initial_set_kickers`
takes typed iterables (`Iterable[ShootoutKick]`, `Iterable[RosterPlayer]`),
never file paths, so the per-kicker orchestrator has no JSONL re-parse and
the JSONL shape lives in `Artifacts`.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import date

from .config import HISTORY_FLOOR, LOOKBACK_WINDOW_YEARS
from .player_history import PlayerPenalty, fetch_player_penalty_history
from .rosters import RosterPlayer
from .shootouts import ShootoutKick

# Re-exported so callers that imported the Initial Set types from
# `penalty_pred.player_history` (the pre-split location) keep working.
__all__ = [
    "InitialSetFetchResult",
    "InitialSetKicker",
    "MissingKicker",
    "fetch_all_initial_set_penalty_history",
    "fetch_all_initial_set_penalty_history_parallel",
    "iter_initial_set_kickers",
]


@dataclass(frozen=True)
class InitialSetKicker:
    """A Kicker in the Initial Set, identified by `player_id`.

    The Initial Set is the union of training kickers (read from
    `shootout_kicks.jsonl`) and prediction kickers (read from
    `wc2026_roster.jsonl`). Both sources carry the player's national
    team id (`team_id`); only the roster source carries a human-readable
    `team_name`. Training kickers not in the roster (e.g. retired players
    from earlier tournaments) have `team_name == ""`.

    We do NOT carry the player slug — FotMob does not use the slug for
    routing, only the `player_id` is authoritative, so the per-kicker
    fetcher uses the no-slug URL form.
    """

    player_id: int
    player_name: str
    team_id: int
    team_name: str  # "" for training kickers not in the prediction roster


@dataclass(frozen=True)
class MissingKicker:
    """An Initial Set Kicker with zero penalty rows in the lookback window.

    Written to `missing_history.jsonl` so downstream slices (#22 features,
    #25 predictions) can decide whether to skip them or use a prior-based
    fallback. We carry `team_name` for parity with the roster (some
    downstream views show the player name + team together).
    """

    player_id: int
    player_name: str
    team_id: int
    team_name: str


@dataclass(frozen=True)
class InitialSetFetchResult:
    """The per-kicker outcome of `fetch_all_initial_set_penalty_history`.

    `kicker` is the input Initial Set Kicker. `rows` is the list of
    `PlayerPenalty` records found in the lookback window (possibly empty).
    `error` is the stringified exception when the per-kicker fetch raised
    (e.g. a transient FotMob 5xx); the kicker is reported as missing in
    that case but the run continues. Successful fetches that yielded zero
    rows have `error=None, rows=[]`.
    """

    kicker: InitialSetKicker
    rows: list[PlayerPenalty]
    error: str | None = None


def iter_initial_set_kickers(
    shootout_kicks: Iterable[ShootoutKick],
    roster: Iterable[RosterPlayer],
) -> Iterator[InitialSetKicker]:
    """Yield the deduplicated union of training and prediction Kickers.

    The seam is typed collections, not the disk. The caller reads the
    JSONL once (via `Artifacts.read_shootout_kicks` /
    `Artifacts.read_roster`) and passes the lists down. The
    deduplication logic operates on the typed rows.

    Dedup + enrichment:
    1. Read the roster into a dict by `player_id` (small, fits in memory).
    2. Yield training kickers first, enriched with the roster's
       `team_name` (and `player_name`/`team_id` if the roster has a more
       up-to-date value) when the kicker is in both sets.
    3. Yield roster-only kickers (not in training) at the end.

    The "training first, roster-only last" ordering is what the caller
    needs to tell apart a kicker the model is going to be trained on
    (a shootout taker from a past tournament) from a kicker we only
    know from the WC roster (no shootout kicks yet).

    Two-level data graph preserved: the Initial Set is built from the
    Training Initial Set (shootout kicks) and the Prediction Initial Set
    (WC roster) — neither of which is derived from per-player penalty
    data. The orchestrator never fans out from the Derived History
    (per-kicker penalty rows) back into the Initial Set.
    """
    roster_by_id: dict[int, RosterPlayer] = {row.player_id: row for row in roster}

    seen: set[int] = set()
    for kick in shootout_kicks:
        kicker_id = kick.kicker_id
        if not kicker_id or kicker_id in seen:
            continue
        seen.add(kicker_id)
        roster_row = roster_by_id.get(kicker_id)
        yield InitialSetKicker(
            player_id=kicker_id,
            player_name=str(roster_row.player_name if roster_row else kick.kicker_name),
            team_id=roster_row.team_id if roster_row else kick.team_id,
            team_name=roster_row.team_name if roster_row else "",
        )

    for player_id, row in roster_by_id.items():
        if player_id in seen:
            continue
        seen.add(player_id)
        yield InitialSetKicker(
            player_id=player_id,
            player_name=row.player_name,
            team_id=row.team_id,
            team_name=row.team_name,
        )


def fetch_all_initial_set_penalty_history(
    client,
    initial_set: Iterable[InitialSetKicker],
    target_date: date | None = None,
    lookback_years: int = LOOKBACK_WINDOW_YEARS,
    history_floor: date = HISTORY_FLOOR,
) -> Iterator[InitialSetFetchResult]:
    """Fan out the per-kicker fetcher across the Initial Set.

    Yields one `InitialSetFetchResult` per kicker, in input order. Per-kicker
    fetch errors (transient FotMob 5xx, malformed player pages, etc.) are
    caught and recorded in `error`; the kicker is reported as missing in
    that case but the run continues. A successful fetch that returned
    zero penalty rows in the window has `error=None, rows=[]`.

    `target_date` defaults to 2022-12-18 (the 2022 WC Final) for the same
    reason as `fetch_player_penalty_history`. The all-Initial-Set slice
    uses `target_date=today_utc()` from `config` so the lookback window
    ends "now" (we want every penalty the player took up to the present,
    not just up to a fixed historical date).

    Two-level data graph preserved: this orchestrator fans out across the
    Initial Set (per-player) only; the per-kicker fetcher fans out
    across per-team-season lookups within the player's career. The Derived
    History (per-match penalty rows) is a leaf in the graph — we never
    recurse from there.
    """
    for kicker in initial_set:
        try:
            rows = list(
                fetch_player_penalty_history(
                    client,
                    player_id=kicker.player_id,
                    target_date=target_date,
                    lookback_years=lookback_years,
                    history_floor=history_floor,
                )
            )
        except Exception as e:  # noqa: BLE001 — boundary: one bad kicker must not abort the run
            yield InitialSetFetchResult(kicker=kicker, rows=[], error=repr(e))
            continue
        yield InitialSetFetchResult(kicker=kicker, rows=rows, error=None)


def fetch_all_initial_set_penalty_history_parallel(
    client,
    initial_set: Iterable[InitialSetKicker],
    target_date: date | None = None,
    lookback_years: int = LOOKBACK_WINDOW_YEARS,
    history_floor: date = HISTORY_FLOOR,
    max_workers: int = 12,
) -> Iterator[InitialSetFetchResult]:
    """Parallel fan-out of the per-kicker fetcher across the Initial Set.

    Same external contract as `fetch_all_initial_set_penalty_history` (one
    `InitialSetFetchResult` per kicker, in input order; the same per-kicker
    error handling), but uses a `ThreadPoolExecutor` to fetch N Kickers
    concurrently. The FotMob HTTP client is stateless across calls
    (per-URL `httpx.Client` in `_cached_get`), so sharing a single client
    across threads is safe — the on-disk cache writes are URL-keyed and
    never collide.

    Yields results in completion order (NOT input order — for input order,
    use the sequential `fetch_all_initial_set_penalty_history`). A slow
    kicker does not block a fast one — the slowest submission determines
    the wall time, not the sum of per-kicker times. The first `max_workers`
    kickers' results become available in roughly the time of the slowest
    one.

    The completion-order yield is a trade-off: it lets the caller stream
    results to disk (a 1359-kicker cold-cache run is ~3h serial, ~15-20
    min parallel, and we don't want to lose that to a Ctrl-C). The
    downstream writer does NOT depend on input order — the kicker's
    `player_id` is on every row and on every missing-history record, so
    the order is recoverable post-hoc.

    Per-kicker fetch errors (transient FotMob 5xx, malformed player
    pages, etc.) are caught and recorded in `error`; the kicker is
    reported as missing in that case but the run continues. A successful
    fetch that returned zero penalty rows in the window has
    `error=None, rows=[]`.

    `max_workers` defaults to 12 (a balance between FotMob's per-IP
    rate-limit and the sequential baseline).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    initial_list = list(initial_set)
    if not initial_list:
        return

    def _fetch_one(kicker: InitialSetKicker) -> InitialSetFetchResult:
        try:
            rows = list(
                fetch_player_penalty_history(
                    client,
                    player_id=kicker.player_id,
                    target_date=target_date,
                    lookback_years=lookback_years,
                    history_floor=history_floor,
                )
            )
        except Exception as e:  # noqa: BLE001 — boundary: one bad kicker must not abort the run
            return InitialSetFetchResult(kicker=kicker, rows=[], error=repr(e))
        return InitialSetFetchResult(kicker=kicker, rows=rows, error=None)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_kicker = {ex.submit(_fetch_one, k): k for k in initial_list}
        for future in as_completed(future_to_kicker):
            kicker = future_to_kicker[future]
            try:
                yield future.result()
            except Exception as e:  # noqa: BLE001 — last-resort boundary
                yield InitialSetFetchResult(kicker=kicker, rows=[], error=repr(e))
