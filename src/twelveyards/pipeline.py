"""End-to-end pipeline coordinator — the single seam between scripts and the library."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from .artifacts import Artifacts
from .config import LOOKBACK_WINDOW_YEARS, SCRAPE_FLOOR, today_utc
from .fotmob.leagues import WC_2026_LEAGUE, WC_2026_SEASON, League
from .model.predict import load_player_history, predict_and_write
from .scraper.initial_set import (
    MissingKicker,
    fetch_all_initial_set_penalty_history_parallel,
    iter_initial_set_kickers,
)
from .scraper.player_history import (
    PlayerMetadata,
    extract_player_metadata,
    fetch_player_data,
)
from .scraper.rosters import fetch_wc_2026_roster

if TYPE_CHECKING:
    from datetime import date
    from pathlib import Path

    from .fotmob.client import FotMobClient


def fetch_and_write_roster(
    client: FotMobClient, output_path: Path,
    league: League = WC_2026_LEAGUE,
    season: int = WC_2026_SEASON,
) -> int:
    """Fetch WC 2026 roster and write to JSONL. Returns count of unique players."""
    rows = fetch_wc_2026_roster(client, league, season)
    return Artifacts().write_roster(rows, path=output_path)


def fetch_and_write_initial_set(
    client: FotMobClient,
    roster_path: Path,
    output_path: Path,
    missing_path: Path,
    target_date: date | None = None,
    lookback_years: int = LOOKBACK_WINDOW_YEARS,
    history_floor: date = SCRAPE_FLOOR,
    max_workers: int = 12,
    *,
    progress_every: int = 25,
) -> tuple[int, int, int, int]:
    """
    Fan out penalty-history fetches across the roster, stream results, write missing.

    Returns (n_kickers, n_rows_written, n_missing, n_errored).
    """
    art = Artifacts()
    roster = art.read_roster(path=roster_path)
    initial_set = list(iter_initial_set_kickers(roster))
    if target_date is None:
        target_date = today_utc()

    n_rows_written = 0
    total = len(initial_set)
    results: list = []
    t0 = time.monotonic()
    with output_path.open("w", encoding="utf-8") as out_f:
        for i, result in enumerate(
            fetch_all_initial_set_penalty_history_parallel(
                client, initial_set,
                target_date=target_date,
                lookback_years=lookback_years,
                history_floor=history_floor,
                max_workers=max_workers,
            ),
            start=1,
        ):
            results.append(result)
            for row in result.rows:
                out_f.write(art.serialize_row(row))
                out_f.write("\n")
                n_rows_written += 1
            out_f.flush()
            if i % progress_every == 0 or i == total:
                time.monotonic() - t0
                n_missing = sum(1 for r in results if not r.rows)
                sum(1 for r in results if r.error)

    missing = [
        MissingKicker(
            player_id=r.kicker.player_id,
            player_name=r.kicker.player_name,
            team_id=r.kicker.team_id,
            team_name=r.kicker.team_name,
        )
        for r in results
        if not r.rows
    ]
    art.write_missing_history(missing, path=missing_path)
    n_missing = len(missing)
    n_errored = sum(1 for r in results if r.error)
    return total, n_rows_written, n_missing, n_errored


def predict(
    client: FotMobClient,
    roster_path: Path,
    player_history_path: Path,
    output_path: Path,
    target_date: date | None = None,
) -> tuple[int, int]:
    """
    Fetch per-kicker metadata, fit TabPFN, predict, write predictions.jsonl.

    Returns (n_predictions, n_no_history).
    """
    art = Artifacts()
    roster = art.read_roster(path=roster_path)
    player_history = load_player_history(player_history_path)
    if target_date is None:
        target_date = today_utc()

    metadata_by_id: dict[int, PlayerMetadata] = {}
    for kicker_id in {*player_history.keys(), *(p.player_id for p in roster)}:
        try:
            payload = fetch_player_data(client, kicker_id)
            metadata = extract_player_metadata(payload)
            if metadata is not None:
                metadata_by_id[kicker_id] = metadata
        except Exception:
            pass

    rows = predict_and_write(
        roster, player_history, metadata_by_id,
        output_path=output_path, target_date=target_date,
    )
    n_no_history = sum(1 for r in rows if r.total_penalties == 0)
    return len(rows), n_no_history
