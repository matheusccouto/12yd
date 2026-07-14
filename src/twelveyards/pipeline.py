"""Shootout kicks dataset pipeline and CLI."""

from __future__ import annotations

import argparse
import contextlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from .fotmob.client import FotMob
from .fotmob.leagues import LEAGUE_BY_ID, LEAGUES
from .fotmob.models import League, PenaltyKick

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass(frozen=True)
class DatasetSpec:
    """Configuration specification for the shootout kicks dataset ingest."""

    leagues: tuple[League, ...]
    date_range: tuple[datetime, datetime]
    lookback_years: int = 4


def load_seen_match_ids(filepath: Path) -> set[str]:
    """
    Scan existing JSONL lines to build seen_match_ids.

    Tolerates a malformed final line by truncating the file to the last good line.
    """
    if not filepath.exists():
        return set()

    seen = set()
    valid_lines: list[bytes] = []
    corrupted = False

    with filepath.open("rb") as f:
        lines = f.readlines()

    for line in lines:
        line_str = line.decode("utf-8").strip()
        if not line_str:
            continue
        try:
            row = json.loads(line_str)
            if "match_id" in row:
                seen.add(str(row["match_id"]))
            valid_lines.append(line)
        except json.JSONDecodeError:
            corrupted = True
            break

    if corrupted:
        with filepath.open("wb") as f:
            for line_bytes in valid_lines:
                f.write(line_bytes)

    return seen


def iter_shootout_kicks(  # noqa: C901, PLR0912
    client: FotMob,
    spec: DatasetSpec,
    seen_match_ids: set[str] | None = None,
) -> Iterator[PenaltyKick]:
    """Yield shootout kicks matching the DatasetSpec filter rules."""
    if seen_match_ids is None:
        seen_match_ids = load_seen_match_ids(Path("data/shootout_kicks.jsonl"))

    start_dt, end_dt = spec.date_range

    for league in spec.leagues:
        seasons = client.get_league_seasons(league.league_id)
        for season in seasons:
            fixtures = client.get_league_matches(
                league.league_id, season.season_name,
            )

            for ref in fixtures:
                ref_date = ref.match_date
                # Align timezones
                if start_dt.tzinfo:
                    s_dt = start_dt.astimezone(ref_date.tzinfo)
                else:
                    s_dt = start_dt.replace(tzinfo=ref_date.tzinfo)

                if end_dt.tzinfo:
                    e_dt = end_dt.astimezone(ref_date.tzinfo)
                else:
                    e_dt = end_dt.replace(tzinfo=ref_date.tzinfo)

                if ref_date < s_dt or ref_date > e_dt:
                    continue
                if not ref.is_shootout:
                    continue
                if ref.match_id in seen_match_ids:
                    continue

                # get_match is the loud point: it fetches full details.
                # If it fails, fail loud.
                match = client.get_match(ref.match_id)

                for shot in match.shotmap:
                    if shot.situation != "Penalty" or shot.period != "PenaltyShootout":
                        continue

                    player_id = shot.player_id
                    team_id = match.player_teams.get(player_id, shot.team_id)
                    is_home = (team_id == match.home_team_id)
                    position = match.player_positions.get(player_id, "")

                    yield PenaltyKick(
                        match_id=ref.match_id,
                        league_id=league.league_id,
                        season=season.season_name,
                        match_date=ref.match_date,
                        player_id=player_id,
                        team_id=team_id,
                        is_home=is_home,
                        x=shot.x,
                        y=shot.y,
                        outcome=shot.outcome,
                        shot_type=shot.shot_type,
                        player_position=position,
                    )


def build_shootout_dataset(
    client: FotMob,
    spec: DatasetSpec,
    filepath: Path = Path("data/shootout_kicks.jsonl"),
) -> pd.DataFrame:
    """Build the shootout kicks dataset.

    Appends new kicks to JSONL and returns a DataFrame.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    seen_match_ids = load_seen_match_ids(filepath)

    with filepath.open("a", encoding="utf-8") as f:
        for kick in iter_shootout_kicks(
            client,
            spec,
            seen_match_ids=seen_match_ids,
        ):
            f.write(kick.model_dump_json() + "\n")
            f.flush()
            seen_match_ids.add(kick.match_id)

    records = []
    if filepath.exists():
        with filepath.open("r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    with contextlib.suppress(json.JSONDecodeError):
                        records.append(json.loads(line_str))

    return pd.DataFrame(records)


def main() -> None:
    """CLI entry point for building the shootout kicks dataset."""
    parser = argparse.ArgumentParser(
        description="Build shootout kicks dataset from FotMob.",
    )
    parser.add_argument(
        "--leagues",
        type=str,
        help=(
            "Comma-separated list of league IDs to query. "
            "Default is all international leagues."
        ),
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2016-01-01T00:00:00Z",
        help="ISO 8601 start date of the range (UTC).",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2030-12-31T23:59:59Z",
        help="ISO 8601 end date of the range (UTC).",
    )
    parser.add_argument(
        "--lookback-years",
        type=int,
        default=4,
        help="Number of lookback years.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/shootout_kicks.jsonl",
        help="Path to write the shootout_kicks.jsonl dataset.",
    )

    args = parser.parse_args()

    # Parse leagues
    if args.leagues:
        league_ids = [int(lid.strip()) for lid in args.leagues.split(",")]
        leagues = tuple(
            LEAGUE_BY_ID[lid]
            if lid in LEAGUE_BY_ID
            else League.model_validate(
                {
                    "id": lid,
                    "slug": "unknown",
                    "name": f"League {lid}",
                    "kind": "unknown",
                },
            )
            for lid in league_ids
        )
    else:
        leagues = LEAGUES

    # Parse dates
    start_dt = datetime.fromisoformat(args.start_date)
    end_dt = datetime.fromisoformat(args.end_date)

    spec = DatasetSpec(
        leagues=leagues,
        date_range=(start_dt, end_dt),
        lookback_years=args.lookback_years,
    )

    client = FotMob()
    build_shootout_dataset(client, spec, filepath=Path(args.output))
    print(f"Shootout kicks dataset successfully built at {args.output}")  # noqa: T201


if __name__ == "__main__":
    main()
