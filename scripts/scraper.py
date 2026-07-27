"""Scraper CLI."""

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import click

from twelveyards.fotmob.client import FotMob

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


@click.group()
def cli() -> None:
    """FotMob Scraper CLI."""


@cli.command()
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option("--timeout-minutes", required=True, type=int, help="Max minutes.")
def seasons(output_path: Path, timeout_minutes: int) -> None:
    """Scrape available leagues and seasons into seasons.jsonl."""
    deadline = datetime.now(tz=UTC) + timedelta(minutes=timeout_minutes)
    client = FotMob()

    with output_path.open(encoding="utf-8") as f:
        skip_keys = {
            (json.loads(line)["league_id"], json.loads(line)["season"])
            for line in f
            if line.strip()
        }

    with output_path.open("a", encoding="utf-8") as f:
        for league in client.get_leagues(max_workers=4):
            if datetime.now(tz=UTC) > deadline:
                logger.warning("Timeout after %s minutes", timeout_minutes)
                return
            for season in league.seasons:
                key = (league.id, season)
                if key in skip_keys:
                    continue
                f.write(
                    json.dumps(
                        {
                            "league_id": league.id,
                            "league_name": league.name,
                            "gender": league.gender,
                            "season": season,
                        },
                    )
                    + "\n",
                )
                f.flush()
                skip_keys.add(key)


def _parse_season_end_year(season: str | int) -> int:
    part = str(season).split("/")[1] if "/" in str(season) else str(season)
    return int(part.split()[0])


def _load_existing_matches(
    output_path: Path,
) -> tuple[set[int], set[int]]:
    skip_ids: set[int] = set()
    scraped_leagues: set[int] = set()
    if output_path.exists():
        with output_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                m = json.loads(line)
                skip_ids.add(int(m["id"]))
                if "league_id" in m:
                    scraped_leagues.add(int(m["league_id"]))
    return skip_ids, scraped_leagues


@cli.command()
@click.argument("seasons_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option("--timeout-minutes", required=True, type=int, help="Max minutes.")
@click.option(
    "--all-seasons",
    is_flag=True,
    help="Force checking all historical seasons.",
)
def matches(
    seasons_path: Path,
    output_path: Path,
    timeout_minutes: int,
    all_seasons: bool = False,  # noqa: FBT001, FBT002
) -> None:
    """Scrape matches for given seasons into matches.jsonl."""
    deadline = datetime.now(tz=UTC) + timedelta(minutes=timeout_minutes)
    client = FotMob()

    skip_ids, scraped_leagues = _load_existing_matches(output_path)

    with seasons_path.open(encoding="utf-8") as f:
        league_seasons = [json.loads(line) for line in f if line.strip()]

    current_year = datetime.now(tz=UTC).year

    with output_path.open("a", encoding="utf-8") as f:
        for item in league_seasons:
            if datetime.now(tz=UTC) > deadline:
                logger.warning("Timeout after %s minutes", timeout_minutes)
                return

            if not all_seasons:
                end_year = _parse_season_end_year(str(item["season"]))
                if end_year < current_year:
                    continue
                if scraped_leagues and int(item["league_id"]) not in scraped_leagues:
                    continue

            for match in client.get_matches(
                item["league_id"],
                item["season"],
                max_workers=1,
                skip_ids=skip_ids,
            ):
                if datetime.now(tz=UTC) > deadline:
                    logger.warning("Timeout after %s minutes", timeout_minutes)
                    return
                f.write(match.model_dump_json() + "\n")
                f.flush()
                skip_ids.add(match.id)


@cli.command()
@click.argument("matches_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option("--timeout-minutes", required=True, type=int, help="Max minutes.")
def players(matches_path: Path, output_path: Path, timeout_minutes: int) -> None:
    """Scrape player profiles into players.jsonl for all kickers in matches.jsonl."""
    deadline = datetime.now(tz=UTC) + timedelta(minutes=timeout_minutes)
    client = FotMob()

    with matches_path.open(encoding="utf-8") as f:
        kicker_ids = {
            p["player_id"]
            for line in f
            if line.strip()
            for p in json.loads(line).get("penalties", [])
        }

    with output_path.open(encoding="utf-8") as f:
        skip_player_ids = {json.loads(line)["id"] for line in f if line.strip()}

    with output_path.open("a", encoding="utf-8") as f:
        for pid in kicker_ids - skip_player_ids:
            if datetime.now(tz=UTC) > deadline:
                logger.warning("Timeout after %s minutes", timeout_minutes)
                return
            f.write(client.get_player(pid).model_dump_json() + "\n")
            f.flush()


if __name__ == "__main__":
    cli()
