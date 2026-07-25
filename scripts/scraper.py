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


@cli.command()
@click.argument("seasons_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option("--timeout-minutes", required=True, type=int, help="Max minutes.")
def matches(seasons_path: Path, output_path: Path, timeout_minutes: int) -> None:
    """Scrape matches for given seasons into matches.jsonl."""
    deadline = datetime.now(tz=UTC) + timedelta(minutes=timeout_minutes)
    client = FotMob()

    with output_path.open(encoding="utf-8") as f:
        skip_ids = {json.loads(line)["id"] for line in f if line.strip()}

    with seasons_path.open(encoding="utf-8") as f:
        league_seasons = [json.loads(line) for line in f if line.strip()]

    with output_path.open("a", encoding="utf-8") as f:
        for item in league_seasons:
            if datetime.now(tz=UTC) > deadline:
                logger.warning("Timeout after %s minutes", timeout_minutes)
                return
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
