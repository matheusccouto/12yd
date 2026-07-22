"""Scraper."""

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import click

from twelveyards.fotmob.client import FotMob

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MATCHES_PATH = DATA_DIR / "matches.jsonl"


@click.command()
@click.option("--timeout-minutes", default=60, help="Max minutes before stopping.")
def main(timeout_minutes: int) -> None:
    """Scrape FotMob matches, appending to matches.jsonl."""
    deadline = datetime.now(tz=UTC) + timedelta(minutes=timeout_minutes)

    with MATCHES_PATH.open(encoding="utf-8") as f:
        skip_ids = {json.loads(line)["id"] for line in f if line.strip()}

    client = FotMob()

    with MATCHES_PATH.open("a", encoding="utf-8") as f:
        for league in client.get_leagues(max_workers=4):
            if datetime.now(tz=UTC) > deadline:
                logger.warning("Timeout after %s minutes", timeout_minutes)
                return
            for season in league.seasons:
                if datetime.now(tz=UTC) > deadline:
                    logger.warning("Timeout after %s minutes", timeout_minutes)
                    return
                for match in client.get_matches(
                    league.id,
                    season,
                    max_workers=1,
                    skip_ids=skip_ids,
                ):
                    if datetime.now(tz=UTC) > deadline:
                        logger.warning("Timeout after %s minutes", timeout_minutes)
                        return
                    f.write(match.model_dump_json() + "\n")
                    f.flush()
                    skip_ids.add(match.id)


if __name__ == "__main__":
    main()
