"""Scraper."""

import logging
from pathlib import Path

from twelveyards.fotmob.client import FotMob

logging.basicConfig(level=logging.WARNING)

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"

client = FotMob()

with (DATA_DIR / "matches.jsonl").open("w", encoding="utf-8") as file:
    for league in client.get_leagues(max_workers=1):
        for season in league.seasons:
            for match in client.get_matches(league.id, season, max_workers=1):
                file.write(match.model_dump_json() + "\n")
