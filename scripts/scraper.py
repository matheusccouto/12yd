"""Scraper."""

import json
import logging
from pathlib import Path

from twelveyards.fotmob.client import FotMob

logging.basicConfig(level=logging.WARNING)

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MATCHES_PATH = DATA_DIR / "matches.jsonl"

with MATCHES_PATH.open(encoding="utf-8") as file:
    skip_ids = {json.loads(line)["id"] for line in file if line.strip()}

client = FotMob()

with MATCHES_PATH.open("a", encoding="utf-8") as file:
    for league in client.get_leagues(max_workers=4):
        for season in league.seasons:
            for match in client.get_matches(
                league.id,
                season,
                max_workers=1,
                skip_ids=skip_ids,
            ):
                file.write(match.model_dump_json() + "\n")
                skip_ids.add(match.id)
