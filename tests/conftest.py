"""Shared pytest fixtures."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Mapping

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_MATCH_PATH = REPO_ROOT / "docs" / "samples" / "match_3370572.json.gz"


@pytest.fixture(scope="session")
def sample_2022_final() -> Mapping[str, Any]:
    """
    Return the full FotMob match JSON for the 2022 FIFA World Cup Final.

    Sourced from `docs/samples/match_3370572.json.gz` (cached by the scraper).
    Tests use this fixture so they do not hit the FotMob API.
    """
    if not SAMPLE_MATCH_PATH.exists():
        pytest.skip(
            f"Sample not present at {SAMPLE_MATCH_PATH}. "
            "Run `uv run python scripts/fetch_2022_final.py` to populate it.",
        )
    return json.loads(gzip.decompress(SAMPLE_MATCH_PATH.read_bytes()))
