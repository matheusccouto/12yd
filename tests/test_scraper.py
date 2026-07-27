"""Tests for scraper CLI and season filtering logic."""

import json
from pathlib import Path

from click.testing import CliRunner

from scripts.scraper import _parse_season_end_year, matches


def test_parse_season_end_year() -> None:
    """Test parsing season end year from various season string formats."""
    assert _parse_season_end_year("2023/2024") == 2024
    assert _parse_season_end_year("2022") == 2022
    assert _parse_season_end_year("2024 - Clausura") == 2024


def test_matches_cli_skips_past_seasons(tmp_path: Path) -> None:
    """Test that matches CLI command skips past historical seasons by default."""
    seasons_file = tmp_path / "seasons.jsonl"
    matches_file = tmp_path / "matches.jsonl"

    seasons_file.write_text(
        json.dumps(
            {
                "league_id": 99999,
                "league_name": "Fake",
                "gender": "men",
                "season": "2000",
            },
        )
        + "\n",
        encoding="utf-8",
    )
    matches_file.write_text("", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        matches,
        [
            str(seasons_file),
            str(matches_file),
            "--timeout-minutes",
            "1",
        ],
    )

    assert result.exit_code == 0
    # Past season 2000 is skipped automatically, matches_file remains empty
    assert matches_file.read_text(encoding="utf-8") == ""
