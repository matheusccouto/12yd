from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from twelveyards.fotmob.client import FotMob
from twelveyards.fotmob.models import League, PenaltyKick
from twelveyards.pipeline import (
    DatasetSpec,
    iter_shootout_kicks,
    load_seen_match_ids,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_iter_shootout_kicks_filtering_real() -> None:
    client = FotMob()
    league = League.model_validate(
        {"id": 77, "slug": "world-cup", "name": "World Cup", "kind": "international"},
    )
    spec = DatasetSpec(
        leagues=(league,),
        date_range=(datetime(2022, 12, 17, tzinfo=UTC), datetime(2022, 12, 19, tzinfo=UTC)),
        lookback_years=1,
    )
    kicks = list(iter_shootout_kicks(client, spec))
    assert len(kicks) > 0
    assert kicks[0].match_id == "3370572"
    assert kicks[0].outcome in ("Goal", "Saved", "Missed")



def test_load_seen_match_ids_truncation(tmp_path: Path) -> None:
    filepath = tmp_path / "shootouts.jsonl"

    # Write 2 valid lines and 1 malformed trailing line
    k1 = PenaltyKick(
        match_id="101", league_id=77, season="2022",
        match_date=datetime(2022, 12, 18, 15, 0, tzinfo=UTC),
        player_id=1, team_id=1, is_home=True, x=0.5, y=0.5,
        outcome="Goal", shot_type="RightFoot", player_position="Forward",
    )
    k2 = PenaltyKick(
        match_id="102", league_id=77, season="2022",
        match_date=datetime(2022, 12, 18, 15, 0, tzinfo=UTC),
        player_id=2, team_id=2, is_home=False, x=0.5, y=0.5,
        outcome="Goal", shot_type="RightFoot", player_position="Forward",
    )

    with filepath.open("w", encoding="utf-8") as f:
        f.write(k1.model_dump_json() + "\n")
        f.write(k2.model_dump_json() + "\n")
        f.write('{"match_id": "103", "league_id": 77, "outcome": "Goa\n') # malformed

    seen = load_seen_match_ids(filepath)
    # Check that we loaded match_ids from the two valid lines and stopped at the malformed one
    assert seen == {"101", "102"}

    # Check that the file was truncated to exactly 2 lines
    with filepath.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 2
    assert "101" in lines[0]
    assert "102" in lines[1]
