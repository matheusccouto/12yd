"""Tests for `validate_shootout_count` and the discrepancy file writer.

The validator is the bridge between the scraper and the RSSSF oracle.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from penalty_pred.artifacts import Artifacts
from penalty_pred.rsssf import load_rsssf_html, parse_rsssf_html
from penalty_pred.shootouts import ShootoutKick
from penalty_pred.tournaments import LEAGUE_SEASONS_PREDICT_WINDOW
from penalty_pred.validate import validate_shootout_count

REPO_ROOT = Path(__file__).resolve().parent.parent
RSSSF_FIXTURE = REPO_ROOT / "docs" / "samples" / "rsssf_penaltiestour.html"


@pytest.fixture(scope="module")
def rsssf_shootouts() -> list[object]:
    return parse_rsssf_html(load_rsssf_html(RSSSF_FIXTURE))


def _write_kicks(path: Path, kicks: list[ShootoutKick]) -> None:
    """Write ShootoutKick records to JSONL via the artifacts adapter."""
    Artifacts(root=path.parent).write_shootout_kicks(kicks, path=path)


def _kicks_for_pairs(pairs: list[tuple[str, int, int]]) -> list[ShootoutKick]:
    """Build one `ShootoutKick` per `(tournament, year, count)` tuple.

    Each match in the count is a distinct match_id so the validator's
    `distinct match_ids` count is `count`. Only the validator-relevant
    fields (match_id, match_date, tournament_name) matter for the
    count; the rest are placeholders that pass the dataclass.
    """
    out: list[ShootoutKick] = []
    next_id = 1
    for tournament, year, count in pairs:
        for _ in range(count):
            out.append(
                ShootoutKick(
                    match_id=next_id,
                    match_date=f"{year}-12-18T15:00:00+00:00",
                    tournament_id=77,
                    tournament_name=tournament,
                    round="Final",
                    kick_number=1,
                    kicker_id=next_id,
                    kicker_name=f"Stub {next_id}",
                    team_id=next_id,
                    is_home=True,
                    x=0.5,
                    side="L",
                    is_on_target=True,
                    outcome="Goal",
                    pen_score_before=[0, 0],
                    pen_score_after=[0, 0],
                    match_score_home=0,
                    match_score_away=0,
                )
            )
            next_id += 1
    return out


IN_SCOPE_PAIRS: list[tuple[str, int, int]] = [
    ("World Cup", 2022, 5),
    ("Euro", 2020, 4),
    ("Euro", 2024, 3),
    ("Copa América", 2021, 3),
    ("Copa América", 2024, 4),
    ("Africa Cup of Nations", 2021, 6),
    ("Africa Cup of Nations", 2023, 5),
    ("Africa Cup of Nations", 2025, 3),
    ("CONCACAF Gold Cup", 2023, 2),
    ("CONCACAF Gold Cup", 2025, 3),
    ("AFC Asian Cup", 2023, 4),
]


def test_match_when_counts_align(tmp_path: Path, rsssf_shootouts: list[object]) -> None:
    """A JSONL with 42 distinct match_ids whose years/tournaments are the
    in-scope pairs should match the RSSSF count exactly."""
    kicks = _kicks_for_pairs(IN_SCOPE_PAIRS)
    jsonl = tmp_path / "kicks.jsonl"
    _write_kicks(jsonl, kicks)

    report = validate_shootout_count(jsonl, rsssf_shootouts, LEAGUE_SEASONS_PREDICT_WINDOW)
    assert report.actual == 42
    assert report.expected == 42
    assert report.match is True


def test_mismatch_writes_discrepancies(tmp_path: Path, rsssf_shootouts: list[object]) -> None:
    """A JSONL with 1 match should not match, and a discrepancies file is written."""
    kicks = _kicks_for_pairs([("World Cup", 2022, 1)])
    jsonl = tmp_path / "kicks.jsonl"
    _write_kicks(jsonl, kicks)
    disc = tmp_path / "discrepancies.json"

    report = validate_shootout_count(
        jsonl, rsssf_shootouts, LEAGUE_SEASONS_PREDICT_WINDOW, discrepancies_path=disc
    )
    assert report.actual == 1
    assert report.expected == 42
    assert report.match is False

    payload = json.loads(disc.read_text())
    assert payload["actual_shootout_count"] == 1
    assert payload["expected_shootout_count"] == 42
    assert payload["delta"] == -41
    # The observed pair is reported.
    assert payload["actual_pairs"] == [{"tournament": "World Cup", "year": 2022}]


def test_match_skips_discrepancy_writing(tmp_path: Path, rsssf_shootouts: list[object]) -> None:
    """When counts match, the discrepancies file is NOT created."""
    kicks = _kicks_for_pairs(IN_SCOPE_PAIRS)
    jsonl = tmp_path / "kicks.jsonl"
    _write_kicks(jsonl, kicks)
    disc = tmp_path / "discrepancies.json"

    report = validate_shootout_count(
        jsonl, rsssf_shootouts, LEAGUE_SEASONS_PREDICT_WINDOW, discrepancies_path=disc
    )
    assert report.match is True
    assert not disc.exists()


def test_match_counts_distinct_match_ids(tmp_path: Path, rsssf_shootouts: list[object]) -> None:
    """Two rows with the same match_id count as one shootout match."""
    kick = _kicks_for_pairs([("World Cup", 2022, 1)])[0]
    kicks = [kick, kick]
    jsonl = tmp_path / "kicks.jsonl"
    _write_kicks(jsonl, kicks)
    report = validate_shootout_count(jsonl, rsssf_shootouts, LEAGUE_SEASONS_PREDICT_WINDOW)
    assert report.actual == 1  # 1 distinct match_id (the second row is a kick)


def test_skipped_refs_included_in_discrepancies(
    tmp_path: Path, rsssf_shootouts: list[object]
) -> None:
    """When the JSONL is short of the expected count, skipped_refs are
    serialised in discrepancies.json for debugging."""
    from penalty_pred.match_ref import MatchRef

    kicks = _kicks_for_pairs([("World Cup", 2022, 1)])
    jsonl = tmp_path / "kicks.jsonl"
    _write_kicks(jsonl, kicks)
    disc = tmp_path / "discrepancies.json"

    skipped = [
        MatchRef(
            match_id=999,
            seo="x-vs-y",
            h2h="abc123",
            round_name="QF",
            home_team_name="X",
            away_team_name="Y",
            match_date="2022-07-01T15:00:00Z",
            score_str="1 - 1",
        )
    ]
    report = validate_shootout_count(
        jsonl,
        rsssf_shootouts,
        LEAGUE_SEASONS_PREDICT_WINDOW,
        discrepancies_path=disc,
        skipped_refs=skipped,
    )
    assert report.match is False
    payload = json.loads(disc.read_text())
    assert len(payload["skipped_refs"]) == 1
    assert payload["skipped_refs"][0]["match_id"] == 999
    assert payload["skipped_refs"][0]["home"] == "X"
    assert payload["skipped_refs"][0]["away"] == "Y"
    assert payload["skipped_refs"][0]["round"] == "QF"
    assert payload["skipped_refs"][0]["match_date"] == "2022-07-01T15:00:00Z"


def test_no_kicks_refs_included_in_discrepancies(
    tmp_path: Path, rsssf_shootouts: list[object]
) -> None:
    """`no_kicks_refs` are also serialised in discrepancies.json."""
    from penalty_pred.match_ref import MatchRef

    kicks = _kicks_for_pairs([("World Cup", 2022, 1)])
    jsonl = tmp_path / "kicks.jsonl"
    _write_kicks(jsonl, kicks)
    disc = tmp_path / "discrepancies.json"

    no_kicks = [
        MatchRef(
            match_id=888,
            seo="a-vs-b",
            h2h="zzz999",
            round_name="SF",
            home_team_name="A",
            away_team_name="B",
            match_date="2022-02-03T19:00:00Z",
            score_str="0 - 0",
        )
    ]
    validate_shootout_count(
        jsonl,
        rsssf_shootouts,
        LEAGUE_SEASONS_PREDICT_WINDOW,
        discrepancies_path=disc,
        no_kicks_refs=no_kicks,
    )
    payload = json.loads(disc.read_text())
    assert len(payload["no_kicks_refs"]) == 1
    assert payload["no_kicks_refs"][0]["match_id"] == 888
    assert payload["no_kicks_refs"][0]["round"] == "SF"


def test_delta_property(tmp_path: Path) -> None:
    """`report.delta` is actual - expected (negative when under-counted)."""
    kicks = _kicks_for_pairs([("World Cup", 2022, 1)])
    jsonl = tmp_path / "kicks.jsonl"
    _write_kicks(jsonl, kicks)
    report = validate_shootout_count(jsonl, [], [(77, 2022)])
    assert report.delta == 1  # 1 actual, 0 expected
    assert report.actual == 1
    assert report.expected == 0
