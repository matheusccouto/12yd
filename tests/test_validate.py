"""Tests for `validate_shootout_count` and the discrepancy file writer.

The validator is the bridge between the scraper and the RSSSF oracle.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from penalty_pred.rsssf import load_rsssf_html, parse_rsssf_html
from penalty_pred.shootouts import LEAGUE_SEASONS_PREDICT_WINDOW
from penalty_pred.validate import validate_shootout_count

REPO_ROOT = Path(__file__).resolve().parent.parent
RSSSF_FIXTURE = REPO_ROOT / "docs" / "samples" / "rsssf_penaltiestour.html"


@pytest.fixture(scope="module")
def rsssf_shootouts() -> list[object]:
    return parse_rsssf_html(load_rsssf_html(RSSSF_FIXTURE))


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row))
            f.write("\n")


def test_match_when_counts_align(tmp_path: Path, rsssf_shootouts: list[object]) -> None:
    """A JSONL with 42 distinct match_ids whose years/tournaments are the
    in-scope pairs should match the RSSSF count exactly."""
    # Build 42 fake rows, one per (tournament, year) match — we just need
    # the right number of distinct match_ids.
    pairs = [
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
    rows: list[dict[str, object]] = []
    next_id = 1
    for tournament, year, count in pairs:
        for _ in range(count):
            rows.append(
                {
                    "match_id": next_id,
                    "match_date": f"{year}-12-18T15:00:00+00:00",
                    "tournament_name": tournament,
                }
            )
            next_id += 1
    jsonl = tmp_path / "kicks.jsonl"
    _write_jsonl(jsonl, rows)

    report = validate_shootout_count(jsonl, rsssf_shootouts, LEAGUE_SEASONS_PREDICT_WINDOW)
    assert report.actual == 42
    assert report.expected == 42
    assert report.match is True


def test_mismatch_writes_discrepancies(tmp_path: Path, rsssf_shootouts: list[object]) -> None:
    """A JSONL with 1 match should not match, and a discrepancies file is written."""
    rows = [
        {
            "match_id": 1,
            "match_date": "2022-12-18T15:00:00+00:00",
            "tournament_name": "World Cup",
        }
    ]
    jsonl = tmp_path / "kicks.jsonl"
    _write_jsonl(jsonl, rows)
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
    pairs = [
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
    rows: list[dict[str, object]] = []
    next_id = 1
    for tournament, year, count in pairs:
        for _ in range(count):
            rows.append(
                {
                    "match_id": next_id,
                    "match_date": f"{year}-12-18T15:00:00+00:00",
                    "tournament_name": tournament,
                }
            )
            next_id += 1
    jsonl = tmp_path / "kicks.jsonl"
    _write_jsonl(jsonl, rows)
    disc = tmp_path / "discrepancies.json"

    report = validate_shootout_count(
        jsonl, rsssf_shootouts, LEAGUE_SEASONS_PREDICT_WINDOW, discrepancies_path=disc
    )
    assert report.match is True
    assert not disc.exists()


def test_match_counts_distinct_match_ids(tmp_path: Path, rsssf_shootouts: list[object]) -> None:
    """Two rows with the same match_id count as one shootout match."""
    rows = [
        {
            "match_id": 1,
            "match_date": "2022-12-18T15:00:00+00:00",
            "tournament_name": "World Cup",
        },
        {
            "match_id": 1,
            "match_date": "2022-12-18T15:00:00+00:00",
            "tournament_name": "World Cup",
        },
    ]
    jsonl = tmp_path / "kicks.jsonl"
    _write_jsonl(jsonl, rows)
    report = validate_shootout_count(jsonl, rsssf_shootouts, LEAGUE_SEASONS_PREDICT_WINDOW)
    assert report.actual == 1  # 1 distinct match_id (the second row is a kick)


def test_skipped_refs_included_in_discrepancies(
    tmp_path: Path, rsssf_shootouts: list[object]
) -> None:
    """When the JSONL is short of the expected count, skipped_refs are
    serialised in discrepancies.json for debugging."""
    from penalty_pred.shootouts import ShootoutMatchRef

    rows = [
        {
            "match_id": 1,
            "match_date": "2022-12-18T15:00:00+00:00",
            "tournament_name": "World Cup",
        }
    ]
    jsonl = tmp_path / "kicks.jsonl"
    _write_jsonl(jsonl, rows)
    disc = tmp_path / "discrepancies.json"

    skipped = [
        ShootoutMatchRef(
            match_id=999,
            seo="x-vs-y",
            h2h="abc123",
            round_name="QF",
            home_name="X",
            away_name="Y",
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
    from penalty_pred.shootouts import ShootoutMatchRef

    rows = [
        {
            "match_id": 1,
            "match_date": "2022-12-18T15:00:00+00:00",
            "tournament_name": "World Cup",
        }
    ]
    jsonl = tmp_path / "kicks.jsonl"
    _write_jsonl(jsonl, rows)
    disc = tmp_path / "discrepancies.json"

    no_kicks = [
        ShootoutMatchRef(
            match_id=888,
            seo="a-vs-b",
            h2h="zzz999",
            round_name="SF",
            home_name="A",
            away_name="B",
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

    rows = [
        {
            "match_id": 1,
            "match_date": "2022-12-18T15:00:00+00:00",
            "tournament_name": "World Cup",
        }
    ]
    jsonl = tmp_path / "kicks.jsonl"
    _write_jsonl(jsonl, rows)
    report = validate_shootout_count(jsonl, [], [(77, 2022)])
    assert report.delta == 1  # 1 actual, 0 expected
    assert report.actual == 1
    assert report.expected == 0
