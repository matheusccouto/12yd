"""Integration tests for the 2022 FIFA World Cup Final shootout extraction.

Issue #17 acceptance: shootout_kicks.jsonl has exactly 8 rows, every row has
match_id = 3370572, x in [0, 2], side in {L, C, R}, the correct kicker names,
and the two missed kicks (Coman, Tchouaméni) have populated x and side.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from penalty_pred.shootouts import extract_shootout_kicks, write_jsonl

# The 8 kickers in canonical shootout order. (3 from France took penalties but
# Mbappé is counted once here; the actual 8-kicker list is below.)
KICKER_ORDER: list[tuple[int, str]] = [
    (701154, "Kylian Mbappé"),
    (30981, "Lionel Messi"),
    (429265, "Kingsley Coman"),
    (325916, "Paulo Dybala"),
    (914458, "Aurélien Tchouaméni"),
    (237606, "Leandro Paredes"),
    (823825, "Randal Kolo Muani"),
    (687008, "Gonzalo Montiel"),
]


def test_extract_returns_eight_kicks(sample_2022_final: Mapping[str, object]) -> None:
    kicks = extract_shootout_kicks(sample_2022_final)
    assert len(kicks) == 8


def test_all_kicks_have_match_id_3370572(sample_2022_final: Mapping[str, object]) -> None:
    for kick in extract_shootout_kicks(sample_2022_final):
        assert kick.match_id == 3370572


def test_x_in_unit_interval(sample_2022_final: Mapping[str, object]) -> None:
    for kick in extract_shootout_kicks(sample_2022_final):
        assert 0.0 <= kick.x <= 2.0


def test_side_in_lcr(sample_2022_final: Mapping[str, object]) -> None:
    for kick in extract_shootout_kicks(sample_2022_final):
        assert kick.side in {"L", "C", "R"}


def test_kickers_in_canonical_order(sample_2022_final: Mapping[str, object]) -> None:
    kicks = extract_shootout_kicks(sample_2022_final)
    assert [(k.kicker_id, k.kicker_name) for k in kicks] == KICKER_ORDER


def test_coman_and_tchouameni_have_populated_x(
    sample_2022_final: Mapping[str, object],
) -> None:
    """PRD: the two MissedPenalty entries (Coman, Tchouaméni) must have x and
    side recovered from shotmap.shots (not from penaltyShootoutEvents, which
    is missing shotmapEvent for non-Goals)."""
    kicks = extract_shootout_kicks(sample_2022_final)
    by_id = {k.kicker_id: k for k in kicks}
    for player_id, _name in (KICKER_ORDER[2], KICKER_ORDER[4]):
        kick = by_id[player_id]
        assert kick.outcome in {"Saved", "Missed"}
        assert 0.0 <= kick.x <= 2.0
        assert kick.side in {"L", "C", "R"}


def test_coman_is_saved_and_tchouameni_is_missed(
    sample_2022_final: Mapping[str, object],
) -> None:
    kicks = extract_shootout_kicks(sample_2022_final)
    by_id = {k.kicker_id: k for k in kicks}
    assert by_id[429265].outcome == "Saved"
    assert by_id[914458].outcome == "Missed"


def test_tchouameni_off_target_clamped_to_post(
    sample_2022_final: Mapping[str, object],
) -> None:
    """Tchouaméni's miss was off-target; x is clamped to the post (0 or 2)."""
    kicks = extract_shootout_kicks(sample_2022_final)
    tch = next(k for k in kicks if k.kicker_id == 914458)
    assert tch.is_on_target is False
    assert tch.x in (0.0, 2.0)


def test_pen_score_walks_correctly(sample_2022_final: Mapping[str, object]) -> None:
    """Argentina 4-2 France at the end; the running score should reach [4, 2]."""
    kicks = extract_shootout_kicks(sample_2022_final)
    last = kicks[-1]
    assert last.pen_score_after == [4, 2]


def test_match_score_3_3(sample_2022_final: Mapping[str, object]) -> None:
    """The 2022 WC Final ended 3-3 after extra time, before the shootout."""
    for kick in extract_shootout_kicks(sample_2022_final):
        assert kick.match_score_home == 3
        assert kick.match_score_away == 3


def test_side_distribution_matches_docs(sample_2022_final: Mapping[str, object]) -> None:
    """docs/fotmob.md line 166: 4L / 0C / 2C / 0R for the 6 goals + 2L for the misses
    = 6L / 2C / 0R across 8 kicks."""
    kicks = extract_shootout_kicks(sample_2022_final)
    distribution = {"L": 0, "C": 0, "R": 0}
    for kick in kicks:
        distribution[kick.side] += 1
    assert distribution == {"L": 6, "C": 2, "R": 0}


def test_kick_number_runs_1_to_8(sample_2022_final: Mapping[str, object]) -> None:
    kicks = extract_shootout_kicks(sample_2022_final)
    assert [k.kick_number for k in kicks] == list(range(1, 9))


def test_tournament_metadata(sample_2022_final: Mapping[str, object]) -> None:
    kicks = extract_shootout_kicks(sample_2022_final)
    for kick in kicks:
        assert kick.tournament_id > 0
        assert kick.tournament_name == "World Cup"
        assert kick.round == "final"
        assert kick.match_date.startswith("2022-12-18")


def test_is_home_flips_per_team(sample_2022_final: Mapping[str, object]) -> None:
    kicks = extract_shootout_kicks(sample_2022_final)
    argentina_team_id = 6706
    france_team_id = 6723
    for kick in kicks:
        if kick.team_id == argentina_team_id:
            assert kick.is_home is True
        elif kick.team_id == france_team_id:
            assert kick.is_home is False
        else:
            pytest.fail(f"Unexpected team_id: {kick.team_id}")


def test_write_and_read_jsonl_roundtrip(
    sample_2022_final: Mapping[str, object], tmp_path: Path
) -> None:
    kicks = extract_shootout_kicks(sample_2022_final)
    out = tmp_path / "kicks.jsonl"
    n = write_jsonl(out, kicks)
    assert n == 8
    # Sanity: every line is valid JSON and has the expected fields.
    with out.open() as f:
        for line in f:
            row = json.loads(line)
            assert row["match_id"] == 3370572
            assert "x" in row
            assert "side" in row
            assert "outcome" in row


def test_write_jsonl_one_record_per_line(
    sample_2022_final: Mapping[str, object], tmp_path: Path
) -> None:
    out = tmp_path / "kicks.jsonl"
    write_jsonl(out, extract_shootout_kicks(sample_2022_final))
    with out.open() as f:
        non_empty = [line for line in f if line.strip()]
    assert len(non_empty) == 8
