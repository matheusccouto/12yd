"""
Tests for `parse_page_url` and `MatchRef.from_fixture`.

`parse_page_url` is the foundation for turning season-fixture entries
into per-match fetchers. `MatchRef.from_fixture` is the single source of
truth for parsing a FotMob fixture entry — both the shootout and roster
pipelines consume it.
"""

from __future__ import annotations

import pytest

from twelveyards.fotmob.match_ref import MatchRef, parse_page_url

# --- parse_page_url ---------------------------------------------------------


@pytest.mark.parametrize(
    ("page_url", "expected"),
    [
        (
            "/matches/japan-vs-croatia/2cq9vk#3370555",
            (3370555, "japan-vs-croatia", "2cq9vk"),
        ),
        (
            "/matches/argentina-vs-france/1hox8a#3370572",
            (3370572, "argentina-vs-france", "1hox8a"),
        ),
        (
            "/matches/argentina-vs-netherlands/1hklvd#3370566",
            (3370566, "argentina-vs-netherlands", "1hklvd"),
        ),
        (
            "/matches/switzerland-vs-spain/1hr85f#2767865",
            (2767865, "switzerland-vs-spain", "1hr85f"),
        ),
    ],
)
def test_parse_page_url_extracts_id_seo_h2h(  # noqa: D103
    page_url: str, expected: tuple[int, str, str],
) -> None:
    assert parse_page_url(page_url) == expected


def test_parse_page_url_no_anchor_raises() -> None:  # noqa: D103
    with pytest.raises(ValueError, match="anchor"):
        parse_page_url("/matches/argentina-vs-france/1hox8a")


def test_parse_page_url_bad_path_raises() -> None:  # noqa: D103
    with pytest.raises(ValueError, match="did not match"):
        parse_page_url("/something/else/1hox8a#3370572")


def test_parse_page_url_no_anchor_in_path_raises() -> None:  # noqa: D103
    with pytest.raises(ValueError, match="anchor"):
        parse_page_url("just-a-string")


def test_parse_page_url_non_integer_anchor_raises() -> None:
    """An anchor that is not an int match_id is rejected."""
    with pytest.raises(ValueError, match="match_id"):
        parse_page_url("/matches/argentina-vs-france/1hox8a#notanumber")


# --- MatchRef.from_fixture --------------------------------------------------


def test_from_fixture_populates_shootout_fields() -> None:
    """The full fixture is parsed into a `MatchRef` with all shootout fields."""
    fixture = {
        "pageUrl": "/matches/argentina-vs-france/1hox8a#3370572",
        "roundName": "Final",
        "home": {"id": "6706", "name": "Argentina"},
        "away": {"id": "9825", "name": "France"},
        "status": {"utcTime": "2022-12-18T15:00:00Z", "scoreStr": "3 - 3"},
    }
    ref = MatchRef.from_fixture(fixture)
    assert ref is not None
    assert ref.match_id == 3370572  # noqa: PLR2004
    assert ref.seo == "argentina-vs-france"
    assert ref.h2h == "1hox8a"
    assert ref.home_team_id == 6706  # noqa: PLR2004
    assert ref.home_team_name == "Argentina"
    assert ref.away_team_id == 9825  # noqa: PLR2004
    assert ref.away_team_name == "France"
    assert ref.round_name == "Final"
    assert ref.match_date == "2022-12-18T15:00:00Z"
    assert ref.score_str == "3 - 3"


def test_from_fixture_round_label_falls_back_to_round() -> None:
    """Some fixtures carry `round` only (e.g. older tournaments)."""
    fixture = {
        "pageUrl": "/matches/argentina-vs-france/1hox8a#3370572",
        "round": "F",
        "home": {"id": "6706", "name": "Argentina"},
        "away": {"id": "9825", "name": "France"},
        "status": {"utcTime": "2022-12-18T15:00:00Z", "scoreStr": "3 - 3"},
    }
    ref = MatchRef.from_fixture(fixture)
    assert ref is not None
    assert ref.round_name == "F"


def test_from_fixture_missing_page_url_returns_none() -> None:
    """A fixture without a `pageUrl` returns None — callers filter, not crash."""
    fixture = {
        "roundName": "Final",
        "home": {"id": "6706", "name": "Argentina"},
        "away": {"id": "9825", "name": "France"},
    }
    assert MatchRef.from_fixture(fixture) is None


def test_from_fixture_malformed_page_url_returns_none() -> None:
    """A fixture with an unparseable `pageUrl` returns None."""
    fixture = {
        "pageUrl": "garbage",
        "home": {"id": "1", "name": "A"},
        "away": {"id": "2", "name": "B"},
    }
    assert MatchRef.from_fixture(fixture) is None


def test_from_fixture_defaults_to_empty_when_status_missing() -> None:
    """A fixture with no `status` block has empty date/score strings."""
    fixture = {
        "pageUrl": "/matches/x-vs-y/abc#1",
        "roundName": "Group A",
        "home": {"id": "100", "name": "X"},
        "away": {"id": "200", "name": "Y"},
    }
    ref = MatchRef.from_fixture(fixture)
    assert ref is not None
    assert ref.match_date == ""
    assert ref.score_str == ""


def test_from_fixture_coerces_string_team_ids() -> None:
    """FotMob returns `home.id` and `away.id` as strings; the ref stores ints."""
    fixture = {
        "pageUrl": "/matches/x-vs-y/abc#1",
        "home": {"id": "12345", "name": "X"},
        "away": {"id": "67890", "name": "Y"},
    }
    ref = MatchRef.from_fixture(fixture)
    assert ref is not None
    assert ref.home_team_id == 12345  # noqa: PLR2004
    assert ref.away_team_id == 67890  # noqa: PLR2004


def test_from_fixture_coerces_unparseable_team_ids_to_zero() -> None:
    """A team id that cannot be parsed as int is stored as 0."""
    fixture = {
        "pageUrl": "/matches/x-vs-y/abc#1",
        "home": {"id": "garbage", "name": "X"},
        "away": {"id": "200", "name": "Y"},
    }
    ref = MatchRef.from_fixture(fixture)
    assert ref is not None
    assert ref.home_team_id == 0
    assert ref.away_team_id == 200  # noqa: PLR2004
