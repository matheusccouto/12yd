"""
Tests for the FotMob-shape coercion helpers in `fotmob_parsing`.

The helpers are the single source of truth for the three FotMob shape
quirks the scraper has to work around. A test that pins the behaviour
of one helper is the contract for every module that imports it.
"""

from __future__ import annotations

import pytest

from twelveyards.fotmob.fotmob_parsing import (
    SHOTMAP_EVENT_TYPE_TO_OUTCOME,
    coerce_int,
    parse_match_date,
)

# --- coerce_int -------------------------------------------------------------


@pytest.mark.parametrize("value", [None, ""])
def test_coerce_int_missing_returns_zero(value: object) -> None:
    """Missing values (None, empty string) coerce to 0."""
    assert coerce_int(value) == 0


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 0),
        (5, 5),
        (5.0, 5),
        (5.9, 5),
        ("5", 5),
        ("12345", 12345),
    ],
)
def test_coerce_int_numeric_value_passes_through(value: object, expected: int) -> None:
    """Clean numeric values (int, float, numeric string) coerce to that int."""
    assert coerce_int(value) == expected


@pytest.mark.parametrize("value", ["abc", "5.5.5", [], {}, object()])
def test_coerce_int_unparseable_returns_zero(value: object) -> None:
    """Unparseable values (non-numeric strings, weird objects) coerce to 0."""
    assert coerce_int(value) == 0


@pytest.mark.parametrize("value", [True, False])
def test_coerce_int_bool_returns_zero(value: bool) -> None:  # noqa: FBT001
    """
    Bools are not accepted as ints — returning 0 is the safe default.

    The FotMob payload occasionally has boolean-shaped fields where the
    scraper expects an int id. Accepting `True` as `1` would be a silent
    data bug; returning 0 makes the mis-typed field visible.
    """
    assert coerce_int(value) == 0


# --- parse_match_date -------------------------------------------------------


def test_parse_match_date_rfc_2822() -> None:
    """Match-detail pages return RFC 2822 (e.g. 'Sun, Dec 18, 2022, 15:00 UTC')."""
    assert (
        parse_match_date("Sun, Dec 18, 2022, 15:00 UTC")
        == "2022-12-18T15:00:00+00:00"
    )


def test_parse_match_date_iso_8601() -> None:
    """Fixture lists return ISO 8601 (e.g. '2022-12-18T15:00:00Z')."""
    assert parse_match_date("2022-12-18T15:00:00Z") == "2022-12-18T15:00:00+00:00"


def test_parse_match_date_iso_8601_with_offset() -> None:
    """An ISO 8601 string with a numeric offset is normalised to UTC."""
    assert parse_match_date("2022-12-18T10:00:00-05:00") == "2022-12-18T15:00:00+00:00"


@pytest.mark.parametrize("value", [None, ""])
def test_parse_match_date_missing_returns_empty(value: object) -> None:
    """Missing values return '' — the column is not always present."""
    assert parse_match_date(value) == ""


def test_parse_match_date_unparseable_returns_input() -> None:
    """Unparseable strings round-trip unchanged — the caller logs the raw value."""
    assert parse_match_date("not a date at all") == "not a date at all"


# --- SHOTMAP_EVENT_TYPE_TO_OUTCOME -----------------------------------------


def test_outcome_map_canonical_values() -> None:
    """The four canonical shotmap event types map to our three outcome labels."""
    assert SHOTMAP_EVENT_TYPE_TO_OUTCOME["Goal"] == "Goal"
    assert SHOTMAP_EVENT_TYPE_TO_OUTCOME["AttemptSaved"] == "Saved"
    assert SHOTMAP_EVENT_TYPE_TO_OUTCOME["Miss"] == "Missed"


def test_outcome_map_post_maps_to_missed() -> None:
    """`Post` (post hit without going in) is a subclass of Miss in our domain."""
    assert SHOTMAP_EVENT_TYPE_TO_OUTCOME["Post"] == "Missed"


def test_outcome_map_unknown_event_type() -> None:
    """Unknown event types fall through to the raw string at the call site."""
    # The map is the FALLBACK table; the call site is `dict.get(..., str(eventType))`.
    assert "BogusEvent" not in SHOTMAP_EVENT_TYPE_TO_OUTCOME
