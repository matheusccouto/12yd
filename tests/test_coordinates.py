"""Tests for the L/C/R coordinate bucketing."""

from __future__ import annotations

import pytest

from penalty_pred.coordinates import (
    LEFT_MAX,
    RIGHT_MIN,
    SIDE_CENTER,
    SIDE_LEFT,
    SIDE_RIGHT,
    side,
)


@pytest.mark.parametrize(
    ("x", "expected"),
    [
        (0.0, SIDE_LEFT),
        (0.1, SIDE_LEFT),
        (0.5, SIDE_LEFT),
        (LEFT_MAX - 1e-9, SIDE_LEFT),
        (LEFT_MAX, SIDE_CENTER),
        (1.0, SIDE_CENTER),
        (RIGHT_MIN, SIDE_CENTER),
        (RIGHT_MIN + 1e-9, SIDE_RIGHT),
        (1.5, SIDE_RIGHT),
        (2.0, SIDE_RIGHT),
    ],
)
def test_side_thresholds(x: float, expected: str) -> None:
    assert side(x) == expected


def test_thresholds_are_ordered() -> None:
    assert 0.0 < LEFT_MAX < 1.0 < RIGHT_MIN < 2.0
