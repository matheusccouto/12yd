"""Tests for the L/C/R coordinate bucketing."""

from __future__ import annotations

import pytest

from penalty_pred.coordinates import LEFT_MAX, RIGHT_MIN, side


@pytest.mark.parametrize(
    ("x", "expected"),
    [
        (0.0, "L"),
        (0.1, "L"),
        (0.5, "L"),
        (LEFT_MAX - 1e-9, "L"),
        (LEFT_MAX, "C"),
        (1.0, "C"),
        (RIGHT_MIN, "C"),
        (RIGHT_MIN + 1e-9, "R"),
        (1.5, "R"),
        (2.0, "R"),
    ],
)
def test_side_thresholds(x: float, expected: str) -> None:
    assert side(x) == expected


def test_thresholds_are_ordered() -> None:
    assert 0.0 < LEFT_MAX < 1.0 < RIGHT_MIN < 2.0
