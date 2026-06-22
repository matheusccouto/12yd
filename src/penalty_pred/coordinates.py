"""L/C/R side bucketing from the continuous goal-mouth coordinate x.

`x` is in [0, 2] from the kicker's perspective: 0 = left post, 1 = centre, 2 = right post.
The side thresholds are the single source of truth for the bucketing and must be
imported everywhere a side is decided (PRD: "Deduplicate coordinate thresholds").
"""

from __future__ import annotations

LEFT_MAX: float = 0.667
RIGHT_MIN: float = 1.333

Side = str  # "L" | "C" | "R"
SIDE_LEFT: Side = "L"
SIDE_CENTER: Side = "C"
SIDE_RIGHT: Side = "R"


def side(x: float) -> Side:
    """Bucket a goal-mouth coordinate in [0, 2] to a side string."""
    if x < LEFT_MAX:
        return SIDE_LEFT
    if x > RIGHT_MIN:
        return SIDE_RIGHT
    return SIDE_CENTER
