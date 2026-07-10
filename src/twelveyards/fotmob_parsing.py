"""Low-level FotMob shape coercions.

The scraper pulls payloads from `https://www.fotmob.com/_next/data/{buildId}/...`
and has to coerce a handful of fields the API serves inconsistently. These
helpers are the single source of truth for that coercion: every other module
imports from here, so a FotMob shape quirk is fixed in one place.

The three concerns are independent:

- `coerce_int(value)` — many FotMob ids arrive as strings on some endpoints
  and ints on others; we accept any of {None, "", 0, "5", 5.0, "abc"} and
  return a clean int (0 for missing/unparseable).
- `parse_match_date(value)` — `matchTimeUTC` is RFC 2822 on the match-detail
  page and ISO 8601 on the season-fixture list. We normalise to ISO 8601 UTC.
- `SHOTMAP_EVENT_TYPE_TO_OUTCOME` — the shotmap's `eventType` strings are
  mapped to our canonical "Goal" / "Saved" / "Missed" outcome labels. `Post`
  is FotMob's tag for shots that hit the post without going in (a miss in
  our domain — the keeper did not concede).
"""

from __future__ import annotations

from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

# shotmap eventType → canonical outcome label.
# `Post` is FotMob's tag for shots that hit the post without going in
# (a miss in our domain — the keeper did not concede). Off-target kicks
# (`isOnTarget=False`) with a non-zero `onGoalShot.x` are also `Post`
# when the shot clipped the post on its way wide. The PRD's
# "Shootout Kick" glossary covers Goals, Saves, and Misses; `Post` is
# a sub-class of Miss, so we map it to `Missed` for the canonical label.
SHOTMAP_EVENT_TYPE_TO_OUTCOME: dict[str, str] = {
    "Goal": "Goal",
    "AttemptSaved": "Saved",
    "Miss": "Missed",
    "Post": "Missed",
}


def coerce_int(value: Any) -> int:  # noqa: ANN401
    """Coerce any FotMob-shaped id to a clean int.

    Returns 0 for `None`, `""`, and unparseable values. Accepts ints, floats
    (truncated), and numeric strings. Bools are NOT accepted as ints —
    returning 0 for `True` would be a surprising data bug. Use `bool(value)`
    explicitly when you need a boolean coercion.
    """
    if value is None or value == "" or isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def parse_match_date(value: Any) -> str:  # noqa: ANN401
    """Coerce a FotMob matchTimeUTC to an ISO 8601 string (UTC, second precision).

    The match-detail page uses RFC 2822 dates like
    "Sun, Dec 18, 2022, 15:00 UTC". The season-fixture list uses ISO 8601
    (e.g. "2022-12-18T15:00:00Z"). We accept both forms and return the
    ISO 8601 form. Returns `""` when the value is missing or unparseable.
    """
    if not value:
        return ""
    text = str(value)
    try:
        return parsedate_to_datetime(text).astimezone(UTC).isoformat()
    except (TypeError, ValueError):
        pass
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).astimezone(UTC).isoformat()
    except ValueError:
        return text
