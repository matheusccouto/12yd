"""Pipeline configuration: scrape floor, train floor, lookback window."""

from __future__ import annotations

from datetime import UTC, date, datetime

SCRAPE_FLOOR: date = date(2016, 1, 1)
TRAIN_FLOOR: date = date(2021, 1, 1)
LOOKBACK_WINDOW_YEARS: int = 5


def today_utc() -> date:
    """Today's date in UTC."""
    return datetime.now(tz=UTC).date()
