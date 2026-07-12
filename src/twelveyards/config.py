"""Pipeline configuration: scrape floor, train floor, lookback window."""

from __future__ import annotations

from datetime import UTC, date, datetime

SCRAPE_FLOOR: date = date(2016, 1, 1)
TRAIN_FLOOR: date = date(2021, 1, 1)
LOOKBACK_WINDOW_YEARS: int = 5

USER_AGENT: str = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

HTTP_TIMEOUT_SECONDS: float = 15.0


def today_utc() -> date:
    """Today's date in UTC."""
    return datetime.now(tz=UTC).date()
