"""Scraper configuration: prediction window, lookback window, history floor.

PRD: The 5-year Lookback Window and the 2016-01-01 History Floor are scraper
config, not dataset properties. Change the config; the dataset is reusable.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

# The current Prediction Window: 2021-01-01 through today.
PREDICT_WINDOW_START: date = date(2021, 1, 1)

# The current Lookback Window: 5 years before each target Shootout Kick's date.
LOOKBACK_WINDOW_YEARS: int = 5

# The current History Floor: 2016-01-01 (giving ≥5y of history for the oldest target).
HISTORY_FLOOR: date = date(2016, 1, 1)

# Default FotMob HTTP cache directory (relative to repo root by the caller).
DEFAULT_CACHE_DIR: str = "data/fotmob_cache"

# User-Agent per docs/fotmob.md. CloudFront returns 403 without it on some builds.
USER_AGENT: str = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

# Default HTTP request timeout (seconds).
HTTP_TIMEOUT_SECONDS: float = 15.0


def today_utc() -> date:
    """Today's date in UTC. Centralised so the scraper is timezone-consistent."""
    return datetime.now(tz=UTC).date()
