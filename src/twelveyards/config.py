"""Pipeline configuration: scrape floor, train floor, lookback window.

PRD-v5: The scrape floor (2016-01-01) bounds which penalties go into
player_history.jsonl. The train floor (2021-01-01) bounds which kicks become
training/test rows. The 5-year lookback window is the rolling feature window
[T - 5y, T) applied to every target kick at time T.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

# Hard lower bound on Training Penalty dates for data ingestion.
# Penalties before this date are not fetched. Provides a 5-year buffer
# for the rolling feature window applied to the oldest training kicks
# (which start at the Train Floor of 2021-01-01).
SCRAPE_FLOOR: date = date(2016, 1, 1)

# Lower bound on kicks that become training/test rows.
# Only kicks on or after this date are used as supervised targets.
# Kicks between the Scrape Floor and the Train Floor exist purely as
# feature-history context for the oldest training rows.
TRAIN_FLOOR: date = date(2021, 1, 1)

# Rolling feature window in years: [T - 5y, T) for a target kick at time T.
LOOKBACK_WINDOW_YEARS: int = 5

# Deprecated aliases — keep until old modules are dropped.
HISTORY_FLOOR: date = SCRAPE_FLOOR
PREDICT_WINDOW_START: date = TRAIN_FLOOR

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
