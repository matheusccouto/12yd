"""Tournament scope: which leagues and seasons the scraper targets.

The current scope is six in-scope national-team cup competitions
(2021–2026) plus the 2026 World Cup roster. Adding a tournament
edits this module — the scrape scripts, the validation oracle, and
the WC 2026 roster fetch all import from here.
"""

from __future__ import annotations

from .leagues import LEAGUE_BY_ID, League

# (FotMob league_id, season) pairs that fall inside the current
# Prediction Window (2021-01-01 → today). The seasons are the FotMob
# `?season=` year values (e.g. Euro 2020 is `season=2020` even though
# it was held in 2021). Held in a module-level constant so the
# orchestrator takes a single parameter and the slice is re-parameterisable
# by editing one line.
LEAGUE_SEASONS_PREDICT_WINDOW: tuple[tuple[int, int], ...] = (
    (LEAGUE_BY_ID[77].league_id, 2022),  # World Cup 2022 (Qatar, Dec 2022)
    (LEAGUE_BY_ID[77].league_id, 2026),  # World Cup 2026 (in progress)
    (LEAGUE_BY_ID[50].league_id, 2020),  # Euro 2020 (held Jun–Jul 2021)
    (LEAGUE_BY_ID[50].league_id, 2024),  # Euro 2024 (Germany)
    (LEAGUE_BY_ID[44].league_id, 2021),  # Copa América 2021
    (LEAGUE_BY_ID[44].league_id, 2024),  # Copa América 2024
    (LEAGUE_BY_ID[289].league_id, 2021),  # AFCON 2021 (held Jan–Feb 2022)
    (LEAGUE_BY_ID[289].league_id, 2023),  # AFCON 2023 (held Jan–Feb 2024)
    (LEAGUE_BY_ID[289].league_id, 2025),  # AFCON 2025 (held Dec 2025–Jan 2026)
    (LEAGUE_BY_ID[298].league_id, 2021),  # Gold Cup 2021 (no shootouts)
    (LEAGUE_BY_ID[298].league_id, 2023),  # Gold Cup 2023
    (LEAGUE_BY_ID[298].league_id, 2025),  # Gold Cup 2025
    (LEAGUE_BY_ID[290].league_id, 2021),  # Asian Cup 2021 (held 2023, no shootouts)
    (LEAGUE_BY_ID[290].league_id, 2023),  # Asian Cup 2023 (held Jan–Feb 2024)
    (LEAGUE_BY_ID[290].league_id, 2025),  # Asian Cup 2025
)

# The six section headings on the RSSSF page, mapped to FotMob league names.
# Used by the RSSSF parser to filter to only the 6 in-scope tournaments
# (the page also lists the Confederations Cup, which is out of scope).
RSSSF_TO_LEAGUE_NAME: dict[str, str] = {
    "World Cup": "World Cup",
    "European Nations' Cup": "Euro",
    "Copa América": "Copa América",
    "African Nations Cup": "Africa Cup of Nations",
    "Gold Cup": "CONCACAF Gold Cup",
    "Asian Nations Cup": "AFC Asian Cup",
}

# FotMob leagueId + slug for the 2026 FIFA World Cup.
WC_2026_LEAGUE: League = LEAGUE_BY_ID[77]

# The FotMob `?season=` value for the 2026 World Cup (start year of the season).
WC_2026_SEASON: int = 2026
