"""Tournament scope: which leagues and seasons the scraper targets.

The current scope is six in-scope national-team cup competitions
(2021–2026) plus seven in-scope club cup competitions (2021–2026, Phase
3 — see `docs/adr/0004-phase-3-data-source.md`) plus the 2026 World
Cup roster. Adding a tournament edits this module — the scrape
scripts, the validation oracle, and the WC 2026 roster fetch all
import from here.
"""

from __future__ import annotations

from typing import Literal

from .leagues import CLUB_LEAGUE_IDS, LEAGUE_BY_ID, League

# Type alias: the `tournament_kind` attribute stamped on every
# `TrainingRow` and `PredictionRow` (Issue #51, Phase 3 schema change).
# The value is derived from the league registry at `build_features` time
# — the source of truth is `TOURNAMENT_KIND_BY_LEAGUE_ID` below.
TournamentKind = Literal["international", "club"]

# (FotMob league_id, season) pairs that fall inside the current
# Prediction Window (2021-01-01 → today). The seasons are the FotMob
# `?season=` year values (e.g. Euro 2020 is `season=2020` even though
# it was held in 2021). Held in a module-level constant so the
# orchestrator takes a single parameter and the slice is re-parameterisable
# by editing one line.
#
# Phase 3 (Issue #51): the 15 in-scope international pairs extend with
# the 7 × 6 = 42 in-scope club (league, season) pairs (Copa Libertadores,
# Champions League, FA Cup, Coupe de France, DFB-Pokal, Coppa Italia,
# Copa del Rey — each over FotMob seasons 2021–2026). The total scope
# is 57 pairs across 13 tournaments. The schema is identical to the
# in-scope international scope; the only new field is
# `tournament_kind` on `TrainingRow` (per the ADR).
INTERNATIONAL_PAIRS: tuple[tuple[int, int], ...] = (
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

# Club (league, season) pairs for Phase 3. Each of the 7 in-scope
# club leagues is registered over 6 FotMob seasons (2021–2026). The
# total is 42 pairs. The seasons are the FotMob `?season=` year
# values, mirroring the international convention.
CLUB_PAIRS: tuple[tuple[int, int], ...] = tuple(
    (league_id, season)
    for league_id in sorted(CLUB_LEAGUE_IDS)
    for season in (2021, 2022, 2023, 2024, 2025, 2026)
)

# The combined shootout scope. The scraper iterates this list. The
# validator's expected count comes from the per-pair RSSSF count.
LEAGUE_SEASONS_PREDICT_WINDOW: tuple[tuple[int, int], ...] = INTERNATIONAL_PAIRS + CLUB_PAIRS

# Per-(league_id, season) list of FotMob `?season=` values for each
# in-scope club league. Convenience for tests and future per-league
# seasons (e.g. if a club tournament holds an extra season in the
# window, the test can parametrize the new (league, season) pair
# without touching the orchestrator).
CLUB_LEAGUE_SEASONS: dict[int, tuple[int, ...]] = dict.fromkeys(sorted(CLUB_LEAGUE_IDS), (2021, 2022, 2023, 2024, 2025, 2026))

# The six section headings on the RSSSF page, mapped to FotMob league names.
# Used by the RSSSF parser to filter to only the 6 in-scope international
# tournaments (the page also lists the Confederations Cup, which is out
# of scope). Club tournaments use a separate map (`RSSSF_TO_CLUB_LEAGUE_NAME`)
# because the same RSSSF page does not list club shootouts; the club
# oracle is a future Phase 4 ADR-driven decision (per the v4 PRD §"Phase
# 3 — More Training Penalties" point 2: "Non-FotMob source").
RSSSF_TO_LEAGUE_NAME: dict[str, str] = {
    "World Cup": "World Cup",
    "European Nations' Cup": "Euro",
    "Copa América": "Copa América",
    "African Nations Cup": "Africa Cup of Nations",
    "Gold Cup": "CONCACAF Gold Cup",
    "Asian Nations Cup": "AFC Asian Cup",
}

# Heading map for the Phase 3 club shootout oracle. The current
# `docs/samples/rsssf_penaltiestour.html` snapshot does not list club
# shootouts; this map is the forward-looking shape for a future Phase
# 4 ingest (e.g. RSSSF's per-cup detail pages — Copa Libertadores,
# UCL, FA Cup, etc.). The keys are the RSSSF heading strings; the
# values are the FotMob league names. A future Phase 4 ADR can fill
# in the actual per-(league, season) RSSSF counts and update the
# `CLUB_EXPECTED_SHOOTOUT_COUNTS` test map (see Issue #50 ADR §"Per-
# tournament handling").
RSSSF_TO_CLUB_LEAGUE_NAME: dict[str, str] = {
    "Copa Libertadores": "Copa Libertadores",
    "UEFA Champions League": "Champions League",
    "FA Cup": "FA Cup",
    "Coupe de France": "Coupe de France",
    "DFB-Pokal": "DFB-Pokal",
    "Coppa Italia": "Coppa Italia",
    "Copa del Rey": "Copa del Rey",
}

# The `tournament_kind` lookup: given a `tournament_id` (FotMob
# leagueId), return the kind string for the `TrainingRow` /
# `PredictionRow` field. Default is `"international"` for any league
# not in the registry (a defensive fallback; in practice the
# in-scope scope never has a missing id).
TOURNAMENT_KIND_BY_LEAGUE_ID: dict[int, TournamentKind] = {
    league.league_id: "club" if league.kind == "club" else "international"
    for league in LEAGUE_BY_ID.values()
    if league.kind in ("international", "club")
}

# FotMob leagueId + slug for the 2026 FIFA World Cup.
WC_2026_LEAGUE: League = LEAGUE_BY_ID[77]

# The FotMob `?season=` value for the 2026 World Cup (start year of the season).
WC_2026_SEASON: int = 2026
