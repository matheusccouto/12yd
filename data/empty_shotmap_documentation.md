# Empty Shotmap Documentation

This file documents the **6 in-scope Shootout Kicks** where FotMob returns an
empty `pageProps.content.shotmap.shots` array — i.e. the match is listed in
the season fixture list as a shootout (status `penalties_short`) and the
response's `matchId` matches the expected matchId, but the `shotmap.shots`
list contains no `period == "PenaltyShootout"` entries.

These 6 cases are accepted as **FotMob data gaps**. The scraper's
`fetch_all_shootout_kicks_with_skips` reports each as
`FetchResult(no_kicks=True, failure_mode="empty_shotmap")` in
`data/skipped_refs_diagnostics.jsonl`. The only fix is to leave FotMob
(StatsBomb Open Data, RSSSF detail scraping), which is **Phase 3** of the
v4 PRD and a separate ADR-driven decision (Issue #50).

The 6 cases account for the divergence between the RSSSF oracle's 42
expected in-scope shootouts and the 36 reachable in-scope shootouts:

- 4 in AFCON 2021 (league_id 289, season 2021): RSSSF expects 6, FotMob
  reachable = 2 (subtracted: Burkina Faso vs Gabon, Mali vs Equatorial
  Guinea, Cameroon vs Egypt, Burkina Faso vs Cameroon).
- 2 in Asian Cup 2023 (league_id 290, season 2023): RSSSF expects 4, FotMob
  reachable = 2 (subtracted: Tajikistan vs UAE, Saudi Arabia vs South Korea).

`tests/test_tournaments.py::EXPECTED_SHOOTOUT_COUNTS` reflects the
reachable count (36 = 42 - 6). The `validate_shootout_count` test pins
`actual == 36` against a fresh re-run; a regression to 18 / 30 / 34 is
caught.

## The 6 cases

Each case has a `matchId` (the live FotMob matchId the (seo, h2h) resolves
to; documented here as `(unknown)` when the URL rotation makes the
historical lookup infeasible), the FotMob URL pattern, a `screenshot_path`
(where a future agent or human can save a screenshot of the live FotMob
match page showing the empty shotmap block), and a one-line explanation.

The matchId is `(unknown)` for cases where the scraper's (seo, h2h) has
been re-rotated by FotMob and the historical lookup is not feasible
without leaving FotMob. A future Phase 3 ingestion (Issue #51) can fill
these in from StatsBomb or RSSSF detail pages.

---

### 1. AFCON 2021 — Round of 16 — Burkina Faso vs Gabon

- **matchId:** `(unknown — see explanation)`
- **URL pattern:** `https://www.fotmob.com/matches/<seo>/<h2h>#<matchId>`
  (the live (seo, h2h) is reachable in `data/skipped_refs_diagnostics.jsonl`
  with `failure_mode="empty_shotmap"`)
- **screenshot_path:** `docs/screenshots/empty_shotmap/01_afcon2021_burkinafaso_gabon.png`
- **explanation:** FotMob has the match listed as a shootout (status
  `penalties_short` on the season fixture list) but the
  `pageProps.content.shotmap.shots` array is empty (no
  `period == "PenaltyShootout"` entries). RSSSF confirms a 7-6 shootout
  on 23 Jan 2022 (Burkina Faso won). FotMob data gap.

### 2. AFCON 2021 — Round of 16 — Mali vs Equatorial Guinea

- **matchId:** `(unknown — see explanation)`
- **URL pattern:** `https://www.fotmob.com/matches/<seo>/<h2h>#<matchId>`
- **screenshot_path:** `docs/screenshots/empty_shotmap/02_afcon2021_mali_equatorial_guinea.png`
- **explanation:** FotMob has the match listed as a shootout but the
  shotmap is empty. RSSSF confirms a 6-5 shootout on 26 Jan 2022
  (Equatorial Guinea won). FotMob data gap.

### 3. AFCON 2021 — Semi-final — Cameroon vs Egypt

- **matchId:** `(unknown — see explanation)`
- **URL pattern:** `https://www.fotmob.com/matches/<seo>/<h2h>#<matchId>`
- **screenshot_path:** `docs/screenshots/empty_shotmap/03_afcon2021_cameroon_egypt_sf.png`
- **explanation:** FotMob has the match listed as a shootout but the
  shotmap is empty. RSSSF confirms a 3-1 shootout on 3 Feb 2022
  (Egypt won, after a 0-0 draw). FotMob data gap.

### 4. AFCON 2021 — Third-place playoff — Burkina Faso vs Cameroon

- **matchId:** `(unknown — see explanation)`
- **URL pattern:** `https://www.fotmob.com/matches/<seo>/<h2h>#<matchId>`
- **screenshot_path:** `docs/screenshots/empty_shotmap/04_afcon2021_burkinafaso_cameroon_34.png`
- **explanation:** FotMob has the match listed as a shootout but the
  shotmap is empty. RSSSF confirms a 5-3 shootout on 5 Feb 2022
  (Cameroon won, after a 3-3 draw). FotMob data gap.

### 5. Asian Cup 2023 — Round of 16 — Tajikistan vs United Arab Emirates

- **matchId:** `(unknown — see explanation)`
- **URL pattern:** `https://www.fotmob.com/matches/<seo>/<h2h>#<matchId>`
- **screenshot_path:** `docs/screenshots/empty_shotmap/05_asiacup2023_tajikistan_uae.png`
- **explanation:** FotMob has the match listed as a shootout but the
  shotmap is empty. RSSSF confirms a 5-3 shootout on 28 Jan 2024
  (Tajikistan won, after a 1-1 draw). FotMob data gap.

### 6. Asian Cup 2023 — Round of 16 — Saudi Arabia vs South Korea

- **matchId:** `(unknown — see explanation)`
- **URL pattern:** `https://www.fotmob.com/matches/<seo>/<h2h>#<matchId>`
- **screenshot_path:** `docs/screenshots/empty_shotmap/06_asiacup2023_saudiarabia_southkorea.png`
- **explanation:** FotMob has the match listed as a shootout but the
  shotmap is empty. RSSSF confirms a 4-2 shootout on 30 Jan 2024
  (South Korea won, after a 1-1 draw). FotMob data gap.

---

## Why these 6 and not other 0-count pairs

The full 15-pair in-scope scope has 4 legitimately empty pairs at the
**season** level (RSSSF reports 0 shootouts for the pair):

- Gold Cup 2021 (298, 2021): 0 RSSSF shootouts — no shootouts happened.
- Asian Cup 2021 (290, 2021): 0 RSSSF shootouts — no shootouts happened.
- Asian Cup 2025 (290, 2025): 0 RSSSF shootouts — no shootouts happened.
- World Cup 2026 (77, 2026): 0 RSSSF shootouts — tournament in progress,
  RSSSF page snapshot is stale.

These 4 are out of scope for this issue — they are NOT data gaps, they
are correct zeros. The 6 documented cases are the **regressions**:
pairs where RSSSF expects a shootout, the season fixture list shows a
shootout, the (seo, h2h) resolves to the right matchId, but the
shotmap payload is empty.

## Schema

Each case in this file follows a fixed format with four fields:

- **matchId:** the FotMob matchId; `(unknown)` if the URL has rotated
- **screenshot_path:** where a screenshot of the live FotMob match page's
  empty shotmap block should be saved (non-empty placeholder; the file
  may not exist yet)
- **URL pattern:** the FotMob URL pattern with placeholders
- **explanation:** one-line explanation of why the shotmap is empty

The shape is parseable by `tests/test_tournaments.py::test_empty_shotmap_documentation_*`
and is human-readable for review.

## References

- Issue #49: `[v4] Document the 6 empty-shotmap shootouts — Phase 2 step 2`
- `docs/PRD-v4.md` Phase 2 step 3
- `docs/model-review.md` Topic 1.4 (the 86.6% no-history prediction rows,
  the noise floor the documentation helps characterise)
- `src/twelveyards/shootouts.py::FetchResult` (`no_kicks=True`,
  `failure_mode="empty_shotmap"`)
- `src/twelveyards/shootouts.py::write_skipped_refs_diagnostics`
  (the JSONL writer that records these in
  `data/skipped_refs_diagnostics.jsonl` on a fresh re-run)
- `tests/test_tournaments.py::EXPECTED_SHOOTOUT_COUNTS` (the per-pair
  reachable count; 36 reachable in-scope shootouts across the 15-pair
  scope)
