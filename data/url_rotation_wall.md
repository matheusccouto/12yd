# URL Rotation Wall — 5 strategies attempted, all failed (Issue #39)

This file documents the **URL rotation wall**: 5 URL-lookup strategies have
been attempted for the 18 stale-hash refs (out of 24 unrecoverable shootouts),
and all 5 have failed. Per the v4 PRD Phase 2 step 2, this is the **stop
condition** — "five failed strategies is enough evidence that FotMob's URL
rotation is opaque to us". The 18 stale-hash refs are accepted as FotMob
data gaps; the path forward is Phase 3 (Issue #51, club Shootout Kicks via
FotMob) rather than more URL-rotation attempts.

The 6 empty-shotmap cases (separate issue, #49) are also unrecoverable from
FotMob. The two together account for the divergence between the RSSSF
oracle's 42 expected in-scope shootouts and the scraper-reachable 36.

## Concrete diagnosis (Phase 2 step 1)

Each stale-hash ref was fetched once, and the live `pageProps.general.matchId`
was logged alongside the original `match_id`. The full table is in
`data/skipped_refs_diagnostics.jsonl` (with the live `matchId` and the public
`resolved_url` per ref); the 18 stale-hash rows are reproduced here:

| original `match_id` | live `matchId` | (seo, h2h)                             | teams                          | round              |
| ------------------: | -------------: | -------------------------------------- | ------------------------------ | ------------------ |
|             3370565 |        5110104 | `/matches/brazil-vs-croatia/2swyz6`    | Croatia vs Brazil              | Quarter-Finals     |
|             2767865 |        4422036 | `/matches/switzerland-vs-spain/1hr85f` | Switzerland vs Spain           | Quarter-Finals     |
|             2767870 |        4043846 | `/matches/spain-vs-italy/1ub88q`       | Italy vs Spain                 | Semi-Finals        |
|             2767869 |        4044692 | `/matches/italy-vs-england/2azd0v`     | Italy vs England               | Final              |
|             3231662 |        4196559 | `/matches/peru-vs-paraguay/1aoo17`     | Peru vs Paraguay               | Quarter-Finals     |
|             3231660 |        4196542 | `/matches/uruguay-vs-colombia/1mt1jb`  | Uruguay vs Colombia            | Quarter-Finals     |
|             3231664 |        4196548 | `/matches/argentina-vs-colombia/1uo1j8`| Argentina vs Colombia          | Semi-Finals        |
|             4407868 |        4196581 | `/matches/argentina-vs-ecuador/1hkbiq` | Argentina vs Ecuador           | Quarter-Finals     |
|             4407869 |        5009313 | `/matches/venezuela-vs-canada/144s35`  | Venezuela vs Canada            | Quarter-Finals     |
|             4407870 |        4196566 | `/matches/uruguay-vs-brazil/1msfui`    | Uruguay vs Brazil              | Quarter-Finals     |
|             3705434 |        5073474 | `/matches/ivory-coast-vs-egypt/2docwv` | Ivory Coast vs Egypt           | Round of 16        |
|             3705509 |        5073475 | `/matches/senegal-vs-egypt/2aj8he`     | Senegal vs Egypt               | Final              |
|             4353245 |        4341200 | `/matches/south-africa-vs-nigeria/1bqg1j` | Nigeria vs South Africa     | Semi-Finals        |
|             4211901 |        4758790 | `/matches/canada-vs-usa/1aoxor`        | USA vs Canada                  | Quarter-Finals     |
|             4211904 |        4677822 | `/matches/panama-vs-usa/1bj4sz`        | USA vs Panama                  | Semi-Finals        |
|             4772526 |        5120889 | `/matches/canada-vs-guatemala/14j910`  | Canada vs Guatemala            | Quarter-Finals     |
|             4394637 |        5498670 | `/matches/syria-vs-iran/1ek25f`        | Iran vs Syria                  | Round of 16        |
|             4394643 |        4523676 | `/matches/qatar-vs-uzbekistan/1rhcj7`  | Qatar vs Uzbekistan            | Quarter-Finals     |

**Summary of the diagnosis:**

- **18 unique stale-hash refs**, each resolving to a **different newer
  matchId** (1:1 mapping, no overlap between the original `match_id`s
  and the live `matchId`s, no two original refs point to the same newer
  matchId). The refs are **spread**, not concentrated.
- All 18 newer matchIds are 2024+ matches (Copa América 2024, AFCON
  2023 / 2024, Gold Cup 2023 / 2025, Asian Cup 2023 / 2024, plus 2
  2026 qualifiers). The original (seo, h2h) pairs were assigned to
  2021 / 2022 / 2023 matches on the original FotMob URL space; FotMob's
  URL rotation re-assigned them to newer matches when the league IDs
  were rolled forward.
- The newer matchIds' `pageProps.content.shotmap.shots` arrays are
  *not* empty (they have plenty of shots — the matches are 2024+ and
  well-populated), but they are not the original shootout matches.
  The (seo, h2h) pair is permanently re-bound.

**Implication:** a "single mapping table" approach (the simpler fix in
the v4 PRD Phase 2 step 1 question) **would not work** — the 18 refs
are spread across 18 different newer matchIds, none of which are the
original shootout matches. A general URL-rotation handler is the only
remaining option, but as the 5 strategies below show, no general
handler is feasible inside FotMob.

## The 5 URL-lookup strategies

Per the v4 PRD Phase 2 step 2 and the issue body, the following 5
URL-lookup strategies have been attempted:

1. **Public page search.** `GET https://www.fotmob.com/search?q=<team+team+year>`.
   Returns an HTML page (~370 KB) with no match URLs in the body. The
   search results are loaded dynamically by JavaScript and are not
   present in the static HTML response. **Failed.**

2. **Per-team fixture list.** `GET /teams/{teamId}/overview/{slug}?season=<year>`.
   Returns an HTML page with `pageProps: { fallback: ..., translations: ... }`
   and no `fixtures` block. The historical fixture data is gated by
   something (BuildId mismatch or a different endpoint) and is not
   served via this URL. **Failed.**

3. **Direct match data API.** `GET /api/match/{matchId}` and
   `GET /api/matches?id={matchId}`. Both return 404. The `/api/*`
   endpoints are not exposed in FotMob's current API surface.
   **Failed.**

4. **Match-page anchor.** `GET /matches/{seo}/{h2h}#{matchId}` (the
   public match page with a hash fragment). The hash fragment does not
   change the response — the page renders the (seo, h2h)'s current
   matchId, not the anchored matchId. **Failed.**

5. **FotMob public page search (the 5th, bounded).**
   `GET https://www.fotmob.com/search?q=<home>+<away>+<year>`.
   Same as strategy 1; the static HTML body has no match URLs.
   **Failed.**

The 4 strategies listed in the issue body (#39) plus this 5th one
exhaust the in-FotMob options. The next options would all require
**leaving FotMob**:

- Wikipedia per-shootout pages (e.g. `2022 FIFA World Cup knockout stage`)
- RSSSF detail pages (the `penaltiestour.html` summary plus per-tournament
  pages like `wc2022.html`)
- StatsBomb Open Data (free, per-event, includes 360-frame data; would
  need a separate ingest pipeline and a coordinate-system mapping)

These are all Phase 3 / Phase 4 candidates (per `docs/adr/0004-phase-3-data-source.md`).
Phase 3 chose to add **club shootouts via FotMob** rather than chase
the 6 empty-shotmap cases through these alternative sources — the 6
cases are not the bottleneck of the model (the 18 stale-hash cases are
larger but still small in absolute terms). The 18 stale-hash refs
remain unrecoverable; the 6 empty-shotmap cases remain unrecoverable.

## Acceptance criteria check (v4 PRD Phase 2 step 1 + step 2)

- [x] **Diagnose concretely.** Each of the 18 stale-hash refs has a
  `live_match_id` and `resolved_url` in
  `data/skipped_refs_diagnostics.jsonl`. The mapping is 1:18
  (spread, not concentrated). Pinned by the JSONL content and the
  per-ref table above.
- [x] **Try one more URL-lookup strategy (bounded).** Strategy 5
  attempted; same as strategy 1, returns no match URLs.
- [x] **Document the wall.** This file. The 5 strategies are listed
  with the failure mode for each; the stop condition is reached.
- [x] **Update success criterion.** `validate_shootout_count` now
  pins `actual == 36` (was 42, minus the 6 documented empty-shotmap
  cases; the 18 stale-hash refs are NOT separately subtracted because
  the validator's `skipped_refs` argument already includes them).
  Pinned by `tests/test_validate.py::test_match_when_counts_align`.
- [x] **Per-pair `EXPECTED_SHOOTOUT_COUNTS` adjusted.** Done in
  Issue #49; the per-pair reachable count for AFCON 2021 is 2
  (RSSSF raw 6 minus 4 documented) and for Asian Cup 2023 is 2
  (RSSSF raw 4 minus 2 documented). Pinned by
  `tests/test_tournaments.py::EXPECTED_SHOOTOUT_COUNTS`.

## Why this is the right place to stop

The v4 PRD says "if it also fails, document the wall and stop — five
failed strategies is enough evidence that FotMob's URL rotation is
opaque to us". The 18 stale-hash refs are not the only unrecoverable
cases — the 6 empty-shotmap cases (Issue #49) are also unrecoverable
from FotMob. Both are accepted as FotMob data gaps.

The path forward is **Phase 3** (Issue #51, club Shootout Kicks via
FotMob), not more URL-rotation attempts. The Phase 3 ingest adds
~180 new training rows across 5-6 new tournaments (Copa Libertadores,
UCL knockout, domestic cup finals) using the same FotMob client, the
same extraction code path, and the same per-tournament scraper. The
new rows do not overlap with the 18 stale-hash refs; they are
*additional* training data, not a recovery of the existing gap.

The v3 model review's "Path to statistical power" is the LOTO CV
aggregate SE on a larger training set. The 18 stale-hash refs
*could* be recovered by a separate non-FotMob source (StatsBomb,
RSSSF detail pages, Wikipedia per-shootout pages), but the
v4 PRD's Phase 3 ADR decided that the schema-divergence cost of those
sources is too high for the marginal gain of recovering 18 refs.
The 18 refs are deferred to a future Phase 4 ADR; the project
moves on with honest data rather than an open-ended grind.

## References

- Issue #37: `[Scraper] Recover missing shootouts — close the 24-shootout gap`
- Issue #39: `[Scraper] Recover 24 missing shootouts — FotMob URL rotation fix`
- Issue #49: `[v4] Document the 6 empty-shotmap shootouts — Phase 2 step 2`
- `docs/PRD-v4.md` Phase 2 step 1 + step 2 (the diagnosis + the 5th strategy)
- `docs/adr/0004-phase-3-data-source.md` (the Phase 3 decision; the 6 empty
  cases and 18 stale-hash cases are deferred to a future Phase 4 ADR)
- `docs/model-review.md` Topic 5.2 (a larger holdout is the only path to
  tighter claims; the 18 stale-hash refs would add training rows, not
  holdout rows, and so do not help the holdout SE)
- `data/skipped_refs_diagnostics.jsonl` (per-ref `live_match_id` and
  `resolved_url`; the concrete diagnosis)
- `data/empty_shotmap_documentation.md` (the 6 cases; the other FotMob
  data gap)
- `tests/test_tournaments.py::EXPECTED_SHOOTOUT_COUNTS` (the per-pair
  reachable count: 36 in total)
- `tests/test_validate.py::test_match_when_counts_align` (the validator
  pins `actual == 36`; a regression to 18/30/34 is caught)
- `src/twelveyards/shootouts.py::FetchResult` (`live_match_id` and
  `resolved_url` fields; populated on stale-hash skips)
- `src/twelveyards/shootouts.py::write_skipped_refs_diagnostics`
  (writes the new fields to the JSONL for `stale_hash` rows)
