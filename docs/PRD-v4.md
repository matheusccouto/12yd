# Penalty Shootout Prediction — v4 PRD

v4 follows v3 (`docs/PRD-v3.md`), which shipped dashboard simplification, model schema refactor, scraper diagnostics, and an independent model review across 7 closed issues. v3's audit (`docs/model-review.md`) and card (`docs/model-card-v3.md`) are the input.

This PRD is decision-rich but does not commit to specific file paths or code. The implementation is fully independent — the agent that picks up these issues makes its own decisions on ordering and trade-offs, and the issues are the durable task list.

## Problem Statement

Two gaps separate the deployed v3 product from the one we want to ship.

**1. The dashboard is not a useful tool for a goalkeeper during a match.** It is a 9-column `st.dataframe` table: Kicker, Team, Foot, P(L), P(C), P(R), Recommended Dive. The probabilities are kicker-PoV (per `CONTEXT.md`, `Side` is "the horizontal half of the goal from the kicker's perspective") but the "Recommended Dive" column reads as a keeper-PoV label to a human viewer — the L/R letter invites the viewer to re-anchor it to themselves. A goalkeeper looking at the page during a match has to mentally translate the L/R letter, look at a row of small numbers, and decide which way to dive, all while the match is happening. The visual goal-drawing card pattern used by tools like Safonov's penalty card (player photo + preferred foot + a 3-segment goal with the most-likely side highlighted) is a better fit for the use case.

**2. The training set is half-empty and the scraper is unreliable.** Of the 42 in-scope Shootout Kicks the RSSSF oracle expects, 18 are silently missing (FotMob URL rotation — the `(seo, h2h)` pair in the v2 season-fixture list has been re-assigned to a newer matchId, and four URL-lookup strategies tried so far have all failed) and 6 are empty-shotmap (FotMob returns no `pageProps.content.shotmap.shots` data and there is no fix without leaving FotMob). The 18 reachable Shootout Kicks per tournament-season on average × 10 kicks per Shootout Kick ≈ 180 missing Training Penalties. The training table has 179 rows today; recovering the 18 Shootout Kicks would roughly double it. 2023 has 0 kicks in the current data — a structural hole in the Training Penalty window. Without a reliable scraper, every model iteration is on a noisy foundation.

**3. The model work is ready to plan but blocked on data.** Issues #42 (drop the A1 rolling-side-counts block — the v3 review's ablation shows no measurable benefit) and #46 (anti-classifier — align the training loss with the deployment policy) are both open from the v3 review. They don't help until the data is solid; running them on 179 rows would repeat the same noise-floor analysis the review already did.

## Solution

Three phases, in strict dependency order. Each phase is a single issue on the tracker; the Ralph loop picks them up one at a time and is gated on the previous phase's acceptance criteria. No model work in v4.

### Phase 1 — Ship the new dashboard

Replace the current `st.dataframe` table with a card-based layout. The design is locked from `docs/prototype-card-layout.png` (the screenshot of the throwaway HTML prototype the user iterated on before signing off; the prototype code itself has been removed, the screenshot is the durable design artifact).

The card has four parts, all rendered with native Streamlit elements — no `st.markdown(unsafe_allow_html=True)` for cards, no custom JS, no third-party card components:

1. **A 64-px circular player photo placeholder** (a coloured SVG circle with the kicker's initials; real FotMob photos are out of scope for v4).
2. **The kicker's name, preferred foot, and career penalty count** on a single line, where the foot is a small coloured pill (left = pink, right = blue, both = yellow) and the penalty count is "N career penalties".
3. **A Plotly goal-drawing figure** rendered via `st.plotly_chart` — three equal-width rectangles (L, C, R) coloured by the kicker-PoV probability using a light-to-deep blue colormap, with a thick accent border and a `★` annotation on the most-likely Side. The figure is built inline in `app.py` as a small helper.
4. **A one-line prediction row**: "Kicker will aim: L 55%  ·  GK dive: R ↔" — kicker-PoV throughout, the dive hint is the opposite Side in the same frame.

The match selector moves to the sidebar as a single `st.selectbox`. The Sidebar has no round dropdown, no model/data block, no explainer paragraph above the cards, no legend. A `st.caption` at the bottom of the main area shows the artifact's last-updated timestamp.

Cards for kickers with no penalty history in the 5-year Lookback Window render with three near-equal light cells — the visual is honest about the absence of signal, no badge, no special flag.

**Kicker-PoV pivot (no model change):** the probabilities and the dive hint are both in the kicker-PoV frame, end to end. The "Recommended Dive" column in the current dashboard becomes "Kicker will aim: L 55%". A docstring + caption update on the existing `recommended_dive` function makes the kicker-PoV frame explicit; the function body and value are unchanged. A one-sentence frame pin in the model card and review documents the convention.

### Phase 2 — Scraper reliability

The dashboard is honest about the data we have, but the data is half-empty. Until the scraper is solid, every downstream iteration is on a noisy foundation. Phase 2 is the bottleneck of the project.

The investigation, in order:

1. **Diagnose the 18 URL-rotation failures concretely.** For each `stale_hash` ref, log the live `pageProps.matchId` and the URL the `(seo, h2h)` pair actually resolves to. Are all 18 re-assigned to a single newer matchId (a simple mapping table would fix it) or spread across many (a general URL-rotation handler is needed)? The existing `data/skipped_refs_diagnostics.jsonl` artifact (added in v3) carries the per-ref `failure_mode`; this step adds per-ref `live_match_id` and `resolved_url` for diagnosability.
2. **Try one more URL-lookup strategy (bounded).** Query the FotMob public page search endpoint with the original `(home_team_name, away_team_name, kickoff_date)` triple, returning the current `(seo, h2h)`. This is the 5th strategy attempted; if it also fails, **document the wall and stop** — five failed strategies is enough evidence that FotMob's URL rotation is opaque to us.
3. **Document the 6 empty-shotmap Shootout Kicks as untouchable.** For each, render the live FotMob match page, screenshot the `shotmap` block (empty), and attach the screenshot + per-Shootout-Kick explanation to the diagnostic file. These are accepted as FotMob data gaps.
4. **Update the success criterion.** The `validate_shootout_count` test pins the per-pair RSSSF count to `actual == expected`. With 6 empty-shotmap Shootout Kicks documented as out-of-scope, the expected count is `42 - 6 = 36`. The test updates to `actual == 36` and the per-pair `EXPECTED_SHOOTOUT_COUNTS` map adjusts.
5. **Add scraper instrumentation.** The existing `failure_mode` recording is good; extend it with retry counts, HTTP latency, ETag hits/misses, and a per-tournament success-rate summary written to a small CSV or JSONL artifact that the dashboard could (in a future cycle) surface.

The scraper work extends the existing shootout-orchestration module rather than introducing a new layer. The fix is one focused helper in the existing pipeline, not a re-architecture.

### Phase 3 — More Training Penalties (after Phase 2 is provably solid)

Only start Phase 3 once Phase 2's acceptance criteria hold. The intent is to roughly double the Training Penalty count and give the LOTO CV a tighter SE. Likely sources, in order of cost:

1. **Club shootouts** (Copa Libertadores, UCL knockout, domestic cup finals) — same FotMob client, new league registry, same extraction code path. Pro: consistent schema, no new client. Con: club Shootout Kicks are rarer and the in-game stakes differ from international tournaments, so the LOTO CV grouping may need a new attribute (`tournament_kind` ∈ {international, club}).
2. **Non-FotMob source** (StatsBomb Open Data, RSSSF detail pages) — covers the 6 empty-shotmap cases. Pro: closes the data gap entirely. Con: schema divergence, new client, new validator, new ADR.

Either path requires a new ADR (`docs/adr/000N-*.md`) that documents the source decision, the schema differences, and the per-tournament handling. New league registrations in the leagues module, new JSONL fields if needed, and a new entry in the tournament registry.

### v5 — Model work (deferred)

Issues #42 (drop A1) and #46 (anti-classifier) stay open and move to v5. They don't help until the data is solid; running them on 179 rows would repeat the same noise-floor analysis the v3 review already did. Issue #44 (per-keeper data) is closed as a v4 candidate — FotMob doesn't publish keeper dive direction. v5 will need its own PRD after v4 Phase 3 lands.

## User Stories

### Phase 1 — UI

1. As a Goalkeeper, I want the match selector in the sidebar as a single dropdown, so I can focus on one match at a time without scrolling.
2. As a Goalkeeper, I want each likely Kicker rendered as a card with their photo, name, preferred foot, and career penalty count, so I can scan the squad quickly before a Shootout.
3. As a Goalkeeper, I want the goal drawing on each card to show three equal-width segments (L, C, R) coloured by the Kicker-PoV probability, so I can see at a glance where the Kicker is most likely to aim.
4. As a Goalkeeper, I want the most-likely Side marked with a star and a thick accent border, so I don't have to read the percentages to find the prediction.
5. As a Goalkeeper, I want the prediction row in the Kicker-PoV frame ("Kicker will aim: L 55% · GK dive: R ↔"), so I don't have to mentally translate the L/R letter from the Kicker to myself.
6. As a Goalkeeper, I want a "last updated" caption at the bottom of the dashboard, so I know how fresh the prediction is.
7. As a Goalkeeper, I want a kicker with no penalty history to render with three near-equal light cells, so the dashboard is honest about the absence of signal — no special badge, no reassurance.
8. As a Goalkeeper, I want the page to render without HTML injection or custom JavaScript, so the dashboard is robust on the Streamlit Cloud deployment.
9. As a maintainer, I want the dashboard to use only native Streamlit elements (`st.container`, `st.columns`, `st.plotly_chart`, `st.markdown`, `st.caption`, `st.selectbox`), so the page is testable, debuggable, and upgrade-safe.
10. As a maintainer, I want the Plotly goal-drawing figure built inline in `app.py` as a small helper, so the dashboard module stays thin and the figure is unit-testable.
11. As a maintainer, I want the Kicker-PoV pivot applied to the column header, caption, and `recommended_dive` docstring, with the function body unchanged, so the dashboard reads consistently kicker-PoV without any model or HF artifact change.
12. As a maintainer, I want the Kicker-PoV frame pin documented in the model card and the model review, so a future maintainer doesn't accidentally re-frame to the Goalkeeper's PoV.
13. As a viewer, I want the deployed page at the Streamlit Cloud URL to render the new layout, so I can use the tool during a live match.
14. As a maintainer, I want the round dropdown, model block, data block, legend, and explainer paragraph removed from the dashboard, so the page is minimal and the visual goal drawing is self-explanatory.

### Phase 2 — Scraper reliability

15. As a maintainer, I want the 18 URL-rotation failures diagnosed concretely — each one points to a specific FotMob matchId that the `(seo, h2h)` pair now resolves to — so the fix is targeted, not a guess.
16. As a maintainer, I want one bounded attempt at a 5th URL-lookup strategy (the FotMob public page search), with a clear go/no-go decision after the attempt, so we don't grind indefinitely on a blocked path.
17. As a maintainer, I want the 6 empty-shotmap Shootout Kicks documented as untouchable (with screenshots of the live FotMob match page and per-Shootout-Kick explanations attached to the diagnostic file), so the discrepancy between expected and actual is explained and not silently tolerated.
18. As a maintainer, I want the `validate_shootout_count` test to pin `actual == 36` (the 42 RSSSF Shootout Kicks minus the 6 documented empty-shotmap cases), so the validator reflects the new reality and a regression to 18 / 30 / 34 is caught.
19. As a maintainer, I want a one-shot scraper re-run that completes in under an hour on a single connection (no manual retries), so the model retraining cycle is fast.
20. As a maintainer, I want per-tournament scraper success rate metrics written to a diagnostic artifact (CSV or JSONL), so I can spot regressions on individual tournaments in future runs.
21. As a maintainer, I want the scraper fix implemented in the existing shootout-orchestration module (no new layer), so the project's seam count doesn't grow for a feature that may end up small.
22. As a maintainer, I want Phase 2 to have a documented stop condition ("five strategies tried, no fix; this is the wall"), so the project moves to Phase 3 with honest data rather than an open-ended grind.

### Phase 3 — More Training Penalties

23. As a maintainer, I want a new ADR for the Phase 3 data source decision (club Shootout Kicks via FotMob, or a non-FotMob source like StatsBomb), so the choice is documented with the trade-offs explicit.
24. As a maintainer, I want new league registrations in the leagues module for any new FotMob-based source, so the pipeline picks them up automatically on the next run.
25. As a maintainer, I want the LOTO CV aggregate SE to drop by ≥ 30% after Phase 3, so the cross-tournament evaluation is statistically meaningful.
26. As a maintainer, I want a new `tournament_kind` attribute on the Training Row (international / club) if club Shootout Kicks are added, so the LOTO CV can group sensibly and a future analyst can compare international vs club behaviour.

## Implementation Decisions

- **The dashboard is rewritten in `app.py` as a thin Streamlit page over the existing `dashboard.py` library.** The library's `predictions_for_match`, `load_upcoming_knockouts`, `MatchContext`, and `KickerPrediction` are unchanged inputs. The change is in the rendering layer only.
- **The Plotly goal-drawing figure is built inline in `app.py` as a small helper** (a function that takes `p_L, p_C, p_R` and returns a `plotly.graph_objects.Figure` with 3 `add_shape` rectangles and 4 `add_annotation` calls — one per Side, plus a star on the most-likely). The figure is passed to `st.plotly_chart` per card. This keeps the dashboard seam at one level (the Streamlit page) and avoids a new module for what is, structurally, one rendering concern.
- **The card layout uses native `st.container(border=True)` for the card chrome**, with a `st.columns([1, 2.5])` layout for the meta block (left) and the goal + prediction block (right). The team-color stripe on the card's left edge is a thin border set on the container.
- **The Kicker-PoV pivot is purely cosmetic.** No model change, no scraper change, no training pipeline change, no HF artifact change. The `p_L`, `p_C`, `p_R` columns on `predictions.jsonl` are unchanged. The `recommended_dive` function (a `dashboard.py` helper) keeps its signature and its return value; only its docstring is updated to make the Kicker-PoV frame explicit. The "Recommended Dive" column in the current `st.dataframe` is replaced with the inline "Kicker will aim: L 55%" line.
- **The Sidebar is minimal.** Just the match selector (`st.selectbox` over the upcoming + both-teams-decided fixtures), the page title, and a small caption. No round, no model, no data block, no explainer.
- **The "last updated" timestamp comes from the artifact on HF.** A small read of the `predictions.jsonl` file's mtime (or a metadata field if the artifact exposes one) — surfaced via `st.caption` at the bottom of the main area.
- **The scraper fix extends the existing shootout-orchestration module** with one focused helper: `resolve_stale_hash(ref, *, search_endpoint) -> MatchRef | None`. The helper is bounded — it tries the 5th URL-lookup strategy once, returns `None` on failure, and the orchestrator records the `failure_mode` as before. No new top-level module, no new layer.
- **The success criterion is `actual_shootout_count == 36`**, not `42`. The 6 empty-shotmap cases are documented and excluded. The per-pair `EXPECTED_SHOOTOUT_COUNTS` map in the tournaments test adjusts accordingly.
- **HF push remains manual.** A successful re-run + `hf upload couto/12yd . couto/12yd` is the close condition.
- **No new runtime dependencies.** The Plotly figure uses the Plotly library that Streamlit already depends on; the scraper fix uses the existing HTTP client.
- **The Kicker-PoV frame is documented once** — in the existing `CONTEXT.md` glossary (where `Side` is already defined as Kicker-PoV) and reaffirmed in the model card and model review with a one-sentence pin. No new ADR; the glossary is the source of truth.
- **Phase 3 starts only after Phase 2's acceptance criteria hold.** No parallel work; the scraper has to be provably solid before adding more data sources that depend on it.
- **`docs/prototype-card-layout.png` is the source of truth for the card spec.** The agent implementing Phase 1 should mirror the screenshot's structure and styling (the colormap, the border, the star, the prediction line, the team-color stripe, the sidebar match selector, the last-updated caption) — the prototype is decision-rich and visual decisions should not be remade in code.

## Testing Decisions

**What makes a good test:**

- Test external behaviour, not implementation. A test that asserts "the dashboard shows 5 cards for team A and 5 for team B" is good; a test that asserts "the inner `st.columns` ratio is `[1, 2.5]`" is fragile.
- Independent ground truth where possible. RSSSF remains the Shootout-Kick-count oracle. The 2022 World Cup Final remains the per-kick ground truth.
- Tests cross the same seam as production. If the test surface differs from the production surface, the test is wrong.

**Phase 1 — UI:**

- A new test file (or extension of the existing dashboard test) asserts that the new dashboard rendering produces one card per Kicker, in the same order as the current `predictions_for_match` output (sorted by `total_penalties` descending, name as tiebreaker). The card count is 5+5 (Brazil vs France) or whatever the mock data is.
- A unit test on the inline Plotly goal-drawing helper asserts: the figure has 3 `shapes` (one per Side), 4 `annotations` (one per Side + the star), the most-likely shape has a `line.color` of the accent (and the others do not), the star annotation is positioned over the most-likely Side. The test runs without a browser — the figure is built in memory and asserted on its structure.
- A test on `dashboard.py`'s `recommended_dive` docstring asserts the docstring contains the phrase "Kicker-PoV" or equivalent — a cheap way to keep the frame-pin honest through future edits.
- Visual smoke: a manual screenshot of the deployed page, compared to the prototype screenshot. The agent takes the screenshot, attaches it to the issue, and the maintainer (the user) signs off before the issue closes.

**Phase 2 — Scraper reliability:**

- The 18 `stale_hash` refs each get a per-ref `live_match_id` and `resolved_url` field in `data/skipped_refs_diagnostics.jsonl`. A test asserts the artifact exists, has 18 rows, and each row has both new fields populated.
- The 6 empty-shotmap Shootout Kicks each get a per-Shootout-Kick record in the diagnostic file (or a separate `data/empty_shotmap_documentation.md` checked into the repo for visibility). A test asserts the file exists and has 6 records.
- The `validate_shootout_count` test pins `actual == 36` and the per-pair `EXPECTED_SHOOTOUT_COUNTS` map adjusts. The test runs against a fresh re-run; if the count is 18 / 30 / 34, the test fails (catching regressions).
- The per-tournament success rate artifact (CSV or JSONL) is exercised in a test that asserts every reachable Shootout Kick in the in-scope pairs is present and has a non-empty kick array.

**Phase 3 — More Training Penalties:**

- A new ADR is checked into `docs/adr/`. A test asserts the ADR file exists and is non-empty.
- The LOTO CV aggregate SE assertion is in the existing metrics report. A test pins the SE is ≤ 70% of the v3 SE (i.e. ≥ 30% reduction).
- If club Shootout Kicks are added, a new test asserts every Training Row has a `tournament_kind` attribute and the value is one of `{international, club}`.

**Modules touched across v4:**

- `app.py` (dashboard rendering — full rewrite of the table into cards)
- `src/penalty_pred/dashboard.py` (docstring update on `recommended_dive`)
- The shootout-orchestration module (Phase 2 — one focused helper)
- The tournaments test module (Phase 2 — `EXPECTED_SHOOTOUT_COUNTS` map adjusts to 36)
- `docs/model-card-v3.md` and `docs/model-review.md` (Kicker-PoV frame pin, one sentence each)
- `docs/PRD-v4.md` (this file)
- New: `docs/adr/000N-*.md` (Phase 3 source decision)

## Out of Scope

- **Model work.** Anti-classifier (issue #46), drop A1 (issue #42), per-keeper data (issue #44) — all deferred to v5. v4 is a data + UI cycle; running the model work on 179 rows would repeat the same noise-floor analysis the v3 review already did.
- **Real FotMob player photos.** The cards use coloured SVG circles with initials for v4. Real photos require a new image-fetch dependency and are a Phase 4 or v5 item.
- **A "low confidence" badge or warning on cards with no history.** The user explicitly rejected this — the dim cells are the honest signal.
- **Multi-language UI.** English only.
- **Light/dark mode toggle.** Use the Streamlit theme the user has set; no toggle.
- **A "winner prediction" for undecided knockout slots.** The dashboard hides matches until both teams are decided, unchanged from v3.
- **A per-Keeper Dive prior.** The model has no opponent or Keeper feature (issue #44 is closed; FotMob doesn't publish keeper dive direction). The Kicker-PoV prediction is the deployment policy; the per-keeper dimension is a v5+ item.
- **Multi-task heads (regression on `x` in addition to classification on Side).** Unchanged from v1, v2, v3.
- **CI / scheduled push to HF.** The push remains manual. Adding CI is a v5+ item.
- **AutoML / neural-net alternatives.** LightGBM is the model. The review's architecture alternatives are a v5+ discussion.
- **Phase 3 work before Phase 2's acceptance criteria hold.** Strict dependency order. The user explicitly said: "make sure we are able to scrape all data, then plan on how to have more data".
- **Group-stage matches.** The dashboard only shows matches with both teams decided, unchanged from v3.
- **A "Most likely" column on the `st.dataframe`.** The new layout doesn't use a `st.dataframe`; the card is the new unit. The text-style "Most likely" label inside the prediction row is the kicker-PoV headline; no separate column is rendered.

## Further Notes

- **v3 history.** v3 shipped 5 issues across 3 workstreams (dashboard simplify, model schema refactor, scraper diagnostics, independent review) and produced `docs/model-card-v3.md` and `docs/model-review.md`. The v3 PRD (`docs/PRD-v3.md`) is preserved for history; v4 references it.
- **The data and model artifacts are gitignored and live on HF.** Re-runs read the on-disk `data/fotmob_cache/` first, then fall back to live FotMob. The recovered Training Penalties (Phase 2) flow through the existing pipeline; the dashboard cards read `predictions.jsonl` directly, unchanged.
- **The Kicker-PoV pivot is the smallest possible change to the code that closes the gap between the data and the dashboard's reading.** Per `CONTEXT.md`, `Side` is "the horizontal half of the goal from the kicker's perspective" — the code is already Kicker-PoV throughout (`predictions.jsonl`, `coordinates.py`, the model output, `evaluate.py`'s `recommended_dive`). The only friction is the "Recommended Dive" column header inviting the viewer to re-anchor the L/R letter to themselves. The pivot is one column rename, one caption update, one docstring update, three one-sentence frame pins in the docs, and a new card layout that surfaces the Kicker-PoV prediction directly.
- **Why the design is locked from the prototype screenshot.** The user iterated on a throwaway HTML prototype across 3 variants (A, B, C) and a v2 simplification pass before signing off on the card layout. The prototype code has been removed; the screenshot at `docs/prototype-card-layout.png` is the durable design artifact. The card spec — photo + name + foot pill + 3 equal-width coloured segments + star on most-likely + one-line prediction + Kicker-PoV throughout — is in the screenshot. Re-deciding these in code would re-litigate decisions the user already made.
- **Why the scraper fix stops at the wall.** Four URL-lookup strategies have already been tried (public page search, per-team fixture list, direct match data API, page anchor) and all have failed. The 5th attempt (public page search with structured query) is bounded — if it fails, the project lives with 36/42 Shootout Kicks and documents it. The 6 empty-shotmap cases are a hard wall regardless: FotMob's live page simply has no data for them, and the only fixes (StatsBomb, RSSSF detail scraping) require leaving FotMob, which the user wants to defer to Phase 3.
- **Why the model work is v5, not v4.** The v3 review's headline finding is that the v3 model loses to random on the LOTO CV aggregate (0.374 vs 0.405, within one SE of 0.036). The two candidate fixes — drop A1 (#42) and anti-classifier (#46) — both reduce the model's degrees of freedom; running them on 179 rows would not have the statistical power to tell whether they worked. The LOTO CV aggregate SE scales as `1/sqrt(n)`; recovering 180 Training Penalties in Phase 2 brings the SE down by `sqrt(360/179) ≈ 1.42x`, which is what gives the v5 model work its statistical foundation.
- **Why "show the dim cells honestly" is the no-history UX.** The user explicitly rejected the "low confidence" badge option in the planning discussion. The reasoning: a badge is reassurance dressed as honesty; dim cells are honesty dressed as dim cells. The v3 review's "limitation, not a bug" recommendation matches this.
- **Why no `progress.txt`.** v1 used one; v2 removed it. v3 and v4 follow v2: the PRD is the durable plan, the issue tracker is the durable task list, the code is the durable state. The agent's session log is not committed.
- **The v4 cycle is a planning cycle, not a sprint.** The user said: "we are not implementing here, but just planning". The intent is to lock the plan before any code moves, so the implementation cycle (when it comes) has a clear brief.
- **The next v5 candidate.** v5 picks up #42 (drop A1), #46 (anti-classifier), and whatever data shape Phase 3 leaves. It will need its own PRD; v4 doesn't pre-write v5.

## References

- `docs/PRD-v3.md` — the v3 PRD (now complete)
- `docs/model-card-v3.md` — v3 model card (the Kicker-PoV frame pin is added here)
- `docs/model-review.md` — v3 independent review (the source of issues #40, #41, #42, #43, #45, #46; the Kicker-PoV frame pin is added here)
- `docs/PRD.md` — the v2 PRD (preserved for history)
- `CONTEXT.md` — domain glossary (the source of truth for `Side` being kicker-PoV)
- `docs/prototype-card-layout.png` — the locked-in card layout (the design source of truth)
- `docs/adr/0001-hugging-face-for-persistence.md`, `docs/adr/0002-streamlit-cloud-for-hosting.md`, `docs/adr/0003-single-hf-repo-with-subpaths.md` — existing ADRs
