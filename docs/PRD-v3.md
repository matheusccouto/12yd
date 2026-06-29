# Penalty Shootout Prediction — v3 PRD

v3 continues from the v2 PRD (`docs/PRD.md`), which shipped Phase 0 (codebase simplification), Phase 1 (Hugging Face persistence), and Phase 2 (Streamlit dashboard) across 34 closed issues. v3 addresses three issues found in the deployed app and the published model card.

The implementation is fully independent — the agent that picks up these issues makes its own decisions, including scope, ordering, and trade-offs. The PRD and issues below are decision-rich enough to be self-contained.

## Problem Statement

Three issues are visible today in the deployed system at https://twelveyards.streamlit.app/ and on the published model card at https://huggingface.co/couto/12yd:

1. **The dashboard's match selector is empty even though the World Cup 2026 knockout stage has started.** Today (2026-06-29) there are 2 Round of 32 matches with real teams (Brazil vs Japan, Germany vs Paraguay) and 15 more in the next 6 days. The dashboard's filter at `src/penalty_pred/dashboard.py:64` defines `KNOCKOUT_ROUNDS = frozenset({"1/8", "1/4", "1/2", "final"})` — it predates the 48-team format, which adds a Round of 32 (FotMob round code `"1/16"`) as the first knockout round. Every R32 match is dropped before the team-decided check. The user-facing message is "No upcoming knockout matches with both teams decided" — the empty state is correct given the current code, but the code itself is wrong: at this stage of the tournament, every upcoming match is a knockout match. The fix is to drop the round allowlist entirely and rely on the `is_placeholder_team` check alone, which already correctly filters out R16+ matches whose teams are still placeholders.

2. **`preferred_foot` is not read from FotMob's payload.** 1080 of 1247 players in `data/predictions.jsonl` have `kicking_foot="Unknown"`. The current value is *inferred* from the mode of each kicker's penalty `shotType` history (`src/penalty_pred/features.py:242-258`), so any player with no senior-team penalty history in the 5-year lookback window gets `"Unknown"`. The FotMob `__next/data/.../players/{id}/{slug}.json` payload already includes the declared preferred foot in `pageProps.data.playerInformation[]` (key `preferred_foot`, value `left`/`right`/`both`). The scraper already fetches and caches this payload; it just never reads the field. The model also has a `b3_round` feature (`src/penalty_pred/model.py:129`) that is round-specific but has never been trained on an R32 round (none of the 6 in-scope tournaments — WC, Euro, Copa América, AFCON, Gold Cup, Asian Cup — has had an R32). When the dashboard starts passing `b3_round="1/16"`, LightGBM treats it as missing and the model falls back to the training prior — a fragility we want to remove by dropping the feature entirely.

3. **The training set is half the size it should be.** The published `data/shootout_kicks.jsonl` has 179 kicks across 18 shootouts. The verification oracle (`data/discrepancies.json`) reports the expected count is 42 shootouts — the scraper is missing 24. The 18 "skipped refs" are concentrated at QF/SF/Final rounds (9 QF, 4 SF, 2 Final, 3 R16). The remaining 6 missing shootouts are from tournaments that aren't in the current scope (AFCON 2021, Copa América 2021, Gold Cup 2021, Gold Cup 2023, Asian Cup 2023 are all plausible). At ~10 kicks per shootout, recovering the missing 24 would roughly double the training set (~180 → ~360+ kicks) and meaningfully improve the model on the 28-row holdout. The model card's `0.214` accuracy is also misleading on a 28-row holdout (SE ≈ 0.09) — the actual deployment KPI is **counterfactual save rate** (the GK dives to the side with the lowest predicted probability, not the highest), where the model is the strongest of all reported baselines.

The model is functional. The dashboard works. The data is mostly right. v3 is the simplification pass that turns a 19-feature model with 2 inferred attributes and a 4-round allowlist into a simpler, more honest system that survives the live tournament and the next one.

## Solution

Four workstreams, in suggested dependency order. Issues 1, 2, and 3 are independent of each other; Issue 4 is the audit and depends on the others.

### 1. Dashboard match selector — drop the round allowlist

Drop `KNOCKOUT_ROUNDS` entirely. `load_upcoming_knockouts` becomes a single filter on `is_placeholder_team` plus the existing upcoming-time check. The 4-round allowlist is over-specific (it lists rounds 1/8, 1/4, 1/2, final — and misses 1/16, the new R32). Once dropped, the selector automatically adapts to whatever the tournament currently is: during R32, it shows the 15 R32 matches with real teams; after R32 ends and R16 slots are decided, it shows those; and so on. The `is_placeholder_team` function (`src/penalty_pred/dashboard.py:119-143`) already handles all the placeholder shapes (empty name, `Winner ...` / `Loser ...`, `/`-joined group opponents, id=0).

Bonus simplification: the re-score path in `dashboard.py` (`predict_match`, `_roster_for_match`, the per-match inference) is no longer needed once the model refactor drops `b3_round` — every match shows the same per-kicker probabilities from `predictions.jsonl`, so the dashboard can read the artifact directly. (This bonus falls out of Issue 2; it's mentioned here for context.)

### 2. Model schema refactor — drop `b3_round`, swap `kicking_foot` → `preferred_foot`

- **Drop `b3_round` from the feature schema.** Remove it from `NUMERIC_FEATURES` / `CATEGORICAL_FEATURES` in `src/penalty_pred/model.py`, from `FeatureRow` in `src/penalty_pred/features.py`, and from the feature build path. The model's only round-specific feature goes away; every prediction becomes round-agnostic. The dashboard's re-score path is then a no-op (passes the same `predictions.jsonl` numbers regardless of match) and can be deleted.
- **Read `preferred_foot` from the existing cached player-page JSON.** Extend `extract_player_metadata` in `src/penalty_pred/player_history.py` to scan `pageProps.data.playerInformation[]` for `translationKey="preferred_foot"` and store it on `PlayerMetadata`. No re-scraping needed — the data is already on disk at `data/fotmob_cache/players/{id}/{slug}.json`.
- **Replace `kicking_foot` with `preferred_foot` in the model schema and `predictions.jsonl`.** The A3 feature is renamed; the inferred value goes away. The dashboard's per-kicker table shows the new column (or keeps the same name with the new meaning — pick one and document it). 1080 of 1247 rows that were `"Unknown"` will get real values.
- **Retrain and re-publish.** Refit LightGBM on the new 18-feature schema, refit the logistic-regression baseline on the same, regenerate `predictions.jsonl`, push the new `model/lightgbm.pkl` and `model/metrics.json` to HF.
- **Update the model card on HF** to lead with **save rate** as the headline KPI, with a one-paragraph caveat that 0.214-style top-1 accuracy on a 28-row holdout is statistically uninformative (standard error ≈ 0.09) and that the deployment policy is `argmin` (dive the lowest-probability side), not argmax. The card's table can keep top-1 for completeness, but the lede changes.

Apply the `/improve-codebase-architecture` skill on the affected modules (`features.py`, `model.py`, `predict.py`, `player_history.py`, `dashboard.py`) during the refactor — look for deepening opportunities (e.g. the feature schema, the predict entry point, the re-score collapse) and present them as the work unfolds.

### 3. Scraper data recovery — close the 24-shootout gap

Recover the 24 shootouts the scraper missed. Two failure modes are visible from `data/discrepancies.json`:

- **18 skipped refs** — specific matches the scraper attempted but failed on. Diagnose the root cause (most likely: round-name normalization for QF/SF/Final doesn't match what `fetch_all_shootouts` expects, or the `pageProps.content.shotmap.shots` array is empty for some reason). Fix and re-run. Each fix should be auditable in a test.
- **6 missing tournaments** — tournaments that aren't even in the current scope. Likely candidates: AFCON 2021 (Jan-Feb 2022), Copa América 2021 (Jun-Jul 2021), Gold Cup 2021 (Jul-Aug 2021), Gold Cup 2023 (Jun-Jul 2023), Asian Cup 2023 (Jan-Feb 2024), Asian Cup 2024. Add them to the tournament config in `src/penalty_pred/tournaments.py` (the `LEAGUE_SEASONS_PREDICT_WINDOW` map) and re-run the pipeline.

Target: `actual_shootout_count` matches `expected_shootout_count` (42 → 42). After the data is recovered, retrain the model and regenerate `predictions.jsonl` (the same training step as Issue 2's retrain — coordinate so it's one training run, not two). The model card's `n_train` and `n_holdout` will update.

### 4. Independent, unbiased model review

A fresh agent (not the one who built or refactored the model) reviews the model end-to-end after Issues 1–3 land. The review is published to `docs/model-review.md` and covers:

- **Train/serve skew** — does any feature the model sees at inference time have a different distribution from training? (The 18-feature schema post-refactor is the test surface.)
- **Feature engineering** — are the A1 (rolling side counts), B1/B2 (in-shootout context), C1/C2 (player attributes) features doing useful work, or are some of them noise on a 360-row training set?
- **Metric choice** — is save rate the right headline? Are there better calibration metrics (Brier, ECE) we should also report?
- **Baseline comparison** — class-conditional prior, last-side mode, logreg, the new model. Is the new model actually adding value over a much simpler baseline?
- **Holdout statistical power** — at n=28, the SE on accuracy is ≈ 0.09. What does the model need to do to make the headline number statistically meaningful? (Probably: more data, which Issue 3 helps with.)
- **Architecture alternatives** — would a small neural net or a different gradient-boosting library do better? Is the `argmin` deployment policy still optimal?

The review is a written report with concrete recommendations. It can suggest follow-up issues (label `ready-for-agent`) but does not commit to implementing them.

## User Stories

### Issue 1 — Dashboard match selector

1. As a viewer, I want the match selector to show every upcoming match with both teams decided, regardless of round, so the dashboard is useful from R32 through the final, including any future tournament format changes.
2. As a maintainer, I want the match filter to be a single check on team-decided-ness plus the upcoming-time check, so the code is shorter and the round-taxonomy knowledge lives in one place (the placeholder detector), not duplicated in a round allowlist.
3. As a viewer, I want the empty state message to be removed (or made conditional on a "no real teams yet" check), so the page doesn't say "the dashboard will populate" when there are 2 matches kicking off in 3 hours.

### Issue 2 — Model schema refactor

4. As a maintainer, I want `b3_round` removed from the model schema, so the model has no unseen categories at inference time and the `argmin` deployment policy is a stable function of the per-kicker probabilities alone.
5. As a maintainer, I want `preferred_foot` read from the cached FotMob player payload and stored on `PlayerMetadata`, so the next slice pipeline run produces a dataset where every player has a declared foot (no more 86% "Unknown" rows).
6. As a maintainer, I want `kicking_foot` (inferred) replaced with `preferred_foot` (declared) in the model's A3 feature and in `predictions.jsonl`, so the dataset has one consistent semantic for "which foot does this player use".
7. As a maintainer, I want the dashboard's re-score path deleted, so the dashboard reads `predictions.jsonl` directly and the per-match inference machinery (28 lines for a no-op) is gone.
8. As a viewer, I want the model card on Hugging Face to lead with save rate, so a reader sees the deployment KPI first, not the noisy top-1 number on a 28-row holdout.
9. As a maintainer, I want the new model trained on the recovered data (Issue 3) in a single retrain run, so the published artifact reflects the new schema and the larger training set at the same time.
10. As a maintainer, I want `/improve-codebase-architecture` applied to the affected modules during the refactor, with any deepening opportunities either folded into the PR or filed as follow-up issues, so the simplification theme is consistent.

### Issue 3 — Scraper data recovery

11. As a maintainer, I want the 18 skipped refs in `data/discrepancies.json` to be diagnosable — each one should point to a specific failure mode (round-name mismatch, missing shotmap, parse error, etc.), so the fix is targeted, not a guess.
12. As a maintainer, I want the 6 missing-tournament shootouts added to the in-scope tournaments in `src/penalty_pred/tournaments.py`, so the next pipeline run includes them and the 5.5-year training window is dense, not gappy (currently 2023 has 0 kicks).
13. As a maintainer, I want `validate_shootout_count` to pass (`actual == expected == 42`), so the discrepancies file is empty and the next time the pipeline runs, the verification oracle confirms the data is complete.
14. As a maintainer, I want the recovered data to flow through the new 18-feature schema (Issue 2) in one retrain, so the published artifact and the new training set land together.

### Issue 4 — Independent model review

15. As a project owner, I want a written review of the model that questions the train/serve consistency, the feature engineering, the metric choice, and the baseline comparison, so a future maintainer has a third-party assessment of what the model is good at and where it's fragile.
16. As a project owner, I want the review to be honest about the holdout's statistical power, so the model card's claims (and the dashboard's marketing) are calibrated to what 28 rows of 2026 holdout can actually tell us.
17. As a project owner, I want the review to recommend concrete next steps, label them as follow-up issues (`ready-for-agent`), and leave them for future cycles, so the v3 work is a foundation, not a final state.

## Implementation Decisions

- **The dashboard's match selector becomes one filter, not three.** `is_placeholder_team` is the only "is this match ready" check; `kickoff > now` is the only "is this match upcoming" check; the round is a display attribute, not a filter. The `MatchContext` dataclass keeps the `round` field (so the table can still say "Round of 32" or "Quarter-finals"), but the filter ignores it.
- **`b3_round` is dropped, not retrained with `1/16` as a new category.** Retraining would require synthesizing R32 data (which doesn't exist in any of the 6 in-scope tournaments). Dropping is the only honest option; the model's round-specificity is small on a 360-row training set anyway.
- **`preferred_foot` and `kicking_foot` are not both kept in the dataset.** The user explicitly chose to replace, not add. The semantics of the A3 feature change from "which foot did the kicker use for past penalties" to "the kicker's declared preferred foot". Retraining on the new feature is required.
- **The retrain happens once**, after both Issues 2 and 3 land. Two retrain runs are wasted work; one retrain on the 18-feature schema, on the recovered 42-shootout data, with a single new holdout cutoff (2026-01-01 is fine; the holdout is unchanged), is the right shape.
- **The HF model card is a markdown file** (`model/README.md` on `couto/12yd`), not the YAML frontmatter. The headline paragraph leads with save rate; the table has accuracy, log loss, save rate, n_kicks, and the four baselines (random, last-side, logreg, lightgbm).
- **The independent review is delivered as `docs/model-review.md`**, not in the model card. The model card stays short (one screen); the review is a separate, deeper document.
- **No new runtime dependencies.** The `preferred_foot` read uses the same JSON path the scraper already parses; the model schema drop removes code, not adds it.
- **HF push remains manual.** A successful retrain + regenerate + `hf upload couto/12yd . couto/12yd` is the close condition. The Ralph loop can run the upload.
- **PRD is the durable plan; issues are the durable task list.** The agent does not need to maintain its own notes; the issue's `## Further Notes` section is where any mid-cycle thoughts go.

## Testing Decisions

**What makes a good test (unchanged from v1, v2):**

- Test external behaviour, not implementation. A test that asserts "predictions sum to 1" is good; a test that asserts "the column order is exactly this list" is fragile.
- Independent ground truth where possible. RSSSF remains the shootout-count oracle. The 2022 World Cup Final remains the per-kick ground truth.
- Tests cross the same seam as production. If the test surface differs from the production surface, the test is wrong.

**Issue 1 — Dashboard:**

- `test_dashboard.py` keeps its `is_placeholder_team` parametrization and gains a parametrized test for `load_upcoming_knockouts` over the four placeholder shapes (`"Winner N"`, `"Loser N"`, `"A/B"`, empty name) with multiple round codes (`"1/16"`, `"1/8"`, `"1/4"`, `"1/2"`, `"final"`). All of them should be dropped — the round is no longer a filter.
- The empty-state UI test pins the message to a "no real-team matches in the next 30 days" condition, not a "this round isn't in the allowlist" condition.
- The 28-line `MatchContext` test surface shrinks (no more `round`-specific assertions) and the `_roster_for_match` / `predict_match` tests are deleted (the re-score path is gone).

**Issue 2 — Model schema:**

- `test_features.py` drops its `b3_round` assertions.
- `test_model.py` updates `FEATURE_COLUMNS` to the 18-feature list; the unseen-categorical test (`test_lightgbm.py:217-231`) now tests a different unseen feature (e.g. a new `position`) since `b3_round` is gone.
- `test_player_history.py` adds a test that `extract_player_metadata` reads `preferred_foot` from the cached `player_30981_messi.json.gz` sample (Messi is `left`; an end-to-end test exercises the full JSON → metadata path).
- The retrain step (`scripts/train_lightgbm.py`) gains a smoke test: run on the existing 179-kick table, verify the new metrics.json is produced, verify `predictions.jsonl` no longer has any `"Unknown"` `kicking_foot` rows (or whatever the new column is named — pick one and document it).

**Issue 3 — Scraper data recovery:**

- The 18 skipped refs in `data/discrepancies.json` get a one-line diagnostic per match (round-name mismatch / missing shotmap / parse error / unknown), recorded in a new `data/skipped_refs_diagnostics.jsonl` artifact.
- The 6 missing tournaments get a parametrized test in `test_tournaments.py` that asserts each in-scope tournament has at least one season with a shootout in the prediction window.
- `validate_shootout_count` is exercised in the test suite against a fresh run — the test pins `actual == expected == 42` (or whatever the new count is, but the assertion is "no delta").

**Issue 4 — Review:**

- The review document is checked into `docs/model-review.md`. It does not need unit tests; it's a report, not a feature. The agent that writes it can use the existing test suite + the published metrics.json as inputs.

**Modules touched across v3:**

- `dashboard.py` (delete `KNOCKOUT_ROUNDS`, delete re-score path)
- `features.py` (drop `b3_round`, swap A3 source)
- `model.py` (drop `b3_round`, update `FEATURE_COLUMNS`)
- `predict.py` (delete `predict_roster_with_context` if it's only used for re-score)
- `player_history.py` (read `preferred_foot`)
- `tournaments.py` (add missing seasons)
- `validate.py` (no schema change; the count goes up)
- `tests/test_dashboard.py`, `tests/test_features.py`, `tests/test_model.py`, `tests/test_lightgbm.py`, `tests/test_player_history.py`, `tests/test_tournaments.py`
- `docs/PRD-v3.md` (this file)
- `model/README.md` on HF (publish after retrain)

## Out of Scope

- **Real-time retraining during the WC.** The model is frozen before the WC starts (unchanged from v1, v2). Issue 3's retrain happens before the next round of matches; Issue 2 + Issue 3 are one retrain.
- **Opponent / goalkeeper features.** The model has no opponent or keeper feature. The "Recommended Dive" is `argmin` over a uniform-prior policy for the keeper. Issue 4's review may suggest adding a keeper feature, but that is a follow-up, not v3.
- **Multi-task heads (regression on `x` in addition to classification on L/C/R).** Unchanged from v1, v2.
- **AutoML / neural-net alternatives.** LightGBM is the model. The review may suggest alternatives, but switching is a v4 decision.
- **Photos, country flags, historical-tournament tabs in the dashboard.** Visual polish is out of scope; the v3 dashboard is a simplification, not a beautification.
- **Group-stage matches.** Even after the round allowlist is dropped, the dashboard only shows matches with both teams decided. Group-stage matches where the opponent is TBD (none, in the 2026 format) would be shown — but this is hypothetical.
- **A "winner prediction" for undecided R16+ slots.** Out of scope. The dashboard hides matches until both teams are decided; a "predict group winners" model is a different project.
- **CI / scheduled push to HF.** The push remains manual. Adding CI is a v4 item.

## Further Notes

- **v2 history.** v2 shipped 34 issues across 3 phases (codebase simplification, HF persistence, Streamlit dashboard). The v2 PRD (`docs/PRD.md`) is preserved for history; v3 references it.
- **The data and model artifacts are gitignored and live on HF.** Re-runs read the on-disk `data/fotmob_cache/` first, then fall back to live FotMob. The recovered data (Issue 3) flows through the existing pipeline; the new feature schema (Issue 2) flows through the existing training script.
- **Why "simplification" is the theme.** v2 was a simplification pass too (Phase 0). v3 continues the same spirit: drop a fragile categorical, drop a 4-round allowlist, drop a re-score path, drop 24 "Unknown" rows in favor of a real attribute. The simplify work earns its keep because the live tournament is here and the data is half-empty.
- **Why the independent review is its own issue.** The user is the only person who has both built and consumed this model. A second pair of eyes, looking for things the builder can't see (calibration drift, baseline alternatives, holdout power), is a small investment for a much more honest model card.
- **Why the implementation is independent.** The agent that picks up these issues makes its own decisions. The PRD and issues are decision-rich (this file has ~5 specific recommendations per issue), but the agent is free to make different calls and document why. If a future cycle finds a different simplification, that's fine — the issues are the work order, the decisions are the agent's.
- **Why no `progress.txt`.** v1 used one; v2 removed it. v3 follows v2: the PRD is the durable plan, the issue tracker is the durable task list, the code is the durable state. The agent's session log is not committed.
- **The next v4 candidate.** If Issues 1–4 land, the obvious next step is per-keeper data (so the "Recommended Dive" can be a per-kicker-and-per-keeper policy, not `argmin` over a uniform-prior keeper). Out of scope for v3; v4 PRD.
