# Penalty Shootout Prediction — v5 PRD

v5 pivots away from trained models (LightGBM) toward a Tabular Foundation Model (TabPFN), collapses the pipeline into a single scheduled GitHub Actions workflow, and simplifies the scraper and app. Previous PRDs (v2, v3, v4) are archived as history; v5 is a clean-sheet simplification.

## Problem Statement

Five problems drive this pivot:

1. **The trained model is at the noise floor.** The v3 LightGBM is statistically indistinguishable from uniform random on the 6-fold LOTO CV aggregate (0.374 vs 0.405, SE 0.036, see `docs/model-review.md`). Issues #42 (drop A1), #46 (anti-classifier), and #51 (Phase 3 more data) are all model-levers the v3 review concluded cannot beat the noise floor at n=179. The model stack (LightGBM + logreg + LOTO CV + metrics pipeline) is ~2K LOC of complexity that produces no statistically meaningful signal.

2. **The scraper is overly complex.** ~3,500 LOC across 14 library modules and 5 scripts. Three pipelines (shootouts, roster, per-kicker history), an RSSSF validation oracle, hand-rolled diagnostic artifacts, and a 719-LOC feature builder whose match-context features (B1, B2) are not needed under a player-only prediction regime. The per-kicker fan-out takes ~3 hours on a cold cache with no resume mechanism.

3. **The pipeline is manual and multi-system.** Data lives on HuggingFace, the model is trained locally, the predictions are uploaded manually, and there is no scheduled GitHub Actions workflow. The only CI (ralph.yml) is a coding-agent driver, not a pipeline.

4. **The match selector is fixture-based and fragile.** It fetches FotMob fixtures live, filters by kickoff timestamp (strict `<=` drops a match the instant it kicks off), and requires both-teams-decided logic. The Q7 matches-of-the-day fix never landed.

5. **"Don't repeat inference" is not achievable with the current architecture.** The user is on TabPFN's free tier (50M tokens/day); re-scoring players daily wastes quota needlessly. The cost is vanishingly small (~90K tokens/full-run = 0.18% of daily quota), but the architecture should be idempotent regardless.

## Solution

Six changes, in dependency order:

1. **Replace LightGBM with TabPFN** (TabPFNClassifier, cheapest mode: `thinking_mode=False`, `n_estimators=8`, one batched `fit + predict_proba` call). Drop the entire trained-model stack (LightGBM, logreg, LOTO CV, metrics pipeline). TabPFN is a zero-shot tabular foundation model that `fit`s on the historical penalty data once and `predict`s the full roster in one call — no gradient-descent training, no hyperparameter tuning, no held-out validation.

2. **Simplify the feature set to player-only, match-agnostic features.** Drop B1 (kick_number, scores), B2 (is_decisive), and any match-context. Keep A1 (side distribution over 5-year rolling time window, 3 columns: p_L/p_C/p_R), A2 (last_side), A3 (preferred_foot), A4 (career_penalty_count in window), C1 (position). Total ~7 columns. The prediction is match-agnostic: each roster player is scored once, and the same prediction row serves any match they appear in.

3. **Collapse the scraper.** Drop the shootout extraction pipeline (`shootouts.py`, `shootout_kicks.jsonl`), the RSSSF validation oracle (`rsssf.py`, `validate.py`), and the per-kicker delta-tracking complexity. The scraper becomes: (a) fetch the WC 2026 roster, (b) fetch each player's penalty history from FotMob (careerHistory → per-(team,season) fixtures → match shotmaps with penalties), (c) write `player_history.jsonl` as a deterministic full-rewrite. "Skip already-scraped data points" is delegated to the HTTP ETag cache: strip the FotMob buildId from the cache key so cached payloads survive buildId rotations, enable HTTP keep-alive, and persist the cache across Actions runs via `actions/cache` with a `run_id`-keyed unique key and `restore-keys` prefix fallback.

4. **Store everything in GitHub, drop HuggingFace.** The JSONL artifacts (`wc2026_roster.jsonl`, `player_history.jsonl`, `predictions.jsonl`) are tracked in the repo under `data/`. The scheduled workflow commits and pushes them. The app reads them directly from the cloned working tree (Streamlit Cloud clones the repo on deploy). Drop `huggingface_hub`, drop the HF account, drop the `huggingface-cli` install step. ADRs 0001, 0003, and 0004 are superseded by a new ADR-0005 documenting the GitHub storage decision.

5. **One GitHub Actions workflow (`scrape-and-predict.yml`).** Daily cron at 06:00 UTC + `workflow_dispatch`. Steps: checkout → restore fotmob-cache (run_id key + restore-keys) → uv sync --group pipeline --no-dev → run scraper (roster + player-history) → run predict.py (TabPFN fit+predict) → git commit + push → save cache (if: always()). Single workflow, single job, ~1-3 minute wall-time after the first backfill.

6. **Simplify the app to two independent team dropdowns.** Drop the live FotMob fixture fetch, `load_upcoming_knockouts`, `MatchContext`, `is_placeholder_team`, `_parse_kickoff_utc`, and the match header. Replace with two `st.selectbox` dropdowns populated from the distinct `(team_id, team_name)` pairs in `predictions.jsonl`. No conditional filtering (selecting Brazil for Team A does not remove Brazil from Team B). The view renders both teams' kicker cards side-by-side, filtered by the two selected `team_id`s, sorted by `total_penalties` descending. Drop the Plotly goal-drawing figure (use `st.bar_chart` natively).

## User Stories

### Scraper simplification

1. As a maintainer, I want the FotMob HTTP cache to survive buildId rotations, so cache files are reusable across runs instead of being orphaned on every deployment.
2. As a maintainer, I want the FotMobClient to reuse one httpx connection pool instead of creating a new one per request, so 304 revalidation calls are sub-millisecond instead of TLS-handshake-bound.
3. As a maintainer, I want the scraper's HTTP cache persisted across GitHub Actions runs via `actions/cache`, so only the first-ever run is cold; subsequent runs complete in minutes.
4. As a maintainer, I want `player_history.jsonl` written as a deterministic full-rewrite (no append-ledger, no per-kicker dedup), so the output is idempotent given a stable cache and simpler to reason about.
5. As a maintainer, I want the shootout extraction pipeline (`shootouts.py`, `shootout_kicks.jsonl`) dropped, so the scraper's responsibility is narrowed to roster + per-kicker penalty history only.
6. As a maintainer, I want the RSSSF validation oracle (`rsssf.py`, `validate.py`, `discrepancies.json`) dropped, so there is no completeness-assertion subsystem to maintain.
7. As a maintainer, I want the WC 2026 roster re-fetched on every run (it's cheap), so late callups and squad changes surface automatically without detecting "file exists" state.

### Model replacement

8. As a maintainer, I want the LightGBM, logistic regression, and metrics pipelines (`model.py`, `evaluate.py`, training scripts, `model/` artifacts) dropped, so the codebase sheds ~2K LOC of noise-floor model complexity.
9. As a maintainer, I want TabPFNClassifier as the sole estimator (cheapest mode: no thinking, `n_estimators=8`), so predictions are free-tier-safe and the code is a thin ~50 LOC module.
10. As a maintainer, I want one batched `predict_proba` call over all roster rows (never row-by-row), so the 5,000-token per-call floor doesn't waste quota.
11. As a maintainer, I want only player-only features (A1/A2/A3/A4/C1) on a 5-year rolling time window, so predictions are match-agnostic and each player is scored once.
12. As a maintainer, I want the feature table built from `player_history.jsonl` alone (each penalty is a training row: features from prior 5-year window → label = this kick's side), so the training set has the same semantics as the test set and no shootout-specific data is needed.

### Feature engineering

13. As a maintainer, I want `SCRAPE_FLOOR = 2016-01-01` as the data corpus floor (which penalties go into `player_history.jsonl`), so the corpus has enough history for a 5-year rolling window applied to the oldest training kicks.
14. As a maintainer, I want `TRAIN_FLOOR = 2021-01-01` as the label floor (only kicks on or after this date become training/test rows), so the oldest training row has a complete 5-year feature window from 2016–2021.
15. As a maintainer, I want `LOOKBACK_WINDOW_YEARS = 5` as a time-based rolling feature window: A1 is the side distribution over `[T - 5 years, T)` for each target kick at time T.
16. As a maintainer, I want A1 collapsed to one horizon (the full 5-year window, no sub-windows like 5/10/20 kicks), so the feature block is simpler to compute and reason about.
17. As a maintainer, I want `features.py` rewritten with only A1/A2/A3/A4/C1 helpers and lean docstrings, dropping the B-group and match-context helpers entirely.

### GitHub Actions pipeline

18. As a maintainer, I want one `.github/workflows/scrape-and-predict.yml` workflow on a daily cron + `workflow_dispatch`, so the pipeline runs automatically and can be triggered manually.
19. As a maintainer, I want the workflow to restore the `fotmob-cache` from the most recent run (via prefix `restore-keys`), scrape, predict, commit+push, then save a fresh cache, so each run is warm after the first backfill.
20. As a maintainer, I want the pipeline to full-refresh predictions every run (re-fit TabPFN on the cumulative training set, batched `predict_proba` on all roster rows, overwrite `predictions.jsonl`), so predictions stay fresh as training data accumulates.
21. As a maintainer, I want the JSONL artifacts (`wc2026_roster.jsonl`, `player_history.jsonl`, `predictions.jsonl`) tracked in git and committed+push by the workflow using the `github-actions[bot]` identity, so the dataset is version-controlled alongside the code.

### App simplification

22. As a viewer, I want two independent team dropdowns (Team A, Team B) populated from the distinct teams in `predictions.jsonl`, so I can pick any two World Cup teams and see their kickers' penalty predictions.
23. As a viewer, I want no conditional filtering between the two dropdowns (selecting Brazil for Team A does not remove Brazil from Team B), so the UI is simple and predictable.
24. As a viewer, I want kicker cards rendered side-by-side in two columns (one per team), sorted by `total_penalties` descending, with the same card layout as v4 (name + foot pill + penalty count + bar chart).
25. As a maintainer, I want the live FotMob fixture fetch, `load_upcoming_knockouts`, `MatchContext`, `is_placeholder_team`, `_parse_kickoff_utc`, and the match header dropped from the app, so the app has zero live network calls.
26. As a maintainer, I want the Plotly goal-drawing figure and `plotly` dependency dropped, so the app uses only native `st.bar_chart` and `st.container` elements.
27. As a maintainer, I want the app to read JSONLs directly from the working tree (no HTTP fetch, no `hf_hub_download`), so Streamlit Cloud's clone-and-run convention is the full deployment story.

### Dependencies and tooling

28. As a maintainer, I want `pyproject.toml` restructured with PEP 735 dependency groups: shared deps in `[project] dependencies`, `dev` (pytest+ruff+ty), `app` (streamlit), and `pipeline` (tabpfn-client), so contributors can install only what they need.
29. As a maintainer, I want `lightgbm`, `scikit-learn`, `plotly`, and `huggingface_hub` dropped from the project's dependencies, so only what is actually used is declared.
30. As a maintainer, I want `tabpfn-client>=0.3` added to the `pipeline` dependency group, so the TabPFN classifier is installable in CI and locally.
31. As a maintainer, I want ruff set to `select = ["ALL"]` with a path-based ignore of S101 (assert) only on `tests/**/*.py`, so linting is strict and comprehensive without the single well-known test-only false positive.

### Docs cleanup

32. As a maintainer, I want `model-card-v3.md`, `model-card-v4.md`, and `model-review.md` deleted (model is gone), so the docs reflect the current architecture.
33. As a maintainer, I want ADR-0005 written, superseding 0001 (HF persistence), 0003 (HF subpaths), and 0004 (Phase 3 data source), documenting the GitHub storage decision and the architectural pivot, so future contributors understand the "why."
34. As a maintainer, I want `CONTEXT.md` updated to reflect the TabPFN-classifier regime, GitHub storage, and the dropped concepts (ShootoutKick, RSSSF oracle, training artifacts), so the domain glossary stays accurate.
35. As a maintainer, I want `model/lightgbm.pkl`, `model/metrics.json`, and `model/README.md` deleted, so the repo no longer carries model artifacts that are being replaced.

### Deferred

36. As a maintainer, I want a standalone offline notebook (outside the pipeline, outside the app) that backtests predictions against holdout kicks and computes Brier/ECE/save-rate metrics, so there is a quality signal without burdening the scheduled workflow.

## Implementation Decisions

### Model

- TabPFNClassifier with defaults (`thinking_mode=False`, `thinking_effort=None`, `n_estimators=8`, `auto_scale_n_estimators=True`, `model_path="auto"`). Free tier: 50M tokens/day, fit is free, only predict costs. One full run ≈ 90K tokens ≈ 0.18% of daily quota. A separate thinking-mode quota (20/month) exists but is not consumed.
- Authentication: `TABPFN_TOKEN` env var, read automatically by `tabpfn_client.init()`.
- Always batch all roster rows in one `predict_proba` call. The 5,000-token per-call floor makes row-by-row vastly more expensive.
- Fit+predict per Actions run (full re-fit on cumulative training set, full re-predict on all roster rows, overwrite `predictions.jsonl`). The cost is negligible at n≈1000 and provides fresh predictions as training data grows.
- Calibration tuning (`softmax_temperature`, `balance_probabilities`) is deferred to a future iteration. Default probabilities are well-calibrated for this size dataset.

### Feature schema

Seven columns (~7 numerical + categorical), match-agnostic, player-only:
- `SCRAPE_FLOOR = 2016-01-01` — penalties in `player_history.jsonl` start here.
- `TRAIN_FLOOR = 2021-01-01` — only kicks at or after this date become training/test rows.
- `LOOKBACK_WINDOW_YEARS = 5` — time-based rolling feature window `[T - 5y, T)`.
- A1: `(p_L, p_C, p_R)` — side distribution over the 5-year window before the target kick.
- A2: `last_side` — the side of the most recent kick in the window.
- A3: `preferred_foot` — declared foot from FotMob player metadata.
- A4: `career_penalty_count` — count of kicks in the window.
- C1: `position` — from FotMob player metadata.
- TabPFN handles categoricals natively via `categorical_features_indices`.
- No feature scaling or one-hot encoding (TabPFN handles internals).
- The training table is built from `player_history.jsonl`: each player's kicks are sorted chronologically, and each kick becomes a training row whose features are derived from prior kicks within the 5-year window and whose label is the kick's side.

### Scraper

- BuildId-stripped cache keys in `client.py:_cache_key`. Strip `/_next/data/<buildId>/` before computing the filesystem key so cache files survive FotMob deployment rotations.
- Shared `httpx.Client` on the `FotMobClient` instance (lazy-init, reuse across all calls). Replaces per-call `with httpx.Client(...)`.
- `player_history.jsonl` full-rewrite (`"w"`) — deterministic from a stable cache, no append-ledger, no per-kicker dedup. The existing `(team_id, league_id, season_year, player_id)` dedup within a kicker's walk is preserved.
- Drop `shootouts.py`, `rsssf.py`, `validate.py`, modules.
- Keep `client.py`, `player_history.py`, `initial_set.py`, `rosters.py`, `artifacts.py` (slimmed), `config.py`, `leagues.py`, `tournaments.py`, `match_ref.py`, `fotmob_parsing.py`, `coordinates.py`.
- `data/fotmob_cache/` stays gitignored. `data/wc2026_roster.jsonl`, `data/player_history.jsonl`, `data/predictions.jsonl` are tracked in git.

### GitHub Actions cache strategy

- Key: `fotmob-cache-${{ github.run_id }}` (unique per run — growing cache needs new keys).
- `restore-keys: | fotmob-cache-` (prefix-match picks up the most recent run's cache).
- Split `actions/cache/restore` + `actions/cache/save` so `if: always()` can save partial progress.
- `ubuntu-latest`, `timeout-minutes: 60`.
- Warm-run wall-time: ~1-3 minutes. Cold-run (cache evicted after 7 days): ~15-20 minutes.
- BuildId-strip fix is the linchpin: without it, every run after a buildId rotation is effectively cold.

### App

- Two `st.selectbox` dropdowns in the sidebar, options = distinct `(team_id, team_name)` from `predictions.jsonl`.
- No restrictions: same team in both dropdowns is allowed.
- Drop the live FotMob client, fixture fetch, match selector, match header, round display.
- Read JSONLs from `data/predictions.jsonl` and `data/player_history.jsonl` via `Artifacts` adapter (local files, Streamlit Cloud clones the repo).
- Cards render via `st.container(border=True)` + `st.bar_chart` — same card layout as v4 minus the plotly goal figure.
- `dashboard.py` keeps `predictions_for_match` and card-rendering helpers; drops `load_upcoming_knockouts`, `MatchContext`, `is_placeholder_team`, `_parse_kickoff_utc`.

### Dependencies

```toml
[project]
dependencies = ["httpx>=0.27", "numpy>=2.0", "pandas>=2.0", "packaging>=24.0"]

[dependency-groups]
dev = ["pytest>=8.0", "ruff>=0.6", "ty>=0.0.1a5"]
app = ["streamlit>=1.30"]
pipeline = ["tabpfn-client>=0.3"]
```

Dropped: `huggingface_hub`, `plotly`, `scikit-learn`, `lightgbm`.

Workflow install: `uv sync --group pipeline --no-dev` (pipeline job), `uv sync --group app --no-dev` (app deploy).

### Modules dropped

Library: `shootouts.py`, `rsssf.py`, `validate.py`, `model.py`, `evaluate.py`.

Scripts: `fetch_2022_final.py`, `fetch_all_shootouts.py`, `fetch_player_history.py`, `build_training_table.py`, `train_baseline.py`, `train_lightgbm.py`, `evaluate_cv.py`, old `predict.py`.

Data: `model/lightgbm.pkl`, `model/metrics.json`, `model/README.md`.

Docs: `model-card-v3.md`, `model-card-v4.md`, `model-review.md`.

ADRs superseded: 0001, 0003, 0004. ADR-0002 (Streamlit Cloud) retained.

### Modules rewritten

- `features.py` — A1/A2/A3/A4/C1 only, lean docstrings.
- `app.py` — two-team dropdowns, local file reads, no live FotMob.
- `dashboard.py` — drop fixture-path helpers, keep `predictions_for_match` + cards.
- `artifacts.py` — slim to only paths used by surviving artifacts.
- `config.py` — `SCRAPE_FLOOR`, `TRAIN_FLOOR`, `LOOKBACK_WINDOW_YEARS`.

### Modules added

- `tabpfn.py` (~50 LOC): `init()`, `fit(X_train, y_train)`, batched `predict_proba(X_test)`.
- `scripts/predict.py` (new): loads `player_history.jsonl`, builds training/test matrices, fits TabPFN, writes `predictions.jsonl`.
- `.github/workflows/scrape-and-predict.yml`.

### Schema changes

**`PlayerMetadata`** gains two fields:
- `photo_url: str` — constructed from `f"https://images.fotmob.com/image_resources/playerimages/{player_id}.png"` (confirmed live; not present in any FotMob API response, constructed from the player ID).
- `short_name: str` — derived from the last word of `pageProps.data.name` (e.g. `"Kylian Mbappé"` → `"Mbappé"`). Falls back to the full name if the name is a single word.

**`PredictionRow`** gains two fields, passed through from `PlayerMetadata`:
- `photo_url: str`
- `short_name: str`

### Directory structure (post-pivot)

```
data/
  wc2026_roster.jsonl       (tracked)
  player_history.jsonl      (tracked)
  predictions.jsonl         (tracked)
  fotmob_cache/             (gitignored, Actions-cached)
src/twelveyards/
  tabpfn.py                 (new)
  client.py                 (buildId-strip fix + shared httpx)
  features.py               (rewritten, player-only)
  dashboard.py              (slimmed)
  ... (surviving lib modules)
scripts/
  fetch_wc_2026_roster.py
  fetch_initial_set_player_history.py
  predict.py                (new)
app.py                      (rewritten, two-team dropdowns)
.github/workflows/
  ralph.yml                 (retained)
  scrape-and-predict.yml    (new)
```

## Testing Decisions

### Seams

The seams to test at (highest practical level):

1. **TabPFN module**: `fit_and_predict(X, y, X_test) -> np.array` — the external boundary of the estimator. Tests assert shape `(n_test, 3)`, rows sum to 1.0, deterministic given fixed random_state. Use a small synthetic dataset so no network call is needed (stub or monkeypatch TabPFNClassifier if the free-tier quota matters in CI).

2. **Feature builder**: `compute_features(kicker_id, history, target_date) -> dict` — same signature as existing, but only A1/A2/A4 returned. Tests assert 7 total features, A1 always sums to 1.0, A2 is in `{L, C, R}`, A4 is non-negative.

3. **Scraper cache fix**: `_cache_key(url)` — tests assert buildId-stripped paths are identical for the same resource across different buildIds. `FotMobClient` instance — tests assert a single httpx client is reused across calls.

4. **Predict script output**: `predictions.jsonl` — same `PredictionRow` schema as today (player_id, player_name, team_id, team_name, p_L/p_C/p_R). Tests assert idempotent overwrite, rows sum to 1.0, every player in the roster has a row, no duplicate `player_id`.

5. **App**: same approach as existing `test_app.py` / `test_dashboard.py` — stub the JSONL readers, assert two selectboxes render, assert the view renders N cards per selected team, assert same-team selection works. Drop tests for live fixture fetch and match header.

6. **Workflow**: no unit test (it's CI infra). Manual smoke: `gh workflow run scrape-and-predict.yml --ref main`, observe run completes within timeout, check `data/predictions.jsonl` is updated.

### What to test vs not test

- Test external behaviour, not internal layout (`st.columns` ratio, CSS classes).
- Test the `data/predictions.jsonl` schema (parity with current contract).
- Test the `_cache_key` URL stripping (the linchpin fix).
- Drop tests for `shootouts.py`, `rsssf.py`, `validate.py`, `model.py`, `evaluate.py`, old `predict.py`, `load_upcoming_knockouts`, `MatchContext`, plotly figure.
- Drop LightGBM/logreg-specific tests in `test_model.py`, `test_lightgbm.py`, `test_evaluate.py`, `test_predict.py` (replaced by corresponding TabPFN tests).
- Drop `test_shootouts_*.py`, `test_rsssf.py`, `test_validate.py`, `test_tournaments.py` (shootout-scope tests).

### Prior art

Tests follow the existing pattern: `tests/_factories.py` for synthetic data, `conftest.py` for session fixtures, `FakeFotMobClient` stubs replacing live HTTP. The `Artifacts` adapter is the I/O seam.

## Out of Scope

- Thinking mode / extra compute on TabPFN (user is on free tier, wants cheapest).
- Per-keeper dive-direction features (FotMob does not expose keeper dive data; ADR from v4's issue #44 is moot under the pivot).
- Club shootout data expansion (issue #51, ADR-0004 — the shootout pipeline is dropped, and TabPFN doesn't target a trained-model accuracy threshold).
- Anti-classifier (issue #46 — LightGBM-specific, dropped with the model).
- Drop A1 feature block (issue #42 — A1 is kept, collapsed to a single 5-year time window).
- Multi-task regression head (x-coordinate prediction alongside side classification).
- SVG photo placeholder (real FotMob player photos are in scope — constructed from player ID CDN URL).
- Player short name fallback from lineup objects (derivation from `pageProps.data.name` last word is in scope; lineup `firstName`/`lastName` is deferred).
- Per-keeper delta-tracking / resume state file (the cache IS the resume mechanism).
- A match-level predictor (score-line, shootout occurrence probability).
- HF model card, HF Dataset surface, HF Spaces deployment (HF is dropped).
- ADR-0002 (Streamlit Cloud) — retained, unchanged.

## Further Notes

### Why TabPFN at n≈500–1000

TabPFN's documented strength is small tabular data (100% win-rate vs default XGBoost at ≤10K rows per the v2.5 model report). A 5-column, 3-class problem at ~500–1000 rows is inside its sweet spot. The API path (tabpfn-client) uses the Prior Labs cloud and a free-tier quota of 50M tokens/day — a full 1243-row predict costs ≈90K tokens (0.18% of a day's quota), making daily refresh effectively free.

### Why GitHub over HuggingFace

The dataset is small (under ~10 MB for all three JSONLs). Storing it in the repo eliminates a second system (HF account, `huggingface-cli`, separate push step, separate secret), gives the dataset version history alongside the code, and lets the Streamlit Cloud app read files directly from the cloned working tree. The Streamlit Cloud clone-and-run convention means zero configuration at deploy time.

### Why the buildId strip is the linchpin

FotMob's `/_next/data/<buildId>/` path segment rotates every few hours. With the buildId baked into the cache key, every cache file is orphaned on the next run. Stripping it means the same logical resource has the same cache key across deployments, so `If-None-Match` (ETag) revalidation works as intended. Without this fix, "warm cache" is a fiction — each run after a buildId rotation is a full cold re-walk. The fix is one-line regex in `_cache_key`.

### Why a full re-fit every run

TabPFN's `fit` is free (no tokens consumed). The `predict` cost is `(train_rows + test_rows) × cols × n_estimators`, floored at 5,000 per call. At ~1000 train + 1243 test × 7 cols × 8 estimators ≈ 125K tokens — well under 1% of daily quota. Re-fitting every run gives the freshest predictions as training data accumulates, with zero cost downside. Switching to "score each player once" (appending only) would save ~125K tokens per run but add dedup complexity for negligible savings. Simplicity wins: overwrite `predictions.jsonl` every run.

### Why `player_history.jsonl` is a full-rewrite, not append-only

Given the same cache and deterministic inputs, `fetch_player_penalty_history` produces the same `PlayerPenalty` rows in the same order. The orchestrator already dedups `(team_id, league_id, season_year, player_id)` lookups within a kicker. An append-ledger would add a read-before-write step, a per-kicker dedup set, and a state file — all to solve a problem (non-determinism) that the current architecture doesn't have. New penalties the kicker took since the last run surface automatically when ETag revalidation returns 200. The rewrite is already idempotent.

## References

- `docs/PRD-v2.md` (v2), `docs/PRD-v3.md`, `docs/PRD-v4.md` — archived predecessor PRDs
- `docs/model-review.md` — v3 independent review (the source of the noise-floor finding)
- `CONTEXT.md` — domain glossary (updated in this cycle)
- `docs/adr/0002-streamlit-cloud-for-hosting.md` — retained ADR
- `docs/adr/0005-github-storage-and-architectural-pivot.md` — new ADR superseding 0001, 0003, 0004
- `docs/fotmob.md` — FotMob API reference (unchanged)
- `docs/prototype-card-layout.png` — card layout design (unchanged)
