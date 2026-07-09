# Penalty Shootout Prediction

Predicts which side (L / C / R) a player will kick a penalty in a shootout, so a goalkeeper can pick the lowest-probability side to dive. The model is a LightGBM multiclass classifier trained on FotMob-derived data. A Streamlit dashboard surfaces the predictions live, fed by a model and dataset pinned to Hugging Face.

This PRD replaces v1. v1 scoped the scraper + model + a deferred dashboard. v2 adds three things: a **codebase simplification pass** that the v1 implementation earned but did not absorb, a **Hugging Face persistence** layer so the model and its inputs survive cold starts, and a **Streamlit dashboard** for live shootout predictions.

## Problem Statement

Penalty shootouts are decided by ~5–10 kicks per match. The goalkeeper's pre-kick dive direction is a major lever; today, keepers have no per-kicker, per-side probability to inform their dive. The v1 system solves this end-to-end: a scraper pulls 179 shootout kicks across 6 in-scope national-team tournaments (2021–2026), a LightGBM model outputs P(L), P(C), P(R) for any kicker, and a slice script produces per-kicker predictions for the 1243-player 2026 World Cup roster.

But v1 is not deployable in its current shape:

- **The codebase has accumulated duplication and shallow modules.** Eight separate `Path("output/...")` blocks are scattered across nine slice scripts; five `write_jsonl` and three `read_jsonl` functions live in five different modules; the FotMob client keeps `BuildId` as a module-level mutable that needs a test-only autouse fixture to reset; the predict path constructs a 17-field synthetic `ShootoutKick` (with six fields the docstring admits are "unused") just to feed the feature builder. The code works, but reading it costs more than it should and adding the next layer (persistence, dashboard) on top of it would entrench the friction.
- **The model and data are not persisted anywhere reachable.** Both `output/lightgbm.pkl` and `output/predictions.jsonl` are gitignored, the FotMob HTTP cache is gitignored, and there is no remote copy. The v1 run had to reproduce the entire pipeline from a cold start to verify it still works, and a future deployment of the dashboard has no model to load.
- **There is no viewer.** The 1243-row `predictions.jsonl` is a deliverable for a human, but humans don't read JSONL. A dashboard is the missing consumer.

## Solution

Three phases, ordered by dependency.

**Phase 0 — Codebase simplification.** Deepen the v1 modules so the next layer can sit on a clean seam. The goal is a smaller, easier-to-read codebase — not a feature change. Concretely: one `Artifacts` adapter owns the on-disk layout (paths, I/O format, FotMob cache); one `PredictionTarget` value object replaces the synthetic `ShootoutKick`; a `fotmob_parsing` module consolidates the triplicated `_int` / `_parse_match_date` / outcome map; a `match_ref` module unifies the near-twin `ShootoutMatchRef` / `RosterMatchRef`; `BuildId` becomes an instance field of `FotMobClient`; the 786-line `player_history.py` splits into `player_history` (FotMob fan-out) and `initial_set` (JSONL merge); the model layer collapses to one row type and one matrix builder; tests share one factory and a schema derived from the dataclass; the dead exports, defensive imports, over-extracted helpers, and the misnamed `kicker_most_frequent_save_rate` are deleted or renamed.

**Phase 1 — Hugging Face persistence.** One repo, `couto/12yd`, with two subpaths: `model/` (the frozen `lightgbm.pkl` + a minimal model card) and `data/` (the raw inputs — `shootout_kicks.jsonl`, `player_history.jsonl`, `wc2026_roster.jsonl` — plus the `predictions.jsonl` artifact the dashboard reads). Manual updates via `huggingface-cli` after the slice pipeline re-runs. No new server, no new service: HF is a file store with a model-card surface, and the dashboard reads from it at startup.

**Phase 2 — Streamlit dashboard.** A single-page Streamlit app on Streamlit Cloud that surfaces live shootout predictions. At load time, the app fetches the WC 2026 fixture list from FotMob (live, via the persistent ETag/gzip cache), filters to upcoming matches with both teams decided, and lets the user pick a match from a selectbox. For the selected match, the app loads `lightgbm.pkl` from HF, builds the 9-feature row for each likely kicker on each team (using the match's actual `round` for the B3 feature), re-scores, and shows a per-kicker table: name, team, kicking foot, P(L), P(C), P(R), and the recommended dive (`argmin`). (v3 simplified this filter to drop the per-round allowlist; see PRD-v3 Issue 1.)

**Precondition (not a phase).** The slice pipeline (`output/lightgbm.pkl`, `output/predictions.jsonl`, the data layer's JSONLs) is not on disk. Before Phase 1 can push to HF, the user re-runs the slice scripts in the natural order (roster → shootouts → player history → training table → train_lightgbm → predict). This is a one-time recovery, not a new phase; the v1 slice scripts are unchanged and idempotent.

## User Stories

### Phase 0 — Codebase simplification

1. As a maintainer, I want one `Artifacts` adapter that owns the on-disk layout (paths to every artifact, JSONL read/write, the FotMob cache), so that no script or test has to know where files live.
2. As a maintainer, I want the `Artifacts` adapter to expose typed read/write methods (e.g. `artifacts.read_shootout_kicks()`, `artifacts.write_predictions(rows)`), so that callers cross one seam, not three (Path + format + cache).
3. As a maintainer, I want the 9 slice scripts to default their `--output` and `--cache-dir` flags from `Artifacts` instead of hardcoding `Path("output/...")`, so that renaming a path touches one module.
4. As a maintainer, I want the 5 duplicated `write_jsonl` functions and 3 duplicated `read_jsonl` functions to be deleted in favour of the `Artifacts` adapter, so the JSONL shape lives in one place.
5. As a maintainer, I want `validate.py` to consume the data layer's `read_shootout_kicks()` instead of re-parsing the JSONL with two ad-hoc readers, so the validator stops duplicating the file format.
6. As a maintainer, I want the model layer to never read a sibling JSONL by relative path, so that `load_training_table` takes `is_on_target` as an explicit argument and the data layer's directory layout stops leaking into the model layer.
7. As a maintainer, I want a `PredictionTarget` value object that carries only the 9 feature inputs, so that `predict.py` no longer constructs a 17-field fake `ShootoutKick` (with 6 fields marked "unused") to feed the feature builder.
8. As a maintainer, I want the feature builder to accept `PredictionTarget` (or, equivalently, the model's row type) instead of the data layer's `ShootoutKick`, so that the data layer's shape does not leak across the model-layer seam.
9. As a maintainer, I want a `fotmob_parsing` module that owns the triplicated `_int`, `_parse_match_date`, and the shotmap outcome map, so that a FotMob-shape quirk (e.g. the `Post` → `Missed` remap) is fixed in one place.
10. As a maintainer, I want a `match_ref` module that owns `parse_page_url` and a single `MatchRef` dataclass (replacing the near-twin `ShootoutMatchRef` and `RosterMatchRef`), so that the URL parser no longer lives in `shootouts.py` and is imported by two non-shootout modules.
11. As a maintainer, I want `FotMobClient.build_id` to be an instance field (set lazily on first use), so that the test suite no longer needs the autouse `_reset_build_id_cache` fixture.
12. As a maintainer, I want the 786-line `player_history.py` to split into `player_history` (FotMob fan-out) and `initial_set` (JSONL merge), so each module has one responsibility and a name that matches it.
13. As a maintainer, I want the `player_history` orchestrator to take typed `Iterable[InitialSetKicker]`, not two file paths, so that the per-kicker fetcher has no JSONL re-parse and the seam is the data shape, not the disk.
14. As a maintainer, I want the model layer to collapse to one row type and one matrix builder, so that `_training_row_from_table_row` (the 19-field bridge with `label="L"` / `is_on_target=True` dummies) and `rows_to_predict_matrix` (a near-copy of `build_feature_matrix`) are deleted.
15. As a maintainer, I want the `predict_proba` dispatch shim (3 lines with an unreachable `TypeError` branch) to be deleted in favour of a `Protocol` on the model artifacts, so the call site is `model.predict_proba(X)` and the contract is in the type.
16. As a maintainer, I want one `tests/_factories.py` with one builder per row type, so that the 4 private `_make_row` / `_make_features` helpers in `test_model.py`, `test_lightgbm.py`, `test_evaluate.py`, `test_predict.py` are deleted.
17. As a maintainer, I want `REQUIRED_COLUMNS` to be derived from `dataclasses.fields(TrainingTableRow)`, so the 19/12/9/26 field sets that are hand-typed in 5 places are derived from the dataclass.
18. As a maintainer, I want tests to assert on the public interface (`predict_proba`, `client.get`, `build_features`), not on `_coerce_lightgbm_categoricals` / `_cache_key` / `_decompress`, so the tests pin behaviour at the seam, not implementation.
19. As a maintainer, I want the dead exports `SIDE_LEFT` / `SIDE_CENTER` / `SIDE_RIGHT` and the `Side = str` alias in `coordinates.py` deleted, so the module's surface matches its use.
20. As a maintainer, I want the defensive local imports of `LEAGUE_BY_ID` (in `player_history.py` and `rsssf.py`, both annotated "avoid a cycle") lifted to module top, so the import graph is honest and not cargo-cult.
21. As a maintainer, I want `_coalesce_int` (18 lines for 2 call sites) inlined, so the merge logic is at the call site and the helper's tests stop.
22. As a maintainer, I want `fetch_all_shootout_kicks` (the pass-through that drops the `FetchResult` envelope) deleted, so callers go through `fetch_all_shootout_kicks_with_skips` and the test surface matches production.
23. As a maintainer, I want `fetch_match_data` (a 3-line wrapper used by exactly one call site) inlined, so the one caller is self-documenting.
24. As a maintainer, I want `last_side` (5 lines for 1 line of logic) inlined, so the helper and its test are gone.
25. As a maintainer, I want `kicker_most_frequent_save_rate` either renamed to `last_side_save_rate` or rewritten to compute the per-kicker mode, so the name matches the body.
26. As a maintainer, I want the tournament scope (`LEAGUE_SEASONS_PREDICT_WINDOW`, the `RSSSF_TO_LEAGUE_NAME` heading map, the WC 2026 season) lifted to a `tournaments` config module, so adding a tournament edits one file.
27. As a maintainer, I want a documented deletion-test for each of the above: deleting the module concentrates complexity (or doesn't), so future reviews know which simplifications earned their keep.

### Phase 1 — Hugging Face persistence

28. As a viewer of the dashboard, I want the frozen LightGBM to live on Hugging Face at `couto/12yd/model/lightgbm.pkl`, so that the Streamlit app can load it without re-training.
29. As a maintainer, I want a minimal model card (name, one-paragraph description, the 2026-holdout metrics, a 5-line usage example with `huggingface_hub.hf_hub_download`) at the root of `couto/12yd`, so a future reader (human or agent) knows what the model is and how to load it.
30. As a maintainer, I want the raw scrape outputs (`shootout_kicks.jsonl`, `player_history.jsonl`, `wc2026_roster.jsonl`) and the per-player prediction artifact (`predictions.jsonl`) on `couto/12yd/data/`, so the dashboard reads the dashboard-ready file and the model is reproducible from the inputs.
31. As a maintainer, I want `model_artifacts` and `data_artifacts` to be the only subpaths in the HF repo, so a single repo serves both the model and the data.
32. As a maintainer, I want updates to the HF repo to be manual (`huggingface-cli upload couto/12yd . couto/12yd`) after the slice pipeline re-runs, so there is no CI complexity and the push matches a known-good local run.
33. As a maintainer, I want no new service for persistence: HF is a file store, the dashboard reads from it, no server process is added.

### Phase 2 — Streamlit dashboard

34. As a viewer, I want a Streamlit app at `matheusccouto/12yd` (deployed to Streamlit Cloud) that lists upcoming 2026 World Cup knockout matches (any round) and shows per-kicker P(L), P(C), P(R) for a selected match, so I can pick a side to dive during a live shootout.
35. As a viewer, I want the app to fetch the WC 2026 fixture list from FotMob at load time (via `huggingface_hub.hf_hub_download` for the model and `twelveyards.client.FotMobClient` for the fixtures), so the schedule is always fresh within the cache window.
36. As a viewer, I want only matches where both teams are decided to appear in the selector, so the app doesn't show "Winner Group A vs Winner Group B" placeholders.
37. As a viewer, I want the per-kicker predictions to be re-scored with the match's actual round (e.g. "Quarter-finals" for an R16 match, "Final" for the F), so I see round-specific predictions, not the round-agnostic ones from `predictions.jsonl`.
38. As a viewer, I want the re-score to use the existing `predict.py` path extended with a `predict_roster_with_context(roster, history, model, context)` entry point, so the dashboard reuses the library instead of duplicating the predict logic.
39. As a viewer, I want the per-kicker table to show name, team, kicking foot, P(L), P(C), P(R), and the recommended dive (`argmin` of the three), so I can read the recommendation at a glance.
40. As a viewer, I want both teams' kickers on the same page (sorted by `total_penalties` descending), so a head-to-head comparison takes one screen.
41. As a maintainer, I want a thin `app.py` at the repo root (Streamlit Cloud's default entry point) that imports the dashboard logic from `src/twelveyards/dashboard.py`, so the dashboard's tests can run without launching Streamlit.
42. As a maintainer, I want the dashboard's data loading and re-scoring logic in the library, with the UI as a thin layer, so the same code can be unit-tested (re-scoring, fixture filtering, TBD-hiding) and exercised end-to-end through Streamlit.

## Implementation Decisions

### Phase 0 — Codebase simplification

**Module layout (one new module, six modifications):**

- **New: `src/twelveyards/fotmob_parsing.py`.** Owns `coerce_int(value)`, `parse_match_date(value)`, `SHOTMAP_EVENT_TYPE_TO_OUTCOME`, and any other FotMob-shape coercion. Both `shootouts.py` and `player_history.py` import from this module; the duplicates are deleted.
- **New: `src/twelveyards/match_ref.py`.** Owns `parse_page_url` (the `_PAGE_URL_RE` regex moves here) and a single `MatchRef` dataclass. The `MatchRef` has optional fields for the data each consumer needs: `home_team_id`/`home_team_name`/`away_team_id`/`away_team_name` (roster), `round_name`/`score_str` (shootout), `match_date` (both). A single `MatchRef.from_fixture(fixture)` builder replaces `ShootoutMatchRef.from_fixture` and `_extract_roster_match_refs`.
- **New: `src/twelveyards/artifacts.py`.** Owns the on-disk layout. `Artifacts(root=Path("output"), cache_dir=Path("data/fotmob_cache"))` exposes path accessors (`artifacts.shootout_kicks`, `artifacts.lightgbm_model`, etc.), typed read/write methods (`artifacts.read_shootout_kicks()`, `artifacts.write_predictions(rows)`), and a `artifacts.fotmob_client()` factory. The 5 `write_jsonl` and 3 `read_jsonl` functions are deleted; the 9 `Path("output/...")` defaults in the slice scripts are replaced with `Artifacts()` defaults.
- **New: `src/twelveyards/dashboard.py`** (Phase 2; built on top of `Artifacts` and the extended `predict.py`).
- **Modified: `src/twelveyards/client.py`.** `build_id` becomes an instance field, set lazily on first `get()`. The module-level `_build_id_cache` and `reset_build_id_cache()` are deleted. The autouse fixture in `tests/test_client.py` is deleted.
- **Modified: `src/twelveyards/player_history.py` → split into `player_history.py` (FotMob fan-out, ~400 lines) and `initial_set.py` (JSONL merge + Initial Set assembly, ~150 lines).** The per-kicker orchestrator takes `Iterable[InitialSetKicker]`; the script does the JSONL read once and passes typed iterables down.
- **Modified: `src/twelveyards/predict.py`.** Adds `PredictionTarget` value object (9 fields, exactly the model inputs); adds `predict_roster_with_context(roster, history, model, context)`. The synthetic `ShootoutKick` construction is removed.
- **Modified: `src/twelveyards/features.py`.** The feature builder accepts `PredictionTarget` (or, equivalently, the model's row type) instead of `ShootoutKick`. The synthetic-kick construction moves to a test fixture.
- **Modified: `src/twelveyards/model.py`.** `TrainingTableRow` and `TrainingRow` collapse to one type (or, equivalently, the feature builder emits the model's row type). `build_feature_matrix` and `rows_to_predict_matrix` collapse to one. `predict_proba` is removed; the model artifacts implement a `PredictProba` protocol. `_training_row_from_table_row` is removed.
- **Modified: `src/twelveyards/shootouts.py`.** `_int`, `_parse_match_date`, `_SHOTMAP_EVENT_TYPE_TO_OUTCOME`, `parse_page_url`, `_PAGE_URL_RE` deleted (moved to `fotmob_parsing` / `match_ref`). `fetch_season_fixtures` is consumed by `rosters.py` via `match_ref` / `fotmob_parsing`, not duplicated. `fetch_match_data` inlined at its single call site in `scripts/fetch_2022_final.py`. `fetch_all_shootout_kicks` (the pass-through) deleted. `ShootoutMatchRef` deleted in favour of `match_ref.MatchRef`. `LEAGUE_SEASONS_PREDICT_WINDOW` lifted to `tournaments.py`.
- **Modified: `src/twelveyards/rosters.py`.** `_int` deleted. `RosterMatchRef` deleted in favour of `match_ref.MatchRef`. The defensive local `from .shootouts import parse_page_url` lifted to module top (no cycle).
- **Modified: `src/twelveyards/rsssf.py`.** The defensive local `from .leagues import LEAGUE_BY_ID` lifted to module top. `RSSSF_TO_LEAGUE_NAME` lifted to `tournaments.py` (this module is now pure parsing).
- **Modified: `src/twelveyards/coordinates.py`.** `SIDE_LEFT`, `SIDE_CENTER`, `SIDE_RIGHT`, and the `Side = str` alias deleted (no callers).
- **Modified: `src/twelveyards/evaluate.py`.** `kicker_most_frequent_save_rate` either renamed to `last_side_save_rate` or rewritten to compute the per-kicker mode.
- **New: `src/twelveyards/tournaments.py`.** Owns `LEAGUE_SEASONS_PREDICT_WINDOW`, `WC_2026_SEASON`, and the `RSSSF_TO_LEAGUE_NAME` heading map. Adding a tournament edits one module.
- **New: `tests/_factories.py`.** One builder per row type (`make_training_row`, `make_history_row`, `make_metadata`). Imports the schema from `dataclasses.fields(TrainingTableRow)`. The 4 private `_make_row` / `_make_features` helpers are deleted.
- **Modified: `tests/test_*.py`.** Live smoke tests use `Artifacts` defaults (not `Path("output/...")` directly). Private-coupled tests (`_coerce_lightgbm_categoricals`, `_cache_key`, `_decompress`) are replaced with public-interface tests (`predict_proba`, `client.get`, gzip roundtrip).

**Depth discipline (apply the deletion test to every change):**

- If deleting a module concentrates complexity (e.g. a true pass-through), it earns deletion. If deleting it just moves the complexity into N callers, it earns its keep.
- The new modules (`fotmob_parsing`, `match_ref`, `artifacts`, `tournaments`, `tests/_factories.py`, `dashboard.py`) each have one responsibility whose name matches the module name. Anything that doesn't fit the name is a code smell.
- Two adapters justify a seam; one adapter means a hypothetical seam. `fotmob_parsing` and `match_ref` are real seams (multiple consumers, real FotMob-shape quirks). `artifacts` is a real seam (every script and test crosses it). The other new modules are leaf utilities.

**Schema changes:**

- `Artifacts` doesn't introduce a new on-disk format — the JSONL shapes for `ShootoutKick`, `RosterPlayer`, `PlayerPenalty`, `InitialSetKicker`, `MissingKicker`, `TrainingTableRow`, `PredictionRow` are unchanged. The interface is a re-statement of what already exists, not a re-design.
- `PredictionTarget` is a new in-memory type, not a new on-disk format. It carries the 9 feature inputs the model uses (A1 over 5/10/20, A2 total penalties, A3 kicking foot, A4 last side, B1 score & kick number, B2 is_decisive, B3 round, C1 position, C2 age).

**API contracts (preserved):**

- The 9 slice scripts' CLI surfaces (`--output`, `--cache-dir`, `--target-date`, `--holdout-cutoff`, `--lookback-years`, `--history-floor`, `--num-leaves`, `--learning-rate`, `--n-estimators`, `--min-child-samples`, `--C`, `--class-weight`) are unchanged. The default values are read from `Artifacts` instead of inline `Path(...)` literals.
- The 9 live smoke tests' pinned paths (`output/shootout_kicks.jsonl`, `output/lightgbm.pkl`, etc.) are still pinned — they just come from `Artifacts` instead of inline strings.

### Phase 1 — Hugging Face persistence

**Repository layout (single repo, two subpaths):**

- `couto/12yd/` (the repo, owned by user `couto` on HF)
  - `README.md` — the minimal model card
  - `model/lightgbm.pkl` — the frozen LightGBM (LGBMClassifier inside a `LightGBMClassifierWrapper`), trained on all 179 rows, with the feature column order recorded
  - `data/shootout_kicks.jsonl` — 179 target kicks
  - `data/player_history.jsonl` — 745 rows, 265 unique kickers (10-year lookback)
  - `data/wc2026_roster.jsonl` — 1243 unique players across 48 teams
  - `data/predictions.jsonl` — 1243 rows, one per WC player, the artifact the dashboard reads

**The model card (root `README.md`) is minimal:**

- One-paragraph description: "Multiclass classifier (L/C/R) on 9 per-kick features; trained on 179 shootout kicks across 6 national-team tournaments (2021–2026). Frozen deployment artifact for `matheusccouto/12yd`."
- Metrics: the 2026 holdout numbers (log loss 1.551, save rate 0.464, etc.).
- Usage:
  ```python
  from huggingface_hub import hf_hub_download
  p = hf_hub_download("couto/12yd", "model/lightgbm.pkl")
  import pickle; model = pickle.load(open(p, "rb"))
  ```
- A one-line note that the model is round-aware (B3 feature) and that round-agnostic predictions are not what it was trained for.

**Update workflow (manual, no automation):**

- After the slice pipeline re-runs and `output/lightgbm.pkl` + `output/*.jsonl` are refreshed, the user runs:
  ```bash
  huggingface-cli upload couto/12yd . couto/12yd
  ```
- No CI, no scheduled job, no GitHub Action. The push is a known-good artifact on a known-good run.

**Why one repo (not separate model + dataset repos):**

- The dashboard reads from the same place the model lives; a single repo removes the cross-repo coordination.
- HF allows a repo to be a "model" repo with sub-paths; the model card at the root is the model surface, the `data/` subdir is just files. We do not load `data/` as a HF `Dataset` — the dashboard reads the JSONLs through `huggingface_hub.hf_hub_download`.
- The dashboard is the only consumer at read time; an external retrain would clone the repo, not load it as a HF Dataset.

### Phase 2 — Streamlit dashboard

**Module layout:**

- **New: `src/twelveyards/dashboard.py`.** The dashboard's data and re-scoring logic. Functions:
  - `load_upcoming_knockouts(client) -> list[MatchContext]` — fetches the WC 2026 fixture list, filters to upcoming + knockout + both teams decided
  - `predict_match(roster, history, model, context) -> list[KickerPrediction]` — re-scores the match's likely kickers with the match's actual round (B3)
  - `recommended_dive(p_L, p_C, p_R) -> str` — `argmin` over the three
- **New: `app.py` at the repo root.** The Streamlit entry point. ~30 lines:
  - Loads the model from HF (`hf_hub_download("couto/12yd", "model/lightgbm.pkl")`)
  - Loads the roster and player history from HF
  - Builds a `FotMobClient` and calls `load_upcoming_knockouts`
  - Renders the selectbox + per-kicker table
- **Modified: `src/twelveyards/predict.py`.** Adds `predict_roster_with_context(roster, history, model, context)`. The existing `predict_roster(roster, history, model)` becomes a thin wrapper that uses an empty context (backward compatible with the v1 slice).

**Match filter:**

- Upcoming: `kickoff > now` (UTC).
- Both teams decided: each side has a non-placeholder team id (skip "Winner Group A", "Loser QF 1", etc.). The round is a display attribute, not a filter (v3 dropped the round allowlist; see PRD-v3 Issue 1).

**Fixture source:**

- Live FotMob fetch via `twelveyards.client.FotMobClient` at dashboard load. The `leagues/77/overview/world-cup` endpoint is the source of truth. The persistent ETag/gzip cache means re-loads are 304 hits.
- No pre-fetch; no `data/wc2026_fixtures.jsonl` artifact.

**Re-score path (round-aware):**

- For each (kicker, match) pair, build the 9-feature row with `b3_round=match.round` (instead of `""`).
- Call `predict_roster_with_context(roster, history, model, context)` where `context = {round: match.round, kick_number: 1, pen_score_before: [0, 0], is_decisive: False}`.
- The model's `predict_proba` returns `p_L`, `p_C`, `p_R`. The recommended dive is `argmin`.

**Hosting: Streamlit Cloud.**

- `app.py` at the repo root is the default entry point.
- `pyproject.toml` (already present) is the dependency manifest; Streamlit Cloud supports it.
- The model + data are loaded from HF at app startup (cached by Streamlit's `@st.cache_resource` for the lifetime of the process).

**UI surface (single page):**

- A selectbox of upcoming knockout matches (sorted by kickoff).
- A per-kicker table for the selected match: kicker name, team, kicking foot, P(L), P(C), P(R), recommended dive. Both teams in one table, sorted by `total_penalties` descending.
- No historical-tournaments tab, no per-kicker photos, no country flags in v1. The minimal render is the data; polish is later.

### Architectural decisions (cross-phase)

- **Two-level data graph** (Initial Set → Derived History) preserved from v1. No recursion past two levels. The new `initial_set.py` enforces this at the seam: the orchestrator takes `Iterable[InitialSetKicker]`, never fetches per-row.
- **JSONL** as the on-disk format. Unchanged from v1. `Artifacts` is the seam that owns the shape.
- **LightGBM**, not AutoML, not a neural net. Unchanged. The frozen model is the same `.pkl` from v1.
- **The 5-year Lookback Window and the 2016-01-01 History Floor** are scraper config, not dataset properties. Unchanged.
- **RSSSF** is a verification oracle, never a data source. Unchanged.
- **HF is a file store, not a service.** The dashboard reads from it like a CDN; the scraper pushes to it like a backup. No HF-specific protocol (no HF `Dataset` loading, no HF `Inference` API).

## Testing Decisions

**What makes a good test (unchanged from v1):**

- Test external behaviour, not implementation. A test that asserts "predictions sum to 1" is good; a test that asserts "the column order is exactly this list" is fragile.
- Independent ground truth where possible. RSSSF is the ground truth for shootout count. The 2022 World Cup Final (sample at `docs/samples/match_3370572.json.gz`) is the ground truth for per-kick data.
- Tests cross the same seam as production. If the test surface differs from the production surface, the test is wrong.

**What changes in Phase 0:**

- The 4 duplicated `_make_row` / `_make_features` helpers are deleted. The 4 test files import from `tests/_factories.py`. The factory derives the schema from `dataclasses.fields(TrainingTableRow)` — so adding a field is a one-line change, not a five-place hunt.
- The private-coupled tests (`_coerce_lightgbm_categoricals`, `_cache_key`, `_decompress`) are replaced with public-interface tests (`predict_proba` on unseen categoricals, `client.get` cache roundtrip, gzip detection on edge-case bodies). The new tests pin the same behaviour with less coupling.
- The 9 live smoke tests use `Artifacts` defaults. The pinned artifact paths are unchanged; the source of the path string changes from inline `Path("output/...")` to `Artifacts().<artifact>`. Re-runs that move the artifact directory (e.g. `--output=/tmp/foo`) now pass without code changes.
- New tests for the new modules:
  - `test_fotmob_parsing.py`: `coerce_int` (None, "", 0, "5", "abc", 5.0), `parse_match_date` (RFC 2822, ISO 8601, malformed, empty), the outcome map (Goal, AttemptSaved, Miss, Post, unknown).
  - `test_match_ref.py`: `parse_page_url` (3-segment, 2-segment, malformed, with anchor), `MatchRef.from_fixture` (shootout fields, roster fields, missing fields).
  - `test_artifacts.py`: path accessors, JSONL roundtrip with NaN→null, pickle roundtrip, JSON roundtrip with MetricsReport, FotMobClient factory.
  - `test_dashboard.py`: match filter (upcoming, knockout, both-teams-known), re-score with context, recommended dive, integration with the extended `predict.py`.

**What changes in Phase 1:**

- No new tests. The HF push is a manual `huggingface-cli upload`; the dashboard's tests in Phase 2 cover the read path.

**What changes in Phase 2:**

- The dashboard's logic (filtering, re-scoring, recommended dive) is tested in `test_dashboard.py` against fixture data. The Streamlit UI itself is not unit-tested (Streamlit apps aren't typically unit-tested; the seam is the library).
- The end-to-end flow (HF model load → FotMob fixture fetch → re-score → table) is verified manually after deployment to Streamlit Cloud, with a one-page checklist: every upcoming knockout match with both teams decided shows in the selectbox, kicker lists populate, predictions sum to 1, recommended dive is `argmin`. (v3 dropped the round allowlist — see PRD-v3 Issue 1 — so the checklist is "every round", not a fixed count.)

**Modules tested in Phase 0 + Phase 2:**

- `fotmob_parsing` (new)
- `match_ref` (new)
- `artifacts` (new)
- `tournaments` (new)
- `initial_set` (new, split from `player_history`)
- `dashboard` (new in Phase 2)
- The 12 existing modules (unchanged in interface, but their tests now use `Artifacts` defaults)

## Out of Scope

- **Real-time retraining during the WC.** The model is frozen before the WC starts (unchanged from v1). Phase 1 is one-shot, not continuous.
- **Opponent / goalkeeper features.** The model has the round (B3) but no opponent or keeper feature. The "Recommended Dive" is `argmin` over a uniform-prior policy for the keeper, not a per-keeper adjustment. The dashboard shows what the model knows, which is the kicker.
- **Per-keeper dive direction data.** The "actual keeper" baseline in `metrics.json` is `null` (v1 explicitly notes this is N/A until a data source like StatsBomb adds the data). Not in scope for v2.
- **Multi-task heads (regression on `x` in addition to classification on L/C/R).** Unchanged from v1.
- **AutoML / neural-net alternatives.** LightGBM is the model. Revisit if it underperforms.
- **Photos, country flags, historical-tournament tabs in the dashboard.** The minimal dashboard shows the per-kicker predictions; visual polish is later.
- **All 48 teams at once (group stage).** The dashboard is knockout-only. Group-stage shootouts don't exist (matches end in regulation or extra time, not penalties).
- **A "winner prediction" for undecided R16 slots.** The dashboard hides matches until both teams are decided. A "predict group winners" model is a different project.
- **HF Spaces (instead of Streamlit Cloud).** Considered in the architecture review; Streamlit Cloud chosen for simpler deployment. HF Spaces remains an option if requirements change.
- **CI / scheduled push to HF.** The push is manual. Adding CI is a later iteration.

## Further Notes

- **v1 history.** The v1 work is in the git history: scraper (Issues #17–#21), features (Issue #22), baseline (Issue #23), LightGBM (Issue #24), live predictions (Issue #25). The v1 PRD (`docs/PRD.md` before this rewrite) defined 31 user stories and the scraper + model + deferred-dashboard scope. v2 supersedes it.
- **The data and model artifacts are gitignored.** They are not in the repository and not in any remote. Before Phase 1, the slice pipeline re-runs to regenerate them. The pipeline is idempotent and re-runnable; the cache (`data/fotmob_cache/`) is gitignored and not required for the regenerate (a cold run is ~minutes-to-hours depending on the slices).
- **The architecture review.** The Phase 0 cards were surfaced by a 3-sub-agent scan of the v1 codebase (data layer / model layer / caller view) using the `/improve-codebase-architecture` and `/codebase-design` skills. The full report is at `/tmp/architecture-review-1782684706.html` (8 cards, ranked by leverage; the disk/artifact layout was the top recommendation). The review is the source of the 7 simplification moves in Phase 0.
- **The two ADRs.** Phase 1 + Phase 2 each make architectural decisions worth recording: (1) HF for persistence, (2) Streamlit Cloud for hosting, (3) single HF repo with `model/` + `data/` subpaths. The ADRs live in `docs/adr/` and reference the v1's git history as the prior art.
- **Why no `progress.txt` going forward.** v1 used `progress.txt` as an agent's working memory; the user removed it because it was gitignored and not persistent. v2 has no equivalent: the PRD is the durable plan, the ADRs are the durable decisions, the issue tracker is the durable task list, and the code is the durable state. If a future session needs a working log, it should live in the session, not in the repo.
- **Why the dashboard is in the same PRD (not a follow-up).** v1 deferred the dashboard to "a follow-up PRD". v2 pulls it in because the persistence layer (Phase 1) is a precondition for the dashboard, and the user wants the v2 plan as a single document.
