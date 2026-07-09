# GitHub storage and architectural pivot

The v5 PRD pivots the project away from trained models (LightGBM) toward a Tabular Foundation Model (TabPFN, via `tabpfn-client`), collapses the pipeline into a single scheduled GitHub Actions workflow, and simplifies the scraper and app. Three previous ADRs are superseded by this one.

## Context

The v3 independent review found the LightGBM model statistically indistinguishable from a uniform-random dive policy on the 6-fold LOTO CV aggregate (0.374 vs 0.405, SE 0.036). Three open issues (#42 drop A1, #46 anti-classifier, #51 Phase 3 more data) are all model-levers that the review concluded cannot beat the current noise floor at n=179. The model stack (LightGBM + logreg + LOTO CV + metrics pipeline) represents ~2K LOC of code producing no statistically meaningful signal. Meanwhile, the scraper has ~3,500 LOC across 14 modules, the pipeline is manual and multi-system (FotMob data → local model training → HF upload), and the HuggingFace persistence layer (ADRs 0001, 0003) adds a second system with a separate push step and a separate account.

## Decision

**1. Replace the trained-model stack with TabPFN.** TabPFN is a zero-shot tabular foundation model accessed via `tabpfn-client` (Prior Labs cloud API, free tier 50M tokens/day, daily reset at 00:00 UTC). The client provides a sklearn-compatible `TabPFNClassifier` with `fit` + `predict_proba`. Fit is free; only predict consumes tokens. One full run over ~1000 training rows + 1243 test rows × ~7 columns × 8 estimators costs ~125K tokens — roughly 0.25% of one day's free quota. The model is called in cheapest mode: `thinking_mode=False`, `thinking_effort=None`, `n_estimators=8` (defaults). Authentication is via the `TABPFN_TOKEN` environment variable, read automatically by `tabpfn_client.init()`.

The training set is built from `player_history.jsonl`: each player's chronological penalties become training rows where features are derived from prior kicks within a 5-year rolling time window and the label is the current kick's side (L/C/R). The test set is every player on the WC 2026 roster, scored once with the same feature derivation applied to their full penalty history (match-agnostic — no opponent features, no match-context features). The prediction output is the same `predictions.jsonl` schema as before (`PredictionRow` with `p_L/p_C/p_R` per player), preserving the app contract.

Trade-off: TabPFN is a black-box cloud API (data is sent to Prior Labs servers; fit runs on their infrastructure). The free-tier daily reset at 00:00 UTC is ample headroom (a full run is ≪1% of daily quota). The alternative — continuing to iterate on LightGBM — would require significantly more training data to overcome the noise floor, which the v3 review already rejected as infeasible without the Phase 3 club-shootout expansion (a scraper complexity increase, not a model fix).

**2. Store artifacts in GitHub, drop HuggingFace.** The three JSONL artifacts (`wc2026_roster.jsonl`, `player_history.jsonl`, `predictions.jsonl`) are tracked in the repo under `data/`. The scheduled workflow commits and pushes them. The Streamlit Cloud app reads them directly from the cloned working tree (no HTTP fetch, no `hf_hub_download`). This drops one external system (HF account, `huggingface-cli` dependency, separate push step), one secret (`HF_TOKEN`), and one runtime dependency (`huggingface_hub`).

The dataset is small: roster ~1.2 MB, player_history ~5–10 MB, predictions ~300 KB — total well under 50 MB, safely within GitHub's file-size and repo-size limits. JSONL is line-based and diffs cleanly, so git history is useful (every commit records a dataset snapshot) and the churn is bounded (one commit per daily run).

Trade-off: GitHub is not purpose-built for datasets (no model card, no HuggingFace Dataset surface, no HF Spaces integration). But the project's app is a Streamlit Cloud deployment that clones the repo, making GitHub the zero-configuration choice. If a future iteration needs a model card or a public dataset surface, it can be re-added as a separate HF repo with no structural cost — the source of truth stays in the git history.

Considered: keeping HuggingFace (would preserve the existing HF integration and the model-card surface, but adds a second system, a separate CLI install step, and an extra account/secret for a dataset small enough to live in the repo). Also considered: S3 (more setup than HF with no model card), git-LFS (overkill at this size, adds LFS quota management).

**3. Drop the shootout extraction pipeline and the RSSSF validation oracle.** With TabPFN as the estimator and a player-only, match-agnostic feature set, `shootout_kicks.jsonl` is no longer needed (the training set comes from `player_history.jsonl` alone). The RSSSF completeness assertion (42 expected shootouts → 36 reachable → validate count) was a quality gate for a model trained on shootout-specific data; under a TabPFN in-context learning regime, the completeness concern shifts to "do we have enough per-player penalty history," which is self-correcting (the scraper accumulates history over time, and the HTTP cache makes re-runs cheap). Dropping these two subsystems removes ~1,300 LOC of scraper complexity.

**4. Single GitHub Actions pipeline (`scrape-and-predict.yml`).** Daily cron at 06:00 UTC + `workflow_dispatch`. The workflow restores the FotMob HTTP cache from the most recent run via `actions/cache` (run_id-keyed with prefix restore-keys), scrapes the roster and per-kicker penalty history, fits+predicts with TabPFN, commits and pushes the JSONLs, and saves a fresh cache. The cache's buildId-stripped keying (a fix in `client.py:_cache_key`) and the shared httpx connection pool (a fix in `client.py`) make warm re-runs ~1-3 minutes instead of ~3 hours.

**5. Simplify the app to two independent team dropdowns.** Drop the live FotMob fixture fetch; replace with two `st.selectbox` dropdowns derived from the distinct teams in `predictions.jsonl`. No conditional filtering, no match header, no round context, no plotly. The app becomes a static dataset viewer with zero live network calls.

## Superseded ADRs

- **ADR-0001 (Hugging Face for persistence)** — replaced by GitHub storage. The HF account (`couto/12yd`) and the `huggingface_hub` dependency are no longer needed.
- **ADR-0003 (Single HF repo with subpaths)** — replaced by GitHub storage. The `model/` and `data/` subpath convention on HF is replaced by a single `data/` directory in the repo.
- **ADR-0004 (Phase 3 data source: club shootouts)** — the shootout pipeline is dropped. The club-shootout expansion was a data-lever to improve LightGBM's statistical power; TabPFN as a foundation model does not target a trained-model accuracy threshold, and the shootout-specific data path is gone.

ADR-0002 (Streamlit Cloud for hosting) is retained, unchanged.

## Consequences

- The HuggingFace repository `couto/12yd` becomes archival. No new artifacts are pushed to it. The existing model card and data files there remain as a historical snapshot.
- The `huggingface-cli` install and `HF_TOKEN` secret are removed from `ralph.yml` and the new workflow.
- The `huggingface_hub`, `scikit-learn`, `lightgbm`, and `plotly` packages are dropped from `pyproject.toml`. `tabpfn-client` is added.
- Dependency groups (PEP 735) segregate app-only deps (`streamlit`) from pipeline-only deps (`tabpfn-client`) from dev deps, so Contributors install only what they need.
- The scraper's HTTP cache (`data/fotmob_cache/`) stays gitignored; it is the persisted state across Actions runs. BuildId-stripped cache keys and a shared httpx connection pool are the two code changes that make this work.
- A standalone offline metrics notebook (holdout log-loss, Brier/ECE, save rate) is deferred to a future ticket — it is an analysis tool, not part of the scheduled pipeline.

## References

- `docs/PRD-v5.md` — the v5 PRD (this cycle)
- `docs/model-review.md` — v3 independent review (the source of the noise-floor finding; now archived)
- `docs/adr/0001-hugging-face-for-persistence.md` — superseded
- `docs/adr/0003-single-hf-repo-with-subpaths.md` — superseded
- `docs/adr/0004-phase-3-data-source.md` — superseded
- `docs/adr/0002-streamlit-cloud-for-hosting.md` — retained
