# 12yd

Predicts which side (L / C / R) a penalty kicker will aim for, so a goalkeeper
can choose the lowest-probability side to dive. Uses **TabPFN** (a Tabular
Foundation Model) on per-player penalty history from FotMob. Predictions are
match-agnostic: each roster player is scored once, and the same prediction row
serves any match they appear in.

## Setup

```bash
uv sync
```

## Usage

The whole pipeline runs as a scheduled GitHub Actions workflow
(`scrape-and-predict.yml`), daily at 06:00 UTC. To run it manually:

```bash
# Fetch the WC 2026 roster
uv run python scripts/fetch_wc_2026_roster.py

# Fetch penalty history for every roster player
uv run python scripts/fetch_initial_set_player_history.py

# Fit TabPFN and write predictions
uv run python scripts/predict.py
```

To launch the Streamlit dashboard locally:

```bash
uv run streamlit run app.py
```

## Layout

- `src/twelveyards/` — library code (HTTP client, feature builder, TabPFN wrapper, prediction pipeline).
- `scripts/` — thin CLI entry points for the pipeline steps.
- `app.py` — Streamlit dashboard with two-team dropdowns.
- `data/` — runtime artifacts:
  - `wc2026_roster.jsonl` — WC 2026 squad (tracked in git).
  - `player_history.jsonl` — per-player penalty history (tracked in git).
  - `predictions.jsonl` — TabPFN predictions (tracked in git).
  - `fotmob_cache/` — persistent HTTP cache (gitignored, cached in Actions).
- `tests/` — pytest suite.
- `.github/workflows/scrape-and-predict.yml` — scheduled pipeline.
