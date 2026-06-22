# 12yd

Predicts the side (L / C / R) a player will kick a penalty in a shootout, so a goalkeeper can pick the lowest-probability side to dive.

## Setup

```bash
uv sync
```

## Usage

Fetch a known shootout and write its kicks to JSONL:

```bash
uv run python scripts/fetch_2022_final.py
```

## Layout

- `src/penalty_pred/` — library code (HTTP client, leagues, shootout extractor, coordinates, config).
- `scripts/` — thin CLI entry points per issue.
- `tests/` — pytest suite.
- `data/fotmob_cache/` — persistent HTTP cache (gitignored).
- `output/` — generated JSONL artifacts (gitignored).
