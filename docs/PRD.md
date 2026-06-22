## Problem Statement

Penalty shootouts are decided by ~5–10 kicks per match, with the goalkeeper's pre-kick dive direction being a major lever. Today, keepers have no per-kicker, per-side probability to inform their dive. We want a model that, for any given kicker in an upcoming shootout, outputs P(left), P(center), P(right) for their intended kick side — so a keeper can dive the lowest-probability side (or stay center) instead of guessing.

## Solution

Build a two-stage system:

1. **Scraper** that pulls penalty-shootout data from FotMob (`docs/fotmob.md`) for six major national-team tournaments (World Cup, Euro, Copa América, CONCACAF Gold Cup, AFC Asian Cup, Africa Cup of Nations), plus each kicker's penalty history (shootout + in-match) over a 5-year lookback, floored at 2016-01-01. Output: JSONL files.
2. **LightGBM classifier** trained on the scraped data, taking 9 per-kick features (kicker's recent side distribution, kicking foot, kick number, current score, round, position, age, career penalty count, last kick's side) and outputting P(L), P(C), P(R) via softmax. The keeper's optimal action is `argmin` over the three probabilities.

The model is trained once on all available data (2021-01 → 2026-06-11) and frozen before the 2026 World Cup. Predictions are produced live as knockout rounds approach. The 2026 World Cup is in progress as of 2026-06-22 (group stage, no shootouts yet), so we scrape WC squad players' penalty history now and produce predictions on demand once a shootout is imminent.

A Streamlit dashboard (separate PRD) will display these predictions per player.

## User Stories

### Scraper
1. As a data engineer, I want a FotMob HTTP client with gzip + ETag + persistent disk cache, so that re-runs are free of network cost.
2. As a data engineer, I want to fetch all shootout matches in the 6 in-scope tournaments between 2021-01-01 and today, so that I have a complete list of target Shootout Kicks.
3. As a data engineer, I want to filter shootout matches by `status.reason.shortKey == "penalties_short"` at the league-fixture level, so that I don't pay for fetching non-shootout match details.
4. As a data engineer, I want to extract the full kick list per shootout from `pageProps.content.shotmap.shots` filtered to `period == "PenaltyShootout"`, so that I get `onGoalShot.x` for every kick including missed/saved ones (not just from the `penaltyShootoutEvents` summary, which omits the shotmap for non-Goals).
5. As a data engineer, I want each shootout kick in JSONL with: `match_id`, `match_date`, `tournament_id`, `tournament_name`, `round`, `kick_number`, `kicker_id`, `kicker_name`, `team_id`, `is_home`, `x` (continuous, [0, 2]), `side` (L/C/R bucketed), `is_on_target`, `outcome` (Goal / Saved / Missed), `pen_score_before`, `pen_score_after`, `match_score_home`, `match_score_away`.
6. As a data engineer, I want to fetch the 2026 World Cup squad list as the Prediction Initial Set, so the dashboard has the candidate Kickers to score even before any WC shootouts occur.
7. As a data engineer, I want, for every Kicker in the union of Training and Prediction Initial Sets, to fetch all their penalty kicks (shootout + in-match) between 2016-01-01 and today via `careerHistory` (senior + national team stints only) → per-team-season fixtures → per-match details.
8. As a data engineer, I want the player's `position` and `dateOfBirth` from `pageProps.data.playerInformation` on the player page, so I can compute the `C1` and `C2` features.
9. As a data engineer, I want each player's penalty history in JSONL with: `kicker_id`, `match_id`, `match_date`, `league_id`, `league_name`, `team_id`, `is_home`, `x`, `side`, `is_on_target`, `outcome`, `shot_type` (RightFoot / LeftFoot).
10. As a data engineer, I want a scraped-data validator that asserts the count of shootout matches I found equals the count of shootouts in the RSSSF page (`https://www.rsssf.org/miscellaneous/penaltiestour.html`) for the in-scope tournaments in the prediction window, so that I have an independent completeness oracle.
11. As a data engineer, I want a scraped-data validator that asserts every shootout kick has a populated `x` and `side`, so that I know my label coverage is complete.
12. As a data engineer, I want a scraped-data validator that asserts every Kicker in the Initial Set has at least one penalty in the lookback window, so I know my features are not all-NaN.
13. As a data engineer, I want the scraper to be re-runnable and idempotent (deterministic output for the same inputs), so that re-running after a fix doesn't double-count.
14. As a data engineer, I want a single scraper config that lists the 6 in-scope tournaments (`{leagueId, slug}` pairs), the prediction window bounds, the lookback window, and the history floor, so that the same code works when we extend to other tournaments or earlier years.

### Model
15. As an ML engineer, I want a feature builder that, given a target Shootout Kick and the player's penalty history, produces a feature row with the 9 features (A1, A2, A3, A4, B1, B2, B3, C1, C2), so that each target kick has a complete feature vector.
16. As an ML engineer, I want A1 (historical side distribution) computed over the last 5, 10, and 20 kicks before the target kick date (anchored to the History Floor), so that the model has multi-horizon recency signals.
17. As an ML engineer, I want A3 (kicking foot) derived from the mode of `shot_type` in the player's history (with a fallback to "Unknown" if no history), so that the feature is robust to short histories.
18. As an ML engineer, I want C2 (age) computed as `target_match_date - dateOfBirth` in years, so the model has a per-kick age rather than a per-today age.
19. As an ML engineer, I want a LightGBM multiclass classifier (3 classes: L, C, R) trained on all shootout kicks in the prediction window, with class-weighted cross-entropy as the loss, so that the model outputs calibrated probabilities.
20. As an ML engineer, I want the training to use class weights inversely proportional to class frequency, so the model doesn't collapse to predicting the majority class (C is rarer than L or R).
21. As an ML engineer, I want a held-out evaluation: train on shootout kicks before 2025-12-31, evaluate on shootout kicks from 2026-01-01 onward, reporting log loss, accuracy, and counterfactual save rate, so I have a metric before deploying to the live WC.
22. As an ML engineer, I want a final model retrained on all shootout kicks from 2021-01-01 through 2026-06-11 (the day before the WC), so the deployed model uses as much data as possible.
23. As an ML engineer, I want the trained model serialized as a single artifact (e.g. `model.pkl` with the LightGBM booster and the feature column order), so the prediction step is self-contained.
24. As an ML engineer, I want a `predict.py` that, given a Kicker id and a target date, returns `{kicker_id, p_L, p_C, p_R}` using that kicker's scraped history, so the dashboard can call it.

### Evaluation
25. As an analyst, I want a counterfactual save rate: for each historical shootout kick, the model's recommended dive is `argmin(p_L, p_C, p_R)`; if the kicker's actual side matches, that's a save (if on-target) or a miss (off-target, always a save); the metric is total saves / total kicks, so I can quantify the model's keeper-side value.
26. As an analyst, I want three baselines computed on the same set: random (33.3%), kicker's most-frequent historical side, and the actual keeper's dive (when recoverable), so I can put the model's save rate in context.
27. As an analyst, I want the headline metrics reported in `metrics.json` after each training run, so they're auditable.

### Dashboard (deferred to a follow-up PRD)
28. As a viewer, I want a Streamlit app that lists 2026 WC players with their P(L), P(C), P(R), so I can see the model's predictions for the upcoming knockout rounds.
29. As a viewer, I want each player row to show name, photo, country, country flag, and preferred foot, so I can quickly identify the kicker.
30. As a viewer, I want to filter by team, so I can focus on a country's likely taker list.
31. As a viewer, I want historical tournaments available as a secondary tab, so I can sanity-check the model against past shootouts.

## Implementation Decisions

### Module layout (no specific file paths; the engineer picks)

- **`fotmob_client`**: HTTP client encapsulating the `__next/data` pattern, gzip request, ETag revalidation, and persistent disk cache. Header constants per `docs/fotmob.md`. BuildId discovery is one-time per process; cached for the lifetime of the run.
- **`leagues`**: Constant table of the 6 in-scope tournaments (`{leagueId, slug}` pairs). The leagueId is the FotMob integer ID, e.g. World Cup = 77, Euro = 50, Copa América = 44, Gold Cup = 298, Asian Cup = 290, AFCON = 289.
- **`shootouts`**: Orchestrates the league-fixture fetch for each tournament × season in the prediction window, filters to `status.reason.shortKey == "penalties_short"`, fetches each match's details, and extracts the shootout kick records from `pageProps.content.shotmap.shots` filtered to `period == "PenaltyShootout"`. Each kick carries the per-kick fields listed in story 5.
- **`rosters`**: Fetches the 2026 World Cup squad list. Uses the WC 2026 league fixtures (leagueId 77) to discover match lineups, then deduplicates to a unique player list. (The exact endpoint may be the WC 2026 `overview` page or the per-match `lineup` block — to be confirmed by prototype.)
- **`player_history`**: For each Kicker in the union of Training and Prediction Initial Sets, fans out via `careerHistory` (`careerItems.senior` and `careerItems["national team"]`; skip `careerItems.youth`). For each (team, season) overlap with the lookback window, fetches that team's season fixtures, then each match's `shotmap.shots` filtered to penalty shots where the kicker is the shooter. Output: one row per kicker-penalty.
- **`features`**: Given a target Shootout Kick and the kicker's penalty history (filtered to before the target date), produces the 9-feature row. Bucket for side: `L = x < 0.667`, `C = 0.667 ≤ x ≤ 1.333`, `R = x > 1.333`. The kicking-foot feature is the mode of `shot_type` over history (tie → "RightFoot", since the population is right-foot-dominant).
- **`model`**: LightGBM multiclass (3 classes). Loss: multiclass log-loss. Class weights: inverse frequency computed on the training fold. Random seed: fixed. Hyperparameters: conservative defaults (`num_leaves=31`, `learning_rate=0.05`, `n_estimators=500`, `min_child_samples=20`) — no aggressive tuning in v1.
- **`evaluate`**: Computes log loss, per-class precision/recall, and the counterfactual save rate against three baselines (random, kicker's most-frequent side, and where recoverable, the actual keeper's dive). Writes `metrics.json`.
- **`pipeline`**: Top-level entry point that runs (1) shootout fetch, (2) roster fetch, (3) player history fetch, (4) feature build, (5) train, (6) evaluate, (7) emit a frozen model artifact and a `predictions.jsonl` for the WC roster.

### Schemas (decision-rich shape; not full code)

**`shootout_kicks.jsonl` row**:
```
{
  "match_id": int,            // FotMob eventId
  "match_date": ISO8601,
  "tournament_id": int,       // FotMob leagueId
  "tournament_name": str,
  "round": str,               // e.g. "Final", "Quarter-finals", "2R"
  "kick_number": int,         // 1..N within the shootout
  "kicker_id": int,
  "kicker_name": str,
  "team_id": int,
  "is_home": bool,
  "x": float,                 // continuous, [0, 2], from shotmap.shots
  "side": "L"|"C"|"R",        // bucketed
  "is_on_target": bool,
  "outcome": "Goal"|"Saved"|"Missed",
  "pen_score_before": [int, int],
  "pen_score_after": [int, int],
  "match_score_home": int,
  "match_score_away": int
}
```

**`player_history.jsonl` row** (kicker-history penalty):
```
{
  "kicker_id": int,
  "match_id": int,
  "match_date": ISO8601,
  "league_id": int,
  "league_name": str,
  "team_id": int,
  "is_home": bool,
  "x": float,
  "side": "L"|"C"|"R",
  "is_on_target": bool,
  "outcome": "Goal"|"Saved"|"Missed",
  "shot_type": "RightFoot"|"LeftFoot"
}
```

**`predictions.jsonl` row** (WC player scored by frozen model):
```
{
  "player_id": int,
  "player_name": str,
  "team_id": int,
  "team_name": str,
  "country_code": str,        // ISO 3166-1 alpha-3
  "kicking_foot": "LeftFoot"|"RightFoot"|"Unknown",
  "p_L": float,
  "p_C": float,
  "p_R": float
}
```

### API/data contracts

- The FotMob match route is `GET /_next/data/{buildId}/matches/{seo}/{h2h}.json` (two segments), not the single-segment `{slug}` form documented in the older `docs/fotmob.md` line 51. The doc needs updating.
- Shootout kick placement comes from `pageProps.content.shotmap.shots` filtered to `period == "PenaltyShootout"`, NOT from `penaltyShootoutEvents[*].shotmapEvent` (the latter is missing for missed/saved kicks).
- The "kicker history" is per-kicker, per (team, season) overlap with the lookback window; no recursion past two levels (Initial Set → Derived History).

### Architectural decisions

- **Two-level data graph** (Initial Set → Derived History), with no recursion. A scraper that fans out from Derived History is a bug, not a feature.
- **JSONL** as the on-disk format. One row per record, easy to `head`, easy to stream, plays well with pandas/polars.
- **LightGBM**, not AutoML, not a neural net. Tabular, small dataset (~336 training rows, 9 features), interpretable, fast to retrain, easy to deploy.
- **The 5-year Lookback Window and the 2016-01-01 History Floor are scraper config, not dataset properties.** Change the config; the dataset is reusable.
- **RSSSF** is a verification oracle, never a data source. We assert our scraper finds the same count of shootouts RSSSF lists, and we investigate discrepancies.

## Testing Decisions

- **Test external behavior, not implementation.** For the scraper, that means: the JSONL files exist, have the right schema, the right count of shootouts vs. RSSSF, the right count of shootout kicks per match (5–10), every kick has a populated `x` and `side`. For the model, that means: predictions are valid probability distributions (sum to 1, all non-negative), the metric on the 2026 holdout beats the random baseline by a meaningful margin, and the metric does not regress between training runs of the same code.
- **What makes a good test**: independent ground truth. RSSSF is the ground truth for the count of shootouts. The 2022 World Cup Final (sample at `docs/samples/match_3370572.json.gz`) is the ground truth for per-kick data — every test case should include at least one assertion that the scraper reproduces the 8 kicks with their correct `x` values.
- **Modules tested**: shootouts (count vs. RSSSF, per-kick completeness), player_history (per-kicker overlap with lookback), features (monotonicity, no NaNs on the 2022 final's 8 kickers), model (probabilities sum to 1, model outperforms random on the 2026 holdout), evaluate (counterfactual save rate matches a hand-computed value for a known fixture).
- **Prior art for tests**: there is no prior test infrastructure in the repo (`.gitignore` is the only existing config file). The pattern to follow is a small pytest suite where each test reads a fixture (sample JSONL or a known match) and asserts a specific property of the scraper/model output. No mocking — the scraper is the integration point, and the FotMob cache layer makes re-runs deterministic.
- **The validation we skip**: live in-WC metrics (we won't have shootouts to score against for a few weeks). The 2026 holdout covers most of what we'd learn from live tracking; if the model beats random on the holdout, we have a reasonable expectation for the WC.

## Out of Scope

- The Streamlit dashboard. A separate PRD will follow once the model is validated.
- Club cup competitions (UEFA Champions League, Europa League, Copa Libertadores, etc.). The fetcher's per-league architecture makes this a config addition later.
- UEFA Nations League Finals. Discovered shootouts from the league fixture endpoint don't include the Finals; out of scope for v1.
- Youth team data. The `careerHistory.youth` block is skipped.
- Per-keeper identity features (e.g., the keeper's historical dive rate against a specific kicker). Requires keeper attribution per shootout, which is not in the current data path.
- Real-time retraining during the WC. The model is frozen before the WC starts.
- Multi-task heads (regression on `x` in addition to classification on L/C/R). The classification head is sufficient for v1.
- AutoML / neural-net alternatives. LightGBM is the v1 model; revisit if it underperforms.

## Further Notes

- The 2026 World Cup starts 2026-06-11 and is currently in the group stage. The prediction window includes the WC. There are no shootouts yet; the scraper should scrape the 2026 WC squad list now and the penalty history for every squad player, so the dashboard can produce predictions the moment a knockout round is imminent.
- Two follow-up doc fixes needed (now resolved via issue #16): (1) `docs/fotmob.md` line 51–53 documents the single-segment `matches/{slug}.json` route; the live route is two-segment `matches/{seo}/{h2h}.json`. (2) `docs/fotmob.md` line 129 says missed kicks have no shotmapEvent; this is true only of `penaltyShootoutEvents[*].shotmapEvent`, not of `pageProps.content.shotmap.shots`, which carries `onGoalShot` for every shootout kick. Both corrections have been applied and the live route verified: `GET /_next/data/5JrXFqDcvBep-L0Qv6mBO/matches/argentina-vs-france/1hox8a.json` returns 200 with all 8 shootout kicks.
- Verification against RSSSF: the expected count of shootouts in the 6 in-scope tournaments in [2021-01-01, today] is 42 (5 WC 2022 + 4 Euro 2020 + 3 Euro 2024 + 3 Copa 2021 + 4 Copa 2024 + 6 AFCON 2021 + 5 AFCON 2023 + 3 AFCON 2025 + 2 Gold Cup 2023 + 3 Gold Cup 2025 + 4 Asian Cup 2023). If the scraper returns a different count, that's the first thing to investigate.
- Domain glossary is at `CONTEXT.md`. The fetcher's two-level graph (Initial Set → Derived History) is captured there as the recursion guard. Any future change to those terms should be made there first.
- The keeper's optimal action is `argmin(p_L, p_C, p_R)` — i.e., dive the lowest-probability side, or stay center if C is the minimum. The model never needs to know the keeper's identity; it just outputs the kicker's distribution.
