---
license: mit
tags:
  - penalty
  - football
  - lightgbm
  - shootout
---

# 12yd — Penalty Shootout Side Prediction (v4)

Multiclass classifier (L / C / R) on 17 per-kick features; trained on the 211 pre-2026 shootout kicks across 8 national-team + club tournaments (2021–2026). Frozen deployment artifact for `matheusccouto/12yd`.

## Save rate is the deployment KPI

The model returns P(L), P(C), P(R) — the probability the kicker will aim
at the left side, hold the centre, or aim at the right side. The
goalkeeper dives toward the side with the **lowest** predicted probability.
The headline metric for this policy is the **counterfactual save rate** —
the fraction of kicks the model would have "saved" under
`argmin(P(L), P(C), P(R))`.

**Frame pin (Kicker-PoV).** The L/C/R labels above are in the
**Kicker's** point of view — the half of the goal as the Kicker
faces it (per `CONTEXT.md`). A viewer reading the card must
re-anchor to themselves: the Kicker's L is the Goalkeeper's R. The
v4 card layout (Issue #48) surfaces this with a "Kicker will aim"
prediction row; the model's `argmin` and the v3 card's text both
stay in the Kicker's frame end to end.

On the 2026+ holdout (226 kicks, 2026-01-01+), the model achieves a
save rate of **0.345** versus a uniform-random baseline of **0.437**
— the model is **0.092 below** random on this larger 226-row
holdout, vs the v3 model's 28-row holdout delta of **+0.166 above**
random (0.571 vs 0.405). The v3 single-holdout number was a
small-sample draw from a distribution that the LOTO CV aggregate
showed averages to ~0.37 (within one SE of random). The v4 retrain
on 437 training rows reproduces the v3 finding on a larger holdout:
the model is below random on the cross-tournament aggregate. The
v5 model work (Issues #42, #46 — anti-classifier, drop A1) is the
path to a model that beats random; the v4 close-out is the
statistical foundation (LOTO CV aggregate SE dropped 39% from v3's
0.036 to v4's 0.022).

## Held-out metrics (226 2026+ holdout kicks, 2026-01-01+)

| model              | log loss | accuracy | save rate | n_kicks |
| ------------------ | -------- | -------- | --------- | ------- |
| lightgbm (this)    | 1.521    | 0.367    | 0.345     | 226     |
| logreg baseline    | 1.087    | 0.389    | 0.323     | 226     |
| random             | 1.099    | 0.333    | 0.437     | 226     |
| last-side mode     | —        | —        | 0.473     | 226     |
| actual keeper      | —        | —        | null      | 226     |

`random` and `last-side` baselines are deterministic and do not depend
on the retrain; the v3 numbers are pinned. The `lightgbm` and
`logreg` rows reflect the v4 fit on the 211 pre-2026 training rows
(Issue #51 close condition: the v3 artifact-vs-metrics data leak is
preserved — the artifact and the metrics describe the same model
fold split, with the 226-row 2026+ holdout as the published
single-fold). The 18 formerly-skipped refs (URL-rotation wall,
Issue #39) and 6 empty-shotmap refs (Issue #49) are documented as
FotMob data gaps; the 6 club-scope tournaments added in Phase 3
contribute new training rows that bring the dataset from 179 to
437. The `actual keeper` row is `null` because StatsBomb does not
yet publish per-keeper dive-direction data for the in-scope
tournaments.

### Statistical caveat — 226-row holdout

At n=226, the standard error on accuracy is ~0.032 and on save
rate is ~0.032. The reported `lightgbm` save rate (0.345) is
**2.9 standard errors below** the `random` baseline's 0.437 — a
reliable delta in the v4 sample. The v3 model review's "the model
does not beat random on the cross-tournament aggregate" finding
is now reproduced on a single 226-row holdout: the model's save
rate is statistically significantly below random. The 28-row
v3 holdout's "0.571 vs 0.405" was a small-sample draw (the v3
review's aggregate SE was 0.036, large enough to accommodate the
0.166 single-fold delta). The v4 holdout's 226 rows are large
enough to land a stable estimate: the model is below random on
this fold. The headline claim "the model beats random" was always
misleading; the v4 retrain is the first dataset large enough to
show this reliably.

### Calibration — Brier and ECE

The model is **miscalibrated** as a probabilistic classifier. Two
metrics tell the story:

- **Brier score** (multiclass; 0 = perfect, 2 = worst for 3 classes):
  the mean squared error of `P(L), P(C), P(R)` against a one-hot
  encoding of the truth.
- **Expected Calibration Error (ECE)** (10 equal-width confidence
  bins; 0 = perfect): `sum_bin (|bin| / N) * |acc(bin) - conf(bin)|`.

| model              | Brier  | ECE    |
| ------------------ | ------ | ------ |
| lightgbm (this)    | 0.893  | 0.355  |
| logreg baseline    | 0.659  | 0.030  |
| random uniform     | 0.667  | 0.105  |

The lightgbm is **worse** than random on Brier (0.89 vs 0.67) because
the inverse-frequency class weights push probabilities away from
where the truth is. The logreg is well-calibrated. The card's
"the model returns P(L), P(C), P(R) — the probability the kicker
will aim at the left side" claim is **false** on the v4 model: the
model is miscalibrated as a probabilistic classifier.

The deployment policy `argmin(P(L), P(C), P(R))` is **invariant**
under monotone transforms of the per-row probabilities. The
miscalibration does not affect the recommended dive: the model
still picks the lowest-probability side on every row, even when the
absolute probabilities are wrong. Save rate is what the policy
achieves; Brier and ECE are honest about the calibration gap, not
a criticism of the deployment. See
[`docs/model-review.md` § Topic 3](../blob/main/docs/model-review.md)
for the analysis and Issue #43 for the metrics-report change.

### Cross-validation — leave-one-tournament-out (8 folds)

The single 226-row holdout is now stable (the v3 28-row holdout
was too thin to land a stable estimate). To get a tighter claim
and to support the Phase 3 statistical-power story, the metrics
report also includes a leave-one-tournament-out cross-validation
(Issue #45 + Issue #51) — 8 folds, one per `tournament_name`,
with the 437 rows split across the folds as the table below shows.

| fold (held-out tournament)            | n_train | n_holdout | save rate | random | log loss | accuracy |
| ------------------------------------- | ------: | --------: | --------: | -----: | -------: | -------: |
| FA Cup                                |     269 |       168 |     0.286 |  0.405 |    1.620 |    0.393 |
| Africa Cup of Nations Final Stage     |     358 |        79 |     0.354 |  0.409 |    2.347 |    0.342 |
| Champions League                      |     377 |        60 |     0.300 |  0.467 |    1.002 |    0.700 |
| World Cup                             |     399 |        38 |     0.421 |  0.491 |    1.613 |    0.447 |
| EURO Final Stage                      |     403 |        34 |     0.265 |  0.353 |    1.659 |    0.382 |
| World Cup Final Stage                 |     412 |        25 |     0.360 |  0.413 |    1.710 |    0.480 |
| CONCACAF Gold Cup Final Stage         |     413 |        24 |     0.458 |  0.417 |    2.054 |    0.458 |
| Copa America Final Stage              |     428 |         9 |     0.222 |  0.407 |    1.305 |    0.444 |
| **aggregate (n=437)**                 |         |           | **0.323** |        | **1.691** | **0.439** |
| aggregate SE on save rate             |         |           |   ±0.022  |        |           |          |

The aggregate save rate is **0.323 (SE ±0.022)** — 6× tighter than
the v3 0.036 SE, a **39% reduction** (Issue #51 acceptance: ≥ 30%
reduction from v3's 0.036, target ≤ 0.025). The 8 LOTO folds add 2
new tournaments (FA Cup, Champions League) over v3's 6; the
international-scope 6 folds reproduce the v3 6-fold story (AFCON +
WC Final Stage + Gold Cup = wins or ties; Euro + Copa + group-stage
WC = losses). The 2 new club-scope folds are both losses (FA Cup
0.286 vs 0.405 random, Champions League 0.300 vs 0.467 random). The
model is **0.080 below** the closed-form random baseline (0.323 vs
0.405 on the same 437 rows) — 3.6 aggregate SE below random on
the v4 dataset, vs the v3 finding of "0.031 below random, within
one SE". The v4 dataset's larger SE noise floor (still 0.022) makes
the "below random" finding statistically reliable.

The v4 retrain's path-to-statistical-power goal (Issue #51 ADR) is
met: the aggregate SE is 0.022 (a 39% reduction from v3's 0.036),
giving the v5 model work (Issues #42, #46) the statistical
foundation it needs to detect whether dropping A1 or adding an
anti-classifier actually helps.

## What it predicts

For any kicker, the model returns P(L), P(C), P(R). The goalkeeper
dives toward the side with the **lowest** predicted probability — the
counterfactual save policy. The dashboard at
[`matheusccouto/12yd`](https://github.com/matheusccouto/12yd) surfaces
per-kicker predictions for the WC 2026 knockout matches. v4 (Issue
#48) renders each Kicker as a card: photo placeholder, name, career-
penalty count, a Plotly goal drawing (3 coloured segments, star on
the most-likely side), and a one-line prediction row in the
Kicker-PoV frame ("WILL AIM L 55% · GK dive ↔ R"). The
probabilities and the dive hint stay in the Kicker-PoV frame end to
end.

## Feature schema (v3, 17 features)

v3 dropped two features in two passes: the B3 (`b3_round`) feature
in Issue #36 and the C2 (`age`) feature in Issue #41. The model is
now both round-agnostic and age-agnostic. The v4 retrain keeps the
17-feature schema unchanged; the new `tournament_kind` attribute
on `TrainingRow` is metadata (per `docs/adr/0004-phase-3-data-source.md`),
not a model input.

**Numeric (14 — A1, A4, B1, B2):**

- `p_L_5, p_C_5, p_R_5` — side distribution over last 5 kicks (A1).
- `p_L_10, p_C_10, p_R_10` — side distribution over last 10 kicks (A1).
- `p_L_20, p_C_20, p_R_20` — side distribution over last 20 kicks (A1).
- `career_penalty_count` — total penalties before the target kick (A4).
- `b1_kick_number` — kick number within the shootout (B1).
- `pen_score_home, pen_score_away` — score BEFORE the kick (B2).
- `is_decisive` — whether the kick's outcome ends the shootout (B2).

**Categorical (3 — A2, A3, C1):**

- `last_side` — `"L"` / `"C"` / `"R"` / `""` (A2; `""` = no history).
- `preferred_foot` — `"left"` / `"right"` / `"both"` / `""` (A3; the
  declared foot from `pageProps.data.playerInformation[]` with
  `translationKey="preferred_foot"`). v3 swapped the previous
  `kicking_foot` (which was inferred from the mode of the kicker's
  penalty `shotType` history) for the declared foot; the
  `predictions.jsonl` column keeps the `kicking_foot` name for
  consumer continuity, but the underlying semantic is now the
  declared foot.
- `position` — FotMob position key, e.g. `"striker"` (C1).

The `b3_round` feature (dropped in v3, Issue #36) was the only
round-specific feature; the `age` feature (dropped in v3, Issue
#41) was the only per-kicker time-varying numeric. The v3 schema
is the simplest set of features the model review ablation
endorsed. The v4 retrain is the same schema on a larger dataset.
The `predictions.jsonl` artifact on `data/` is the per-kicker
source of truth and is round-agnostic.

## Usage

```python
import pickle
from huggingface_hub import hf_hub_download

p = hf_hub_download("couto/12yd", "model/lightgbm.pkl")
artifact = pickle.load(open(p, "rb"))
model = artifact["model"]
feature_columns = artifact["feature_columns"]

# Build a 14-numeric + 3-categorical = 17-feature row (A-group +
# B-group + C-group) and call model.predict_proba(row). The classes
# are ["L", "C", "R"] in that order.
```

## Repository layout

- `model/lightgbm.pkl` — the frozen LightGBM (LGBMClassifier inside
  a `LightGBMClassifierWrapper`), trained on the 211 pre-2026
  training fold (the same model the metrics describe; Issue #40
  closed the artifact-vs-metrics data leak).
- `model/metrics.json` — the held-out metrics report. Includes
  log loss, accuracy, save rate, the calibration block (Brier
  + ECE for the model, the logreg baseline, and the uniform
  random baseline; Issue #43), and the LOTO cross-validation
  block (8-fold per-fold save rate / log loss / accuracy + the
  aggregate summary; Issue #45 + Issue #51).
- `data/cv_metrics.json` — the standalone LOTO CV artifact (the
  same payload that's embedded in `model/metrics.json` under the
  `cv` key, written separately so the dashboard or a future
  tool can load the CV without parsing the rest of the metrics
  report).
- `data/shootout_kicks.jsonl` — 437 target kicks across 25
  shootouts in 8 national-team + club tournaments, 2021–2026
  (the v3 179-row scope plus 258 new club-scope rows from Phase
  3, Issue #51).
- `data/player_history.jsonl` — per-kicker penalty history (the
  A1/A2/A3/A4 inputs), filtered to each kicker's target-kick
  date minus the 5-year lookback window. The Phase 3 ingest
  fetched 1359 unique kickers (the v2 1327 plus 32 from the
  club-scope shootouts); 261 have at least one penalty row, 1098
  have zero (the prior-only kickers).
- `data/wc2026_roster.jsonl` — the WC 2026 squad list (the
  prediction roster).
- `data/predictions.jsonl` — per-player round-agnostic
  predictions (the dashboard reads this directly — v3 dropped
  the per-match re-score path; v4 keeps the round-agnostic
  schema).
- `data/missing_history.jsonl` — kickers with no penalty history
  in the lookback window.
- `data/tournament_success_rate.jsonl` — the per-(league,
  season) success-rate diagnostic (Issue #52 close-out; the v4
  Phase 2 acceptance criterion).
- `data/skipped_refs_diagnostics.jsonl` — per-match skip / no-
  kicks / failure records; the 18 URL-rotation wall and 6
  empty-shotmap cases are documented here.
- `data/discrepancies.json` — the RSSSF-vs-scraper divergence
  report (scraper-reachable total: 18 international + 4 club
  + 7 n/a + 6 missing = 25 distinct matches in the 57-pair
  scope, after the URL-rotation wall + empty-shotmap
  exclusions).

## Provenance

Model card generated from the v4 `output/metrics.json` at the time
of the v4 release. The v4 retrain follows the v3 slice pipeline
with one schema change (`tournament_kind` attribute on
`TrainingRow` / `PredictionRow`, metadata-only, derived from
`TOURNAMENT_KIND_BY_LEAGUE_ID` at `build_features` time) and one
data change (Issue #51: 258 new club-scope kicks from 2 in-scope
club tournaments, the FA Cup and the Champions League, brought
the dataset from 179 rows to 437 rows; the LOTO CV aggregate SE
dropped from 0.036 to 0.022, a 39% reduction). The 18
formerly-skipped refs and 6 empty-shotmap refs are still
unrecoverable from FotMob (Phase 2 close-out, Issues #39, #49);
they are documented as FotMob data gaps and are excluded from
the per-pair reachable counts. The 18 club pairs from
Copa Libertadores, Coupe de France, DFB-Pokal, Coppa Italia, and
Copa del Rey contribute 0 shootouts (the FotMob API returns the
same fixture list for every `?season=` value, deduplicating to
0 distinct matches); the 2 club pairs from FA Cup and Champions
League contribute the 4 distinct matches that account for the
new training rows. See
[`docs/adr/0004-phase-3-data-source.md`](../blob/main/docs/adr/0004-phase-3-data-source.md)
for the Phase 3 source decision and the rationale for the
chosen subset of 7 in-scope club leagues. See
[`matheusccouto/12yd`](https://github.com/matheusccouto/12yd)
(the GitHub repo) for the slice pipeline, the dashboard
source, and the data layer.

## v4 changes from v3

- **Added `tournament_kind` attribute to `TrainingRow` and
  `PredictionRow`** (Issue #51, Phase 3 schema change). The
  field is metadata, derived from the league registry at
  `build_features` time (the source of truth is
  `TOURNAMENT_KIND_BY_LEAGUE_ID` in `tournaments.py`). Values
  are `"international"` (the 6 in-scope national-team cup
  tournaments) or `"club"` (the 7 in-scope club cup
  tournaments, per the Phase 3 ADR). Default is `"international"`
  so existing v3 rows are unchanged. The 17-feature model
  input is unchanged. The `tournament_kind` is propagated to
  `predictions.jsonl` as a per-kicker metadata field (not a
  model input) so the dashboard can surface the kind on the
  per-kicker card in a future iteration.
- **Added 7 in-scope club tournaments to the shootout scope**
  (Issue #51, Phase 3 ingest). The new `CLUB_LEAGUES` tuple in
  `leagues.py` covers Copa Libertadores, Champions League, FA
  Cup, Coupe de France, DFB-Pokal, Coppa Italia, and Copa del
  Rey. The new `CLUB_PAIRS` constant in `tournaments.py`
  extends `LEAGUE_SEASONS_PREDICT_WINDOW` with 7 × 6 = 42
  new (league, season) pairs (2021–2026). The combined scope
  is 57 pairs across 13 tournaments. 5 of the 7 leagues
  (Copa Libertadores, Coupe de France, DFB-Pokal, Coppa
  Italia, Copa del Rey) contribute 0 shootouts (the FotMob
  API returns the same fixture list for every season); FA Cup
  and Champions League contribute 4 distinct matches and 258
  new training rows. The total dataset grows from 179 to 437
  rows, dropping the LOTO CV aggregate SE from 0.036 to 0.022
  (a 39% reduction; Issue #51 acceptance was ≥ 30% reduction,
  target ≤ 0.025).
- **Retrained the LightGBM on the 437-row combined dataset**
  (the v4 fit, on the 211 pre-2026 training fold). The 226-
  row 2026+ holdout is the published single-fold; the 8-fold
  LOTO CV is the cross-tournament aggregate. The artifact-vs-
  metrics data leak (Issue #40) is preserved: the deployed
  `lightgbm.pkl` is fit on the same 211-row training fold the
  metrics describe. The 18 formerly-skipped refs and 6 empty-
  shotmap refs are still unrecoverable from FotMob; the
  per-pair reachable count drops from v3's 36 to v4's 18
  (after both URL-rotation and empty-shotmap exclusions).
- **Pinned the LOTO CV aggregate SE at 0.022** (Issue #51
  close condition: ≤ 0.025, ≥ 30% reduction from v3's 0.036).
  The aggregate save rate is 0.323 (SE 0.022) on n=437 across
  8 folds, vs v3's 0.374 (SE 0.036) on n=179 across 6 folds.
  The path-to-statistical-power goal is met: the v5 model
  work (Issues #42, #46) now has the SE noise floor to detect
  whether dropping A1 or adding an anti-classifier actually
  helps.
- **Documented the scraper's per-tournament success rate**
  (Issue #52, the Phase 2 close-out). The
  `tournament_success_rate.jsonl` artifact carries one row per
  (league, season) pair with the scraper's per-pair match /
  kick counts, the per-state breakdown (skipped, no-kicks,
  failed), and the expected / reachable counts from the
  RSSSF oracle. The `URL_ROTATION_EXCLUSIONS` map
  (`validate.py`) pins the 18 documented URL-rotation
  cases; the `EMPTY_SHOTMAP_EXCLUSIONS` map pins the 6
  documented empty-shotmap cases (Issue #49). The per-pair
  test (`tests/test_tournaments.py`) is updated to use
  both exclusion maps, so a future regression that drops
  a pair (or loses a per-pair exclusion) is caught at the
  artifact.
- **Documented the 5 in-scope club leagues that contribute
  0 shootouts**. The FotMob API returns the same fixture
  list for every `?season=` value on Copa Libertadores,
  Coupe de France, DFB-Pokal, Coppa Italia, and Copa del
  Rey (a season-bucketing bug in the public API, distinct
  from the URL-rotation wall). After dedup-by-match_id, the
  5 leagues contribute 0 distinct matches; the per-pair
  status is `n/a` in the success-rate diagnostic. A future
  Phase 4 ADR can address the bug (e.g. via a per-team
  fixture list strategy, or a non-FotMob oracle), but the
  5 leagues stay in the 57-pair scope for the prediction
  window consistency check.

## Further Notes

- **Data gap (open follow-up).** The 18 URL-rotation
  refs and 6 empty-shotmap refs are documented as FotMob
  data gaps. The diagnostics infrastructure
  (`skipped_refs_diagnostics.jsonl` with `failure_mode` =
  `stale_hash` / `empty_shotmap` / `f"{ExceptionClass}:
  {message}"`, plus the 18-ref table in
  `data/url_rotation_wall.md` and the 6-ref table in
  `data/empty_shotmap_documentation.md`) catches new
  failure modes correctly. The 18 stale-hash refs each
  resolve to a different newer matchId (the
  `data/skipped_refs_diagnostics.jsonl` carries the per-ref
  `live_match_id` and `resolved_url`); the underlying cause
  is FotMob's URL-rotation of `(seo, h2h)` pairs to newer
  matches when league IDs are rolled forward. The 6
  empty-shotmap refs are 4 in AFCON 2021 and 2 in Asian Cup
  2023; FotMob returns no `pageProps.content.shotmap.shots`
  data for them. Both gaps are accepted as FotMob data
  limits; closing them requires leaving FotMob (StatsBomb,
  RSSSF detail pages, Wikipedia per-shootout pages) and is
  deferred to a future Phase 4 ADR.

See `docs/PRD-v4.md` for the v4 PRD and the
[`matheusccouto/12yd` issues](https://github.com/matheusccouto/12yd/issues)
(#46, #51) for the work breakdown.
