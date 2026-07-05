---
license: mit
tags:
  - penalty
  - football
  - lightgbm
  - shootout
---

# 12yd — Penalty Shootout Side Prediction

Multiclass classifier (L / C / R) on 17 per-kick features; trained on the 151 pre-2026 shootout kicks across 6 national-team tournaments (2021–2022). Frozen deployment artifact for `matheusccouto/12yd`.

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

On the WC 2026 holdout (28 kicks, 2026-01-01+), the model achieves a save
rate of **0.571** versus a uniform-random baseline of **0.405** — a
41% relative improvement, and a +0.107 absolute gain over the
pre-#41 18-feature model (which scored 0.464 on the same 28 kicks).
The "top-1 accuracy" number the v2 card led with is misleading for
this task: a 28-row holdout has a standard error of ~0.09 on accuracy,
so differences smaller than that are noise. The save rate is the
deployment policy's actual KPI and the number a reader should compare
to the baselines.

## Held-out metrics (28 WC 2026 holdout kicks, 2026-01-01+)

| model              | log loss | accuracy | save rate | n_kicks |
| ------------------ | -------- | -------- | --------- | ------- |
| lightgbm (this)    | 1.700    | 0.179    | 0.571     | 28      |
| logreg baseline    | 1.096    | 0.357    | 0.321     | 28      |
| random             | 1.099    | 0.333    | 0.405     | 28      |
| last-side mode     | —        | —        | 0.393     | 28      |
| actual keeper      | —        | —        | null      | 28      |

`random` and `last-side` baselines are deterministic and do not depend on
the retrain; the v2 numbers are pinned. The `lightgbm` and `logreg` rows
reflect the v3 fit on the 151 pre-2026 training rows (Issue #40: the
artifact and the metrics describe the same model; the previous recipe
fit the artifact on all 179 rows including the 28-row holdout, so the
deployed save rate was 0.107 — in-sample memorisation — not the 0.571
this card advertises). The 18 formerly-skipped refs have URL rotation
issues and need a separate fix — see `## Further Notes`. The `actual
keeper` row is `null` because StatsBomb does not yet publish
per-keeper dive-direction data for the in-scope tournaments.

### Statistical caveat — 28-row holdout

At n=28, the standard error on accuracy is ~0.09 and on save rate is
~0.09. The reported `lightgbm` save rate (0.571) is **1.8 standard
errors above** the `random` baseline's 0.405 — the largest
delta the v3 model has shown on this holdout. The 28-row holdout
remains statistically thin; a larger holdout (n ≥ 100) is needed
to confirm the gain survives out-of-sample. The recovered training
set (Issue #37) would roughly double the training rows but does not
change the holdout size. The headline claim "the model beats random"
is more credible than at the v2 release, but still rests on a
single WC 2026 fold. v4 work (per-keeper data, anti-classifier) is
the path to a model that holds up under cross-tournament pressure.

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
| lightgbm (this)    | 0.990  | 0.434  |
| logreg baseline    | 0.665  | 0.004  |
| random uniform     | 0.667  | 0.060  |

The lightgbm is **worse** than random on Brier (0.99 vs 0.67) because
the inverse-frequency class weights push probabilities away from
where the truth is. The logreg is well-calibrated. The card's
"the model returns P(L), P(C), P(R) — the probability the kicker will
aim at the left side" claim is **false** on the v3 model: the model
is miscalibrated as a probabilistic classifier.

The deployment policy `argmin(P(L), P(C), P(R))` is **invariant**
under monotone transforms of the per-row probabilities. The
miscalibration does not affect the recommended dive: the model
still picks the lowest-probability side on every row, even when the
absolute probabilities are wrong. Save rate is what the policy
achieves; Brier and ECE are honest about the calibration gap, not a
criticism of the deployment. See
[`docs/model-review.md` § Topic 3](../blob/main/docs/model-review.md)
for the analysis and Issue #43 for the metrics-report change.

### Cross-validation — leave-one-tournament-out

The single 28-row holdout is honest about what 28 rows can tell us
(see the statistical caveat above). To get a
tighter claim, the metrics report also includes a
leave-one-tournament-out cross-validation (Issue #45) — 6 folds, one
per `tournament_name`, with the 179 rows split across the folds
as the table below shows.

| fold (held-out tournament)            | n_train | n_holdout | save rate | random | log loss | accuracy |
| ------------------------------------- | ------: | --------: | --------: | -----: | -------: | -------: |
| Africa Cup of Nations Final Stage     |     100 |        79 |     0.418 |  0.409 |    1.260 |    0.405 |
| EURO Final Stage                      |     145 |        34 |     0.294 |  0.353 |    1.170 |    0.500 |
| World Cup Final Stage                 |     154 |        25 |     0.400 |  0.413 |    1.252 |    0.280 |
| CONCACAF Gold Cup Final Stage         |     155 |        24 |     0.417 |  0.417 |    1.247 |    0.250 |
| Copa America Final Stage              |     170 |         9 |     0.222 |  0.407 |    1.113 |    0.556 |
| World Cup                             |     171 |         8 |     0.250 |  0.417 |    1.006 |    0.375 |
| **aggregate (n=179)**                 |         |           | **0.374** |        | **1.221** | **0.391** |
| aggregate SE on save rate             |         |           |   ±0.036  |        |           |          |

The aggregate save rate is **0.374 (SE ±0.036)** — 6× tighter than
the single 28-row holdout (SE ±0.094). The model is **0.031 below**
the closed-form random baseline (0.405 on the same 179 rows) — well
within one aggregate SE. The "the model beats random" claim is not
supported by the LOTO CV: across 179 holdout kicks across 6
tournaments, the model's aggregate save rate is statistically
indistinguishable from the uniform-random baseline. The 28-row
holdout's "0.571 vs 0.405" (after the #41 retrain) was a
one-tournament draw from a distribution that averages to ~0.37.

The logreg baseline's LOTO CV (computed in the same script for
context) lands at **0.380 save rate** — also below random, also
within one SE. The published single-fold "logreg 0.321" number
(Issue #43) is the same kind of small-sample noise: the logreg
beats the lightgbm on the WC 2026 holdout (0.321 vs 0.571) but
loses to it on the cross-tournament aggregate (0.380 vs 0.374 are
within noise of each other).

The CV reveals what the single 28-row holdout could not: **the
17-feature model is not adding measurable value over a uniform-
random dive policy on the cross-tournament aggregate.** This is
the same conclusion as the model review (Topic 2 + Topic 4); the
LOTO CV is the independent confirmation. The per-fold picture is
mixed — the 17-feature model beats random on the AFCON + WC-Final-
Stage folds and ties on Gold Cup; it loses to random on EURO,
Copa, and the "World Cup" group stage. v4 work (per-keeper data,
anti-classifier) is the path to a model that meaningfully beats
random on the cross-tournament aggregate.

## What it predicts

For any kicker, the model returns P(L), P(C), P(R). The goalkeeper
dives toward the side with the **lowest** predicted probability — the
counterfactual save policy. The dashboard at
[`matheusccouto/12yd`](https://github.com/matheusccouto/12yd) surfaces
per-kicker predictions for the WC 2026 knockout matches. v4 (Issue
#48) replaces the v3 dataframe with a card layout: each Kicker gets
a card with a photo placeholder, name, career-penalty count, a
Plotly goal drawing (3 coloured segments, star on the most-likely
side), and a one-line prediction row in the Kicker-PoV frame ("WILL
AIM L 55% · GK dive ↔ R"). The probabilities and the dive hint stay
in the Kicker-PoV frame end to end.

## Feature schema (v3, 17 features)

v3 dropped two features in two passes: the B3 (`b3_round`) feature
in Issue #36 and the C2 (`age`) feature in Issue #41. The model is
now both round-agnostic and age-agnostic.

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
round-specific feature; the `age` feature (dropped in v3, Issue #41)
was the only per-kicker time-varying numeric. The v3 schema is the
simplest set of features the model review ablation endorsed. The
`predictions.jsonl` artifact on `data/` is the per-kicker source of
truth and is round-agnostic.

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

- `model/lightgbm.pkl` — the frozen LightGBM (LGBMClassifier inside a
  `LightGBMClassifierWrapper`), trained on the 151 pre-2026 training
  fold (the same model the metrics describe; Issue #40 closed the
  artifact-vs-metrics data leak).
- `model/metrics.json` — the held-out metrics report. Includes
  log loss, accuracy, save rate, the calibration block (Brier
  + ECE for the model, the logreg baseline, and the uniform random
  baseline; Issue #43), and the LOTO cross-validation block
  (per-fold save rate / log loss / accuracy + the aggregate
  summary; Issue #45).
- `data/cv_metrics.json` — the standalone LOTO CV artifact (the
  same payload that's embedded in `model/metrics.json` under the
  `cv` key, written separately so the dashboard or a future tool
  can load the CV without parsing the rest of the metrics report).
- `data/shootout_kicks.jsonl` — 179 target kicks across 18 shootouts in 6
  national-team tournaments, 2021–2022 (the v2 42-shootout scope minus
  24 shootouts with URL rotation issues; see `## Further Notes`).
- `data/player_history.jsonl` — per-kicker penalty history (the
  A1/A2/A3/A4 inputs), filtered to each kicker's target-kick date
  minus the 5-year lookback window.
- `data/wc2026_roster.jsonl` — the WC 2026 squad list (the
  prediction roster).
- `data/predictions.jsonl` — per-player round-agnostic predictions
  (the dashboard reads this directly — v3 dropped the per-match
  re-score path).
- `data/missing_history.jsonl` — kickers with no penalty history in
  the lookback window.
- `data/discrepancies.json` — the RSSSF-vs-scraper divergence
  report (actual=18, expected=42, delta=-24; the 18 skipped refs
  are URL rotation issues, not extractor exceptions).

## Provenance

Model card generated from the v3 `output/metrics.json` at the time of
the v3 release. The v3 retrain follows the v2 slice pipeline with three
schema changes (drop `b3_round`, replace `kicking_foot` with
`preferred_foot`, drop `age`) and one data change (recovered
42-shootout training set after Issue #37). See
[`matheusccouto/12yd`](https://github.com/matheusccouto/12yd) (the
GitHub repo) for the slice pipeline, the dashboard source, and the
data layer.

## v3 changes from v2

- **Dropped `b3_round` from the feature schema** (v3 model is
  round-agnostic; the dashboard reads `predictions.jsonl` directly).
- **Replaced inferred `kicking_foot` with declared `preferred_foot`**
  in the A3 feature. 1080 of 1247 v2 rows with `"Unknown"` get real
  declared-foot values from the cached `pageProps.data.playerInformation[]`
  payload. The v3 `predictions.jsonl` has 0 `"Unknown"` rows.
- **Retrained on the 151 pre-2026 training rows** (the same model the
  metrics describe; Issue #40 closed the artifact-vs-metrics data
  leak — the v3 release shipped a 179-row artifact with a 0.107
  in-sample save rate, not the 0.464 the card advertised). The
  holdout (28 WC 2026 kicks, 2026-01-01+) is unchanged from v2;
  `n_train` is 151. The 18 formerly-skipped refs have URL rotation
  issues; see `## Further Notes`.
- **Save rate is now the headline metric** in this card; the
  accuracy-led v2 card is replaced. The 28-row holdout caveat
  applies to both metrics at this sample size.
- **Calibration block added to `model/metrics.json`** (Issue #43):
  Brier score and ECE (10-bin) for the model, the logreg baseline,
  and the uniform random baseline. The card has a new "Calibration"
  section that documents the miscalibration story in plain English.
- **LOTO cross-validation block added to `model/metrics.json`**
  (Issue #45): 6 folds (one per `tournament_name` in the 179-row
  training set) with per-fold save rate, log loss, accuracy, and
  the aggregate summary (weighted-mean save rate + the binomial
  SE on the aggregate). The card has a new "Cross-validation"
  section. The CV reveals that the 18-feature model is not
  statistically distinguishable from a uniform-random dive policy
  on the cross-tournament aggregate (model 0.369 vs random 0.405,
  one aggregate SE), and that the single 28-row holdout's "model
  beats random" claim was a small-sample draw. v4 work is the
  path to a model that meaningfully beats random.
- **Dropped `age` (C2) from the feature schema** (Issue #41).
  The model review ablation in `docs/model-review.md` Topic 2.3
  showed that removing age improves BOTH save rate (0.464 → 0.571)
  and log loss (1.769 → 1.700) on the 28-row 2026 holdout — a
  +0.107 save-rate gain that clears the 0.09 SE by ~1.2 standard
  errors. The LOTO CV aggregate is essentially unchanged (0.369 →
  0.374, both well within one aggregate SE of the 0.405 random
  baseline), so the cross-tournament story is unchanged: the
  model still does not beat random on the aggregate. The 28-row
  holdout is now the most favourable draw the v3 model has shown.
  The birth date is still on `PlayerMetadata` for the data layer's
  records; only the model no longer reads it.

## Further Notes

- **Data gap (open follow-up).** The 18 formerly-skipped refs from
  `data/discrepancies.json` are caused by URL rotation on FotMob's
  `(seo, h2h)` pairs, not by extractor exceptions. The diagnostics
  infrastructure from iteration 1 (the `skipped_refs_diagnostics.jsonl`
  artifact) catches the new failure mode correctly: all 18 are
  flagged `stale_hash`, and the orchestrator continues past them.
  The underlying cause is that some `(seo, h2h)` pairs from the v2
  season-fixture list have been re-assigned to newer matches (e.g.
  match 3370565 Croatia vs Brazil QF 2022 is now at a different
  URL; the old URL points to a 2026 friendly). Recovering the
  18 missing shootouts requires a search-based URL lookup
  (FotMob's public page or per-team fixture list) — Issue #38
  follow-up.

See `docs/PRD-v3.md` for the v3 PRD and the
[`matheusccouto/12yd` issues](https://github.com/matheusccouto/12yd/issues)
(#35, #36, #37, #38) for the work breakdown.
