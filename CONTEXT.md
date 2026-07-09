# Penalty Shootout Prediction

Predicts which side a player will kick a penalty, so a goalkeeper can pick the best side to dive. Uses TabPFN (a Tabular Foundation Model) as the classifier on per-player penalty-history features. Predictions are match-agnostic: each roster player is scored once, and the same prediction row serves any match.

## Language

**Side**:
The horizontal half of the goal from the kicker's perspective: `L` (left), `C` (centre), or `R` (right). Bucketed from the continuous goal-mouth coordinate `x ∈ [0, 2]` with `1` as centre.
_Avoid_: Direction, corner, post

**Kicker**:
The outfield player taking a penalty kick.
_Avoid_: Shooter, taker, penalty taker

**Goalkeeper (GK)**:
The defending player in a penalty shootout. Chooses a Side to dive toward (or stay in the centre) before the kick.
_Avoid_: Keeper, goalie

**Dive**:
The Goalkeeper's action before a kick: choose Side `L`, stay in `C`, or choose `R`. A Dive is optimal when it picks the Side with the lowest predicted probability for the Kicker.
_Avoid_: Guess, action

**Training Penalty**:
Any past penalty kick used to build a Kicker's history. Sourced from the Kicker's career on FotMob (careerHistory → per-(team,season) league fixtures → match shotmaps with penalties), not limited to a single competition.
_Avoid_: Historical penalty, past kick

**Scrape Floor**:
The hard lower bound on Training Penalty dates for data ingestion. Set to `2016-01-01` — penalties before this date are not fetched. Provides a 5-year buffer for the rolling feature window applied to the oldest training kicks (which start at the Train Floor of 2021-01-01).
_Avoid_: Minimum date, history cutoff

**Train Floor**:
The lower bound on kicks that become training/test rows. Set to `2021-01-01`. Kicks between the Scrape Floor and the Train Floor exist purely as feature-history context for the oldest training rows.
_Avoid_: Label start, training cutoff

**Lookback Window**:
The time-based rolling window `[T - 5 years, T)` for feature derivation at a target kick at time T. Features A1/A2/A4 are computed over kicks within this window. The window size is 5 years, matching the buffer between Scrape Floor and Train Floor so every training row has a complete feature window.
_Avoid_: History depth, training horizon

**Initial Set**:
The candidate Kickers the model is prepared to score. Identified from the WC 2026 Tournament Roster. All roster players are scored — there is no separate training-initial set under the TabPFN regime.
_Avoid_: Seed set, target players

**Tournament Roster**:
The list of players registered for the 2026 World Cup. Source for the Prediction Initial Set. Fetched once per Actions run from FotMob lineups.
_Avoid_: Squad, lineup

**Player History**:
The Training Penalties fetched for each Kicker in the Initial Set, bounded by the Scrape Floor. Stored in `data/player_history.jsonl`. The source of both the training matrix (features + labels for prior kicks) and the test matrix (features for roster players).
_Avoid_: Penalty history, lookback data

**Prediction**:
A per-player row in `data/predictions.jsonl` with `p_L`, `p_C`, `p_R` (the predicted probability distribution over Sides) and metadata (player_id, player_name, team_id, team_name). Match-agnostic — no opponent or match-context features.
_Avoid_: Score, recommendation

**TabPFN**:
The Tabular Foundation Model (TabPFNClassifier from `tabpfn-client`, Prior Labs cloud API) used as the estimator. Operates in cheapest mode: no thinking, `n_estimators=8`. Fit is free; predict costs tokens (50M/day free-tier quota). One full run costs ~0.25% of daily quota.
_Avoid_: Model, classifier, LightGBM

**GitHub Actions Pipeline**:
The single scheduled workflow (`scrape-and-predict.yml`) that runs the scraper (roster + player history), fits TabPFN, writes predictions, and commits the JSONLs. Triggered daily at 06:00 UTC and via manual `workflow_dispatch`.
_Avoid_: CI, cron job, automation

**FotMob HTTP Cache**:
The on-disk ETag+gzip cache at `data/fotmob_cache/` (gitignored). Persisted across Actions runs via `actions/cache`. The linchpin is buildId-stripped cache keys so stored responses survive FotMob deployment rotations, making re-runs warm (~1-3 minutes) after the initial backfill. "Already-scraped" is delegated to HTTP 304 revalidation — no per-kicker state file exists.
_Avoid_: Response cache, disk cache
