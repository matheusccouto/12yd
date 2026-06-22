# Penalty Shootout Prediction

Predicts which side a player will kick a penalty in a shootout, so a goalkeeper can pick the best side to dive.

## Language

**Side**:
The horizontal half of the goal from the kicker's perspective: `L` (left), `C` (centre), or `R` (right). Bucketed from the continuous goal-mouth coordinate `x ∈ [0, 2]` with `1` as centre.
_Avoid_: Direction, corner, post

**Kicker**:
The outfield player taking a penalty kick in a shootout.
_Avoid_: Shooter, taker, penalty taker

**Goalkeeper (GK)**:
The defending player in a penalty shootout. Chooses a Side to dive toward (or stay in the centre) before the kick.
_Avoid_: Keeper, goalie

**Dive**:
The Goalkeeper's action before a kick: choose Side `L`, stay in `C`, or choose `R`. A Dive is optimal when it picks the Side with the lowest predicted probability for the Kicker.
_Avoid_: Guess, action

**Shootout Kick**:
One penalty taken as part of a tiebreaker sequence after a knockout match ends level. Every Shootout Kick has a known Side: sourced from `pageProps.content.shotmap.shots` filtered by `period == "PenaltyShootout"`, which is populated for Goals, Saves, and Misses alike (off-target kicks have `onGoalShot.x` clamped to [0, 2] at the post).
_Avoid_: Penalty, spot kick

**In-Match Penalty**:
A penalty awarded and taken during normal or extra time, not as part of a tiebreaker. Used as training data only — never a prediction target.
_Avoid_: Regular penalty, open-play penalty

**Training Penalty**:
Any past penalty kick (Shootout Kick or In-Match Penalty) used to build a Kicker's history. Sourced from the Kicker's career, not limited to a single competition.
_Avoid_: Historical penalty, past kick

**Lookback Window**:
The duration before a target Shootout Kick's match date from which a Kicker's Training Penalties are drawn. Configurable in the scraper (default: 5 years). Lives in code, not in the dataset — the same dataset is reusable with a different window.
_Avoid_: History depth, training horizon

**History Floor**:
The hard lower bound on Training Penalty dates, independent of the Lookback Window. For the current prediction window starting 2021-01, the floor is 2016-01-01 — giving at least 5 years of history before the oldest target Shootout Kick. Recomputed automatically when the prediction window changes.
_Avoid_: Minimum date, history cutoff

**Prediction Window**:
The date range from which target Shootout Kicks are drawn. The current Prediction Window is 2021-01-01 to today, spanning 5.5 years of national-team cup competitions. Recomputed automatically when the target competition list changes.
_Avoid_: Test period, target date range

**Initial Set**:
The candidate Kickers the model is prepared to score. Has two flavors:
- **Training Initial Set** — the players who kicked in past Shootout Kicks; used to build the training table.
- **Prediction Initial Set** — a Tournament Roster (e.g. 2026 WC squads); used to build the prediction table before any shootout kicks are known.

Identified statically for each prediction window; never derived from Training Penalty data.
_Avoid_: Seed set, target players

**Tournament Roster**:
The list of players registered for a tournament (e.g. 2026 World Cup squad). Source for the Prediction Initial Set when target Shootout Kicks do not yet exist.
_Avoid_: Squad, lineup

**Derived History**:
The Training Penalties fetched for each Kicker in the Initial Set, bounded by the Lookback Window relative to that Kicker's target Shootout Kick. A leaf in the data graph: no further fetches originate from it.
_Avoid_: Penalty history, lookback data
