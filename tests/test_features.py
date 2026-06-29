"""Tests for the feature builder (slice #6, Issue #22; refined #36, #41).

The tests cover five layers:

1. **Pure helpers** — `side_distribution`, `is_decisive_kick`,
   `age_in_years` (kept for backwards compatibility; the model no
   longer uses it after Issue #41), `filter_history`,
   `index_kicks_done`. No network, no I/O. v3 (Issue #36) dropped
   the `mode_kicking_foot` helper — the A3 feature now reads from
   `PlayerMetadata.preferred_foot` instead of the per-penalty
   `shot_type` mode.

2. **Per-kick feature builder** — `build_features` (the packaging
   function) and `compute_features` (the A-group/C-group
   computation) against a constructed `ShootoutKick` and a
   constructed `PlayerPenalty` history. Verifies the
   A1/A2/A3/A4/B1/B2/C1 features and the no-history fallback.
   v3 dropped B3 (`b3_round`, Issue #36) and C2 (`age`, Issue #41).

3. **Orchestration** — `build_training_table` against a stubbed
   `MetadataFetcher` that returns canned metadata per kicker. Verifies
   the row count, the sort order, and the JSONL roundtrip.

4. **JSONL helpers** — `Artifacts.write_training_table` roundtrip; no
   NaN handling needed (the v3 row has no NaN-valued fields).

5. **Live smoke test** — `output/training_table.jsonl` (skipped if
   absent): schema, row count matches the live shootout kicks file,
   A1 monotonicity, every training kicker has at least one row.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from penalty_pred.artifacts import Artifacts
from penalty_pred.features import (
    PRIOR_PROB,
    BGroupContext,
    KickIndex,
    PredictionTarget,
    TrainingRow,
    age_in_years,
    build_features,
    build_training_table,
    compute_features,
    filter_history,
    index_kicks_done,
    is_decisive_kick,
    load_player_history,
    side_distribution,
)
from penalty_pred.player_history import PlayerMetadata
from penalty_pred.shootouts import ShootoutKick
from tests._factories import (
    TRAINING_ROW_FIELDS,
    make_history_row,
    make_shootout_kick,
)

# ---------------------------------------------------------------------------
# side_distribution
# ---------------------------------------------------------------------------


def test_side_distribution_empty_returns_uniform_prior() -> None:
    """An empty history returns (1/3, 1/3, 1/3)."""
    assert side_distribution([], 5) == PRIOR_PROB
    assert side_distribution([], 20) == PRIOR_PROB


def test_side_distribution_single_side() -> None:
    """A single L returns (1.0, 0.0, 0.0) regardless of horizon (the
    horizon only matters when there are more kicks than the horizon)."""
    assert side_distribution(["L"], 5) == (1.0, 0.0, 0.0)
    assert side_distribution(["L"], 20) == (1.0, 0.0, 0.0)


def test_side_distribution_takes_last_n_chronologically() -> None:
    """`side_distribution(["L"] * 10 + ["R"] * 10, 5)` returns all R,
    because the last 5 entries are the R's."""
    p_L, p_C, p_R = side_distribution(["L"] * 10 + ["R"] * 10, 5)
    assert (p_L, p_C, p_R) == (0.0, 0.0, 1.0)


def test_side_distribution_horizons_nest() -> None:
    """The last-5 window is a subset of last-10, which is a subset of
    last-20. Verify on a constructed sequence where P(L) is
    monotonically non-decreasing across horizons: 10 L's, then 5 C's,
    then 5 R's. Last 5 = 5R, last 10 = 5C+5R, last 20 = 10L+5C+5R —
    so P(L) = 0, 0, 0.5."""
    sides = ["L"] * 10 + ["C"] * 5 + ["R"] * 5  # chronological
    p5 = side_distribution(sides, 5)
    p10 = side_distribution(sides, 10)
    p20 = side_distribution(sides, 20)
    assert p5 == (0.0, 0.0, 1.0)  # last 5 = 5R
    assert p10 == (0.0, 0.5, 0.5)  # last 10 = 5C + 5R
    assert p20 == (0.5, 0.25, 0.25)  # last 20 = 10L + 5C + 5R
    # Monotonic non-decreasing in P(L) across horizons.
    assert p5[0] <= p10[0] + 1e-9 <= p20[0] + 1e-9


def test_side_distribution_history_shorter_than_horizon() -> None:
    """When there are fewer kicks than the horizon, all kicks are used."""
    p_L, p_C, p_R = side_distribution(["L", "C"], 20)
    assert p_L == 0.5
    assert p_C == 0.5
    assert p_R == 0.0


def test_side_distribution_handles_only_relevant_values() -> None:
    """Unexpected side strings (none in our data) are simply not counted
    in the numerator, so the proportions sum to < 1.0. We do not
    regress that behaviour silently — the function trusts the caller
    to have bucketed via `coordinates.side`."""
    p_L, p_C, p_R = side_distribution(["L", "X", "L"], 5)
    assert (p_L, p_C, p_R) == (2 / 3, 0.0, 0.0)


# ---------------------------------------------------------------------------
# is_decisive_kick
# ---------------------------------------------------------------------------


def test_is_decisive_first_kick_is_not_decisive() -> None:
    """The very first kick of a shootout: 0-0, both teams have 5
    remaining. Neither scoring nor missing ends the shootout."""
    assert is_decisive_kick(0, 0, 0, 0, is_home_kicking=True) is False
    assert is_decisive_kick(0, 0, 0, 0, is_home_kicking=False) is False


def test_is_decisive_2022_final_montiel_kick() -> None:
    """The 2022 final kick 8 (Montiel, home, score 3-2): scoring ends
    the shootout (Argentina clinches 4-2 with 1 home kick and 1 away
    kick remaining; 4 > 2+1=3). Missing does not end the shootout
    (3-2 with 1+1 remaining; home is not yet clinched). So the kick is
    decisive (scoring ends it)."""
    # Before kick 8: home 3, away 4 done; score 3-2.
    assert is_decisive_kick(3, 2, home_kicks_done=3, away_kicks_done=4, is_home_kicking=True) is True


def test_is_decisive_2022_final_kolo_muani_kick() -> None:
    """The 2022 final kick 7 (Kolo Muani, away, score 3-1): missing
    ends the shootout (Argentina clinches 3-1 with 2 home and 1 away
    remaining; 3 > 1+1=2). Scoring does not (3-2 with 2+1 remaining;
    3 not > 2+1=3). So the kick is decisive (missing ends it)."""
    assert is_decisive_kick(3, 1, home_kicks_done=3, away_kicks_done=3, is_home_kicking=False) is True


def test_is_decisive_2022_final_paredes_kick_not_decisive() -> None:
    """The 2022 final kick 6 (Paredes, home, score 2-1): neither
    outcome ends the shootout (Argentina cannot clinch by scoring, and
    cannot be eliminated by missing)."""
    assert is_decisive_kick(2, 1, home_kicks_done=2, away_kicks_done=3, is_home_kicking=True) is False


def test_is_decisive_full_round_clinches() -> None:
    """5th home kick, score 4-0, home 4 done, away 4 done. Scoring
    makes it 5-0, away has 1 kick left, max away = 1. 5 > 1. Clinched.
    Missing keeps it 4-0, away has 1 kick, max away = 1. 4 > 1. Clinched.
    So scoring OR missing ends the shootout → decisive."""
    assert is_decisive_kick(4, 0, home_kicks_done=4, away_kicks_done=4, is_home_kicking=True) is True


def test_is_decisive_elimination_in_both_branches() -> None:
    """4th home kick, score 0-3, home 3 done, away 3 done. Home has
    1 kick left, away has 2 left. Home max possible score is 1,
    away already at 3 → home is eliminated. So both scoring and
    missing end the shootout → decisive."""
    assert is_decisive_kick(0, 3, home_kicks_done=3, away_kicks_done=3, is_home_kicking=True) is True


# ---------------------------------------------------------------------------
# age_in_years
#
# Issue #41: the model no longer reads the kicker's age (the C2
# feature was dropped; the ablation in `docs/model-review.md` Topic
# 2.3 showed age actively hurt the save rate on the 28-row 2026
# holdout). The `age_in_years` helper is kept in `features.py` for
# backwards compatibility with downstream consumers and so a future
# re-introduction of the feature can lift the helper into
# `compute_features` without re-deriving the date math. The tests
# below pin the helper's behaviour.
# ---------------------------------------------------------------------------


def test_age_in_years_simple() -> None:
    """Messi: born 1987-06-24, target 2022-12-18 → 35 years (he had
    his 35th birthday in June 2022)."""
    assert age_in_years("1987-06-24", "2022-12-18T15:00:00+00:00") == 35.0


def test_age_in_years_before_birthday() -> None:
    """Born 1987-06-24, target 2022-01-01 → 34 years (he hadn't had
    his 35th birthday yet)."""
    assert age_in_years("1987-06-24", "2022-01-01T00:00:00+00:00") == 34.0


def test_age_in_years_empty_returns_nan() -> None:
    assert math.isnan(age_in_years("", "2022-12-18T15:00:00+00:00"))


def test_age_in_years_malformed_returns_nan() -> None:
    assert math.isnan(age_in_years("not-a-date", "2022-12-18T15:00:00+00:00"))
    assert math.isnan(age_in_years("1987-06-24", "not-a-date"))


# ---------------------------------------------------------------------------
# filter_history
# ---------------------------------------------------------------------------


_penalty = make_history_row


def test_filter_history_excludes_target_date() -> None:
    """Penalties with the same match_date as the target are excluded
    (strict `<`)."""
    history = [
        _penalty(1, "2022-11-22T10:00:00+00:00", side="L"),
        _penalty(2, "2022-12-18T15:00:00+00:00", side="R"),
        _penalty(3, "2022-12-18T15:00:00+00:00", side="C"),
    ]
    out = filter_history(history, "2022-12-18T15:00:00+00:00")
    assert [p.match_id for p in out] == [1]


def test_filter_history_sorts_chronologically() -> None:
    history = [
        _penalty(2, "2022-12-13T19:00:00+00:00", side="L"),
        _penalty(1, "2022-11-22T10:00:00+00:00", side="R"),
        _penalty(3, "2022-12-09T19:00:00+00:00", side="C"),
    ]
    out = filter_history(history, "2022-12-18T15:00:00+00:00")
    assert [p.match_id for p in out] == [1, 3, 2]


def test_filter_history_empty() -> None:
    assert filter_history([], "2022-12-18T15:00:00+00:00") == []


# ---------------------------------------------------------------------------
# index_kicks_done
# ---------------------------------------------------------------------------


_shootout_kick = make_shootout_kick


def test_index_kicks_done_walks_match_in_order() -> None:
    """For the 2022 final, kick 1 is away, kick 2 is home, ... The
    index records (home_kicks_done, away_kicks_done) BEFORE each kick."""
    # Build the kicks in canonical 2022-final order: kick k is_home = (k % 2 == 0).
    kicks = [
        _shootout_kick(1, k, is_home=(k % 2 == 0)) for k in range(1, 9)
    ]
    idx = index_kicks_done(kicks)
    assert idx[(1, 1)] == KickIndex(home_kicks_done=0, away_kicks_done=0)
    assert idx[(1, 2)] == KickIndex(home_kicks_done=0, away_kicks_done=1)
    assert idx[(1, 3)] == KickIndex(home_kicks_done=1, away_kicks_done=1)
    assert idx[(1, 8)] == KickIndex(home_kicks_done=3, away_kicks_done=4)


def test_index_kicks_done_handles_multiple_matches() -> None:
    """Kicks_done counts are per-match."""
    kicks = [
        _shootout_kick(1, 1, is_home=False),
        _shootout_kick(1, 2, is_home=True),
        _shootout_kick(2, 1, is_home=True),  # different match
    ]
    idx = index_kicks_done(kicks)
    assert idx[(1, 1)] == KickIndex(0, 0)
    assert idx[(1, 2)] == KickIndex(0, 1)
    assert idx[(2, 1)] == KickIndex(0, 0)


# ---------------------------------------------------------------------------
# build_features
# ---------------------------------------------------------------------------


def _target(
    *,
    match_id: int = 1,
    kick_number: int = 1,
    kicker_id: int = 1,
    is_home: bool = True,
    side: str = "L",
    match_date: str = "2022-12-18T15:00:00+00:00",
) -> ShootoutKick:
    return _shootout_kick(
        match_id=match_id,
        kick_number=kick_number,
        kicker_id=kicker_id,
        is_home=is_home,
        side=side,
        match_date=match_date,
    )


def test_build_features_with_no_history_uses_prior() -> None:
    """No history → A1 = (1/3, 1/3, 1/3), A2 = "", A3 = "" (no metadata),
    A4 = 0.

    v3 (Issue #36): A3 (`preferred_foot`) comes from `PlayerMetadata`
    directly, not the per-penalty `shot_type` mode. With the default
    `PlayerMetadata(preferred_foot="")`, A3 is the empty string — not
    the historical "Unknown" sentinel. (The PredictionRow keeps
    `kicking_foot=""` for kickers with no metadata, so the dashboard
    renders an empty cell rather than "Unknown".)
    """
    target = _target(kicker_id=1, side="L")
    features = compute_features(
        history=[],
        metadata=PlayerMetadata(player_id=1, player_name="X", position_key="striker", birth_date="1990-01-01"),
        target_date=target.match_date,
        b_group=BGroupContext(
            kick_number=target.kick_number,
            pen_score_home=target.pen_score_before[0],
            pen_score_away=target.pen_score_before[1],
            is_home=target.is_home,
        ),
        kicks_done=KickIndex(0, 0),
    )
    row = build_features(
        features,
        match_id=target.match_id,
        kick_number=target.kick_number,
        kicker_id=target.kicker_id,
        kicker_name=target.kicker_name,
        match_date=target.match_date,
        tournament_id=target.tournament_id,
        tournament_name=target.tournament_name,
        round=target.round,
        team_id=target.team_id,
        is_home=target.is_home,
        label=target.side,
        is_on_target=target.is_on_target,
    )
    assert row.label == "L"
    assert (row.p_L_5, row.p_C_5, row.p_R_5) == PRIOR_PROB
    assert (row.p_L_10, row.p_C_10, row.p_R_10) == PRIOR_PROB
    assert (row.p_L_20, row.p_C_20, row.p_R_20) == PRIOR_PROB
    assert row.last_side == ""
    assert row.preferred_foot == ""
    assert row.career_penalty_count == 0


def test_build_features_with_history_computes_a1_a2_a3_a4() -> None:
    """A 5-kick history: A1 over last 5 = (0.6, 0.0, 0.4); A2 = "R";
    A4 = 5. A3 (`preferred_foot`) comes from metadata, not the
    per-penalty mode — the test metadata passes `preferred_foot="right"`,
    so the row carries the declared foot regardless of the per-penalty
    `shot_type` history."""
    history = [
        _penalty(1, "2022-01-01T00:00:00+00:00", side="L"),
        _penalty(2, "2022-02-01T00:00:00+00:00", side="L"),
        _penalty(3, "2022-03-01T00:00:00+00:00", side="R"),
        _penalty(4, "2022-04-01T00:00:00+00:00", side="L"),
        _penalty(5, "2022-05-01T00:00:00+00:00", side="R"),
    ]
    target = _target(kicker_id=1, side="L")
    features = compute_features(
        history=history,
        metadata=PlayerMetadata(
            player_id=1,
            player_name="X",
            position_key="striker",
            birth_date="1990-01-01",
            preferred_foot="right",
        ),
        target_date=target.match_date,
        b_group=BGroupContext(
            kick_number=target.kick_number,
            pen_score_home=target.pen_score_before[0],
            pen_score_away=target.pen_score_before[1],
            is_home=target.is_home,
        ),
        kicks_done=KickIndex(0, 0),
    )
    row = build_features(
        features,
        match_id=target.match_id,
        kick_number=target.kick_number,
        kicker_id=target.kicker_id,
        kicker_name=target.kicker_name,
        match_date=target.match_date,
        tournament_id=target.tournament_id,
        tournament_name=target.tournament_name,
        round=target.round,
        team_id=target.team_id,
        is_home=target.is_home,
        label=target.side,
        is_on_target=target.is_on_target,
    )
    assert (row.p_L_5, row.p_C_5, row.p_R_5) == (0.6, 0.0, 0.4)
    assert row.last_side == "R"
    assert row.preferred_foot == "right"
    assert row.career_penalty_count == 5


def test_build_features_a1_horizons_nest() -> None:
    """A1 monotonicity: with 20 L's then 10 R's then 5 L's, the last
    5 = 5L, last 10 = 5L+5R, last 20 = 15L+5R. P(L) is monotonically
    non-decreasing across horizons (1.0, 0.5, 0.75)... wait, that's
    NOT monotonic in this case. Use a sequence where it IS
    monotonic."""
    history = (
        [_penalty(i, f"2021-{i:02d}-01T00:00:00+00:00", side="L") for i in range(1, 21)]
        + [_penalty(100 + i, f"2022-{i:02d}-01T00:00:00+00:00", side="L") for i in range(1, 6)]
    )
    target = _target(kicker_id=1)
    features = compute_features(
        history=history,
        metadata=PlayerMetadata(player_id=1, player_name="X", position_key="striker", birth_date="1990-01-01"),
        target_date=target.match_date,
        b_group=BGroupContext(
            kick_number=target.kick_number,
            pen_score_home=target.pen_score_before[0],
            pen_score_away=target.pen_score_before[1],
            is_home=target.is_home,
        ),
        kicks_done=KickIndex(0, 0),
    )
    row = build_features(
        features,
        match_id=target.match_id,
        kick_number=target.kick_number,
        kicker_id=target.kicker_id,
        kicker_name=target.kicker_name,
        match_date=target.match_date,
        tournament_id=target.tournament_id,
        tournament_name=target.tournament_name,
        round=target.round,
        team_id=target.team_id,
        is_home=target.is_home,
        label=target.side,
        is_on_target=target.is_on_target,
    )
    # Last 5 = 5L → p_L_5 = 1.0. Last 10 = 10L → 1.0. Last 20 = 20L → 1.0.
    # So monotonicity holds trivially.
    assert row.p_L_5 == 1.0
    assert row.p_L_10 == 1.0
    assert row.p_L_20 == 1.0


def test_build_features_c1_from_metadata() -> None:
    """Position (C1) comes from the metadata. Issue #41 dropped the
    C2 (`age`) feature, so the model no longer reads the kicker's
    birth date. The birth date is still on `PlayerMetadata` for
    the data layer's records (and `age_in_years` is still tested
    directly), but the model's `PredictionTarget` no longer carries
    an `age` field."""
    target = _target()
    features = compute_features(
        history=[],
        metadata=PlayerMetadata(
            player_id=1,
            player_name="X",
            position_key="centreback",
            birth_date="1995-05-01",
        ),
        target_date=target.match_date,
        b_group=BGroupContext(
            kick_number=target.kick_number,
            pen_score_home=target.pen_score_before[0],
            pen_score_away=target.pen_score_before[1],
            is_home=target.is_home,
        ),
        kicks_done=KickIndex(0, 0),
    )
    row = build_features(
        features,
        match_id=target.match_id,
        kick_number=target.kick_number,
        kicker_id=target.kicker_id,
        kicker_name=target.kicker_name,
        match_date=target.match_date,
        tournament_id=target.tournament_id,
        tournament_name=target.tournament_name,
        round=target.round,
        team_id=target.team_id,
        is_home=target.is_home,
        label=target.side,
        is_on_target=target.is_on_target,
    )
    assert row.position == "centreback"
    # Issue #41: no `age` attribute on the unified row type.
    assert not hasattr(row, "age")


def test_build_features_c1_handles_missing_metadata() -> None:
    """`metadata=None` → C1 = "" (no position known). The C2 (`age`)
    feature is gone; the test asserts position is the empty string
    and the row has no `age` attribute."""
    target = _target()
    features = compute_features(
        history=[],
        metadata=None,
        target_date=target.match_date,
        b_group=BGroupContext(
            kick_number=target.kick_number,
            pen_score_home=target.pen_score_before[0],
            pen_score_away=target.pen_score_before[1],
            is_home=target.is_home,
        ),
        kicks_done=KickIndex(0, 0),
    )
    row = build_features(
        features,
        match_id=target.match_id,
        kick_number=target.kick_number,
        kicker_id=target.kicker_id,
        kicker_name=target.kicker_name,
        match_date=target.match_date,
        tournament_id=target.tournament_id,
        tournament_name=target.tournament_name,
        round=target.round,
        team_id=target.team_id,
        is_home=target.is_home,
        label=target.side,
        is_on_target=target.is_on_target,
    )
    assert row.position == ""
    assert not hasattr(row, "age")


def test_build_features_b1_b2_pass_through() -> None:
    """B1 = kick_number, B2 = pen_score_before + is_decisive.

    v3 (Issue #36) dropped B3 (`b3_round`) from the feature schema;
    the `round` field on `ShootoutKick` is identifier-only (kept in
    the `TrainingRow` for traceability but not in the model matrix).
    """
    target = ShootoutKick(
        match_id=99,
        match_date="2024-07-15T20:00:00+00:00",
        tournament_id=50,
        tournament_name="Euro",
        round="Quarter-finals",
        kick_number=7,
        kicker_id=1,
        kicker_name="X",
        team_id=1,
        is_home=True,
        x=0.5,
        side="R",
        is_on_target=True,
        outcome="Goal",
        pen_score_before=[2, 3],
        pen_score_after=[2, 3],
        match_score_home=2,
        match_score_away=2,
    )
    features = compute_features(
        history=[],
        metadata=None,
        target_date=target.match_date,
        b_group=BGroupContext(
            kick_number=target.kick_number,
            pen_score_home=target.pen_score_before[0],
            pen_score_away=target.pen_score_before[1],
            is_home=target.is_home,
        ),
        # Before kick 7: home 3 done, away 3 done. Score 2-3.
        # Scoring (3-3) → not clinched, not eliminated. Missing (2-3) → not clinched, not eliminated. Not decisive.
        kicks_done=KickIndex(3, 3),
    )
    row = build_features(
        features,
        match_id=target.match_id,
        kick_number=target.kick_number,
        kicker_id=target.kicker_id,
        kicker_name=target.kicker_name,
        match_date=target.match_date,
        tournament_id=target.tournament_id,
        tournament_name=target.tournament_name,
        round=target.round,
        team_id=target.team_id,
        is_home=target.is_home,
        label=target.side,
        is_on_target=target.is_on_target,
    )
    assert row.b1_kick_number == 7
    assert row.pen_score_home == 2
    assert row.pen_score_away == 3
    assert row.is_decisive is False
    # The round is identifier-only; no model column.
    assert row.round == "Quarter-finals"


def test_compute_features_returns_prediction_target() -> None:
    """`compute_features` returns a `PredictionTarget` (not a `TrainingRow`).

    The split (Issue #30) separates the computation from the
    packaging: `compute_features` produces the 18 model features; the
    caller wraps them in a `TrainingRow` via `build_features`. The
    `PredictionTarget` is the value object the prediction slice uses
    directly, with no synthetic `ShootoutKick`.
    """
    target = _target()
    features = compute_features(
        history=[],
        metadata=PlayerMetadata(
            player_id=1,
            player_name="X",
            position_key="striker",
            birth_date="1990-01-01",
        ),
        target_date=target.match_date,
        b_group=BGroupContext(
            kick_number=target.kick_number,
            pen_score_home=target.pen_score_before[0],
            pen_score_away=target.pen_score_before[1],
            is_home=target.is_home,
        ),
        kicks_done=KickIndex(0, 0),
    )
    assert isinstance(features, PredictionTarget)
    assert (features.p_L_5, features.p_C_5, features.p_R_5) == PRIOR_PROB
    assert features.last_side == ""
    assert features.preferred_foot == ""  # no metadata
    assert features.b1_kick_number == target.kick_number
    # v3: no b3_round column (B3 was dropped).


def test_b_group_context_neutral_has_neutral_b_group() -> None:
    """`BGroupContext.neutral()` produces the values the prediction
    slice uses: kick_number=1, score=0-0, is_decisive=False.

    This is what the prediction slice feeds to `compute_features` so
    the model's B-group features don't leak shootout-state info that
    doesn't exist for a prediction target. v3 dropped the `round`
    field — the model is round-agnostic.
    """
    ctx = BGroupContext.neutral()
    assert ctx.kick_number == 1
    assert ctx.pen_score_home == 0
    assert ctx.pen_score_away == 0
    assert ctx.is_home is True
    # v3: no `round` attribute on BGroupContext.


# ---------------------------------------------------------------------------
# load_player_history
# ---------------------------------------------------------------------------


def test_load_player_history_groups_by_kicker(tmp_path: Path) -> None:
    """`load_player_history` returns a dict keyed by kicker_id, with
    each value the list of penalties for that kicker."""
    path = tmp_path / "h.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for kicker_id, match_id in ((1, 100), (1, 101), (2, 200)):
            row = {
                "kicker_id": kicker_id,
                "match_id": match_id,
                "match_date": "2022-01-01T00:00:00+00:00",
                "league_id": 77,
                "league_name": "WC",
                "team_id": 1,
                "is_home": True,
                "x": 0.5,
                "side": "L",
                "is_on_target": True,
                "outcome": "Goal",
                "shot_type": "RightFoot",
            }
            f.write(json.dumps(row) + "\n")
    history = load_player_history(path)
    assert set(history.keys()) == {1, 2}
    assert len(history[1]) == 2
    assert len(history[2]) == 1


# ---------------------------------------------------------------------------
# build_training_table + Artifacts.write_training_table
# ---------------------------------------------------------------------------


def _stub_metadata(kicker_id: int) -> PlayerMetadata | None:
    """Return canned metadata: all kickers are strikers born 1995-05-01
    except kicker 99, who is missing (returns None)."""
    if kicker_id == 99:
        return None
    return PlayerMetadata(
        player_id=kicker_id,
        player_name=f"Player {kicker_id}",
        position_key="striker",
        birth_date="1995-05-01",
    )


def test_build_training_table_returns_one_row_per_kick() -> None:
    """3 kicks across 2 matches → 3 rows."""
    kicks = [
        _shootout_kick(1, 1, kicker_id=1),
        _shootout_kick(1, 2, kicker_id=2),
        _shootout_kick(2, 1, kicker_id=1),
    ]
    history = {1: [], 2: []}
    rows = build_training_table(kicks, history, _stub_metadata)
    assert len(rows) == 3


def test_build_training_table_sorts_by_match_then_kick_number() -> None:
    """The output is sorted by (match_date, match_id, kick_number) for
    a stable, idempotent order."""
    kicks = [
        _shootout_kick(1, 2, kicker_id=2, match_date="2022-12-18T15:00:00+00:00"),
        _shootout_kick(1, 1, kicker_id=1, match_date="2022-12-18T15:00:00+00:00"),
        _shootout_kick(2, 1, kicker_id=1, match_date="2024-07-15T20:00:00+00:00"),
    ]
    history = {1: [], 2: []}
    rows = build_training_table(kicks, history, _stub_metadata)
    keys = [(r.match_id, r.kick_number) for r in rows]
    assert keys == [(1, 1), (1, 2), (2, 1)]


def test_build_training_table_idempotent() -> None:
    """Re-running with the same inputs yields the same output."""
    kicks = [
        _shootout_kick(1, 1, kicker_id=1),
        _shootout_kick(1, 2, kicker_id=2),
    ]
    history = {1: [], 2: []}
    rows1 = build_training_table(kicks, history, _stub_metadata)
    rows2 = build_training_table(kicks, history, _stub_metadata)
    assert [asdict_payload(r) for r in rows1] == [asdict_payload(r) for r in rows2]


def asdict_payload(r: TrainingRow) -> dict[str, object]:
    """`asdict` for the row, with no field-level normalisation.

    Issue #41 dropped the `age` column (the only field whose value
    could be `NaN`); the v3 row has no NaN-valued fields, so the
    payload is the raw `dataclasses.asdict(r)` minus nothing.
    """
    return {f: getattr(r, f) for f in r.__dataclass_fields__}  # type: ignore[union-attr]


def test_write_training_table_roundtrip(tmp_path: Path) -> None:
    """Writing and reading the JSONL preserves the v3 17-feature
    schema. No NaN handling is needed (Issue #41 dropped the `age`
    column; all v3 fields are int / float / str / bool)."""
    row = TrainingRow(
        match_id=1,
        kick_number=1,
        kicker_id=1,
        kicker_name="X",
        match_date="2022-12-18T15:00:00+00:00",
        tournament_id=77,
        tournament_name="World Cup",
        round="Final",
        team_id=1,
        is_home=True,
        label="L",
        is_on_target=True,
        p_L_5=1.0,
        p_C_5=0.0,
        p_R_5=0.0,
        p_L_10=1.0,
        p_C_10=0.0,
        p_R_10=0.0,
        p_L_20=1.0,
        p_C_20=0.0,
        p_R_20=0.0,
        last_side="R",
        preferred_foot="right",
        career_penalty_count=5,
        b1_kick_number=1,
        pen_score_home=0,
        pen_score_away=0,
        is_decisive=False,
        position="striker",
    )
    out = tmp_path / "tt.jsonl"
    art = Artifacts(root=tmp_path)
    n = art.write_training_table([row], path=out)
    assert n == 1
    with out.open(encoding="utf-8") as f:
        text = f.read()
    # Issue #41: no NaN handling needed; the JSON has no `null` fields.
    assert "null" not in text
    assert "NaN" not in text
    loaded = json.loads(text)
    # v3 17-feature schema: no `age` key.
    assert "age" not in loaded
    assert loaded["position"] == "striker"
    assert loaded["preferred_foot"] == "right"


def test_write_training_table_empty(tmp_path: Path) -> None:
    """An empty row list writes an empty file and returns 0."""
    out = tmp_path / "tt.jsonl"
    art = Artifacts(root=tmp_path)
    n = art.write_training_table([], path=out)
    assert n == 0
    assert out.read_text(encoding="utf-8") == ""


# ---------------------------------------------------------------------------
# Live smoke test
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (
        Artifacts().shootout_kicks.exists()
        and Artifacts().player_history.exists()
        and Artifacts().training_table.exists()
    ),
    reason="output/ JSONL artifacts not present (run the slice first)",
)
def test_training_table_jsonl_schema_smoke() -> None:
    """Smoke test against the live `output/training_table.jsonl`:

    1. Every row has the unified `TrainingRow` schema (derived from
       `dataclasses.fields(TrainingRow)` in `tests._factories`).
    2. Row count equals the count in `shootout_kicks.jsonl`.
    3. A1 sums to 1.0 (within 1e-6) for every row.
    4. `label` is in {L, C, R}.
    5. `is_decisive` is a bool.
    6. Every training kicker (from `shootout_kicks.jsonl`) has at
       least one row (sanity: no kickers dropped).
    7. No `age` key (Issue #41 dropped the column from the v3 schema).
    """
    art = Artifacts()
    n_target = 0
    with art.shootout_kicks.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n_target += 1
    assert n_target > 0

    n_rows = 0
    kickers: set[int] = set()
    training_kickers: set[int] = set()
    with art.shootout_kicks.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            training_kickers.add(int(json.loads(line)["kicker_id"]))

    with art.training_table.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            assert TRAINING_ROW_FIELDS <= set(row.keys()), (
                f"row missing fields: {TRAINING_ROW_FIELDS - set(row.keys())}"
            )
            assert row["label"] in {"L", "C", "R"}, f"bad label: {row['label']!r}"
            assert isinstance(row["is_decisive"], bool)
            for col in ("p_L_5", "p_C_5", "p_R_5"):
                assert 0.0 <= row[col] <= 1.0
            # A1 sum check (use last-5 horizon, the strictest).
            assert abs(row["p_L_5"] + row["p_C_5"] + row["p_R_5"] - 1.0) < 1e-6, (
                f"A1 last-5 doesn't sum to 1: {row['p_L_5']} + {row['p_C_5']} + {row['p_R_5']}"
            )
            # Issue #41: no `age` key in the v3 JSONL.
            assert "age" not in row, "v3 schema should not carry an `age` field"
            n_rows += 1
            kickers.add(int(row["kicker_id"]))

    assert n_rows == n_target, f"row count {n_rows} != target count {n_target}"
    assert training_kickers <= kickers, (
        f"missing training kickers: {training_kickers - kickers}"
    )


@pytest.mark.skipif(
    not Artifacts().training_table.exists(),
    reason="output/training_table.jsonl not present (run the slice first)",
)
def test_training_table_a1_monotonicity_smoke() -> None:
    """A1 sanity check on the live data: P(L) over last 5 ≤ P(L) over
    last 10 ≤ P(L) over last 20 (within 1e-6) for the same kicker.
    The check is per-kicker: for each kicker, compare the
    corresponding rows in chronological order.

    Monotonicity does NOT hold for every kicker in general (e.g. a
    kicker who takes 15 R's then 5 L's has P_L_5 = 1.0, P_L_10 = 0.5,
    P_L_20 = 0.25). The check is a soft sanity, not a strict
    guarantee — the test passes as long as the data is plausible
    (most kickers have monotone P(L) across horizons, OR the kicker
    is too sparse to check).
    """
    by_kicker: dict[int, list[dict[str, object]]] = {}
    with Artifacts().training_table.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            by_kicker.setdefault(int(row["kicker_id"]), []).append(row)

    n_checked = 0
    n_monotone = 0
    for _kicker_id, rows in by_kicker.items():
        rows.sort(key=lambda r: (r["match_date"], r["match_id"], r["kick_number"]))
        for row in rows:
            p5 = float(row["p_L_5"])
            p10 = float(row["p_L_10"])
            p20 = float(row["p_L_20"])
            # Skip degenerate (uniform-prior) cases — no signal.
            if p5 == p10 == p20 and p5 == 1 / 3:
                continue
            n_checked += 1
            if p5 <= p10 + 1e-6 and p10 <= p20 + 1e-6:
                n_monotone += 1
    # Soft check: at least half the checked rows are monotone. We don't
    # require all to be monotone (the property doesn't hold in general
    # — see test docstring).
    if n_checked > 0:
        assert n_monotone >= n_checked // 2, (
            f"Only {n_monotone}/{n_checked} checked rows are monotone in P(L) "
            f"across A1 horizons — A1 sanity check failed."
        )
