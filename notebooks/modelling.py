"""Modelling."""

# %%

import os
from pathlib import Path

import pandas as pd
import tabpfn_client
from sklearn.model_selection import train_test_split
from tabpfn_client import TabPFNClassifier

from twelveyards.fotmob.client import FLOOR_DATETIME as SCRAPER_FLOOR

ROOT_DIR = Path(__file__).parent.parent
DATA_PATH = ROOT_DIR / "data" / "matches.jsonl"

# %%

df = (
    pd.read_json(DATA_PATH, lines=True)
    .filter(
        items=["id", "start_at", "status", "penalties", "players"],
        axis="columns",
    )
    .convert_dtypes()
    .rename(columns={"id": "match_id"})
    .assign(start_at=lambda x: pd.to_datetime(x["start_at"]))
)

df.head()

# %%

df_status = (
    df.drop(columns=["status", "players"])
    .reset_index(drop=True)
    .join(pd.json_normalize(df["status"], sep="_"))
)
df_status.head()

# %%

df_pen = (
    df_status.explode("penalties").dropna(subset=["penalties"]).reset_index(drop=True)
)

df_pen = (
    df_pen.join(pd.json_normalize(df_pen["penalties"], sep="_"))
    .rename(columns={"id": "penalty_id"})
    .dropna(subset=["penalty_id"])
    .drop(columns="penalties")
    .convert_dtypes()
)

df_pen.head()

# %%

df_player = (
    df[["match_id", "players"]]
    .explode("players")
    .dropna(subset=["players"])
    .reset_index(drop=True)
)

df_player = (
    df_player.join(
        pd.json_normalize(df_player["players"], sep="_").add_prefix("player_"),
    )
    .rename(columns={"id": "player_id"})
    .drop(columns="players")
    .convert_dtypes()
    .set_index(["match_id", "player_id"])
)

df_player.head()


# %%

df_pen = df_pen.assign(
    is_goal=df_pen["outcome"].eq("Goal").astype("Int64"),
    shot_x_normalized=df_pen["shot_x"] * df_pen["shot_zoom"],
)

side = pd.cut(
    df_pen["shot_x_normalized"],
    bins=[0.0, 2 / 3, 4 / 3, 2.0],
    labels=["left", "center", "right"],
    right=False,
)

df_pen = df_pen.assign(
    side=side.astype("category"),
    **pd.get_dummies(side, prefix="shot", dtype="Int64"),
).drop(columns="shot_x_normalized")

# %%

# Sort by group and time to guarantee 1-to-1 row order alignment.
df_pen = df_pen.sort_values(["player_id", "start_at"]).reset_index(drop=True)

# Group once with sort=False (preserves pre-sorted order).
group = df_pen.set_index("start_at").groupby(
    "player_id",
    group_keys=False,
    sort=False,
)

# 5-year rolling window per kicker [T - 5y, T); closed="left" excludes the
# current kick so the side never leaks into the features.
roll = (
    pd.concat(
        {
            "attempts_5y": group["is_goal"].rolling("1825D", closed="left").count(),
            "left_5y": group["shot_left"].rolling("1825D", closed="left").sum(),
            "center_5y": group["shot_center"].rolling("1825D", closed="left").sum(),
            "right_5y": group["shot_right"].rolling("1825D", closed="left").sum(),
            "goals_5y": group["is_goal"].rolling("1825D", closed="left").sum(),
        },
        axis=1,
    )
    .assign(conversion_5y=lambda x: x["goals_5y"] / x["attempts_5y"])
    .reset_index(drop=True)
)

# Side the kicker took on their immediately previous penalty (one-hot). First
# kick of each kicker -> all-zero row (identified by absent history).
last_side = pd.get_dummies(
    group["side"].shift(1),
    prefix="last_side",
    dtype="Int64",
).reset_index(drop=True)

df_pen = df_pen.assign(**roll, **last_side)

df_pen.head()

# %%

# Join player attributes recorded at the match (age, market value, position).
df_pen = df_pen.join(
    df_player[["player_age", "player_market_value", "player_position_id"]],
    on=["match_id", "player_id"],
    how="inner",
)
# market_value is missing for ~45% of rows; flag unknowns and zero-fill so the
# estimator gets a clean numeric column.
df_pen = df_pen.assign(
    market_value_known=df_pen["player_market_value"].notna().astype("Int64"),
    player_market_value=df_pen["player_market_value"].fillna(0),
)

df_pen.head()

# %%

# Training floor: scraper floor + 5 years, so every training row has a full
# 5-year feature window of kicks behind it. Earlier kicks exist only as history.
TRAIN_FLOOR = SCRAPER_FLOOR + pd.DateOffset(years=5)

# Target: the side of the goal the kicker chooses, bucketed from goal-mouth x.
FEATURES = [
    "attempts_5y",
    "left_5y",
    "center_5y",
    "right_5y",
    "conversion_5y",
    "last_side_left",
    "last_side_center",
    "last_side_right",
    "player_age",
    "player_market_value",
    "market_value_known",
    "player_position_id",
]

df_model = df_pen.loc[
    df_pen["start_at"] >= TRAIN_FLOOR,
    [*FEATURES, "side", "start_at"],
]

# First kick of each kicker has no history. Drop, can't score.
df_model = df_model.dropna(subset=FEATURES).reset_index(drop=True)
df_model.head()

# %%

# Chronological split: train on older kicks, test on recent ones (no shuffle).
# Matches the deployment task — predict future kicks from past tendencies.

df_model = df_model.sort_values("start_at").reset_index(drop=True)
X_train, X_test, y_train, y_test = train_test_split(
    df_model[FEATURES].astype("float64"),
    df_model["side"],
    test_size=0.2,
    shuffle=False,
    random_state=42,
)

(
    len(X_train),
    len(X_test),
    y_train.value_counts().to_dict(),
    y_test.value_counts().to_dict(),
)

# %%

tabpfn_client.set_access_token(os.environ["PRIOR_LABS_API_KEY"])
clf = TabPFNClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# %%

pd.crosstab(y_test, y_pred, rownames=["actual"], colnames=["pred"], margins=True)

# %%


# %%

# Probability distribution over sides
y_proba = clf.predict_proba(X_test)
pd.DataFrame(y_proba, columns=clf.classes_, index=X_test.index).head()
