"""Modelling."""

# %%

from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).parent.parent
DATA_PATH = ROOT_DIR / "data" / "matches.jsonl"

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

df_pen = df_pen.assign(is_goal=df_pen["outcome"].eq("Goal").astype("Int64")).assign(
    **pd.get_dummies(
        pd.cut(
            df_pen["shot_x"] * df_pen["shot_zoom"],
            bins=[0.0, 2 / 3, 4 / 3, 2.0],
            labels=["left", "center", "right"],
            right=False,
        ),
        prefix="shot",
    ),
)

# %%

# Sort by group and time upfront to guarantee 1-to-1 row order alignment
df_pen = df_pen.sort_values(["player_id", "start_at"]).reset_index(drop=True)

# Group once with sort=False (preserves pre-sorted order)
group = df_pen.set_index("start_at").groupby("player_id", group_keys=False, sort=False)

# Last year rolling metrics
roll_365d = (
    pd.concat(
        {
            "goals_last_year": group["is_goal"].rolling("365D").sum(),
            "attempts_last_year": group["is_goal"].rolling("365D").count(),
            "left_last_year": group["shot_left"].rolling("365D").sum(),
            "center_last_year": group["shot_center"].rolling("365D").sum(),
            "right_last_year": group["shot_right"].rolling("365D").sum(),
        },
        axis=1,
    )
    .assign(
        conversion_last_year=lambda x: x["goals_last_year"] / x["attempts_last_year"]
    )
    .reset_index(drop=True)  # Strips MultiIndex to align directly by row position
)

# Last 10 shots rolling metrics
roll_10 = (
    pd.concat(
        {
            "goals_last_10": group["is_goal"].rolling(10, min_periods=1).sum(),
            "attempts_last_10": group["is_goal"].rolling(10, min_periods=1).count(),
            "left_last_10": group["shot_left"].rolling(10, min_periods=1).sum(),
            "center_last_10": group["shot_center"].rolling(10, min_periods=1).sum(),
            "right_last_10": group["shot_right"].rolling(10, min_periods=1).sum(),
        },
        axis=1,
    )
    .assign(conversion_last_10=lambda x: x["goals_last_10"] / x["attempts_last_10"])
    .reset_index(drop=True)
)

# Join both feature sets directly into df_pen (row order matters)
df_pen = df_pen.assign(
    **roll_365d,
    **roll_10,
)

df_pen.head()

# %%

# TODO: TabPFN
