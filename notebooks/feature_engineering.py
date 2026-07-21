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

df_pen.join(df_player, on=["match_id", "player_id"]).iloc[0]

# %%

_side = pd.cut(
    df_pen["shot_x"] * df_pen["shot_zoom"],
    bins=[0.0, 2 / 3, 4 / 3, 2.0],
    labels=["L", "C", "R"],
    right=False,
)
df_pen = df_pen.assign(shot_side=_side).join(
    pd.get_dummies(_side, prefix="shot").astype("Int64"),
)

# %%

df_pen = (
    df_pen.assign(is_goal=df_pen["outcome"].eq("Goal").astype("Int64"))
    .sort_values("start_at")
    .set_index("start_at")
)

# %%

_g = df_pen.groupby("player_id", group_keys=False)

(
    pd.concat(
        {
            "goals_last_year": _g["is_goal"].rolling("365D").sum(),
            "attempts_last_year": _g["is_goal"].rolling("365D").count(),
            "L_last_year": _g["shot_L"].rolling("365D").sum(),
            "C_last_year": _g["shot_C"].rolling("365D").sum(),
            "R_last_year": _g["shot_R"].rolling("365D").sum(),
        },
        axis=1,
    )
    .assign(
        conversion_last_year=lambda x: x["goals_last_year"] / x["attempts_last_year"],
    )
    .head(12)
)

# %%

# Rolling features based on attempts with groupby (e.g. conversion rate
# over the last 10 attempts per player, ordered by match date).
# df_pen is already sorted by start_at and datetime-indexed from above.

_df_rolling = (
    pd.concat(
        {
            "goals_last_10": _g["is_goal"].rolling(10, min_periods=1).sum(),
            "attempts_last_10": _g["is_goal"].rolling(10, min_periods=1).count(),
            "L_last_10": _g["shot_L"].rolling(10, min_periods=1).sum(),
            "C_last_10": _g["shot_C"].rolling(10, min_periods=1).sum(),
            "R_last_10": _g["shot_R"].rolling(10, min_periods=1).sum(),
        },
        axis=1,
    )
    .assign(
        conversion_last_10=lambda x: x["goals_last_10"] / x["attempts_last_10"],
    )
)
_df_rolling.head(12)
