"""
Prediction CLI script.

Fits TabPFN model on historical penalties and outputs predictions.jsonl.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import click
import pandas as pd
import tabpfn_client
from pydantic import BaseModel, Field
from tabpfn_client import TabPFNClassifier

from twelveyards.fotmob.client import FLOOR_DATETIME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MATCHES_PATH = DATA_DIR / "matches.jsonl"
PREDICTIONS_PATH = DATA_DIR / "predictions.jsonl"

# Training floor is 5 years after the scrape floor
TRAIN_FLOOR = FLOOR_DATETIME + pd.DateOffset(years=5)

FEATURES = [
    "attempts_5y",
    "left_5y",
    "center_5y",
    "right_5y",
    "conversion_5y",
    "attempts_1y",
    "left_1y",
    "center_1y",
    "right_1y",
    "conversion_1y",
    "last_side_left",
    "last_side_center",
    "last_side_right",
    "player_age",
    "player_market_value",
    "market_value_known",
    "player_position_id",
]


class PredictionRow(BaseModel):
    """Prediction output row for a single kicker."""

    player_id: int
    player_name: str
    short_name: str
    team_id: int
    team_name: str
    kicking_foot: str | None = None
    photo_url: str
    total_penalties: int
    p_l: float = Field(alias="p_L")
    p_c: float = Field(alias="p_C")
    p_r: float = Field(alias="p_R")

    model_config = {"populate_by_name": True}


def prepare_features(matches_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load matches.jsonl and prepare feature matrices.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (df_train, df_predict)

    """
    if not matches_path.exists():
        err_msg = f"Matches file not found at {matches_path}"
        raise FileNotFoundError(err_msg)

    raw_df = pd.read_json(matches_path, lines=True)
    if raw_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = (
        raw_df.filter(
            items=["id", "start_at", "status", "penalties", "players"],
            axis="columns",
        )
        .convert_dtypes()
        .rename(columns={"id": "match_id"})
        .assign(start_at=lambda x: pd.to_datetime(x["start_at"]))
    )

    # 1. Unnest penalties
    df_pen = df.explode("penalties").dropna(subset=["penalties"]).reset_index(drop=True)
    if df_pen.empty:
        return pd.DataFrame(), pd.DataFrame()

    df_pen = (
        df_pen.join(pd.json_normalize(df_pen["penalties"], sep="_"))
        .rename(columns={"id": "penalty_id"})
        .dropna(subset=["penalty_id"])
        .drop(columns="penalties")
        .convert_dtypes()
    )

    # 2. Unnest player info for match-level metadata
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
    )

    # Pick representative metadata per player across matches
    player_meta = (
        df_player.sort_values("match_id")
        .groupby("player_id")
        .last()
        .reset_index()
    )

    # Normalize penalty shot outcome & target side
    df_pen = df_pen.assign(
        is_goal=df_pen["outcome"].eq("Goal").astype("Int64"),
        shot_x_normalized=df_pen["shot_x"] * df_pen["shot_zoom"],
    )

    side_bins = pd.cut(
        df_pen["shot_x_normalized"],
        bins=[0.0, 2 / 3, 4 / 3, 2.0],
        labels=["left", "center", "right"],
        right=False,
    )

    df_pen = df_pen.assign(
        side=side_bins.astype("category"),
        **pd.get_dummies(side_bins, prefix="shot", dtype="Int64"),
        is_prediction_dummy=False,
    ).drop(columns="shot_x_normalized")

    # 3. Append a dummy prediction row for each player with >= 1 penalty
    now_dt = df_pen["start_at"].max() + pd.DateOffset(seconds=1)
    unique_players = df_pen["player_id"].unique()

    dummy_rows: list[dict[str, Any]] = [
        {
            "player_id": pid,
            "team_id": df_pen.loc[df_pen["player_id"] == pid, "team_id"].iloc[-1],
            "start_at": now_dt,
            "side": None,
            "is_goal": 0,
            "shot_left": 0,
            "shot_center": 0,
            "shot_right": 0,
            "is_prediction_dummy": True,
        }
        for pid in unique_players
    ]

    df_dummies = pd.DataFrame(dummy_rows).astype(
        {
            "player_id": "int64",
            "team_id": "int64",
            "is_goal": "Int64",
            "shot_left": "Int64",
            "shot_center": "Int64",
            "shot_right": "Int64",
            "is_prediction_dummy": "bool",
        },
    )
    df_dummies["side"] = pd.Series(dtype=df_pen["side"].dtype)
    df_combined = pd.concat([df_pen, df_dummies], ignore_index=True)

    # Sort by player and start_at to compute rolling features
    df_combined = df_combined.sort_values(
        ["player_id", "start_at"],
    ).reset_index(drop=True)

    group = df_combined.set_index("start_at").groupby(
        "player_id",
        group_keys=False,
        sort=False,
    )

    # 5-year rolling window [T - 5y, T); closed="left"
    roll_5y = (
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
        .assign(
            conversion_5y=lambda x: (x["goals_5y"] / x["attempts_5y"]).fillna(0.0),
        )
        .reset_index(drop=True)
    )

    # 1-year rolling window [T - 1y, T); closed="left"
    roll_1y = (
        pd.concat(
            {
                "attempts_1y": group["is_goal"].rolling("365D", closed="left").count(),
                "left_1y": group["shot_left"].rolling("365D", closed="left").sum(),
                "center_1y": group["shot_center"].rolling("365D", closed="left").sum(),
                "right_1y": group["shot_right"].rolling("365D", closed="left").sum(),
                "goals_1y": group["is_goal"].rolling("365D", closed="left").sum(),
            },
            axis=1,
        )
        .assign(
            conversion_1y=lambda x: (x["goals_1y"] / x["attempts_1y"]).fillna(0.0),
        )
        .reset_index(drop=True)
    )

    last_side = pd.get_dummies(
        group["side"].shift(1),
        prefix="last_side",
        dtype="Int64",
    ).reset_index(drop=True)

    df_combined = df_combined.assign(**roll_5y, **roll_1y, **last_side)

    # Join player attributes
    meta_cols = [
        "player_id",
        "player_name",
        "player_age",
        "player_market_value",
        "player_position_id",
    ]
    df_combined = df_combined.merge(
        player_meta[meta_cols],
        on="player_id",
        how="left",
    )

    df_combined = df_combined.assign(
        market_value_known=df_combined["player_market_value"].notna().astype("Int64"),
        player_market_value=df_combined["player_market_value"].fillna(0.0),
        player_age=df_combined["player_age"].fillna(25).astype("float64"),
        player_position_id=df_combined["player_position_id"].fillna(0).astype("int64"),
    )

    # Separate training set and prediction set
    df_train = df_combined.loc[
        (~df_combined["is_prediction_dummy"])
        & (df_combined["start_at"] >= TRAIN_FLOOR)
        & df_combined["side"].notna(),
    ].dropna(subset=FEATURES).reset_index(drop=True)

    df_predict = df_combined.loc[
        df_combined["is_prediction_dummy"]
    ].reset_index(drop=True)

    return df_train, df_predict


def run_pipeline(matches_path: Path, output_path: Path) -> list[PredictionRow]:
    """
    Run feature extraction, model fitting, and save predictions.

    Overwrites output_path on every execution.
    """
    df_train, df_predict = prepare_features(matches_path)
    if df_train.empty or df_predict.empty:
        logger.warning(
            "Empty dataset. No predictions generated.",
        )
        output_path.write_text("", encoding="utf-8")
        return []

    x_train = df_train[FEATURES].astype("float64")
    y_train = df_train["side"].astype(str)

    x_pred = df_predict[FEATURES].astype("float64")

    token = os.environ.get("PRIOR_LABS_API_KEY")
    if token:
        tabpfn_client.set_access_token(token)

    clf = TabPFNClassifier(random_state=42)
    clf.fit(x_train, y_train)

    probas = clf.predict_proba(x_pred)
    classes = list(clf.classes_)

    idx_l = classes.index("left") if "left" in classes else 0
    idx_c = classes.index("center") if "center" in classes else 1
    idx_r = classes.index("right") if "right" in classes else 2

    total_penalties_series = df_train.groupby("player_id").size()

    results: list[PredictionRow] = []

    for i, row in df_predict.iterrows():
        pid = int(row["player_id"])
        name = str(row.get("player_name") or f"Player {pid}")
        team_id = int(row.get("team_id") or 0)
        p_l = float(probas[i, idx_l])
        p_c = float(probas[i, idx_c])
        p_r = float(probas[i, idx_r])
        tot = int(total_penalties_series.get(pid, 1))

        photo_url = f"https://images.fotmob.com/image_resources/playerimages/{pid}.png"
        pred = PredictionRow(
            player_id=pid,
            player_name=name,
            short_name=name,
            team_id=team_id,
            team_name=f"Team {team_id}",
            kicking_foot=None,
            photo_url=photo_url,
            total_penalties=tot,
            p_L=p_l,
            p_C=p_c,
            p_R=p_r,
        )
        results.append(pred)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(r.model_dump_json(by_alias=True) + "\n")

    return results


@click.command()
@click.option(
    "--matches-path",
    type=click.Path(exists=True, path_type=Path),
    default=MATCHES_PATH,
    help="Path to matches.jsonl input file.",
)
@click.option(
    "--output-path",
    type=click.Path(path_type=Path),
    default=PREDICTIONS_PATH,
    help="Path to predictions.jsonl output file.",
)
def main(matches_path: Path, output_path: Path) -> None:
    """Run feature extraction, TabPFN fitting, and write predictions.jsonl."""
    logger.info("Running prediction pipeline with input %s", matches_path)
    predictions = run_pipeline(matches_path, output_path)
    logger.info(
        "Successfully generated %d predictions to %s",
        len(predictions),
        output_path,
    )


if __name__ == "__main__":
    main()
