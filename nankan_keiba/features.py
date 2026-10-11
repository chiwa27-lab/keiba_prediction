"""SQLite のレース結果から予想用の特徴量を作る.

特徴量はすべて「そのレースより前に確定した結果」だけから計算する。
予想対象（出走表）の行は着順が空のまま履歴の後ろに連結し、同じ処理で特徴量化する。
"""

from __future__ import annotations

import re
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

NO_RUN_STATUSES = frozenset({"取消", "除外"})

ENTRY_COLUMNS = [
    "race_id",
    "race_date",
    "place_code",
    "race_number",
    "race_class",
    "surface",
    "distance",
    "weather",
    "track_condition",
    "num_horses",
    "finish_position",
    "finish_status",
    "bracket_number",
    "horse_number",
    "horse_id",
    "horse_name",
    "sex",
    "age",
    "weight_carried",
    "jockey_id",
    "jockey_name",
    "trainer_id",
    "trainer_name",
    "finish_time_sec",
    "popularity",
    "win_odds",
    "last_3f",
    "horse_weight",
    "horse_weight_diff",
]

CATEGORICAL_FEATURES = ["place_code", "surface", "track_condition", "weather", "sex", "class_group"]

BASE_FEATURES = [
    *CATEGORICAL_FEATURES,
    "distance",
    "num_horses",
    "race_number",
    "month",
    "bracket_number",
    "horse_number",
    "age",
    "weight_carried",
    "horse_weight",
    "horse_weight_diff",
    "horse_starts",
    "horse_win_rate",
    "horse_top3_rate",
    "days_since_last",
    "distance_change",
    "prev_rel_finish",
    "prev_time_behind",
    "prev_last3f",
    "prev_last3f_rank",
    "prev_popularity",
    "rel_finish_mean3",
    "rel_finish_mean5",
    "time_behind_mean3",
    "time_behind_mean5",
    "last3f_rank_mean3",
    "best_rel_finish5",
    "place_starts",
    "place_top3_rate",
    "distance_starts",
    "distance_top3_rate",
    "jockey_starts",
    "jockey_win_rate",
    "jockey_top3_rate",
    "trainer_starts",
    "trainer_win_rate",
    "trainer_top3_rate",
    "jockey_horse_starts",
    "rel_finish_mean5_race_rank",
    "time_behind_mean5_race_rank",
    "jockey_top3_rate_race_rank",
    "horse_top3_rate_race_rank",
]

ODDS_FEATURES = ["win_odds", "popularity", "win_odds_race_rank"]

LABELS = ["is_win", "is_top3"]

_RACE_RANK_SOURCES = ["rel_finish_mean5", "time_behind_mean5", "jockey_top3_rate", "horse_top3_rate"]


def feature_columns(use_odds: bool) -> list[str]:
    return [*BASE_FEATURES, *ODDS_FEATURES] if use_odds else list(BASE_FEATURES)


def load_history(db_path: Path) -> pd.DataFrame:
    """races と results を結合した 1 頭 1 行の履歴."""
    if not db_path.exists():
        raise FileNotFoundError(f"DB が見つかりません: {db_path}")
    columns = ", ".join(f"ra.{c}" if c in _RACE_COLUMNS else f"r.{c}" for c in ENTRY_COLUMNS)
    query = f"SELECT {columns} FROM results r JOIN races ra ON ra.race_id = r.race_id"
    with closing(sqlite3.connect(db_path)) as conn:
        return pd.read_sql_query(query, conn)


def load_win_payouts(db_path: Path) -> pd.DataFrame:
    """単勝の払戻（race_id, horse_number, payout_yen）."""
    with closing(sqlite3.connect(db_path)) as conn:
        df = pd.read_sql_query(
            "SELECT race_id, combination, payout_yen FROM payouts WHERE bet_type = '単勝'",
            conn,
        )
    df["horse_number"] = pd.to_numeric(df["combination"], errors="coerce")
    return df.dropna(subset=["horse_number"])[["race_id", "horse_number", "payout_yen"]]


_RACE_COLUMNS = frozenset(
    {
        "race_date",
        "place_code",
        "race_number",
        "race_class",
        "surface",
        "distance",
        "weather",
        "track_condition",
        "num_horses",
    }
)


def class_group(race_class: object) -> str:
    """'サラ系一般 B2B3' → 'B' のように、クラスを大まかに分類する."""
    if not isinstance(race_class, str):
        return "other"
    if m := re.search(r"([ABC])\d", race_class):
        return m.group(1)
    for key in ("2歳", "3歳", "オープン", "OP"):
        if key in race_class:
            return "OP" if key == "オープン" else key
    return "other"


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """履歴（＋予想対象）の各行に特徴量とラベルを付けて返す.

    予想対象の行は ``finish_status`` を欠損にしておく。ラベル列は出走した行だけ 0/1、
    それ以外（取消・除外・予想対象）は NaN になる。
    """
    df = frame.reindex(columns=ENTRY_COLUMNS).copy()
    df["race_date"] = pd.to_datetime(df["race_date"])
    df = df.sort_values(["race_date", "race_id", "horse_number"], kind="mergesort").reset_index(drop=True)

    for col in ("distance", "num_horses", "race_number", "bracket_number", "horse_number", "age"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in (
        "finish_position",
        "weight_carried",
        "finish_time_sec",
        "popularity",
        "win_odds",
        "last_3f",
        "horse_weight",
        "horse_weight_diff",
    ):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    status = df["finish_status"]
    ran = status.notna() & ~status.isin(NO_RUN_STATUSES)
    df["ran"] = ran.astype(int)
    field_size = df.groupby("race_id")["horse_number"].transform("count")
    df["num_horses"] = df["num_horses"].fillna(field_size)
    df["month"] = df["race_date"].dt.month
    df["class_group"] = df["race_class"].map(class_group)

    pos = df["finish_position"]
    df["is_win"] = np.where(ran, (pos == 1).astype(float), np.nan)
    df["is_top3"] = np.where(ran, (pos <= 3).astype(float), np.nan)
    # 競走中止は最下位扱い
    df["rel_finish"] = np.where(ran, (pos / df["num_horses"]).fillna(1.0), np.nan)
    df["time_behind"] = df["finish_time_sec"] - df.groupby("race_id")["finish_time_sec"].transform("min")
    df["last3f_rank"] = df.groupby("race_id")["last_3f"].rank(pct=True)
    df["ran_date"] = df["race_date"].where(ran)
    df["ran_distance"] = df["distance"].where(ran)

    _add_horse_features(df)
    _add_group_rates(df, ["horse_id", "place_code"], "place")
    _add_group_rates(df, ["horse_id", "distance"], "distance")
    _add_prior_day_rates(df, "jockey_id", "jockey")
    _add_prior_day_rates(df, "trainer_id", "trainer")
    _add_prior_day_rates(df, ["jockey_id", "horse_id"], "jockey_horse")

    for col in _RACE_RANK_SOURCES:
        ascending = col in ("rel_finish_mean5", "time_behind_mean5")
        df[f"{col}_race_rank"] = df.groupby("race_id")[col].rank(pct=True, ascending=ascending)
    df["win_odds_race_rank"] = df.groupby("race_id")["win_odds"].rank(pct=True)

    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("string").astype("category")
    return df


def _shifted(
    df: pd.DataFrame, keys: str | list[str], col: str, window: int | None = None, how: str = "mean"
) -> pd.Series:
    def per_group(s: pd.Series) -> pd.Series:
        prev = s.shift(1)
        if window is None:
            return prev.ffill()
        rolling = prev.rolling(window, min_periods=1)
        return rolling.min() if how == "min" else rolling.mean()

    return df.groupby(keys, sort=False, dropna=True)[col].transform(per_group)


def _ran_only(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].where(df["ran"] == 1)


def _add_horse_features(df: pd.DataFrame) -> None:
    key = "horse_id"
    g = df.groupby(key, sort=False, dropna=True)
    df["horse_starts"] = g["ran"].cumsum() - df["ran"]
    wins = g["is_win"].transform(lambda s: s.fillna(0).cumsum().shift(1, fill_value=0))
    top3 = g["is_top3"].transform(lambda s: s.fillna(0).cumsum().shift(1, fill_value=0))
    starts = df["horse_starts"].replace(0, np.nan)
    df["horse_win_rate"] = wins / starts
    df["horse_top3_rate"] = top3 / starts

    df["days_since_last"] = (df["race_date"] - _shifted(df, key, "ran_date")).dt.days
    df["distance_change"] = df["distance"] - _shifted(df, key, "ran_distance")

    for name in ("rel_finish", "time_behind", "last_3f", "last3f_rank", "popularity"):
        df[f"_{name}_ran"] = _ran_only(df, name)
    df["prev_rel_finish"] = _shifted(df, key, "_rel_finish_ran")
    df["prev_time_behind"] = _shifted(df, key, "_time_behind_ran")
    df["prev_last3f"] = _shifted(df, key, "_last_3f_ran")
    df["prev_last3f_rank"] = _shifted(df, key, "_last3f_rank_ran")
    df["prev_popularity"] = _shifted(df, key, "_popularity_ran")
    df["rel_finish_mean3"] = _shifted(df, key, "_rel_finish_ran", 3)
    df["rel_finish_mean5"] = _shifted(df, key, "_rel_finish_ran", 5)
    df["time_behind_mean3"] = _shifted(df, key, "_time_behind_ran", 3)
    df["time_behind_mean5"] = _shifted(df, key, "_time_behind_ran", 5)
    df["last3f_rank_mean3"] = _shifted(df, key, "_last3f_rank_ran", 3)
    df["best_rel_finish5"] = _shifted(df, key, "_rel_finish_ran", 5, how="min")
    df.drop(columns=[c for c in df.columns if c.startswith("_")], inplace=True)


def _add_group_rates(df: pd.DataFrame, keys: list[str], prefix: str) -> None:
    g = df.groupby(keys, sort=False, dropna=True)
    starts = g["ran"].cumsum() - df["ran"]
    top3 = g["is_top3"].transform(lambda s: s.fillna(0).cumsum().shift(1, fill_value=0))
    df[f"{prefix}_starts"] = starts
    df[f"{prefix}_top3_rate"] = top3 / starts.replace(0, np.nan)


def _add_prior_day_rates(df: pd.DataFrame, keys: str | list[str], prefix: str) -> None:
    """騎手・調教師などの成績を、当日より前の開催日までの累計で集計する."""
    key_list = [keys] if isinstance(keys, str) else list(keys)
    tmp = df[[*key_list, "race_date"]].copy()
    tmp["starts"] = df["ran"]
    tmp["wins"] = df["is_win"].fillna(0)
    tmp["top3"] = df["is_top3"].fillna(0)
    daily = tmp.groupby([*key_list, "race_date"], sort=True, dropna=True)[["starts", "wins", "top3"]].sum()
    prior = daily.groupby(level=list(range(len(key_list)))).cumsum() - daily
    prior = prior.reset_index()
    merged = df[[*key_list, "race_date"]].merge(prior, on=[*key_list, "race_date"], how="left")
    starts = merged["starts"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        df[f"{prefix}_starts"] = starts
        if prefix != "jockey_horse":
            df[f"{prefix}_win_rate"] = np.where(starts > 0, merged["wins"].to_numpy(dtype=float) / starts, np.nan)
            df[f"{prefix}_top3_rate"] = np.where(starts > 0, merged["top3"].to_numpy(dtype=float) / starts, np.nan)
