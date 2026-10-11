"""出馬表を取得して、学習済みモデルで着順を予想する."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import asdict

import pandas as pd

from .client import PoliteHttpClient
from .config import NANKAN_PLACES, race_list_url, shutuba_url
from .features import ENTRY_COLUMNS, build_features
from .model import RaceModel
from .models import RaceResult
from .parsers import parse_shutuba, parse_shutuba_ids

logger = logging.getLogger(__name__)

OUTPUT_COLUMNS = [
    "race_id",
    "race_date",
    "place_name",
    "race_number",
    "race_name",
    "distance",
    "pred_rank",
    "horse_number",
    "horse_name",
    "jockey_name",
    "win_prob",
    "top3_prob",
    "win_odds",
    "popularity",
]


def find_race_ids(client: PoliteHttpClient, kaisai_date: str, place_codes: Iterable[str] = NANKAN_PLACES) -> list[str]:
    html = client.get(race_list_url(kaisai_date), use_cache=False)
    return parse_shutuba_ids(html, kaisai_date, place_codes)


def fetch_cards(client: PoliteHttpClient, race_ids: Iterable[str]) -> list[RaceResult]:
    # 出走取消や馬体重の発表で内容が変わるため、出馬表はキャッシュしない
    return [parse_shutuba(client.get(shutuba_url(rid), use_cache=False), rid) for rid in race_ids]


def cards_to_frame(cards: Iterable[RaceResult]) -> pd.DataFrame:
    rows = []
    for card in cards:
        race = asdict(card.race)
        for entry in card.entries:
            row = {**race, **asdict(entry)}
            row["finish_status"] = entry.finish_status or None
            rows.append(row)
    return pd.DataFrame(rows).reindex(columns=[*ENTRY_COLUMNS, "place_name", "race_name"])


def predict_cards(model: RaceModel, history: pd.DataFrame, cards: list[RaceResult]) -> pd.DataFrame:
    """取消・除外を除いた出走予定馬について、勝率・3 着内確率・予想順位を返す."""
    targets = cards_to_frame(cards)
    targets = targets[targets["finish_status"].isna()]
    race_ids = set(targets["race_id"])
    past = history[~history["race_id"].isin(race_ids)]
    combined = pd.concat([past, targets[ENTRY_COLUMNS].dropna(axis=1, how="all")], ignore_index=True)
    features = build_features(combined)
    target_features = features[features["race_id"].isin(race_ids)]
    pred = model.predict(target_features)
    info = targets[["race_id", "horse_number", "race_date", "place_name", "race_number", "race_name", "distance"]]
    extra = targets[["race_id", "horse_number", "jockey_name", "win_odds", "popularity"]]
    out = pred.merge(info, on=["race_id", "horse_number"]).merge(extra, on=["race_id", "horse_number"])
    return out[OUTPUT_COLUMNS].sort_values(["race_id", "pred_rank"]).reset_index(drop=True)


def format_predictions(pred: pd.DataFrame) -> str:
    lines: list[str] = []
    for _, race in pred.groupby("race_id", sort=True):
        head = race.iloc[0]
        lines.append(f"■ {head['race_date']} {head['place_name']}{int(head['race_number'])}R {head['race_name']}")
        for _, r in race.iterrows():
            odds = "" if pd.isna(r["win_odds"]) else f"  予想オッズ {r['win_odds']:.1f}"
            lines.append(
                f"  {int(r['pred_rank']):>2}位  {int(r['horse_number']):>2} {r['horse_name']:<12}"
                f" 勝率 {r['win_prob']:6.1%}  3着内 {r['top3_prob']:6.1%}{odds}"
            )
    return "\n".join(lines)
