"""出馬表を取得して、学習済みモデルで着順を予想する."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from .client import NotFoundError, PoliteHttpClient
from .collector import collect_pedigrees
from .config import NANKAN_PLACES, race_list_url, race_result_url, shutuba_url
from .features import ENTRY_COLUMNS, attach_pedigrees, build_features, load_history, load_pedigrees
from .model import RaceModel
from .models import RaceResult
from .parsers import ParseError, parse_result, parse_shutuba, parse_shutuba_ids
from .storage import RaceStore

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


def fetch_same_day_results(
    client: PoliteHttpClient, race_ids: Iterable[str], known_race_ids: Iterable[str] = ()
) -> list[RaceResult]:
    """予想対象と同じ日・同じ競馬場で、それより前に終わったレースの結果を取得する.

    当日の馬場傾向の特徴量に使う。結果がまだ出ていないレースに当たったら、その競馬場の以降のレースは取りに行かない。
    """
    targets = set(race_ids)
    skip = targets | set(known_race_ids)
    last_target: dict[str, int] = {}
    for rid in targets:
        last_target[rid[:10]] = max(last_target.get(rid[:10], 0), int(rid[10:12]))
    results = []
    for prefix, last in sorted(last_target.items()):
        for number in range(1, last):
            rid = f"{prefix}{number:02d}"
            if rid in skip:
                continue
            url = race_result_url(rid)
            try:
                # 確定前の速報を保存しないよう、当日の結果はキャッシュしない
                results.append(parse_result(client.get(url, use_cache=False), rid))
            except (ParseError, NotFoundError):
                break
    return results


def results_to_frame(results: Iterable[RaceResult]) -> pd.DataFrame:
    rows = [{**asdict(r.race), **asdict(entry)} for r in results for entry in r.entries]
    return pd.DataFrame(rows).reindex(columns=ENTRY_COLUMNS)


def cards_to_frame(cards: Iterable[RaceResult]) -> pd.DataFrame:
    rows = []
    for card in cards:
        race = asdict(card.race)
        for entry in card.entries:
            row = {**race, **asdict(entry)}
            row["finish_status"] = entry.finish_status or None
            rows.append(row)
    return pd.DataFrame(rows).reindex(columns=[*ENTRY_COLUMNS, "place_name", "race_name"])


def predict_cards(
    model: RaceModel,
    history: pd.DataFrame,
    cards: list[RaceResult],
    *,
    same_day: list[RaceResult] | None = None,
    pedigrees: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """取消・除外を除いた出走予定馬について、勝率・3 着内確率・予想順位を返す.

    ``same_day`` は同じ日の前のレースの確定結果、``pedigrees`` は予想対象の馬の血統（どちらも任意）。
    """
    targets = cards_to_frame(cards)
    targets = targets[targets["finish_status"].isna()]
    race_ids = set(targets["race_id"])
    frames = [history[~history["race_id"].isin(race_ids)]]
    if same_day:
        today = results_to_frame(same_day)
        frames = [frames[0][~frames[0]["race_id"].isin(set(today["race_id"]))], today]
    new_rows = pd.concat(
        [f.dropna(axis=1, how="all") for f in [*frames[1:], targets[ENTRY_COLUMNS]]], ignore_index=True
    )
    if pedigrees is not None:
        new_rows = attach_pedigrees(new_rows, pedigrees)
    combined = pd.concat([frames[0], new_rows.dropna(axis=1, how="all")], ignore_index=True)
    features = build_features(combined)
    target_features = features[features["race_id"].isin(race_ids)]
    pred = model.predict(target_features)
    info = targets[["race_id", "horse_number", "race_date", "place_name", "race_number", "race_name", "distance"]]
    extra = targets[["race_id", "horse_number", "jockey_name", "win_odds", "popularity"]]
    out = pred.merge(info, on=["race_id", "horse_number"]).merge(extra, on=["race_id", "horse_number"])
    return out[OUTPUT_COLUMNS].sort_values(["race_id", "pred_rank"]).reset_index(drop=True)


@dataclass
class PredictionRun:
    predictions: pd.DataFrame
    same_day_races: int
    fetched_pedigrees: int


def run_prediction(client: PoliteHttpClient, db_path: Path, model: RaceModel, race_ids: list[str]) -> PredictionRun:
    """出馬表・同日の前のレースの結果・未取得の血統を取得して予想する（CLI と Web 画面の共通処理）."""
    history = load_history(db_path)
    cards = fetch_cards(client, race_ids)
    same_day = fetch_same_day_results(client, race_ids, set(history["race_id"]))
    if same_day:
        logger.info("当日の前のレースの結果 %d レース分を馬場傾向に使います", len(same_day))
    horse_ids = [e.horse_id for r in [*cards, *same_day] for e in r.entries if e.horse_id]
    store = RaceStore(db_path)
    try:
        missing = store.missing_pedigrees(horse_ids)
        if missing:
            logger.info("血統が未取得の馬 %d 頭の血統表を取得します", len(missing))
            collect_pedigrees(client, store, missing)
    finally:
        store.close()
    pred = predict_cards(model, history, cards, same_day=same_day, pedigrees=load_pedigrees(db_path))
    return PredictionRun(pred, len(same_day), len(missing))


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
