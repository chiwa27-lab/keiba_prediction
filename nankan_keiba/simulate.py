"""保存済みのレース結果・払戻で、予想どおりに買った場合の回収率を答え合わせする."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from .features import build_features, load_history
from .model import RaceModel

STAKE_YEN = 100
SCRATCH_STATUSES = ("取消", "除外")
BET_TYPES: dict[str, str] = {
    "単勝": "◎",
    "複勝": "◎",
    "ワイド": "◎-○▲△ 流し（3 点）",
    "馬連": "◎-○▲△ 流し（3 点）",
    "3連複": "◎-○▲-総流し",
    "3連単": "◎→○▲→総流し",
}
ORDERED_BET_TYPES = frozenset({"単勝", "複勝", "3連単"})
SUMMARY_COLUMNS = ["券種", "買い方", "レース数", "購入点数", "投資額", "払戻額", "回収率", "的中レース数", "的中率"]
DETAIL_COLUMNS = ["race_id", "race_date", "券種", "購入点数", "投資額", "払戻額", "的中"]


@dataclass
class Simulation:
    predictions: pd.DataFrame
    summary: pd.DataFrame
    details: pd.DataFrame


def tickets(ranked: list[int]) -> dict[str, set[tuple[int, ...]]]:
    """予想順位の順に並べた出走馬の馬番から、券種ごとの買い目を作る.

    単勝・複勝は ◎、ワイド・馬連は ◎ から ○▲△ への流し、3 連複は ◎・○▲ の 2 頭軸から残り全頭、
    3 連単は 1 着 ◎・2 着 ○▲・3 着 残り全頭。組み合わせの順序を問わない券種は馬番の昇順にそろえる。
    """
    if not ranked:
        return {bet: set() for bet in BET_TYPES}
    first, partners = ranked[0], ranked[1:4]
    seconds = ranked[1:3]
    out: dict[str, set[tuple[int, ...]]] = {
        "単勝": {(first,)},
        "複勝": {(first,)},
        "ワイド": {tuple(sorted((first, p))) for p in partners},
        "馬連": {tuple(sorted((first, p))) for p in partners},
        "3連複": set(),
        "3連単": set(),
    }
    for second in seconds:
        for third in ranked[1:]:
            if third != second:
                out["3連複"].add(tuple(sorted((first, second, third))))
                out["3連単"].add((first, second, third))
    return out


def load_payouts(db_path: Path) -> pd.DataFrame:
    with closing(sqlite3.connect(db_path)) as conn:
        return pd.read_sql_query(
            "SELECT race_id, bet_type, combination, payout_yen FROM payouts WHERE payout_yen IS NOT NULL",
            conn,
        )


def payout_table(payouts: pd.DataFrame) -> dict[tuple[str, str], dict[tuple[int, ...], int]]:
    """(race_id, 券種) → {組み合わせ: 100 円あたりの払戻}."""
    table: dict[tuple[str, str], dict[tuple[int, ...], int]] = {}
    for row in payouts.itertuples(index=False):
        if row.bet_type not in BET_TYPES:
            continue
        try:
            numbers = tuple(int(x) for x in str(row.combination).split("-"))
        except ValueError:
            continue
        key = numbers if row.bet_type in ORDERED_BET_TYPES else tuple(sorted(numbers))
        table.setdefault((str(row.race_id), str(row.bet_type)), {})[key] = int(row.payout_yen)
    return table


def settle(predictions: pd.DataFrame, payouts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """予想（race_id・horse_number・pred_rank）と払戻から、レース別・券種別の収支と券種別の集計を返す.

    発売のなかった券種（払戻が保存されていない券種）はそのレースでは買わなかったものとする。
    """
    table = payout_table(payouts)
    rows: list[dict[str, object]] = []
    for race_id, race in predictions.sort_values(["race_id", "pred_rank"]).groupby("race_id", sort=True):
        ranked = [int(h) for h in race["horse_number"]]
        race_date = race["race_date"].iloc[0] if "race_date" in race else None
        for bet, combos in tickets(ranked).items():
            paid = table.get((str(race_id), bet))
            if not paid or not combos:
                continue
            hits = sorted(c for c in combos if c in paid)
            rows.append(
                {
                    "race_id": race_id,
                    "race_date": race_date,
                    "券種": bet,
                    "購入点数": len(combos),
                    "投資額": len(combos) * STAKE_YEN,
                    "払戻額": sum(paid[c] for c in hits) * STAKE_YEN // 100,
                    "的中": " / ".join("-".join(str(n) for n in c) for c in hits),
                }
            )
    details = pd.DataFrame(rows, columns=DETAIL_COLUMNS)
    return summarize(details), details


def summarize(details: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bet, label in BET_TYPES.items():
        d = details[details["券種"] == bet]
        if d.empty:
            continue
        rows.append(_summary_row(bet, label, d))
    if rows:
        total = _summary_row("合計", "", details)
        rows.append({**total, "的中レース数": None, "的中率": None})
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _summary_row(bet: str, label: str, d: pd.DataFrame) -> dict[str, object]:
    cost, ret = int(d["投資額"].sum()), int(d["払戻額"].sum())
    hit = int((d["的中"] != "").sum())
    return {
        "券種": bet,
        "買い方": label,
        "レース数": d["race_id"].nunique(),
        "購入点数": int(d["購入点数"].sum()),
        "投資額": cost,
        "払戻額": ret,
        "回収率": ret / cost * 100 if cost else 0.0,
        "的中レース数": hit,
        "的中率": hit / len(d) * 100,
    }


def predict_saved_races(model: RaceModel, history: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """DB に保存済みのレースを、取消・除外を除いた出走馬で予想し直す.

    特徴量は予想対象より前の情報だけで作る（同じ日の前のレースの結果は使う＝発走直前の予想と同じ条件）。
    オッズありモデルは確定オッズを使う。
    """
    features = build_features(history)
    day = features["race_date"].dt.date
    target = features[(day >= start) & (day <= end) & ~features["finish_status"].isin(SCRATCH_STATUSES)]
    if target.empty:
        return pd.DataFrame(columns=["race_id", "race_date", "horse_number", "horse_name", "win_prob", "pred_rank"])
    pred = model.predict(target)
    dates = target[["race_id", "race_date"]].drop_duplicates("race_id")
    pred = pred.merge(dates.assign(race_date=dates["race_date"].dt.strftime("%Y-%m-%d")), on="race_id")
    return pred.sort_values(["race_id", "pred_rank"]).reset_index(drop=True)


def run_simulation(db_path: Path, model: RaceModel, start: date, end: date) -> Simulation:
    pred = predict_saved_races(model, load_history(db_path), start, end)
    summary, details = settle(pred, load_payouts(db_path))
    return Simulation(pred, summary, details)


def leaks_training(model: RaceModel, end: date) -> bool:
    """期間がモデルの学習データに含まれている（実際より成績が良く出る）か."""
    trained_until = model.meta.get("trained_until")
    return isinstance(trained_until, str) and date.fromisoformat(trained_until) >= end


def format_summary(summary: pd.DataFrame) -> str:
    lines = []
    for r in summary.itertuples(index=False):
        lines.append(
            f"{r[0]:<4} {r[1]:<20} {r[2]:>4}R {r[3]:>6}点 投資 {r[4]:>9,}円 払戻 {r[5]:>9,}円"
            f" 回収率 {r[6]:6.1f}%" + ("" if pd.isna(r[7]) else f" 的中 {int(r[7])}R（{r[8]:.1f}%）")
        )
    return "\n".join(lines)
