"""Web 画面（app.py）で使う、Streamlit に依存しない処理."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from contextlib import closing
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from .config import NANKAN_PLACES

JST = timezone(timedelta(hours=9), "JST")
MARKS = ("◎", "○", "▲", "△", "△")
ACTUAL_COLUMNS = ["race_id", "horse_number", "finish_position", "finish_status"]


def jst_today(now: datetime | None = None) -> date:
    """開催日は日本時間で決まるので、PC のタイムゾーンによらず日本時間の今日を返す."""
    return (now or datetime.now(JST)).astimezone(JST).date()


def race_label(race_id: str) -> str:
    place = race_id[4:6]
    return f"{NANKAN_PLACES.get(place, place)} {int(race_id[10:12])}R"


def last_race_date(db_path: Path) -> date | None:
    if not db_path.exists():
        return None
    with closing(sqlite3.connect(db_path)) as conn:
        try:
            row = conn.execute("SELECT MAX(race_date) FROM races").fetchone()
        except sqlite3.OperationalError:
            return None
    return date.fromisoformat(row[0]) if row and row[0] else None


def update_period(last: date | None, today: date, *, default_days: int = 365) -> tuple[date, date] | None:
    """DB に無い日から前日までの収集期間。取り込み済みなら None."""
    end = today - timedelta(days=1)
    start = last + timedelta(days=1) if last else end - timedelta(days=default_days)
    return (start, end) if start <= end else None


def load_actual_results(db_path: Path, race_ids: Iterable[str]) -> pd.DataFrame:
    """答え合わせ用に、DB に保存済みの着順を返す."""
    ids = sorted(set(race_ids))
    if not ids or not db_path.exists():
        return pd.DataFrame(columns=ACTUAL_COLUMNS)
    placeholders = ", ".join("?" * len(ids))
    with closing(sqlite3.connect(db_path)) as conn:
        return pd.read_sql_query(
            f"SELECT {', '.join(ACTUAL_COLUMNS)} FROM results WHERE race_id IN ({placeholders})",
            conn,
            params=ids,
        )


def race_table(pred: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    """1 レース分の予想を、予想順位の順に画面表示用の列名で返す。結果があれば着順も付ける."""
    df = pred.sort_values("pred_rank").reset_index(drop=True)
    out = pd.DataFrame(
        {
            "印": [MARKS[i] if i < len(MARKS) else "" for i in range(len(df))],
            "予想順位": df["pred_rank"].astype(int),
            "馬番": df["horse_number"].astype(int),
            "馬名": df["horse_name"],
            "騎手": df["jockey_name"],
            "勝率": df["win_prob"] * 100,
            "3着内率": df["top3_prob"] * 100,
            "オッズ": df["win_odds"],
            "人気": df["popularity"],
        }
    )
    race_actual = actual[actual["race_id"].isin(set(df["race_id"]))]
    if not race_actual.empty:
        by_number = race_actual.set_index(race_actual["horse_number"].astype(int))
        position = df["horse_number"].astype(int).map(by_number["finish_position"])
        status = df["horse_number"].astype(int).map(by_number["finish_status"])
        out["着順"] = [
            str(int(p)) if pd.notna(p) else (str(s) if pd.notna(s) and s else "")
            for p, s in zip(position, status, strict=True)
        ]
    return out
