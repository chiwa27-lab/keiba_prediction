"""SPAT4 投票の買い目作成・設定・記録（画面操作は spat4_browser）.

加入者番号・利用者 ID はこの PC の data/ にだけ保存する（git 管理外）。暗証番号は保存しない。
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from .simulate import BET_TYPES, ORDERED_BET_TYPES, STAKE_YEN, tickets

DEFAULT_DAILY_LIMIT_YEN = 5000
TICKET_COLUMNS = ["race_id", "券種", "買い目", "金額"]

USER_ID_MIN_LEN = 8
USER_ID_MAX_LEN = 32


@dataclass
class Spat4Settings:
    member_number: str = ""
    user_id: str = ""
    daily_limit_yen: int = DEFAULT_DAILY_LIMIT_YEN

    @property
    def complete(self) -> bool:
        return bool(self.member_number and self.user_id)


def settings_path(data_dir: Path) -> Path:
    return data_dir / "spat4_settings.json"


def load_settings(data_dir: Path) -> Spat4Settings:
    path = settings_path(data_dir)
    if not path.exists():
        return Spat4Settings()
    raw = json.loads(path.read_text(encoding="utf-8"))
    return Spat4Settings(
        member_number=str(raw.get("member_number", "")),
        user_id=str(raw.get("user_id", "")),
        daily_limit_yen=int(raw.get("daily_limit_yen", DEFAULT_DAILY_LIMIT_YEN)),
    )


def save_settings(data_dir: Path, settings: Spat4Settings) -> None:
    if not settings.member_number.isdigit():
        raise ValueError("加入者番号は数字で入力してください")
    user_id = settings.user_id
    if not (
        user_id.isascii()
        and user_id.isprintable()
        and " " not in user_id
        and USER_ID_MIN_LEN <= len(user_id) <= USER_ID_MAX_LEN
    ):
        raise ValueError(f"利用者IDは半角英数記号の {USER_ID_MIN_LEN}〜{USER_ID_MAX_LEN} 文字で入力してください")
    if settings.daily_limit_yen < STAKE_YEN:
        raise ValueError(f"1日の上限は {STAKE_YEN} 円以上にしてください")
    data_dir.mkdir(parents=True, exist_ok=True)
    path = settings_path(data_dir)
    path.write_text(json.dumps(asdict(settings), ensure_ascii=False), encoding="utf-8")
    os.chmod(path, 0o600)


def format_combination(bet_type: str, combination: tuple[int, ...]) -> str:
    sep = "→" if bet_type in ORDERED_BET_TYPES and len(combination) > 1 else "-"
    return sep.join(str(n) for n in combination)


def build_tickets(predictions: pd.DataFrame, race_ids: list[str], bet_types: list[str]) -> pd.DataFrame:
    """選んだレース・券種について、回収率タブと同じ買い方の買い目を 1 点 100 円で並べる."""
    rows = []
    for race_id in race_ids:
        race = predictions[predictions["race_id"] == race_id].sort_values("pred_rank")
        ranked = [int(n) for n in race["horse_number"]]
        by_type = tickets(ranked)
        for bet_type in BET_TYPES:
            if bet_type not in bet_types:
                continue
            for combo in sorted(by_type[bet_type]):
                rows.append([race_id, bet_type, format_combination(bet_type, combo), STAKE_YEN])
    return pd.DataFrame(rows, columns=TICKET_COLUMNS)


def log_path(data_dir: Path) -> Path:
    return data_dir / "spat4_votes.sqlite"


def _connect(data_dir: Path) -> sqlite3.Connection:
    data_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(log_path(data_dir))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS votes ("
        "entered_at TEXT NOT NULL, vote_date TEXT NOT NULL, race_id TEXT NOT NULL, "
        "bet_type TEXT NOT NULL, combination TEXT NOT NULL, amount_yen INTEGER NOT NULL)"
    )
    return conn


def load_log(data_dir: Path, vote_date: date) -> pd.DataFrame:
    if not log_path(data_dir).exists():
        return pd.DataFrame(columns=["entered_at", "vote_date", "race_id", "bet_type", "combination", "amount_yen"])
    with closing(_connect(data_dir)) as conn:
        return pd.read_sql_query("SELECT * FROM votes WHERE vote_date = ?", conn, params=[vote_date.isoformat()])


def record(data_dir: Path, ticket_rows: pd.DataFrame, vote_date: date, now: datetime | None = None) -> None:
    """SPAT4 に入力した買い目を記録する（最後の投票ボタンを押したかは分からないため「入力済み」の記録）."""
    stamp = (now or datetime.now()).isoformat(timespec="seconds")
    with closing(_connect(data_dir)) as conn, conn:
        conn.executemany(
            "INSERT INTO votes VALUES (?, ?, ?, ?, ?, ?)",
            [
                (stamp, vote_date.isoformat(), r.race_id, r.券種, r.買い目, int(r.金額))
                for r in ticket_rows.itertuples(index=False)
            ],
        )


@dataclass
class VoteCheck:
    new_tickets: pd.DataFrame
    duplicates: pd.DataFrame
    total_yen: int
    used_today_yen: int
    limit_yen: int

    @property
    def over_limit(self) -> bool:
        return self.used_today_yen + self.total_yen > self.limit_yen


def check(ticket_rows: pd.DataFrame, log: pd.DataFrame, limit_yen: int) -> VoteCheck:
    """同じ日にすでに入力した買い目を除き、1 日の上限を超えないか確かめる."""
    done = set(zip(log["race_id"], log["bet_type"], log["combination"], strict=True))
    keys = list(zip(ticket_rows["race_id"], ticket_rows["券種"], ticket_rows["買い目"], strict=True))
    dup_mask = pd.Series([k in done for k in keys], index=ticket_rows.index, dtype=bool)
    new = ticket_rows[~dup_mask].reset_index(drop=True)
    return VoteCheck(
        new_tickets=new,
        duplicates=ticket_rows[dup_mask].reset_index(drop=True),
        total_yen=int(new["金額"].sum()),
        used_today_yen=int(log["amount_yen"].sum()),
        limit_yen=limit_yen,
    )
